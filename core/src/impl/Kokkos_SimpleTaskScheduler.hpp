/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_SIMPLETASKSCHEDULER_HPP
#define KOKKOS_SIMPLETASKSCHEDULER_HPP

//----------------------------------------------------------------------------

#include <Kokkos_Macros.hpp>
#if defined( KOKKOS_ENABLE_TASKDAG )

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_TaskScheduler_fwd.hpp>
//----------------------------------------------------------------------------

#include <Kokkos_MemoryPool.hpp>
#include <impl/Kokkos_Tags.hpp>

#include <Kokkos_Future.hpp>
#include <impl/Kokkos_TaskQueue.hpp>
#include <impl/Kokkos_SingleTaskQueue.hpp>
#include <impl/Kokkos_TaskQueueMultiple.hpp>
#include <impl/Kokkos_TaskPolicyData.hpp>
#include <impl/Kokkos_TaskTeamMember.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

namespace Impl {

// TODO move this
template <class T>
struct DefaultDestroy {
  T* managed_object;
  KOKKOS_FUNCTION
  void destroy_shared_allocation() {
    managed_object->~T();
  }
};

} // end namespace Impl

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

template <class ExecSpace, class QueueType>
  // requires ExecutionSpace<ExecSpace> && TaskQueue<QueueType>
class SimpleTaskScheduler : public Impl::TaskSchedulerBase {
public:

  using scheduler_type = SimpleTaskScheduler; // tag as scheduler concept
  using execution_space = ExecSpace;
  using task_queue_type = QueueType;
  using memory_space = typename task_queue_type::memory_space;
  using memory_pool = typename task_queue_type::memory_pool;

  using specialization = Impl::TaskQueueSpecialization<SimpleTaskScheduler>;

  using member_type = typename specialization::member_type;

  template <class Functor>
  using runnable_task_type = typename QueueType::template runnable_task_type<Functor, SimpleTaskScheduler>;

  using task_base_type = typename task_queue_type::task_base_type;
  using runnable_task_base_type = typename task_queue_type::runnable_task_base_type;

private:

  template <typename, typename>
  friend class BasicFuture;

  using track_type = Kokkos::Impl::SharedAllocationTracker;

  track_type m_track;
  task_queue_type* m_queue = nullptr;

public:

  //----------------------------------------------------------------------------
  // <editor-fold desc="Constructors, destructor, and assignment"> {{{2

  explicit
  SimpleTaskScheduler(memory_pool const& pool)
  {
    // TODO better encapsulation of this pattern
    using record_type = Impl::SharedAllocationRecord<
      memory_space, Impl::DefaultDestroy<task_queue_type>
    >;

    // Allocate space for the
    auto* record = record_type::allocate(
      memory_space(), "TaskQueue", sizeof(task_queue_type)
    );
    m_queue = new (record->data()) task_queue_type(pool);
    record->m_destroy.managed_object = m_queue;
    m_track.assign_allocated_record_to_uninitialized(record);
  }

  SimpleTaskScheduler(
    memory_space const & arg_memory_space,
    size_t const mempool_capacity,
    unsigned const mempool_min_block_size, // = 1u << 6
    unsigned const mempool_max_block_size, // = 1u << 10
    unsigned const mempool_superblock_size // = 1u << 12
  ) : SimpleTaskScheduler(
    memory_pool(
      arg_memory_space, mempool_capacity, mempool_min_block_size,
      mempool_max_block_size, mempool_superblock_size
    )
  )
  { /* forwarding ctor, must be empty */ }

  // </editor-fold> end Constructors, destructor, and assignment }}}2
  //----------------------------------------------------------------------------

  // Note that this is an expression of shallow constness
  KOKKOS_INLINE_FUNCTION
  task_queue_type& queue() const
  {
    KOKKOS_EXPECTS(m_queue != nullptr);
    return *m_queue;
  }

  KOKKOS_INLINE_FUNCTION
  SimpleTaskScheduler const&
  get_team_scheduler(int rank_in_league) const noexcept { return *this; }

  // TODO Refactor to make this a member function and remove the queue pointer from task
  template <
    class TaskPolicy, // instance of TaskPolicyData, for now
    class FunctorType
  >
  KOKKOS_FUNCTION
  static
  Kokkos::BasicFuture<typename FunctorType::value_type, scheduler_type>
  spawn(
    TaskPolicy&& policy,
    typename runnable_task_base_type::function_type apply_function_ptr,
    typename runnable_task_base_type::destroy_type destroy_function_ptr,
    FunctorType&& functor
  )
  {
    using value_type = typename FunctorType::value_type;
    using future_type = BasicFuture< value_type , scheduler_type > ;
    using task_type = typename task_queue_type::template runnable_task_type<
      FunctorType, scheduler_type
    >;

    task_queue_type* queue_ptr = nullptr;

    if(policy.m_scheduler != nullptr) {
      auto& scheduler = *static_cast<scheduler_type const*>(policy.m_scheduler);
      queue_ptr = &scheduler.queue();
    }
    else {
      queue_ptr = static_cast<task_queue_type*>(
        policy.m_dependence.m_task->ready_queue_base_ptr()
      );
    }

    KOKKOS_ASSERT(queue_ptr != nullptr);

    auto& queue = *queue_ptr;

    future_type rv;

    // Reference count starts at two:
    //   +1 for the matching decrement when task is complete
    //   +1 for the future
    auto* runnable_task = queue.template allocate_and_construct<task_type>(
      /* functor = */ std::forward<FunctorType>(functor),
      /* apply_function_ptr = */ apply_function_ptr,
      /* task_type = */ Impl::TaskType(policy.m_task_type),
      /* priority = */ policy.m_priority,
      /* queue_base = */ &queue,
      /* initial_reference_count = */ 2
    );

    if(policy.m_scheduler == nullptr) { // Another way of saying policy was constructed with a future
      runnable_task->set_predecessor(*policy.m_dependence.m_task);
    }

    rv = future_type(runnable_task);

    Kokkos::memory_fence(); // fence to ensure dependent stores are visible

    queue.schedule_runnable(std::move(*runnable_task));
    // note that task may be already completed even here, so don't touch it again

    return rv;
  }

  template <class FunctorType, class ValueType, class Scheduler>
  KOKKOS_FUNCTION
  static void
  respawn(
    FunctorType* functor,
    BasicFuture<ValueType, Scheduler> const& predecessor,
    TaskPriority priority = TaskPriority::Regular
  ) {
    using task_type = typename task_queue_type::template runnable_task_type<
      FunctorType, scheduler_type
    >;

    auto& task = *static_cast<task_type*>(functor);
    task.set_priority(priority);
    task.set_predecessor(*predecessor.m_task);
    task.set_respawn_flag(true);
  }

  template <class FunctorType, class ValueType, class Scheduler>
  KOKKOS_FUNCTION
  static void
  respawn(
    FunctorType* functor,
    scheduler_type const&,
    TaskPriority priority = TaskPriority::Regular
  ) {
    using task_type = typename task_queue_type::template runnable_task_type<
      FunctorType, scheduler_type
    >;

    auto& task = *static_cast<task_type*>(functor);
    task.set_priority(priority);
    KOKKOS_ASSERT(not task.has_predecessor());
    task.set_respawn_flag(true);
  }


  template <class ValueType>
  KOKKOS_FUNCTION
  static BasicFuture<void, scheduler_type>
  when_all(BasicFuture<ValueType, scheduler_type> const predecessors[], int n_predecessors) {

    using future_type = BasicFuture<void, scheduler_type>;

    using task_type = typename task_queue_type::aggregate_task_type;

    future_type rv;

    if(n_predecessors > 0) {
      task_queue_type* queue_ptr = nullptr;

      // Loop over the predecessors to find the queue and increment the reference
      // counts
      for(int i_pred = 0; i_pred < n_predecessors; ++i_pred) {

        auto* predecessor_task_ptr = predecessors[i_pred].m_task;

        if(predecessor_task_ptr != nullptr) { // TODO figure out when this is allowed to be nullptr
          // Increment reference count to track subsequent assignment.
          // TODO figure out if this reference count increment is necessary
          predecessor_task_ptr->increment_reference_count();

          auto* pred_queue_ptr = static_cast<task_queue_type*>(
            predecessor_task_ptr->ready_queue_base_ptr()
          );

          if(queue_ptr == nullptr) {
            queue_ptr = pred_queue_ptr;
          }
          else {
            KOKKOS_ASSERT(queue_ptr == pred_queue_ptr && "Queue mismatch in when_all");
          }
        }

      } // end loop over predecessors

      // This only represents a non-ready future if at least one of the predecessors
      // has a task (and thus, a queue)(and thus, a queue)(and thus, a queue)(and thus, a queue)(and thus, a queue)(and thus, a queue)(and thus, a queue)(and thus, a queue)(and thus, a queue)
      if(queue_ptr != nullptr) {
        auto& queue = *queue_ptr;

        auto* aggregate_task = queue.template allocate_and_construct_with_vla_emulation<
          task_type, task_base_type*
        >(
          /* n_vla_entries = */ n_predecessors,
          /* aggregate_predecessor_count = */ n_predecessors,
          /* queue_base = */ &queue,
          /* initial_reference_count = */ 2
        );

        rv = future_type(aggregate_task);

        // get the VLA-emulated storage for the aggregate predecessors
        auto* predecessor_ptr_storage = aggregate_task->aggregate_dependences();

        for(int i_pred = 0; i_pred < n_predecessors; ++i_pred) {
          predecessor_ptr_storage[i_pred] = predecessors[i_pred].m_task;
        }

        Kokkos::memory_fence(); // we're touching very questionable memory, so be sure to fence

        queue.schedule_aggregate(std::move(*aggregate_task));
        // the aggregate may be processed at any time, so don't touch it after this
      }
    }

    return rv;
  }

  template <class F>
  KOKKOS_FUNCTION
  BasicFuture<void, scheduler_type>
  when_all(int n_calls, F&& func)
  {
    using future_type = BasicFuture<void, scheduler_type>;
    // later this should be std::invoke_result_t
    using generated_type = decltype(func(0));
    using task_type = typename task_queue_type::aggregate_task_type;

    // TODO check for scheduler compatibility
    static_assert(is_future<generated_type>::value,
      "when_all function must return a Kokkos::Future"
    );

    auto* aggregate_task = m_queue->template allocate_and_construct_with_vla_emulation<
      task_type, task_base_type*
    >(
      /* n_vla_entries = */ n_calls,
      /* aggregate_predecessor_count = */ n_calls,
      /* queue_base = */ m_queue,
      /* initial_reference_count = */ 2
    );

    auto rv = future_type(aggregate_task);

    auto* predecessor_ptr_storage = aggregate_task->aggregate_dependences();

    for(int i_call = 0; i_call < n_calls; ++i_call) {

      auto generated_future = func(i_call);

      if(generated_future.m_task != nullptr) {
        generated_future.m_task->increment_reference_count();
        predecessor_ptr_storage[i_call] = generated_future.m_task;

        KOKKOS_ASSERT(m_queue == generated_future.m_task->ready_queue_base_ptr()
          && "Queue mismatch in when_all"
        );
      }

    }

    Kokkos::memory_fence();

    m_queue->schedule_aggregate(std::move(*aggregate_task));
    // This could complete at any moment, so don't touch anything after this

    return rv;
  }

};


template<class ExecSpace, class QueueType>
inline
void wait(SimpleTaskScheduler<ExecSpace, QueueType> const& scheduler)
{
  using scheduler_type = SimpleTaskScheduler<ExecSpace, QueueType>;
  scheduler_type::specialization::execute(scheduler);
}

} // namespace Kokkos

//----------------------------------------------------------------------------
//---------------------------------------------------------------------------#endif /* #if defined( KOKKOS_ENABLE_TASKDAG ) */

#endif /* #if defined( KOKKOS_ENABLE_TASKDAG ) */
#endif /* #ifndef KOKKOS_SIMPLETASKSCHEDULER_HPP */

