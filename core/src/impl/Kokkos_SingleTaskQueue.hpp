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

#ifndef KOKKOS_IMPL_SINGLETASKQUEUE_HPP
#define KOKKOS_IMPL_SINGLETASKQUEUE_HPP

#include <Kokkos_Macros.hpp>
#if defined( KOKKOS_ENABLE_TASKDAG )


#include <Kokkos_TaskScheduler_fwd.hpp>
#include <Kokkos_Core_fwd.hpp>

#include <Kokkos_MemoryPool.hpp>

#include <impl/Kokkos_TaskBase.hpp>
#include <impl/Kokkos_TaskResult.hpp>

#include <impl/Kokkos_TaskQueueMemoryManager.hpp>
#include <impl/Kokkos_Memory_Fence.hpp>
#include <impl/Kokkos_Atomic_Increment.hpp>
#include <impl/Kokkos_OptionalRef.hpp>
#include <impl/Kokkos_LIFO.hpp>

#include <string>
#include <typeinfo>
#include <stdexcept>


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

template <
  class ExecSpace,
  class MemorySpace,
  class TaskQueueTraits
>
class SingleTaskQueue
  : public TaskQueueMemoryManager<ExecSpace, MemorySpace>
{
private:

  using base_t = TaskQueueMemoryManager<ExecSpace, MemorySpace>;

public:

  using task_queue_type = SingleTaskQueue; // mark as task_queue concept
  using execution_space = ExecSpace;
  using memory_space = MemorySpace;
  using task_queue_traits = TaskQueueTraits;
  using task_base_type = TaskNode<TaskQueueTraits>;
  using ready_queue_type = typename TaskQueueTraits::template ready_queue_type<task_base_type>;
  using device_type = Kokkos::Device<execution_space, memory_space>;
  using memory_pool = Kokkos::MemoryPool<device_type>;

  using runnable_task_base_type = RunnableTaskBase<TaskQueueTraits>;

  template <class Functor, class Scheduler>
    // requires TaskScheduler<Scheduler> && TaskFunctor<Functor>
  using runnable_task_type = RunnableTask<
    task_queue_traits, Scheduler, typename Functor::value_type, Functor
  >;

  using aggregate_task_type = AggregateTask<TaskQueueTraits>;

public:

  // Number of allowed priorities
  static constexpr int NumQueue = 3;

private:

  ready_queue_type m_ready_queues[NumQueue][2];
  int32_t m_ready_count = 0;

  struct _schedule_waiting_tasks_operation {
    SingleTaskQueue& m_queue;
    KOKKOS_INLINE_FUNCTION
    void operator()(task_base_type& task) const noexcept
    {
      if(task.is_runnable()) // KOKKOS_LIKELY
      {
        m_queue.schedule_runnable(task.as_runnable_task());
      }
      else {
        m_queue.schedule_aggregate(task.as_aggregate());
      }
    }
  };

  KOKKOS_FUNCTION
  void _complete_finished_task(task_base_type& task) {
    // This would be more readable with a lambda, but that comes with
    // all the baggage associated with a lambda (compilation times, bugs with
    // nvcc, etc.), so we'll use a simple little helper functor here.
    task.consume_wait_queue(_schedule_waiting_tasks_operation{*this});
  }

public:

  SingleTaskQueue() = delete;
  SingleTaskQueue(SingleTaskQueue const&) = delete;
  SingleTaskQueue(SingleTaskQueue&&) = delete;
  SingleTaskQueue& operator=(SingleTaskQueue const&) = delete;
  SingleTaskQueue& operator=(SingleTaskQueue&&) = delete;

  ~SingleTaskQueue() {
    for(int i_priority = 0; i_priority < NumQueue; ++i_priority) {
      KOKKOS_EXPECTS(m_ready_queues[i_priority][TaskTeam].empty());
      KOKKOS_EXPECTS(m_ready_queues[i_priority][TaskSingle].empty());
    }
    KOKKOS_EXPECTS(m_ready_count == 0);
  }

  explicit
  SingleTaskQueue(memory_pool const& arg_memory_pool)
    : base_t(arg_memory_pool),
      m_ready_count(0)
  { }


  KOKKOS_FUNCTION
  void
  schedule_runnable(runnable_task_base_type& task) {

    bool task_is_ready = true;

    if(task.has_predecessor()) {
      // save the predecessor into a local variable, then clear it from the
      // task before adding it to the wait queue of the predecessor
      // (We have exclusive access to the task's predecessor, so we don't need
      // to do this atomically)
      // TODO document that we expect exclusive access to `task` in this function
      auto& predecessor = task.get_predecessor();
      // This needs a load/store fence here, technically
      // making this a release store would also do this
      task.clear_predecessor();

      // TODO remove this fence in favor of memory orders
      Kokkos::memory_fence(); // for now

      // Try to add the task to the predecessor's waiting queue.  If it fails,
      // the predecessor is already done
      bool predecessor_not_ready = predecessor.try_add_waiting(task);

      // If the predecessor is not done, then task is not ready
      task_is_ready = not predecessor_not_ready;

      if(task.get_respawn_flag()) {
        // Reference count for predecessor was incremented when
        // respawn called set_dependency()
        // so that if predecessor completed prior to the
        // above try_add_waiting(), predecessor would not be destroyed.
        // predecessor reference count can now be decremented,
        // which may deallocate it.
        bool should_delete = predecessor.decrement_and_check_reference_count();
        if(should_delete) {
          // TODO better encapsulation of this!
          this->deallocate(predecessor);
        }
      }
      // Note! predecessor may be destroyed at this point, so don't add anything
      // here
    }

    // clear the respawn flag, since we handled the respawn (if any) here
    task.set_respawn_flag(false);

    // Put it in the appropriate ready queue if it's ready
    if(task_is_ready) {
      // Increment the ready count
      Kokkos::atomic_increment(&m_ready_count);
      // and enqueue the task
      m_ready_queues[int(task.get_priority())][int(task.get_task_type())].push(task);
    }

  }

  KOKKOS_FUNCTION
  void
  schedule_aggregate(aggregate_task_type& task) {
    bool incomplete_dependence_found = false;

    auto predecessor_ptrs = task.aggregate_dependences();

    for(int i = task.dependence_count() - 1; i >= 0 && !incomplete_dependence_found; --i) {
      // swap the task pointer onto the stack; doesn't need to be done
      // atomically because we have exclusive access to the aggregate here
      // TODO document that we expect exclusive access to `task` in this function
      auto pred_ptr = predecessor_ptrs[i];
      predecessor_ptrs[i] = nullptr;

      // if a previous scheduling operation hasn't already set the predecessorendence
      // to nullptr, try to enqueue the aggregate into the predecessorendence's waiting
      // queue
      if(pred_ptr != nullptr) {
        // If adding task to the waiting queue succeeds, the predecessor is not
        // complete
        bool pred_not_ready = pred_ptr->try_add_waiting(task);

        // we found an incomplete dependence, so we can't make task's successors
        // ready yet
        incomplete_dependence_found = pred_not_ready;

        // the reference count for the predecessor was incremented when we put
        // it into the predecessor list, so decrement it here
        bool should_delete = pred_ptr->decrement_and_check_reference_count();
        if(should_delete) {
          // TODO better encapsulation of this!
          // TODO this branch should never be taken, so figure out if/why there's a redundant ref count increment somewhere
          this->deallocate(*pred_ptr);
        }
      }
    }

    if(not incomplete_dependence_found) {
      // all of the predecessors were completed, so we can complete `task`
      complete(task);
    }
    // Note!! task may have been deleted at this point, so don't add anything here!
  }

  KOKKOS_FUNCTION
  void
  complete(runnable_task_base_type& task) {
    if(task.get_respawn_flag()) {
      schedule_runnable(task);
    }
    else {
      _complete_finished_task(task);
      bool should_delete = task.decrement_and_check_reference_count();
      if(should_delete) {
        this->deallocate(task);
      }
    }
    // A runnable task was popped from a ready queue finished executing.
    // If respawned into a ready queue then the ready count was incremented
    // so decrement whether respawned or not.  If finished, all of the
    // tasks waiting on this have been enqueued (either in the ready queue
    // or the next waiting queue, in the case of an aggregate), and the
    // ready count has been incremented for each of those, preventing
    // quiescence.  Thus, it's safe to decrement the ready count here.
    // TODO memory order? (probably relaxed)
    Kokkos::atomic_decrement(&m_ready_count);
  }

  KOKKOS_FUNCTION
  void
  complete(aggregate_task_type& task) {
    // TODO old code has a ifndef __HCC_ACCELERATOR__ here; figure out why
    _complete_finished_task(task);
    bool should_delete = task.decrement_and_check_reference_count();
    if(should_delete) {
      this->deallocate(task);
    }
  }

  KOKKOS_INLINE_FUNCTION
  bool is_done() const noexcept {
    // TODO Memory order, instead of volatile
    return (*(volatile int*)(&m_ready_count)) == 0;
  }

  KOKKOS_FUNCTION
  OptionalRef<task_base_type>
  pop_ready_task()
  {
    OptionalRef<task_base_type> return_value;
    // always loop in order of priority first, then prefer team tasks over single tasks
    for(int i_priority = 0; i_priority < NumQueue; ++i_priority) {

      // Check for a team task with this priority
      return_value = m_ready_queues[i_priority][TaskTeam].pop();
      if(return_value) return return_value;

      // Check for a single task with this priority
      return_value = m_ready_queues[i_priority][TaskSingle].pop();
      if(return_value) return return_value;

    }
    // if nothing was found, return a default-constructed (empty) OptionalRef
    return return_value;
  }

};

} /* namespace Impl */
} /* namespace Kokkos */

////////////////////////////////////////////////////////////////////////////////
// END OLD CODE
////////////////////////////////////////////////////////////////////////////////


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #if defined( KOKKOS_ENABLE_TASKDAG ) */
#endif /* #ifndef KOKKOS_IMPL_SINGLETASKQUEUE_HPP */

