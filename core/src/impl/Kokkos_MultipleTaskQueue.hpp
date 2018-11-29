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

#ifndef KOKKOS_IMPL_MULTIPLETASKQUEUE_HPP
#define KOKKOS_IMPL_MULTIPLETASKQUEUE_HPP

#include <Kokkos_Macros.hpp>
#if defined( KOKKOS_ENABLE_TASKDAG )


#include <Kokkos_TaskScheduler_fwd.hpp>
#include <Kokkos_Core_fwd.hpp>

#include <Kokkos_MemoryPool.hpp>

#include <impl/Kokkos_TaskBase.hpp>
#include <impl/Kokkos_TaskResult.hpp>

#include <impl/Kokkos_TaskQueueMemoryManager.hpp>
#include <impl/Kokkos_TaskQueueCommon.hpp>
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

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

template <
  class ExecSpace,
  class MemorySpace,
  class TaskQueueTraits
>
class MultipleTaskQueueTeamQueue;

template <
  class ExecSpace,
  class MemorySpace,
  class TaskQueueTraits
>
class MultipleTaskQueue
  : public TaskQueueMemoryManager<ExecSpace, MemorySpace>,
    public TaskQueueCommonMixin<MultipleTaskQueue<ExecSpace, MemorySpace, TaskQueueTraits>>,
    private ObjectWithVLAEmulation<
      MultipleTaskQueue<ExecSpace, MemorySpace, TaskQueueTraits>,
      MultipleTaskQueueTeamQueue<ExecSpace, MemorySpace, TaskQueueTraits>
    >
{
private:

  using base_t = TaskQueueMemoryManager<ExecSpace, MemorySpace>;
  using common_mixin_t = TaskQueueCommonMixin<MultipleTaskQueue>;

  template <class, class, class>
  friend class MultipleTaskQueueTeamEntry;

  struct SchedulingInfo {
    using team_queue_id_t = int32_t;
    static constexpr team_queue_id_t NoAssociatedTeam = -1;
    team_queue_id_t team_association = NoAssociatedTeam;
  };

public:

  using task_queue_type = MultipleTaskQueue; // mark as task_queue concept
  using task_queue_traits = TaskQueueTraits;
  using task_base_type = TaskNode<TaskQueueTraits>;
  using ready_queue_type = typename TaskQueueTraits::template ready_queue_type<task_base_type>;

  using scheduling_info_type = SchedulingInfo;

  using runnable_task_base_type = RunnableTaskBase<TaskQueueTraits>;

  template <class Functor, class Scheduler>
    // requires TaskScheduler<Scheduler> && TaskFunctor<Functor>
  using runnable_task_type = RunnableTask<
    task_queue_traits, Scheduler, typename Functor::value_type, Functor
  >;

  using aggregate_task_type = AggregateTask<TaskQueueTraits>;

  // Number of allowed priorities
  static constexpr int NumQueue = 3;


public:

  //----------------------------------------------------------------------------
  // <editor-fold desc="Constructors, destructors, and assignment"> {{{2

  MultipleTaskQueue() = delete;
  MultipleTaskQueue(MultipleTaskQueue const&) = delete;
  MultipleTaskQueue(MultipleTaskQueue&&) = delete;
  MultipleTaskQueue& operator=(MultipleTaskQueue const&) = delete;
  MultipleTaskQueue& operator=(MultipleTaskQueue&&) = delete;

  explicit
  MultipleTaskQueue(typename base_t::memory_pool const& arg_memory_pool)
    : base_t(arg_memory_pool)
  { }

  ~MultipleTaskQueue() = default;

  // </editor-fold> end Constructors, destructors, and assignment }}}2
  //----------------------------------------------------------------------------

  using common_mixin_t::schedule_runnable;

  KOKKOS_FUNCTION
  void
  schedule_runnable(
    runnable_task_base_type&& task,
    scheduling_info_type const& info
  ) {
    auto team_association = info.team_association;
    if(team_association == scheduling_info_type::NoAssociatedTeam) {
      // TODO maybe do a round robin here instead?
      team_association = 0;
    }
    this->_schedule_runnable_to_queue(
      std::move(task),
      this->vla_value_at(team_association).team_queue_for(task)
    );
    // Task may be enqueued and may be run at any point; don't touch it (hence
    // the use of move semantics)
  }

  KOKKOS_FUNCTION
  OptionalRef<task_base_type>
  pop_ready_task(
    scheduling_info_type const& info
  )
  {
    auto return_value = OptionalRef<task_base_type>{};
    auto team_association = info.team_association;
    if(team_association == scheduling_info_type::NoAssociatedTeam) {
      // TODO maybe do a round robin here instead?
      team_association = 0;
    }
    // always loop in order of priority first, then prefer team tasks over single tasks
    auto& team_queue_info = this->vla_value_at(team_association);

    return_value = team_queue_info.pop_ready_task();

    if(not return_value) {

      auto stolen_from = team_association;

      // loop through the rest of the teams and try to steal
      for(auto isteal = team_association+1; isteal != team_association; ++isteal) {
        isteal %= this->n_vla_entries();
        return_value = this->vla_value_at(isteal).pop_ready_task();
        if(return_value) {
          stolen_from = isteal;
          break;
        }
      }

      // if a task was stolen successfully, update the scheduling info
      if(return_value) {
        // Note that this won't update any associated futures, so don't trust
        // the scheduling info from a future to a runnable task
        return_value->as_runnable_task()
          .template scheduling_info_as<scheduling_info_type>()
            .team_association = stolen_from;
      }
    }
    // if nothing was found, return a default-constructed (empty) OptionalRef
    return return_value;
  }



};

template <
  class TaskQueueTraits
>
struct TeamQueueInfo {
public:

  using task_base_type = TaskNode<TaskQueueTraits>;
  using runnable_task_base_type = RunnableTaskBase<TaskQueueTraits>;
  using ready_queue_type = typename TaskQueueTraits::template ready_queue_type<task_base_type>;

private:

  // Number of allowed priorities
  static constexpr int NumQueue = 3;

  ready_queue_type m_ready_queues[NumQueue][2];

public:

  OptionalRef<task_base_type>
  pop_ready_task()
  {
    auto return_value = OptionalRef<task_base_type>{};
    for(int i_priority = 0; i_priority < NumQueue; ++i_priority) {
      // Check for a team task with this priority
      return_value = m_ready_queues[i_priority][TaskTeam].pop();
      if(return_value) return return_value;

      // Check for a single task with this priority
      return_value = m_ready_queues[i_priority][TaskSingle].pop();
      if(return_value) return return_value;
    }
    return return_value;
  }

  ready_queue_type&
  team_queue_for(runnable_task_base_type const& task)
  {
    return m_ready_queues[int(task.get_priority())][int(task.get_task_type())];
  }


};

} /* namespace Impl */
} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #if defined( KOKKOS_ENABLE_TASKDAG ) */
#endif /* #ifndef KOKKOS_IMPL_MULTIPLETASKQUEUE_HPP */

