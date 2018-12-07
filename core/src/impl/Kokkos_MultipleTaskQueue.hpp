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

template <class TaskQueueTraits>
struct MultipleTaskQueueTeamEntry {
public:

  using task_base_type = TaskNode<TaskQueueTraits>;
  using runnable_task_base_type = RunnableTaskBase<TaskQueueTraits>;
  using ready_queue_type = typename TaskQueueTraits::template ready_queue_type<task_base_type>;

private:

  // Number of allowed priorities
  static constexpr int NumPriorities = 3;

  ready_queue_type m_ready_queues[NumPriorities][2];

public:

  KOKKOS_INLINE_FUNCTION
  OptionalRef<task_base_type>
  try_to_steal_ready_task()
  {
    auto return_value = OptionalRef<task_base_type>{};
    // prefer lower priority tasks when stealing
    for(int i_priority = NumPriorities-1; i_priority >= 0; --i_priority) {
      // Check for a single task with this priority
      return_value = m_ready_queues[i_priority][TaskSingle].steal();
      if(return_value) return return_value;

      // Check for a team task with this priority
      return_value = m_ready_queues[i_priority][TaskTeam].steal();
      if(return_value) return return_value;

    }
    return return_value;
  }

  KOKKOS_INLINE_FUNCTION
  OptionalRef<task_base_type>
  pop_ready_task()
  {
    auto return_value = OptionalRef<task_base_type>{};
    for(int i_priority = 0; i_priority < NumPriorities; ++i_priority) {
      // Check for a team task with this priority
      return_value = m_ready_queues[i_priority][TaskTeam].pop();
      if(return_value) return return_value;

      // Check for a single task with this priority
      return_value = m_ready_queues[i_priority][TaskSingle].pop();
      if(return_value) return return_value;
    }
    return return_value;
  }

  KOKKOS_INLINE_FUNCTION
  ready_queue_type&
  team_queue_for(runnable_task_base_type const& task)
  {
    return m_ready_queues[int(task.get_priority())][int(task.get_task_type())];
  }

};

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

template <
  class ExecSpace,
  class MemorySpace,
  class TaskQueueTraits
>
class MultipleTaskQueue final
  : public TaskQueueMemoryManager<ExecSpace, MemorySpace>,
    public TaskQueueCommonMixin<MultipleTaskQueue<ExecSpace, MemorySpace, TaskQueueTraits>>,
    private ObjectWithVLAEmulation<
      MultipleTaskQueue<ExecSpace, MemorySpace, TaskQueueTraits>,
      MultipleTaskQueueTeamEntry<TaskQueueTraits>
    >
{
private:

  using base_t = TaskQueueMemoryManager<ExecSpace, MemorySpace>;
  using common_mixin_t = TaskQueueCommonMixin<MultipleTaskQueue>;
  using vla_emulation_base_t = ObjectWithVLAEmulation<
    MultipleTaskQueue<ExecSpace, MemorySpace, TaskQueueTraits>,
    MultipleTaskQueueTeamEntry<TaskQueueTraits>
  >;

  // Allow private inheritance from ObjectWithVLAEmulation
  friend struct VLAEmulationAccess;

  struct SchedulerInfo {
    using team_queue_id_t = int32_t;
    static constexpr team_queue_id_t NoAssociatedTeam = -1;
    team_queue_id_t team_association = NoAssociatedTeam;

    using scheduler_info_type = SchedulerInfo;

    KOKKOS_INLINE_FUNCTION
    constexpr explicit SchedulerInfo(team_queue_id_t association) noexcept
      : team_association(association)
    { }

    KOKKOS_INLINE_FUNCTION
    SchedulerInfo() = default;

    KOKKOS_INLINE_FUNCTION
    SchedulerInfo(SchedulerInfo const&) = default;

    KOKKOS_INLINE_FUNCTION
    SchedulerInfo(SchedulerInfo&&) = default;

    KOKKOS_INLINE_FUNCTION
    SchedulerInfo& operator=(SchedulerInfo const&) = default;

    KOKKOS_INLINE_FUNCTION
    SchedulerInfo& operator=(SchedulerInfo&&) = default;

    KOKKOS_INLINE_FUNCTION
    ~SchedulerInfo() = default;

  };

  struct EmptyTaskSchedulingInfo { };

public:

  using task_queue_type = MultipleTaskQueue; // mark as task_queue concept
  using task_queue_traits = TaskQueueTraits;
  using task_base_type = TaskNode<TaskQueueTraits>;
  using ready_queue_type = typename TaskQueueTraits::template ready_queue_type<task_base_type>;

  using task_scheduling_info_type = EmptyTaskSchedulingInfo;
  using team_scheduler_info_type = SchedulerInfo;

  using runnable_task_base_type = RunnableTaskBase<TaskQueueTraits>;

  template <class Functor, class Scheduler>
    // requires TaskScheduler<Scheduler> && TaskFunctor<Functor>
  using runnable_task_type = RunnableTask<
    task_queue_traits, Scheduler, typename Functor::value_type, Functor
  >;

  using aggregate_task_type = AggregateTask<task_queue_traits, task_scheduling_info_type>;

  // Number of allowed priorities
  static constexpr int NumPriorities = 3;

  KOKKOS_INLINE_FUNCTION
  constexpr typename vla_emulation_base_t::vla_entry_count_type
  n_queues() const noexcept { return this->n_vla_entries(); }

  // TODO !!!!query this using a property of the execution space rather than using a constexpr member
  // This should query a customization point that defaults to the recommended
  // league size (which should probably also be a property-based customization
  // point)
  static constexpr int num_team_queues = 4;

public:

  //----------------------------------------------------------------------------
  // <editor-fold desc="Constructors, destructors, and assignment"> {{{2

  MultipleTaskQueue() = delete;
  MultipleTaskQueue(MultipleTaskQueue const&) = delete;
  MultipleTaskQueue(MultipleTaskQueue&&) = delete;
  MultipleTaskQueue& operator=(MultipleTaskQueue const&) = delete;
  MultipleTaskQueue& operator=(MultipleTaskQueue&&) = delete;

  MultipleTaskQueue(
    typename base_t::execution_space const& arg_execution_space,
    typename base_t::memory_space const&,
    typename base_t::memory_pool const& arg_memory_pool
  ) : base_t(arg_memory_pool),
      // TODO !!!!query this using a property of the execution space rather than using a constexpr member
      vla_emulation_base_t(num_team_queues)
  { }

  ~MultipleTaskQueue() = default;

  // </editor-fold> end Constructors, destructors, and assignment }}}2
  //----------------------------------------------------------------------------

  KOKKOS_FUNCTION
  void
  schedule_runnable(
    runnable_task_base_type&& task,
    team_scheduler_info_type const& info
  ) {
    auto team_association = info.team_association;
    // Should only not be assigned if this is a host spawn...
    if(team_association == team_scheduler_info_type::NoAssociatedTeam) {
      team_association = 0;
    }
    this->_schedule_runnable_to_queue(
      std::move(task),
      this->vla_value_at(team_association).team_queue_for(task),
      info
    );
    // Task may be enqueued and may be run at any point; don't touch it (hence
    // the use of move semantics)
  }

  KOKKOS_FUNCTION
  OptionalRef<task_base_type>
  pop_ready_task(
    team_scheduler_info_type const& info
  )
  {
    KOKKOS_EXPECTS(info.team_association != team_scheduler_info_type::NoAssociatedTeam);

    auto return_value = OptionalRef<task_base_type>{};
    auto team_association = info.team_association;

    // always loop in order of priority first, then prefer team tasks over single tasks
    auto& team_queue_info = this->vla_value_at(team_association);

    return_value = team_queue_info.pop_ready_task();

    if(not return_value) {

      // loop through the rest of the teams and try to steal
      for(
        auto isteal = (team_association + 1) % this->n_queues();
        isteal != team_association;
        isteal = (isteal + 1) % this->n_queues()
      ) {
        return_value = this->vla_value_at(isteal).try_to_steal_ready_task();
        if(return_value) { break; }
      }

      // if a task was stolen successfully, update the scheduling info
      //if(return_value) {
      //  // Note that this won't update any associated futures, so don't trust
      //  // the scheduling info from a future to a runnable task
      //  return_value->as_runnable_task()
      //    .template scheduling_info_as<task_scheduling_info_type>()
      //      .team_association = info.team_association;
      //}
    }
    // if nothing was found, return a default-constructed (empty) OptionalRef
    return return_value;
  }


  // TODO make this a property-based customization point
  KOKKOS_INLINE_FUNCTION
  team_scheduler_info_type
  initial_team_scheduler_info(int rank_in_league) const noexcept {
    return team_scheduler_info_type{
      typename team_scheduler_info_type::team_queue_id_t(rank_in_league % n_queues())
    };
  }

  // TODO make this a property-based customization point
  static /* KOKKOS_CONSTEXPR_14 */ size_t
  task_queue_allocation_size(
    typename base_t::execution_space const& exec_space,
    typename base_t::memory_space const&,
    typename base_t::memory_pool const&
  )
  {
    // TODO !!!!query this using a property of the execution space rather than using a constexpr member

    return vla_emulation_base_t::required_allocation_size(
      /* num_vla_entries = */ num_team_queues
    );
  }

};


} /* namespace Impl */
} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #if defined( KOKKOS_ENABLE_TASKDAG ) */
#endif /* #ifndef KOKKOS_IMPL_MULTIPLETASKQUEUE_HPP */

