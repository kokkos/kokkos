//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef KOKKOS_IMPL_SINGLETASKQUEUE_HPP
#define KOKKOS_IMPL_SINGLETASKQUEUE_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_TASKDAG)

#include <Kokkos_TaskScheduler_fwd.hpp>
#include <Kokkos_Core_fwd.hpp>

#include <Kokkos_MemoryPool.hpp>

#include <impl/Kokkos_TaskBase.hpp>
#include <impl/Kokkos_TaskResult.hpp>

#include <impl/Kokkos_TaskQueueMemoryManager.hpp>
#include <impl/Kokkos_TaskQueueCommon.hpp>
#include <Kokkos_Atomic.hpp>
#include <impl/Kokkos_OptionalRef.hpp>
#include <impl/Kokkos_LIFO.hpp>

#include <string>
#include <typeinfo>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

template <class ExecSpace, class MemorySpace, class TaskQueueTraits,
          class MemoryPool>
class SingleTaskQueue
    : public TaskQueueMemoryManager<ExecSpace, MemorySpace, MemoryPool>,
      public TaskQueueCommonMixin<SingleTaskQueue<
          ExecSpace, MemorySpace, TaskQueueTraits, MemoryPool>> {
 private:
  using base_t = TaskQueueMemoryManager<ExecSpace, MemorySpace, MemoryPool>;
  using common_mixin_t = TaskQueueCommonMixin<SingleTaskQueue>;

  struct EmptyTeamSchedulerInfo {};
  struct EmptyTaskSchedulingInfo {};

 public:
  using task_queue_type   = SingleTaskQueue;  // mark as task_queue concept
  using task_queue_traits = TaskQueueTraits;
  using task_base_type    = TaskNode<TaskQueueTraits>;
  using ready_queue_type =
      typename TaskQueueTraits::template ready_queue_type<task_base_type>;

  using team_scheduler_info_type  = EmptyTeamSchedulerInfo;
  using task_scheduling_info_type = EmptyTaskSchedulingInfo;

  using runnable_task_base_type = RunnableTaskBase<TaskQueueTraits>;

  template <class Functor, class Scheduler>
  // requires TaskScheduler<Scheduler> && TaskFunctor<Functor>
  using runnable_task_type =
      RunnableTask<task_queue_traits, Scheduler, typename Functor::value_type,
                   Functor>;

  using aggregate_task_type =
      AggregateTask<task_queue_traits, task_scheduling_info_type>;

  // Number of allowed priorities
  static constexpr int NumQueue = 3;

 private:
  ready_queue_type m_ready_queues[NumQueue][2];

 public:
  //----------------------------------------------------------------------------
  // <editor-fold desc="Constructors, destructors, and assignment"> {{{2

  SingleTaskQueue()                                  = delete;
  SingleTaskQueue(SingleTaskQueue const&)            = delete;
  SingleTaskQueue(SingleTaskQueue&&)                 = delete;
  SingleTaskQueue& operator=(SingleTaskQueue const&) = delete;
  SingleTaskQueue& operator=(SingleTaskQueue&&)      = delete;

  explicit SingleTaskQueue(typename base_t::execution_space const&,
                           typename base_t::memory_space const&,
                           typename base_t::memory_pool const& arg_memory_pool)
      : base_t(arg_memory_pool) {}

  ~SingleTaskQueue() {
    for (int i_priority = 0; i_priority < NumQueue; ++i_priority) {
      KOKKOS_EXPECTS(m_ready_queues[i_priority][TaskTeam].empty());
      KOKKOS_EXPECTS(m_ready_queues[i_priority][TaskSingle].empty());
    }
  }

  // </editor-fold> end Constructors, destructors, and assignment }}}2
  //----------------------------------------------------------------------------

  KOKKOS_FUNCTION
  void schedule_runnable(runnable_task_base_type&& task,
                         team_scheduler_info_type const& info) {
    this->schedule_runnable_to_queue(
        std::move(task),
        m_ready_queues[int(task.get_priority())][int(task.get_task_type())],
        info);
    // Task may be enqueued and may be run at any point; don't touch it (hence
    // the use of move semantics)
  }

  KOKKOS_FUNCTION
  OptionalRef<task_base_type> pop_ready_task(
      team_scheduler_info_type const& /*info*/) {
    OptionalRef<task_base_type> return_value;
    // always loop in order of priority first, then prefer team tasks over
    // single tasks
    for (int i_priority = 0; i_priority < NumQueue; ++i_priority) {
      // Check for a team task with this priority
      return_value = m_ready_queues[i_priority][TaskTeam].pop();
      if (return_value) return return_value;

      // Check for a single task with this priority
      return_value = m_ready_queues[i_priority][TaskSingle].pop();
      if (return_value) return return_value;
    }
    // if nothing was found, return a default-constructed (empty) OptionalRef
    return return_value;
  }

  KOKKOS_INLINE_FUNCTION
  constexpr team_scheduler_info_type initial_team_scheduler_info(
      int) const noexcept {
    return {};
  }
};

} /* namespace Impl */
} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #if defined( KOKKOS_ENABLE_TASKDAG ) */
#endif /* #ifndef KOKKOS_IMPL_SINGLETASKQUEUE_HPP */
