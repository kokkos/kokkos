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

// Experimental unified task-data parallel manycore LDRD

#ifndef KOKKOS_IMPL_TASKQUEUE_HPP
#define KOKKOS_IMPL_TASKQUEUE_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_TASKDAG)

#include <Kokkos_TaskScheduler_fwd.hpp>
#include <Kokkos_Core_fwd.hpp>

#include <Kokkos_MemoryPool.hpp>

#include <impl/Kokkos_TaskBase.hpp>
#include <impl/Kokkos_TaskResult.hpp>

#include <Kokkos_Atomic.hpp>
#include <impl/Kokkos_OptionalRef.hpp>
#include <impl/Kokkos_LIFO.hpp>

#include <string>
#include <typeinfo>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

/** \brief  Manage task allocation, deallocation, and scheduling.
 *
 *  Task execution is deferred to the TaskQueueSpecialization.
 *  All other aspects of task management have shared implementation.
 */
template <typename ExecSpace, typename MemorySpace>
class TaskQueue : public TaskQueueBase {
 protected:
  template <class>
  friend struct TaskQueueSpecialization;
  template <class, class>
  friend class TaskQueueSpecializationConstrained;
  template <class, class>
  friend class Kokkos::BasicTaskScheduler;

  using execution_space = ExecSpace;
  using memory_space    = MemorySpace;
  using device_type     = Kokkos::Device<execution_space, memory_space>;
  using memory_pool     = Kokkos::MemoryPool<device_type>;
  using task_root_type  = Kokkos::Impl::TaskBase;
  using team_queue_type = TaskQueue;

  struct Destroy {
    TaskQueue* m_queue;
    void destroy_shared_allocation();
  };

  //----------------------------------------

  enum : int { NumQueue = 3 };

  // Queue is organized as [ priority ][ type ]

  memory_pool m_memory;
  task_root_type* volatile m_ready[NumQueue][2];
  // long                      m_accum_alloc ; // Accumulated number of
  // allocations
  int m_count_alloc = 0;  // Current number of allocations
  int m_max_alloc;        // Maximum number of allocations
  int m_ready_count;      // Number of ready or executing

  //----------------------------------------

  ~TaskQueue();
  TaskQueue()                 = delete;
  TaskQueue(TaskQueue&&)      = delete;
  TaskQueue(TaskQueue const&) = delete;
  TaskQueue& operator=(TaskQueue&&) = delete;
  TaskQueue& operator=(TaskQueue const&) = delete;

  TaskQueue(const memory_pool& arg_memory_pool);

  // Schedule a task
  //   Precondition:
  //     task is not executing
  //     task->m_next is the dependence or zero
  //   Postcondition:
  //     task->m_next is linked list membership
  KOKKOS_FUNCTION void schedule_runnable(task_root_type*);
  KOKKOS_FUNCTION void schedule_aggregate(task_root_type*);

  // Reschedule a task
  //   Precondition:
  //     task is in Executing state
  //     task->m_next == LockTag
  //   Postcondition:
  //     task is in Executing-Respawn state
  //     task->m_next == 0 (no dependence)
  KOKKOS_FUNCTION
  void reschedule(task_root_type*);

  // Complete a task
  //   Precondition:
  //     task is not executing
  //     task->m_next == LockTag  =>  task is complete
  //     task->m_next != LockTag  =>  task is respawn
  //   Postcondition:
  //     task->m_wait == LockTag  =>  task is complete
  //     task->m_wait != LockTag  =>  task is waiting
  KOKKOS_FUNCTION
  void complete(task_root_type*);

  KOKKOS_FUNCTION
  static bool push_task(task_root_type* volatile* const, task_root_type* const);

  KOKKOS_FUNCTION
  static task_root_type* pop_ready_task(task_root_type* volatile* const);

  KOKKOS_FUNCTION static void decrement(task_root_type* task);

 public:
  KOKKOS_INLINE_FUNCTION
  int allocation_count() const noexcept { return m_count_alloc; }

  KOKKOS_INLINE_FUNCTION
  void initialize_team_queues(int /*pool_size*/) const noexcept {}

  KOKKOS_INLINE_FUNCTION
  task_root_type* attempt_to_steal_task() const noexcept { return nullptr; }

  KOKKOS_INLINE_FUNCTION
  team_queue_type& get_team_queue(int /*team_rank*/) { return *this; }

  // void execute() { specialization::execute( this ); }

  template <typename FunctorType>
  void proc_set_apply(typename task_root_type::function_type* ptr) {
    using specialization =
        TaskQueueSpecialization<BasicTaskScheduler<ExecSpace, TaskQueue>>;
    specialization::template proc_set_apply<FunctorType>(ptr);
  }

  // Assign task pointer with reference counting of assigned tasks
  KOKKOS_FUNCTION static void assign(task_root_type** const lhs,
                                     task_root_type* const rhs) {
    if (*lhs) decrement(*lhs);
    if (rhs) {
      Kokkos::Impl::desul_atomic_inc(&rhs->m_ref_count,
                                     Kokkos::Impl::MemoryOrderSeqCst(),
                                     Kokkos::Impl::MemoryScopeDevice());
    }

    // Force write of *lhs

    *static_cast<task_root_type* volatile*>(lhs) = rhs;

    Kokkos::memory_fence();
  }

  KOKKOS_FUNCTION
  size_t allocate_block_size(size_t n);  ///< Actual block size allocated

  KOKKOS_FUNCTION
  void* allocate(size_t n);  ///< Allocate from the memory pool

  KOKKOS_FUNCTION
  void deallocate(void* p, size_t n);  ///< Deallocate to the memory pool

  //----------------------------------------
  /**\brief  Allocation size for a spawned task */

  template <typename FunctorType>
  KOKKOS_FUNCTION size_t spawn_allocation_size() const {
    using value_type = typename FunctorType::value_type;

    using task_type = Impl::Task<execution_space, value_type, FunctorType>;

    constexpr size_t task_size = sizeof(task_type);

    return m_memory.allocate_block_size(task_size);
  }

  /**\brief  Allocation size for a when_all aggregate */

  KOKKOS_FUNCTION
  size_t when_all_allocation_size(int narg) const {
    return m_memory.allocate_block_size(sizeof(task_root_type) +
                                        narg * sizeof(task_root_type*));
  }
};

} /* namespace Impl */
} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #if defined( KOKKOS_ENABLE_TASKDAG ) */
#endif /* #ifndef KOKKOS_IMPL_TASKQUEUE_HPP */
