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

#ifndef KOKKOS_TASKSCHEDULER_FWD_HPP
#define KOKKOS_TASKSCHEDULER_FWD_HPP

//----------------------------------------------------------------------------

#include <Kokkos_Macros.hpp>
#if defined( KOKKOS_ENABLE_TASKDAG )

#include <Kokkos_Core_fwd.hpp>
//----------------------------------------------------------------------------

namespace Kokkos {

// Forward declarations used in Impl::TaskQueue

template <typename ValueType, typename Scheduler>
class BasicFuture;

template <class Space, class Queue>
class SimpleTaskScheduler;

template <class Space, class Queue>
class BasicTaskScheduler;

template< typename Space >
struct is_scheduler : public std::false_type {};

template<class Space, class Queue>
struct is_scheduler<BasicTaskScheduler<Space, Queue>> : public std::true_type {};

template<class Space, class Queue>
struct is_scheduler<SimpleTaskScheduler<Space, Queue>> : public std::true_type {};

enum class TaskPriority : int {
  High = 0,
  Regular = 1,
  Low = 2
};

} // namespace Kokkos

//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template <class TaskQueueTraits>
class TaskNode;

class TaskBase;

/*\brief  Implementation data for task data management, access, and execution.
 *
 *  CRTP Inheritance structure to allow static_cast from the
 *  task root type and a task's FunctorType.
 *
 *    TaskBase< Space , ResultType , FunctorType >
 *      : TaskBase< Space , ResultType , void >
 *      , FunctorType
 *      { ... };
 *
 *    TaskBase< Space , ResultType , void >
 *      : TaskBase< Space , void , void >
 *      { ... };
 */
template< typename Space , typename ResultType , typename FunctorType >
class Task;

class TaskQueueBase;

template< typename Space, typename MemorySpace = typename Space::memory_space >
class TaskQueue;

template< typename ExecSpace, typename MemSpace = typename ExecSpace::memory_space >
class TaskQueueMultiple ;

template< typename ExecSpace, typename MemSpace, typename TaskQueueTraits>
class SingleTaskQueue;

template< typename ExecSpace, typename MemSpace, typename TaskQueueTraits>
class MultipleTaskQueue;

struct TaskQueueTraitsLockBased;

template <size_t CircularBufferSize=64>
struct TaskQueueTraitsChaseLev;

template< typename ResultType >
struct TaskResult;

struct TaskSchedulerBase;

} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------

namespace Kokkos {

template< typename Space >
using DeprecatedTaskScheduler = BasicTaskScheduler<Space, Impl::TaskQueue<Space>> ;

template< typename Space >
using DeprecatedTaskSchedulerMultiple = BasicTaskScheduler<Space, Impl::TaskQueueMultiple<Space>> ;

template< typename Space >
using TaskScheduler = SimpleTaskScheduler<Space, Impl::SingleTaskQueue<Space, typename Space::memory_space, Impl::TaskQueueTraitsLockBased>>;

template< typename Space >
using TaskSchedulerMultiple = SimpleTaskScheduler<Space, Impl::MultipleTaskQueue<Space, typename Space::memory_space, Impl::TaskQueueTraitsLockBased>>;

template< typename Space >
using ChaseLevTaskScheduler = SimpleTaskScheduler<Space, Impl::MultipleTaskQueue<Space, typename Space::memory_space, Impl::TaskQueueTraitsChaseLev<>>>;

template<class Space, class QueueType>
void wait(BasicTaskScheduler<Space, QueueType> const&);

namespace Impl {

struct TaskSchedulerBase { };

class TaskQueueBase { };

template <typename Scheduler, typename EnableIfConstraint=void>
class TaskQueueSpecializationConstrained { };

template <typename Scheduler>
struct TaskQueueSpecialization : TaskQueueSpecializationConstrained<Scheduler> { };

template <int, typename>
struct TaskPolicyData;


} // end namespace Impl

} // namespace Kokkos

//----------------------------------------------------------------------------

#endif /* #if defined( KOKKOS_ENABLE_TASKDAG ) */
#endif /* #ifndef KOKKOS_TASKSCHEDULER_FWD_HPP */

