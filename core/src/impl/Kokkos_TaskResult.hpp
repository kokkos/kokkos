/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
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

// Experimental unified task-data parallel manycore LDRD

#ifndef KOKKOS_IMPL_TASKRESULT_HPP
#define KOKKOS_IMPL_TASKRESULT_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_TASKDAG)

#include <Kokkos_TaskScheduler_fwd.hpp>
#include <Kokkos_Core_fwd.hpp>

#include <impl/Kokkos_TaskBase.hpp>
#include <impl/Kokkos_TaskNode.hpp>

#include <string>
#include <typeinfo>
#include <stdexcept>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template <typename ResultType>
struct TaskResult {
  enum : int32_t { size = sizeof(ResultType) };

  using reference_type = ResultType&;

  template <class CountType>
  KOKKOS_INLINE_FUNCTION static ResultType* ptr(
      PoolAllocatedObjectBase<CountType>* task) {
    return reinterpret_cast<ResultType*>(reinterpret_cast<char*>(task) +
                                         task->get_allocation_size() -
                                         sizeof(ResultType));
  }

  KOKKOS_INLINE_FUNCTION static ResultType* ptr(TaskBase* task) {
    return reinterpret_cast<ResultType*>(reinterpret_cast<char*>(task) +
                                         task->m_alloc_size -
                                         sizeof(ResultType));
  }

  KOKKOS_INLINE_FUNCTION static reference_type get(TaskBase* task) {
    return *ptr(task);
  }

  template <class TaskQueueTraits>
  KOKKOS_INLINE_FUNCTION static reference_type get(
      TaskNode<TaskQueueTraits>* task) {
    return *ptr(task);
  }

  KOKKOS_INLINE_FUNCTION static void destroy(TaskBase* task) {
    get(task).~ResultType();
  }

  // template <class TaskQueueTraits>
  // KOKKOS_INLINE_FUNCTION static
  // void destroy( TaskNode<TaskQueueTraits>* task )
  //{ get(task).~ResultType(); }
};

template <>
struct TaskResult<void> {
  enum : int32_t { size = 0 };

  using reference_type = void;

  template <class TaskQueueTraits>
  KOKKOS_INLINE_FUNCTION static void* ptr(TaskNode<TaskQueueTraits>* task) {
    return nullptr;
  }

  KOKKOS_INLINE_FUNCTION static void* ptr(TaskBase*) { return (void*)nullptr; }

  template <class TaskQueueTraits>
  KOKKOS_INLINE_FUNCTION static reference_type get(
      TaskNode<TaskQueueTraits>* task) { /* Should never be called */
  }

  KOKKOS_INLINE_FUNCTION static reference_type get(TaskBase*) {}

  KOKKOS_INLINE_FUNCTION static void destroy(TaskBase* task) {}

  // template <class TaskQueueTraits>
  // KOKKOS_INLINE_FUNCTION static
  // void destroy( TaskNode<TaskQueueTraits>* task )
  //{ }
};

} /* namespace Impl */
} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #if defined( KOKKOS_ENABLE_TASKDAG ) */
#endif /* #ifndef KOKKOS_IMPL_TASKRESULT_HPP */
