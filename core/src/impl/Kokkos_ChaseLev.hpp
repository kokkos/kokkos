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

// Experimental unified task-data parallel manycore LDRD

#ifndef KOKKOS_IMPL_LOCKFREEDEQUE_HPP
#define KOKKOS_IMPL_LOCKFREEDEQUE_HPP

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_TASKDAG // Note: implies CUDA_VERSION >= 8000 if using CUDA

#include <Kokkos_Core_fwd.hpp>

#include <Kokkos_PointerOwnership.hpp>
#include <impl/Kokkos_OptionalRef.hpp>
#include <impl/Kokkos_Error.hpp> // KOKKOS_EXPECTS
#include <impl/Kokkos_LinkedListNode.hpp> // KOKKOS_EXPECTS

#include <Kokkos_Atomic.hpp>  // atomic_compare_exchange, atomic_fence
#include "Kokkos_LIFO.hpp"

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

/** Based on "Correct and Efficient Work-Stealing for Weak Memory Models,"
 * PPoPP '13, https://www.di.ens.fr/~zappa/readings/ppopp13.pdf
 *
 */
template <
  class T,
  class SizeType = int32_t,
  size_t CircularBufferSize = 64
>
struct ChaseLevDeque {
public:

  using size_type = SizeType;
  using value_type = T;
  // Still using intrusive linked list for waiting queue
  using node_type = SimpleSinglyLinkedListNode<>;

private:

  // TODO @tasking @new_feature DSH variable size circular buffer?


  struct fixed_size_circular_buffer {

    node_type* buffer[CircularBufferSize] = { nullptr };
    static constexpr auto size = size_type(CircularBufferSize);

    KOKKOS_INLINE_FUNCTION
    fixed_size_circular_buffer grow() {
      Kokkos::abort("Circular buffer is fixed size only for now; can't grow");
      return {};
    }

    KOKKOS_INLINE_FUNCTION
    fixed_size_circular_buffer* operator->() { return this; }

  };

  fixed_size_circular_buffer m_array;
  size_type m_top = 0;
  size_type m_bottom = 0;


public:

  KOKKOS_INLINE_FUNCTION
  bool empty() const {
    // TODO @tasking @memory_order DSH memory order
    return m_top > m_bottom - 1;
  }

  KOKKOS_INLINE_FUNCTION
  OptionalRef<T>
  pop() {
    auto b = m_bottom - 1; // atomic load relaxed
    auto& a = m_array; // atomic load relaxed
    m_bottom = b; // atomic store relaxed
    Kokkos::memory_fence(); // memory order seq_cst
    auto t = m_top; // atomic load relaxed
    OptionalRef<T> return_value;
    if(t <= b) {
      /* non-empty queue */
      return_value = *static_cast<T*>(a->buffer[b % a->size]); // relaxed load
      if(t == b) {
        /* single last element in the queue. */
        if(not Impl::atomic_compare_exchange_strong(&m_top, t, t+1, memory_order_seq_cst, memory_order_relaxed)) {
          /* failed race, someone else stole it */
          return_value = nullptr;
        }
        m_bottom = b + 1; // memory order relaxed
      }
    } else {
      /* empty queue */
      m_bottom = b + 1; // memory order relaxed
    }
    return return_value;
  }

  KOKKOS_INLINE_FUNCTION
  bool push(node_type&& node)
  {
    // Just forward to the lvalue version
    return push(node);
  }

  KOKKOS_INLINE_FUNCTION
  bool push(node_type& node)
  {
    auto b = m_bottom; // memory order relaxed
    auto t = Impl::atomic_load(&m_top, memory_order_acquire);
    auto& a = m_array;
    if(b - t > a->size - 1) {
      /* queue is full, resize */
      //m_array = a->grow();
      //a = m_array;
      return false;
    }
    a->buffer[b % a->size] = &node; // relaxed
    Impl::atomic_store(&m_bottom, b + 1, memory_order_release);
    return true;
  }

  KOKKOS_INLINE_FUNCTION
  OptionalRef<T>
  steal() {
    auto t = m_top; // TODO @tasking @memory_order DSH: atomic load acquire
    Kokkos::memory_fence(); // seq_cst fence, so why does the above need to be acquire?
    auto b = Impl::atomic_load(&m_bottom, memory_order_acquire);
    OptionalRef<T> return_value;
    if(t < b) {
      /* Non-empty queue */
      auto& a = m_array; // TODO @tasking @memory_order DSH: technically consume ordered, but acquire should be fine
      Kokkos::load_fence(); // TODO @tasking @memory_order DSH memory order instead of fence
      return_value = *static_cast<T*>(a->buffer[t % a->size]); // relaxed
      if(not Impl::atomic_compare_exchange_strong(&m_top, t, t+1, memory_order_seq_cst, memory_order_relaxed)) {
        return_value = nullptr;
      }
    }
    return return_value;
  }

};

/*
      // The atomicity of this load was more important in the paper's version
      // because that version had a circular buffer that could grow.  We're
      // essentially using the memory order in this version as a fence, which
      // may be unnecessary
      auto buffer_ptr = (node_type***)&m_array.buffer;
      auto a = Impl::atomic_load(buffer_ptr, memory_order_acquire); // technically consume ordered, but acquire should be fine
      return_value = *static_cast<T*>(a[t % m_array->size]); // relaxed; we'd have to replace the m_array->size if we ever allow growth
*/

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

template <size_t CircularBufferSize>
struct TaskQueueTraitsChaseLev {

  template <class Task>
  using ready_queue_type = ChaseLevDeque<Task, int32_t, CircularBufferSize>;

  template <class Task>
  using waiting_queue_type = SingleConsumeOperationLIFO<Task>;

  template <class Task>
  using intrusive_task_base_type =
    typename ready_queue_type<Task>::node_type;

  static constexpr auto ready_queue_insertion_may_fail = true;

};

} // end namespace Impl
} // end namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* defined KOKKOS_ENABLE_TASKDAG */
#endif /* #ifndef KOKKOS_IMPL_LOCKFREEDEQUE_HPP */

