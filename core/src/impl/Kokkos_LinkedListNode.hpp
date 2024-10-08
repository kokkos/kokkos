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

#ifndef KOKKOS_IMPL_LINKEDLISTNODE_HPP
#define KOKKOS_IMPL_LINKEDLISTNODE_HPP

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_TASKDAG

#include <Kokkos_Core_fwd.hpp>

#include <Kokkos_PointerOwnership.hpp>
#include <impl/Kokkos_OptionalRef.hpp>
#include <impl/Kokkos_Error.hpp>  // KOKKOS_EXPECTS

#include <Kokkos_Atomic.hpp>  // atomic_compare_exchange, atomic_fence

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

struct LinkedListNodeAccess;

template <uintptr_t NotEnqueuedValue             = 0,
          template <class> class PointerTemplate = std::add_pointer>
struct SimpleSinglyLinkedListNode {
 private:
  using pointer_type =
      typename PointerTemplate<SimpleSinglyLinkedListNode>::type;  // NOLINT

  pointer_type m_next = reinterpret_cast<pointer_type>(NotEnqueuedValue);

  // These are private because they are an implementation detail of the queue
  // and should not get added to the value type's interface via the intrusive
  // wrapper.

  KOKKOS_INLINE_FUNCTION
  void mark_as_not_enqueued() noexcept {
    // TODO @tasking @memory_order DSH make this an atomic store with memory
    // order
    m_next = (pointer_type)NotEnqueuedValue;
  }

  KOKKOS_INLINE_FUNCTION
  void mark_as_not_enqueued() volatile noexcept {
    // TODO @tasking @memory_order DSH make this an atomic store with memory
    // order
    m_next = (pointer_type)NotEnqueuedValue;
  }

  KOKKOS_INLINE_FUNCTION
  pointer_type& _next_ptr() noexcept { return m_next; }

  KOKKOS_INLINE_FUNCTION
  pointer_type volatile& _next_ptr() volatile noexcept { return m_next; }

  KOKKOS_INLINE_FUNCTION
  pointer_type const& _next_ptr() const noexcept { return m_next; }

  KOKKOS_INLINE_FUNCTION
  pointer_type const volatile& _next_ptr() const volatile noexcept {
    return m_next;
  }

  friend struct LinkedListNodeAccess;

 public:
  // constexpr
  KOKKOS_INLINE_FUNCTION
  bool is_enqueued() const noexcept {
    // TODO @tasking @memory_order DSH make this an atomic load with memory
    // order
    return m_next != reinterpret_cast<pointer_type>(NotEnqueuedValue);
  }

  // constexpr
  KOKKOS_INLINE_FUNCTION
  bool is_enqueued() const volatile noexcept {
    // TODO @tasking @memory_order DSH make this an atomic load with memory
    // order
    return m_next != reinterpret_cast<pointer_type>(NotEnqueuedValue);
  }
};

/// Attorney for LinkedListNode, since user types inherit from it
struct LinkedListNodeAccess {
  template <class Node>
  KOKKOS_INLINE_FUNCTION static void mark_as_not_enqueued(Node& node) noexcept {
    node.mark_as_not_enqueued();
  }

  template <class Node>
  KOKKOS_INLINE_FUNCTION static void mark_as_not_enqueued(
      Node volatile& node) noexcept {
    node.mark_as_not_enqueued();
  }

  template <class Node>
  KOKKOS_INLINE_FUNCTION static typename Node::pointer_type& next_ptr(
      Node& node) noexcept {
    return node._next_ptr();
  }

  template <class Node>
  KOKKOS_INLINE_FUNCTION static typename Node::pointer_type& next_ptr(
      Node volatile& node) noexcept {
    return node._next_ptr();
  }

  template <class Node>
  KOKKOS_INLINE_FUNCTION static typename Node::pointer_type& next_ptr(
      Node const& node) noexcept {
    return node._next_ptr();
  }

  template <class Node>
  KOKKOS_INLINE_FUNCTION static typename Node::pointer_type& prev_ptr(
      Node& node) noexcept {
    return node._prev_ptr();
  }

  template <class Node>
  KOKKOS_INLINE_FUNCTION static typename Node::pointer_type& prev_ptr(
      Node const& node) noexcept {
    return node._prev_ptr();
  }
};

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

}  // end namespace Impl
}  // end namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* defined KOKKOS_ENABLE_TASKDAG */
#endif /* #ifndef KOKKOS_IMPL_LINKEDLISTNODE_HPP */
