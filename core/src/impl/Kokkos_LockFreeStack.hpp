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

#ifndef KOKKOS_IMPL_LOCKFREESTACK_HPP
#define KOKKOS_IMPL_LOCKFREESTACK_HPP

#include <Kokkos_Macros.hpp>

#include <Kokkos_Core_fwd.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
namespace Kokkos {
namespace Impl {

struct InPlaceTag { };

template <class T>
using OwningPtr = T*;

template <class T>
using ObservingPtr = T*;

template <class T>
struct OptionalRef {
private:

  T* m_value = nullptr;

public:

  using value_type = T;

  KOKKOS_INLINE_FUNCTION
  OptionalRef() = default;

  KOKKOS_INLINE_FUNCTION
  OptionalRef(OptionalRef const&) = default;
  
  KOKKOS_INLINE_FUNCTION
  OptionalRef(OptionalRef&&) = default;

  KOKKOS_INLINE_FUNCTION
  OptionalRef& operator=(OptionalRef const&) = default;

  KOKKOS_INLINE_FUNCTION
  OptionalRef& operator=(OptionalRef&&) = default;

  KOKKOS_INLINE_FUNCTION
  ~OptionalRef() = default;

  KOKKOS_INLINE_FUNCTION
  explicit OptionalRef(T& arg_value) : m_value(&arg_value) { }

  KOKKOS_INLINE_FUNCTION
  explicit OptionalRef(nullptr_t) : m_value(nullptr) { }

  KOKKOS_INLINE_FUNCTION
  OptionalRef& operator=(T& arg_value) { m_value = &arg_value; return *this; }

  KOKKOS_INLINE_FUNCTION
  OptionalRef& operator=(nullptr_t) { m_value = nullptr; return *this; }

  //----------------------------------------
  
  KOKKOS_INLINE_FUNCTION
  T& operator*() {
    // TODO assert value is not nullptr
    return *m_value;
  }
   
  KOKKOS_INLINE_FUNCTION
  T const& operator*() const {
    // TODO assert value is not nullptr
    return *m_value;
  }

  KOKKOS_INLINE_FUNCTION
  T volatile& operator*() volatile {
    // TODO assert value is not nullptr
    return *m_value;
  }

  KOKKOS_INLINE_FUNCTION
  T* operator->() {
    // TODO assert value is not nullptr
    return m_value;
  }

  KOKKOS_INLINE_FUNCTION
  T const* operator->() const {
    // TODO assert value is not nullptr
    return m_value;
  }

  KOKKOS_INLINE_FUNCTION
  T volatile* operator->() volatile {
    // TODO assert value is not nullptr
    return m_value;
  }

  KOKKOS_INLINE_FUNCTION
  T* get() {
    return m_value;
  }

  KOKKOS_INLINE_FUNCTION
  T const* get() const {
    return m_value;
  }

  KOKKOS_INLINE_FUNCTION
  T volatile* get() volatile {
    return m_value;
  }

  //----------------------------------------

  KOKKOS_INLINE_FUNCTION
  operator bool() const { return m_value != nullptr; }

  KOKKOS_INLINE_FUNCTION
  operator bool() volatile { return m_value != nullptr; }

  KOKKOS_INLINE_FUNCTION
  bool has_value() const { return m_value != nullptr; }

  KOKKOS_INLINE_FUNCTION
  bool has_value() volatile { return m_value != nullptr; }
  
};

// TODO !!! Tagged pointers to avoid the ABA problems

template <class T>
class TaggedPointerLockFreeStack {

private:

  struct Node {
    Node* m_next;
    T m_value;

    template <class... Args>
    explicit Node(
      InPlaceTag,
      Args&&... args
    ) : m_next(nullptr),
        m_value(std::forward<Args>(args)...)
    { }

  };

public:

  // TODO finish this

};

// TODO use allocator traits for allocation (device not supported by standard library currently)

template <class T, class Allocator>
class LockBasedStack {
public:

  using value_type = T;

private:

  enum : uintptr_t { LockTag = ~uintptr_t(0) };
  enum : uintptr_t { EndTag = ~uintptr_t(0) };

  // Inherit from T so that we can cast back to a Node upon re-enqueue
  struct Node
    : public T
  {
    using value_t = T;

    Node* m_next;

    template <class... Args>
    explicit Node(
      InPlaceTag,
      Args&&... args
    ) : m_next(nullptr),
        value_t(std::forward<Args>(args)...)
    { }

  };

public:

  using allocator_type = typename Allocator::template rebind<Node>::other;

private:

  OwningPtr<Node> m_head = (Node*)EndTag;
  allocator_type m_allocator;

  void _try_push_node(Node& node) {

    auto* const lock = (Node*)LockTag;

    auto* volatile & next = task->m_next;

    // store the head of the queue
    auto* old_head = m_head;

    while (old_head != lock) {

      // TODO this should have a memory order and not a memory fence 

      // set task->next to the head of the queue
      next = old_head;

      // Do not proceed until 'next' has been stored.
      Kokkos::memory_fence();

      // store the old head
      auto* const old_head_tmp = old_head;

      // attempt to swap task with the old head of the queue
      // as if this were done atomically:
      //   if(*queue == old_head) {
      //     *queue = task;
      //   }
      //   old_head = *queue;
      old_head = Kokkos::atomic_compare_exchange(queue, old_head, task);

      if(old_head_tmp == old_head) return true;
    }

    // Failed, replace 'task->m_next' value since 'task' remains
    // not a member of a queue.

    next = zero ;

    // Do not proceed until 'next' has been stored.
    Kokkos::memory_fence();

    return false ;

  }

public:


  OptionalRef<T> pop() {
    Node* const lock = (Node*)LockTag;
    Node* const end = (Node*)EndTag;

    // Retry until the lock is acquired or the queue is empty.

    // Shouldn't this be a relaxed atomic load?
    Node* head = m_head.get();

    while(end != task) {

      // The only possible values for the queue are
      // (1) lock, (2) end, or (3) a valid task.
      // Thus zero will never appear in the queue.
      //
      // If queue is locked then just read by guaranteeing the CAS will fail.

      // Is this assignment necessary? Couldn't we just continue here?
      if(task == lock) task = 0;

      Node* const old_head = head;

      // TODO this should be a weak compare exchange in a loop
      head = Kokkos::atomic_compare_exchange(&m_head, x, lock);

      if(head == old_head) {
        // CAS succeeded and queue is locked
        //
        // This thread has locked the queue and removed 'task' from the queue.
        // Extract the next entry of the queue from 'task->m_next'
        // and mark 'task' as popped from a queue by setting
        // 'task->m_next = lock'.
        //
        // Place the next entry in the head of the queue,
        // which also unlocks the queue.
        //
        // This thread has exclusive access to
        // the queue and the popped task's m_next.

        Node* volatile & next = head->m_next;

        // This algorithm is not lockfree because a adversarial scheduler could
        // context switch this thread at this point and the rest of the threads
        // calling this method would never make forward progress

        // TODO I think this needs to be a atomic store release (and the memory fence needs to be removed)
        m_head = next; // Lock is released here
        next = lock; // Go ahead and mark this as popped

        Kokkos::memory_fence();

        return head;
      }
    }

    return nullptr;

  }

  template <class... Args>
  void emplace(Args&&... args)
    // requires std::is_constructible_v<T, Args&&...>
  {
    static_assert(
      std::is_constructible<T, Args&&...>::value,
      "value_type must be constructible from arguments to emplace()"
    );

    auto* storage = m_allocator.allocate(1);
    auto* node = new ((void*)storage) Node(InPlaceTag{}, std::forward<Args>(args)...);
    while(!_try_push_node(node)) { }
  }

  void push_popped_item(OptionalRef<T> ptr) {
    // TODO assert ptr is not null
    // TODO assert m_next is lock tag
    auto& node = static_cast<Node&>(*ptr);
    while(!_try_push_node(node)) { }
  }

  void delete_popped_item(OptionalRef<T> ptr) const {
    // TODO assert ptr is not null
    // TODO assert m_next is lock tag
    auto& to_delete = static_cast<Node&>(*ptr);
    to_delete.~to_delete();
    m_allocator.deallocate(&to_delete);
  }

};


} // end namespace Impl
} // end namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------



#endif /* #ifndef KOKKOS_IMPL_LOCKFREESTACK_HPP */

