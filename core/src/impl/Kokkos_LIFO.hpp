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

#ifndef KOKKOS_IMPL_LIFO_HPP
#define KOKKOS_IMPL_LIFO_HPP

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_TASKDAG // Note: implies CUDA_VERSION >= 8000 if using CUDA

#include <Kokkos_Core_fwd.hpp>

#include <Kokkos_PointerOwnership.hpp>
#include <impl/Kokkos_OptionalRef.hpp>
#include <impl/Kokkos_Error.hpp> // KOKKOS_EXPECTS

#include <impl/Kokkos_Memory_Fence.hpp>
#include <Kokkos_Atomic.hpp>  // atomic_compare_exchange

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

// TODO !!! Tagged pointers to avoid the ABA problems

//template <class T>
//class TaggedPointerLockFreeStack {
//
//private:
//
//  struct Node {
//    Node* m_next;
//    T m_value;
//
//    template <class... Args>
//    explicit Node(
//      InPlaceTag,
//      Args&&... args
//    ) : m_next(nullptr),
//        m_value(std::forward<Args>(args)...)
//    { }
//
//  };
//
//public:
//
//  // TODO finish this
//
//};

} // end namespace Impl
} // end namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {


// Saves on some template instantiation by putting type-agnostic parts
// in a common base class
struct LockBasedLIFOBase {


};


// Saves on some template instantiation by putting allocator-agnostic parts
// in a common base class
template <class T>
struct AllocatorAgnosticLockBasedLIFOCommon
  : public LockBasedLIFOBase
{
  using base_t = LockBasedLIFOBase;
  using value_type = T;

  struct Node;

  static constexpr Node* LockTag = reinterpret_cast<Node>(~uintptr_t(0));
  static constexpr Node* EndTag = reinterpret_cast<Node>(~uintptr_t(1));

  // Inherit from T so that we can cast back to a Node upon re-enqueue
  struct Node
    : public T
  {
    using value_type = T;

    Node* m_next = LockTag;

    template <class... Args>
    KOKKOS_INLINE_FUNCTION
    explicit Node(
      InPlaceTag,
      Args&&... args
    ) : value_type(std::forward<Args>(args)...)
    { }

    // KOKKOS_CONSTEXPR_14
    KOKKOS_INLINE_FUNCTION
    bool is_enqueued() const noexcept {
      // TODO memory order
      // TODO make this an atomic load
      return m_next != LockTag;
    }

    KOKKOS_INLINE_FUNCTION
    void mark_as_not_enqueued() noexcept {
      // TODO memory order
      // TODO make this an atomic store
      m_next = LockTag;
    }

  };

};

template <class T, class Allocator>
struct LockBasedLIFOCommon
  : public AllocatorAgnosticLockBasedLIFOCommon<T>
{
  using base_t = AllocatorAgnosticLockBasedLIFOCommon<T>;
  using node_type = typename base_t::Node;
  using allocator_type = typename Allocator::template rebind<node_type>::other;

  using base_t::LockTag;
  using base_t::EndTag;

  // TODO use EBCO for stateless allocators
  OwningRawPtr<node_type> m_head = EndTag;
  allocator_type m_allocator = { };

  KOKKOS_INLINE_FUNCTION
  bool _try_push_node(node_type& node) {

    KOKKOS_EXPECTS(!node.is_enqueued());

    auto* volatile & next = node.m_next;

    // store the head of the queue in a local variable
    auto* old_head = m_head;

    // retry until someone locks the queue or we successfully compare exchange
    while (old_head != LockTag) {

      // TODO this should have a memory order and not a memory fence

      // set task->next to the head of the queue
      next = old_head;

      // fence to emulate acquire semantics on next and release semantics on
      // the store of m_head
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
      old_head = Kokkos::atomic_compare_exchange(&m_head, old_head, &node);

      if(old_head_tmp == old_head) return true;
    }

    // Failed, replace 'task->m_next' value since 'task' remains
    // not a member of a queue.

    // TODO this should have a memory order and not a memory fence
    next->mark_as_not_enqueued();

    // fence to emulate acquire semantics on next
    // Do not proceed until 'next' has been stored.
    Kokkos::memory_fence();

    return false;
  }

  bool _is_empty() const noexcept {
    // TODO memory order
    // TODO make this an atomic load
    return this->m_head == base_t::EndTag;
  }

};



// TODO use allocator traits for allocation (device not supported by standard library currently)

template <class T, class Allocator>
class LockBasedLIFO
  : private LockBasedLIFOCommon<T, Allocator>
{
private:

  using base_t = LockBasedLIFOCommon<T, Allocator>;
  using node_type = typename base_t::Node;

public:

  using value_type = typename base_t::value_type; // = T
  using allocator_type = typename base_t::allocator_type;

  LockBasedLIFO() = default;
  LockBasedLIFO(LockBasedLIFO const&) = delete;
  LockBasedLIFO(LockBasedLIFO&&) = delete;
  LockBasedLIFO& operator=(LockBasedLIFO const&) = delete;
  LockBasedLIFO& operator=(LockBasedLIFO&&) = delete;

  ~LockBasedLIFO() = default;


  bool empty() const noexcept {
    // TODO memory order
    return this->_is_empty();
  }

  KOKKOS_INLINE_FUNCTION
  OptionalRef<T> pop()
  {
    using base_t::LockTag;
    using base_t::EndTag;

    // We can't use the static constexpr LockTag directly because
    // atomic_compare_exchange needs to bind a reference to that, and you
    // can't do that with static constexpr variables.
    auto const* const lock_tag = LockTag;

    // TODO shouldn't this be a relaxed atomic load?
    // start with the return value equal to the head
    auto* rv = this->m_head;

    // Retry until the lock is acquired or the queue is empty.
    while(rv != EndTag) {

      // The only possible values for the queue are
      // (1) lock, (2) end, or (3) a valid task.
      // Thus zero will never appear in the queue.
      //
      // If queue is locked then just read by guaranteeing the CAS will fail.
      KOKKOS_ASSERT(rv != nullptr);

      if(rv == LockTag) {
        // TODO this should just be an atomic load followed by a continue
        // just set rv to nullptr for now, effectively turning the
        // atomic_compare_exchange below into a load
        rv = nullptr;
      }

      auto* const old_rv = rv;

      // TODO this should be a weak compare exchange in a loop
      rv = Kokkos::atomic_compare_exchange(&(this->m_head), old_rv, lock_tag);

      if(rv == old_rv) {
        // CAS succeeded and queue is locked
        //
        // This thread has locked the queue and removed 'rv' from the queue.
        // Extract the next entry of the queue from 'rv->m_next'
        // and mark 'rv' as popped from a queue by setting
        // 'rv->m_next = nullptr'.
        //
        // Place the next entry in the head of the queue,
        // which also unlocks the queue.
        //
        // This thread has exclusive access to
        // the queue and the popped task's m_next.

        // TODO check whether the volatile is needed here
        auto* volatile& next = rv->m_next;

        // This algorithm is not lockfree because a adversarial scheduler could
        // context switch this thread at this point and the rest of the threads
        // calling this method would never make forward progress

        // TODO I think this needs to be a atomic store release (and the memory fence needs to be removed)
        // Lock is released here
        this->m_head = next;

        // Mark rv as popped by assigning nullptr to the next
        next->mark_as_not_enqueued();

        Kokkos::memory_fence();

        return { rv };
      }

      // Otherwise, the CAS got a value that didn't match (either because
      // another thread locked the queue and we observed the lock tag or because
      // another thread replaced the head and now we want to try to lock the
      // queue with that as the popped item.  Either way, try again.
    }

    // Return an empty OptionalRef by calling the default constructor
    return { };
  }

  template <class... Args>
  KOKKOS_INLINE_FUNCTION
  void emplace(Args&&... args)
    // requires std::is_constructible_v<T, Args&&...>
  {
    static_assert(
      std::is_constructible<T, Args&&...>::value,
      "value_type must be constructible from arguments to emplace()"
    );

    // TODO use allocator traits
    auto* storage = this->m_allocator.allocate(1);
    auto* node =
      new ((void*)storage) node_type(InPlaceTag{}, std::forward<Args>(args)...);

    while(!this->_try_push_node(node)) { /* retry until success */ }
  }

  // TODO push() implementation constrained by move-constructibility

  KOKKOS_INLINE_FUNCTION
  void push_popped_item(OptionalRef<T> ptr) {
    KOKKOS_ASSERT(ptr.has_value());
    auto& node = static_cast<node_type&>(*ptr);
    KOKKOS_ASSERT(!node.is_enqueued());
    while(!_try_push_node(node)) { }
  }

  KOKKOS_INLINE_FUNCTION
  void delete_popped_item(OptionalRef<T> ptr) const {
    KOKKOS_ASSERT(ptr.has_value());
    auto& to_delete = static_cast<node_type&>(*ptr);
    KOKKOS_ASSERT(!to_delete.is_enqueued());
    to_delete.~to_delete();
    this->m_allocator.deallocate(&to_delete);
  }

};


/** @brief A Multiple Producer, Single Consumer Queue with some special semantics
 *
 * This multi-producer, single consumer queue has the following semantics:
 *
 *   - Any number of threads may call `try_emplace`/`try_push`
 *       + These operations are lock-free.
 *   - Exactly one thread calls `consume()`, and the call occurs exactly once
 *     in the lifetime of the queue.
 *       + This operation is lock-free (and wait-free w.r.t. producers)
 *   - Any calls to `try_emplace`/`try_push` that happen-before the call to
 *     `consume()` will succeed and return an empty `OptionalRef<T>`.
 *   - Any calls to `try_emplace`/`try_push` for which the single call to
 *     `consume()` happens-before those calls will construct a `T` from their
 *     arguments and return it as a `OptionalRef<T>`, analogous to a popped item.
 *
 *
 *
 *
 * @tparam T The type of items in the queue
 * @tparam Allocator The allocator used to construct new entries (via `emplace`)
 *                   or to construct new nodes for move or copy construction
 *                   (via `push`, if `T` is move and/or copy constructible)
 */
template <class T, class Allocator>
class SingleConsumeOperationLIFO
  : private LockBasedLIFOCommon<T, Allocator>
{
private:

  using base_t = LockBasedLIFOCommon<T, Allocator>;
  using node_type = typename base_t::Node;

  // Allows us to reuse the existing infrastructure for
  static constexpr auto ConsumedTag = base_t::LockTag;

public:

  using value_type = typename base_t::value_type; // = T
  using allocator_type = typename base_t::allocator_type;

  KOKKOS_INLINE_FUNCTION
  SingleConsumeOperationLIFO() = default;

  SingleConsumeOperationLIFO(SingleConsumeOperationLIFO const&) = delete;
  SingleConsumeOperationLIFO(SingleConsumeOperationLIFO&&) = delete;
  SingleConsumeOperationLIFO& operator=(SingleConsumeOperationLIFO const&) = delete;
  SingleConsumeOperationLIFO& operator=(SingleConsumeOperationLIFO&&) = delete;

  KOKKOS_INLINE_FUNCTION
  ~SingleConsumeOperationLIFO() = default;

  KOKKOS_INLINE_FUNCTION
  bool empty() const noexcept {
    // TODO memory order
    return this->_is_empty();
  }

  KOKKOS_INLINE_FUNCTION
  bool consumed() const noexcept {
    // TODO memory order?
    return this->m_head == ConsumedTag;
  }

  template <class... Args>
  KOKKOS_INLINE_FUNCTION
  OptionalRef<T> try_emplace(Args&&... args)
    // requires std::is_constructible_v<T, Args&&...>
  {
    static_assert(
      std::is_constructible<T, Args&&...>::value,
      "value_type must be constructible from arguments to emplace()"
    );

    // TODO use allocator traits
    auto* storage = this->m_allocator.allocate(1);
    auto* node =
      new ((void*)storage) node_type(InPlaceTag{}, std::forward<Args>(args)...);

    auto result = this->_try_push_node(node);
    if(result.get() == ConsumedTag) {
      return { node };
    }
    else {
      // Otherwise, the push was successful, so return an empty OptionalRef
      // indicating such
      return { };
    }
  }

  template <class Function>
  KOKKOS_INLINE_FUNCTION
  void consume(Function&& f) {

    // Swap the Consumed tag into the head of the queue:

    // (local variable used for assertion only)
    // TODO this should have memory order release, I think
    auto old_head = Kokkos::atomic_exchange(&(this->m_head), ConsumedTag);

    // Assert that the queue wasn't consumed before this
    // This can't be an expects clause because the acquire fence on the read
    // would be a side-effect
    KOKKOS_ASSERT(old_head != ConsumedTag);

    // We now have exclusive access to the queue; loop over it and call
    // the user function
    while(old_head != base_t::EndTag) {

      // get the Node to make the call with
      auto* call_arg = old_head;

      // advance the head
      old_head = old_head->m_next;

      // Mark as popped before proceeding
      call_arg->mark_as_not_enqueued();

      // Call the user function
      auto& arg = *static_cast<T*>(call_arg);
      f(arg);

    }

  }

};

} // end namespace Impl
} // end namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------



#endif /* #ifndef KOKKOS_IMPL_LIFO_HPP */

