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

#include <Kokkos_Atomic.hpp>  // atomic_compare_exchange, atomic_fence

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

struct LinkedListNodeAccess;

template <
  uintptr_t NotEnqueuedValue = 0,
  template <class> class PointerTemplate = std::add_pointer
>
struct SimpleSinglyLinkedListNode
{

private:

  using pointer_type = typename PointerTemplate<SimpleSinglyLinkedListNode>::type;

  pointer_type m_next = reinterpret_cast<pointer_type>(NotEnqueuedValue);

  // These are private because they are an implementation detail of the queue
  // and should not get added to the value type's interface via the intrusive
  // wrapper.

  KOKKOS_INLINE_FUNCTION
  void mark_as_not_enqueued() noexcept {
    // TODO memory order
    // TODO make this an atomic store
    m_next = (pointer_type)NotEnqueuedValue;
  }

  KOKKOS_INLINE_FUNCTION
  pointer_type& _next_ptr() noexcept {
    return m_next;
  }

  KOKKOS_INLINE_FUNCTION
  pointer_type const& _next_ptr() const noexcept {
    return m_next;
  }

  friend struct LinkedListNodeAccess;

public:

  // KOKKOS_CONSTEXPR_14
  KOKKOS_INLINE_FUNCTION
  bool is_enqueued() const noexcept {
    // TODO memory order
    // TODO make this an atomic load
    return m_next != reinterpret_cast<pointer_type>(NotEnqueuedValue);
  }

};


/// Attorney for LinkedListNode, since user types inherit from it
struct LinkedListNodeAccess
{

  template <class Node>
  KOKKOS_INLINE_FUNCTION
  static void mark_as_not_enqueued(Node& node) noexcept {
    node.mark_as_not_enqueued();
  }

  template <class Node>
  KOKKOS_INLINE_FUNCTION
  static
  typename Node::pointer_type&
  _next_ptr(Node& node) noexcept {
    return node._next_ptr();
  }

  template <class Node>
  KOKKOS_INLINE_FUNCTION
  static
  typename Node::pointer_type&
  _next_ptr(Node const& node) noexcept {
    return node._next_ptr();
  }

};

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

template <class T>
struct LockBasedLIFOCommon
{

  using value_type = T;

  using node_type = SimpleSinglyLinkedListNode<>;

  static constexpr uintptr_t LockTag = ~uintptr_t(0);
  static constexpr uintptr_t EndTag = ~uintptr_t(1);

  OwningRawPtr<node_type> m_head = (node_type*)EndTag;

  KOKKOS_INLINE_FUNCTION
  bool _try_push_node(node_type& node) {

    KOKKOS_EXPECTS(!node.is_enqueued());

    auto* volatile & next = LinkedListNodeAccess::_next_ptr(node);

    // store the head of the queue in a local variable
    auto* old_head = m_head;

    // retry until someone locks the queue or we successfully compare exchange
    while (old_head != (node_type*)LockTag) {

      // TODO this should have a memory order and not a memory fence

      // set task->next to the head of the queue
      next = old_head;

      // fence to emulate acquire semantics on next and release semantics on
      // the store of m_head
      // Do not proceed until 'next' has been stored.
      ::Kokkos::memory_fence();

      // store the old head
      auto* const old_head_tmp = old_head;

      // attempt to swap task with the old head of the queue
      // as if this were done atomically:
      //   if(*queue == old_head) {
      //     *queue = task;
      //   }
      //   old_head = *queue;
      old_head = ::Kokkos::atomic_compare_exchange(&m_head, old_head, &node);

      if(old_head_tmp == old_head) return true;
    }

    // Failed, replace 'task->m_next' value since 'task' remains
    // not a member of a queue.

    // TODO this should have a memory order and not a memory fence
    LinkedListNodeAccess::mark_as_not_enqueued(node);

    // fence to emulate acquire semantics on next
    // Do not proceed until 'next' has been stored.
    ::Kokkos::memory_fence();

    return false;
  }

  bool _is_empty() const noexcept {
    // TODO memory order
    // TODO make this an atomic load
    return this->m_head == (node_type*)EndTag;
  }

};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

template <class T>
class LockBasedLIFO
  : private LockBasedLIFOCommon<T>
{

private:

  using base_t = LockBasedLIFOCommon<T>;
  using node_type = typename base_t::node_type;

public:

  using value_type = typename base_t::value_type; // = T
  using intrusive_node_base_type = SimpleSinglyLinkedListNode<>;

public:

  static_assert(
    std::is_base_of<intrusive_node_base_type, value_type>::value,
    "Intrusive linked-list value_type must be derived from intrusive_node_base_type"
  );

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
    // We can't use the static constexpr LockTag directly because
    // atomic_compare_exchange needs to bind a reference to that, and you
    // can't do that with static constexpr variables.
    auto* const lock_tag = (node_type*)base_t::LockTag;

    // TODO shouldn't this be a relaxed atomic load?
    // start with the return value equal to the head
    auto* rv = this->m_head;

    // Retry until the lock is acquired or the queue is empty.
    while(rv != (node_type*)base_t::EndTag) {

      // The only possible values for the queue are
      // (1) lock, (2) end, or (3) a valid task.
      // Thus zero will never appear in the queue.
      //
      // If queue is locked then just read by guaranteeing the CAS will fail.
      KOKKOS_ASSERT(rv != nullptr);

      if(rv == lock_tag) {
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
        auto* volatile& next = LinkedListNodeAccess::_next_ptr(*rv); //->m_next;

        // This algorithm is not lockfree because a adversarial scheduler could
        // context switch this thread at this point and the rest of the threads
        // calling this method would never make forward progress

        // TODO I think this needs to be a atomic store release (and the memory fence needs to be removed)
        // Lock is released here
        this->m_head = next;

        // Mark rv as popped by assigning nullptr to the next
        LinkedListNodeAccess::mark_as_not_enqueued(*rv);

        ::Kokkos::memory_fence();

        return OptionalRef<T>{ *static_cast<T*>(rv) };
      }

      // Otherwise, the CAS got a value that didn't match (either because
      // another thread locked the queue and we observed the lock tag or because
      // another thread replaced the head and now we want to try to lock the
      // queue with that as the popped item.  Either way, try again.
    }

    // Return an empty OptionalRef by calling the default constructor
    return { };
  }

  KOKKOS_INLINE_FUNCTION
  bool push(node_type& node)
  {
    while(!this->_try_push_node(node)) { /* retry until success */ }
    // for consistency with push interface on other queue types:
    return true;
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
 *   - Any calls to `try_push` that happen-before the call to
 *     `consume()` will succeed and return an true, such that the `consume()`
 *     call will visit that node.
 *   - Any calls to `try_push` for which the single call to `consume()`
 *     happens-before those calls will return false and the node given as
 *     an argument to `try_push` will not be visited by consume()
 *
 *
 * @tparam T The type of items in the queue
 *
 */
template <class T>
class SingleConsumeOperationLIFO
  : private LockBasedLIFOCommon<T>
{
private:

  using base_t = LockBasedLIFOCommon<T>;
  using node_type = typename base_t::node_type;

  // Allows us to reuse the existing infrastructure for
  static constexpr auto ConsumedTag = base_t::LockTag;

public:

  using value_type = typename base_t::value_type; // = T

  KOKKOS_INLINE_FUNCTION
  SingleConsumeOperationLIFO() noexcept = default;

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
  bool is_consumed() const noexcept {
    // TODO memory order?
    return this->m_head == (node_type*)ConsumedTag;
  }

  KOKKOS_INLINE_FUNCTION
  bool try_push(node_type& node)
  {
    return this->_try_push_node(node);
    // Ensures: (return value is true) || (node.is_enqueued() == false);
  }

  template <class Function>
  KOKKOS_INLINE_FUNCTION
  void consume(Function&& f) {
    auto* const consumed_tag = (node_type*)ConsumedTag;

    // Swap the Consumed tag into the head of the queue:

    // (local variable used for assertion only)
    // TODO this should have memory order release, I think
    auto old_head = Kokkos::atomic_exchange(&(this->m_head), consumed_tag);

    // Assert that the queue wasn't consumed before this
    // This can't be an expects clause because the acquire fence on the read
    // would be a side-effect
    KOKKOS_ASSERT(old_head != consumed_tag);

    // We now have exclusive access to the queue; loop over it and call
    // the user function
    while(old_head != (node_type*)base_t::EndTag) {

      // get the Node to make the call with
      auto* call_arg = old_head;

      // advance the head
      old_head = LinkedListNodeAccess::_next_ptr(*old_head);

      // Mark as popped before proceeding
      LinkedListNodeAccess::mark_as_not_enqueued(*call_arg);

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

namespace Kokkos {
namespace Impl {

struct TaskQueueTraitsLockBased
{

  // TODO document what concepts these match

  template <class Task>
  using ready_queue_type = LockBasedLIFO<Task>;

  template <class Task>
  using waiting_queue_type = SingleConsumeOperationLIFO<Task>;

  template <class Task>
  using intrusive_task_base_type =
    typename ready_queue_type<Task>::intrusive_node_base_type;
};


} // end namespace Impl
} // end namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* defined KOKKOS_ENABLE_TASKDAG */
#endif /* #ifndef KOKKOS_IMPL_LIFO_HPP */

