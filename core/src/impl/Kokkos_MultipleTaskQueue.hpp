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

// TODO move this
/** @brief A CRTP base class for a type that includes a variable-length array by allocation
 *
 *  The storage for the derived type must be allocated manually and the objects
 *  (both derived type and VLA objects) must be constructed with placement new.
 *  Obviously, this can't be done for objects on the stack.
 *
 *  Note: Though most uses of this currently delete the copy and move constructor
 *  in the `Derived` type, this type is intended to have value semantics.
 *
 *  @todo elaborate on implications of value semantics for this class template
 *
 */
template <
  class Derived,
  class VLAValueType,
  class EntryCountType = int32_t
>
struct ObjectWithVLAEmulation {
public:

  using object_type = Derived;
  using vla_value_type = VLAValueType;
  using vla_entry_count_type = EntryCountType;

  using iterator = VLAValueType*;
  using const_iterator = typename std::add_const<VLAValueType>::type*;

  static_assert(
    alignof(object_type) >= alignof(vla_value_type),
    "Can't append emulated variable length array of type with greater alignment than"
    "  the type to which the VLA is being appended"
  );

  static_assert(
    not std::is_abstract<vla_value_type>::value,
    "Can't use abstract type with VLA emulation"
  );

  // TODO require that Derived be marked final? (note that std::is_final is C++14)

private:

  vla_entry_count_type m_num_entries;

  // Note: can't be constexpr because of reinterpret_cast
  vla_value_type* _vla_pointer() const {
    return reinterpret_cast<vla_value_type*>(
      static_cast<Derived>(this) + 1
    );
  }

public:

  KOKKOS_INLINE_FUNCTION
  static constexpr size_t
  required_allocation_size(vla_entry_count_type num_vla_entries) {
    KOKKOS_EXPECTS(num_vla_entries >= 0);
    return sizeof(Derived) + num_vla_entries * sizeof(VLAValueType);
  }

  //----------------------------------------------------------------------------
  // <editor-fold desc="Constructors, destructor, and assignment"> {{{2

  // TODO specialization for trivially constructible VLAValueType?
  // TODO constrained this to default contructible vla_value_types
  KOKKOS_INLINE_FUNCTION
  explicit
  ObjectWithVLAEmulation(vla_entry_count_type num_entries)
    noexcept(noexcept(vla_value_type()))
    : m_num_entries(num_entries)
  {
    KOKKOS_EXPECTS(num_entries >= 0);
    for(vla_entry_count_type i = 0; i < m_num_entries; ++i) {
      new (_vla_pointer() + i) vla_value_type();
    }
  }

  // TODO specialization for trivially destructible VLAValueType?
  KOKKOS_INLINE_FUNCTION
  ~ObjectWithVLAEmulation()
    noexcept(noexcept(std::declval<vla_value_type>().~vla_value_type()))
  {
    for(auto&& value : *this) { value.~vla_value_type(); }
  }

  // TODO constrained analogs for move and copy ctors and assignment ops
  // TODO forwarding in_place constructor
  // TODO initializer_list constructor?

  // </editor-fold> end Constructors, destructor, and assignment }}}2
  //----------------------------------------------------------------------------


  KOKKOS_INLINE_FUNCTION
  constexpr EntryCountType n_vla_entries() const noexcept { return m_num_entries; }


  //----------------------------------------------------------------------------
  // <editor-fold desc="Accessing the object and the VLA values"> {{{2

  KOKKOS_INLINE_FUNCTION
  object_type& object() & { return static_cast<Derived&>(*this); }

  KOKKOS_INLINE_FUNCTION
  object_type const& object() const & { return static_cast<Derived const&>(*this); }

  KOKKOS_INLINE_FUNCTION
  object_type&& object() && { return static_cast<Derived&&>(*this); }


  KOKKOS_INLINE_FUNCTION
  vla_value_type& vla_value_at(vla_entry_count_type n) &
  {
    KOKKOS_EXPECTS(n < n_vla_entries());
    return _vla_pointer()[n];
  }

  KOKKOS_INLINE_FUNCTION
  vla_value_type const& vla_value_at(vla_entry_count_type n) const &
  {
    KOKKOS_EXPECTS(n < n_vla_entries());
    return _vla_pointer()[n];
  }

  KOKKOS_INLINE_FUNCTION
  vla_value_type& vla_value_at(vla_entry_count_type n) &&
  {
    KOKKOS_EXPECTS(n < n_vla_entries());
    return _vla_pointer()[n];
  }

  // </editor-fold> end Accessing the object and the VLA values }}}2
  //----------------------------------------------------------------------------


  //----------------------------------------------------------------------------
  // <editor-fold desc="Iterators"> {{{2

  KOKKOS_INLINE_FUNCTION
  iterator begin() noexcept { return _vla_pointer(); }

  KOKKOS_INLINE_FUNCTION
  const_iterator begin() const noexcept { return _vla_pointer(); }

  KOKKOS_INLINE_FUNCTION
  const_iterator cbegin() noexcept { return _vla_pointer(); }

  KOKKOS_INLINE_FUNCTION
  iterator end() noexcept { return _vla_pointer() + m_num_entries; }

  KOKKOS_INLINE_FUNCTION
  const_iterator end() const noexcept { return _vla_pointer() + m_num_entries; }

  KOKKOS_INLINE_FUNCTION
  const_iterator cend() noexcept { return _vla_pointer() + m_num_entries; }

  // </editor-fold> end Iterators }}}2
  //----------------------------------------------------------------------------

};




template <
  class ExecSpace,
  class MemorySpace,
  class TaskQueueTraits
>
class MultipleTaskQueueTeamEntry
{

};




template <
  class ExecSpace,
  class MemorySpace,
  class TaskQueueTraits
>
class MultipleTaskQueue
  : public TaskQueueMemoryManager<ExecSpace, MemorySpace>,
    public TaskQueueCommonMixin<SingleTaskQueue<ExecSpace, MemorySpace, TaskQueueTraits>>,
    public ObjectWithVLAEmulation<
      MultipleTaskQueue<ExecSpace, MemorySpace, TaskQueueTraits>,
      MultipleTaskQueueTeamEntry<ExecSpace, MemorySpace, TaskQueueTraits>
    >
{
private:

  using base_t = TaskQueueMemoryManager<ExecSpace, MemorySpace>;
  using common_mixin_t = TaskQueueCommonMixin<SingleTaskQueue>;

public:

  using task_queue_type = SingleTaskQueue; // mark as task_queue concept
  using task_queue_traits = TaskQueueTraits;
  using task_base_type = TaskNode<TaskQueueTraits>;
  using ready_queue_type = typename TaskQueueTraits::template ready_queue_type<task_base_type>;

  using runnable_task_base_type = RunnableTaskBase<TaskQueueTraits>;

  template <class Functor, class Scheduler>
    // requires TaskScheduler<Scheduler> && TaskFunctor<Functor>
  using runnable_task_type = RunnableTask<
    task_queue_traits, Scheduler, typename Functor::value_type, Functor
  >;

  using aggregate_task_type = AggregateTask<TaskQueueTraits>;

  // Number of allowed priorities
  static constexpr int NumQueue = 3;

public:

  //----------------------------------------------------------------------------
  // <editor-fold desc="Constructors, destructors, and assignment"> {{{2

  MultipleTaskQueue() = delete;
  MultipleTaskQueue(MultipleTaskQueue const&) = delete;
  MultipleTaskQueue(MultipleTaskQueue&&) = delete;
  MultipleTaskQueue& operator=(MultipleTaskQueue const&) = delete;
  MultipleTaskQueue& operator=(MultipleTaskQueue&&) = delete;

  explicit
  MultipleTaskQueue(typename base_t::memory_pool const& arg_memory_pool)
    : base_t(arg_memory_pool)
  { }

  ~MultipleTaskQueue() {
  }

  // </editor-fold> end Constructors, destructors, and assignment }}}2
  //----------------------------------------------------------------------------


  KOKKOS_FUNCTION
  void
  schedule_runnable(runnable_task_base_type&& task) {
    this->_schedule_runnable_to_queue(
      std::move(task),
      m_ready_queues[int(task.get_priority())][int(task.get_task_type())]
    );
    // Task may be enqueued and may be run at any point; don't touch it (hence
    // the use of move semantics)
  }

  KOKKOS_FUNCTION
  OptionalRef<task_base_type>
  pop_ready_task()
  {
    OptionalRef<task_base_type> return_value;
    // always loop in order of priority first, then prefer team tasks over single tasks
    for(int i_priority = 0; i_priority < NumQueue; ++i_priority) {

      // Check for a team task with this priority
      return_value = m_ready_queues[i_priority][TaskTeam].pop();
      if(return_value) return return_value;

      // Check for a single task with this priority
      return_value = m_ready_queues[i_priority][TaskSingle].pop();
      if(return_value) return return_value;

    }
    // if nothing was found, return a default-constructed (empty) OptionalRef
    return return_value;
  }

};

} /* namespace Impl */
} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #if defined( KOKKOS_ENABLE_TASKDAG ) */
#endif /* #ifndef KOKKOS_IMPL_MULTIPLETASKQUEUE_HPP */

