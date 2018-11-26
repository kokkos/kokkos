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

#ifndef KOKKOS_IMPL_TASKBASE_HPP
#define KOKKOS_IMPL_TASKBASE_HPP

#include <Kokkos_Macros.hpp>
#if defined( KOKKOS_ENABLE_TASKDAG )

#include <Kokkos_TaskScheduler_fwd.hpp>
#include <Kokkos_Core_fwd.hpp>

#include <impl/Kokkos_LIFO.hpp>

#include <string>
#include <typeinfo>
#include <stdexcept>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

enum TaskType : int16_t   { TaskTeam = 0 , TaskSingle = 1 , Aggregate = 2 };


/** Intrusive base class for things allocated with a Kokkos::MemoryPool
 *
 *  @warning Memory pools assume that the address of this class is the same
 *           as the address of the most derived type that was allocated to
 *           have the given size.  As a consequence, when interacting with
 *           multiple inheritance, this must always be the first base class
 *           of any derived class that uses it!
 *  @todo Consider inverting inheritance structure to avoid this problem?
 *
 *  @tparam CountType type of integer used to store the allocation size
 */
template <class CountType = int32_t>
class alignas(void*) PoolAllocatedObjectBase {
public:

  using pool_allocation_size_type = CountType;

private:

  pool_allocation_size_type m_alloc_size;

public:


  KOKKOS_INLINE_FUNCTION
  constexpr explicit PoolAllocatedObjectBase(pool_allocation_size_type allocation_size)
    : m_alloc_size(allocation_size)
  { }

  KOKKOS_INLINE_FUNCTION
  CountType get_allocation_size() const noexcept { return m_alloc_size; }

};


// TODO move this?
template <class CountType = int32_t>
class ReferenceCountedBase {
public:

  using reference_count_size_type = CountType;

private:

  reference_count_size_type m_ref_count = 0;

public:

  KOKKOS_INLINE_FUNCTION
  constexpr explicit
  ReferenceCountedBase(reference_count_size_type initial_reference_count)
    : m_ref_count(initial_reference_count)
  {
    // KOKKOS_EXPECTS(initial_reference_count > 0);
  }

  /** Decrement the reference count,
   *  and return true iff this decrement caused
   *  the reference count to become zero
   */
  KOKKOS_INLINE_FUNCTION
  bool decrement_and_check_reference_count()
  {
    // TODO memory order
    auto old_count = Kokkos::atomic_fetch_add(&m_ref_count, -1);

    KOKKOS_ASSERT(old_count > 0 && "reference count greater less than zero!");

    return (old_count == 1);
  }

  KOKKOS_INLINE_FUNCTION
  void increment_reference_count()
  {
    Kokkos::atomic_increment(&m_ref_count);
  }

};

template <class TaskQueueTraits>
class AggregateTask;

template <class TaskQueueTraits>
class RunnableTaskBase;

template <class TaskQueueTraits>
class TaskNode
  : public PoolAllocatedObjectBase<int32_t>, // size 4, must be first!
    public ReferenceCountedBase<int32_t>, // size 4
    public TaskQueueTraits::template intrusive_task_base_type<TaskNode<TaskQueueTraits>> // size 8+
{
public:

  using priority_type = int16_t;

private:

  using task_base_type = TaskNode<TaskQueueTraits>;
  using pool_allocated_base_type = PoolAllocatedObjectBase<int32_t>;
  using reference_counted_base_type = ReferenceCountedBase<int32_t>;
  using task_queue_traits = TaskQueueTraits;
  using waiting_queue_type =
    typename task_queue_traits::template waiting_queue_type<TaskNode>;

  waiting_queue_type m_wait_queue; // size 8+

  // TODO eliminate this???
  TaskQueueBase* m_ready_queue_base;

  TaskType m_task_type;  // size 2
  priority_type m_priority; // size 2

public:

  KOKKOS_INLINE_FUNCTION
  constexpr
  TaskNode(
    TaskType task_type,
    priority_type priority,
    TaskQueueBase* queue_base,
    reference_count_size_type initial_reference_count,
    pool_allocation_size_type allocation_size
  ) : pool_allocated_base_type(
        /* allocation_size = */ allocation_size
      ),
      reference_counted_base_type(
        /* initial_reference_count = */ initial_reference_count
      ),
      m_wait_queue(),
      m_task_type(task_type),
      m_priority(priority),
      m_ready_queue_base(queue_base)
  { }

  TaskNode() = delete;
  TaskNode(TaskNode const&) = delete;
  TaskNode(TaskNode&&) = delete;
  TaskNode& operator=(TaskNode const&) = delete;
  TaskNode& operator=(TaskNode&&) = delete;

  KOKKOS_INLINE_FUNCTION
  bool is_aggregate() const noexcept { return m_task_type == TaskType::Aggregate; }

  KOKKOS_INLINE_FUNCTION
  bool is_runnable() const noexcept { return m_task_type != TaskType::Aggregate; }

  KOKKOS_INLINE_FUNCTION
  bool is_single_runnable() const noexcept { return m_task_type == TaskType::TaskSingle; }

  KOKKOS_INLINE_FUNCTION
  bool is_team_runnable() const noexcept { return m_task_type == TaskType::TaskTeam; }

  KOKKOS_INLINE_FUNCTION
  TaskType get_task_type() const noexcept { return m_task_type; }

  KOKKOS_INLINE_FUNCTION
  RunnableTaskBase<TaskQueueTraits>&
  as_runnable_task() {
    KOKKOS_EXPECTS(this->is_runnable());
    return static_cast<RunnableTaskBase<TaskQueueTraits>&>(*this);
  }

  KOKKOS_INLINE_FUNCTION
  AggregateTask<TaskQueueTraits>&
  as_aggregate() {
    KOKKOS_EXPECTS(this->is_aggregate());
    return static_cast<AggregateTask<TaskQueueTraits>&>(*this);
  }

  KOKKOS_INLINE_FUNCTION
  bool try_add_waiting(task_base_type& depends_on_this) {
    return m_wait_queue.try_push(depends_on_this);
  }

  template <class Function>
  KOKKOS_INLINE_FUNCTION
  void consume_wait_queue(Function&& f) {
    KOKKOS_EXPECTS(not m_wait_queue.is_consumed());
    m_wait_queue.consume(std::forward<Function>(f));
  }

  KOKKOS_INLINE_FUNCTION
  bool wait_queue_is_consumed() const noexcept {
    // TODO memory order
    return m_wait_queue.is_consumed();
  }

  KOKKOS_INLINE_FUNCTION
  TaskQueueBase*
  ready_queue_base_ptr() const noexcept {
    return m_ready_queue_base;
  }


  KOKKOS_INLINE_FUNCTION
  void set_priority(TaskPriority priority) noexcept {
    KOKKOS_EXPECTS(!this->is_enqueued());
    m_priority = (priority_type)priority;
  }

  KOKKOS_INLINE_FUNCTION
  TaskPriority get_priority() const noexcept {
    return (TaskPriority)m_priority;
  }


};


template <class TaskQueueTraits>
class AggregateTask
  : public TaskNode<TaskQueueTraits> // must be first base class for allocation reasons!!!
{
private:

  using base_t = TaskNode<TaskQueueTraits>;
  using task_base_type = TaskNode<TaskQueueTraits>;

  int32_t m_aggregate_dep_count;

public:

  template <class... Args>
  // requires std::is_constructible_v<base_t, Args&&...>
  KOKKOS_INLINE_FUNCTION
  constexpr explicit
  AggregateTask(
    int32_t aggregate_predecessor_count,
    Args&&... args
  ) : base_t(
        TaskType::Aggregate,
        (typename base_t::priority_type)TaskPriority::Regular, // all aggregates are regular priority
        std::forward<Args>(args)...
      ),
      m_aggregate_dep_count(aggregate_predecessor_count)
  { }

  /** Get the list of pointers that this is aggregating as predecessors.  We're
   *  emulating variable-length arrays by storing those values starting at
   *  this+1 to this+m_aggredgate_dep_count+1 (exclusive) as task_base_type*
   */
   // TODO make this a span
   // TODO verify need for volatile here
  KOKKOS_INLINE_FUNCTION
  task_base_type * volatile *
  aggregate_dependences() {
    return reinterpret_cast<task_base_type*volatile*>(this+1);
  }

  KOKKOS_INLINE_FUNCTION
  int32_t dependence_count() const { return m_aggregate_dep_count; }


};


template <class TaskQueueTraits>
class RunnableTaskBase
  : public TaskNode<TaskQueueTraits> // must be first base class for allocation reasons!!!
{
private:

  using base_t = TaskNode<TaskQueueTraits>;

public:

  using task_base_type = TaskNode<TaskQueueTraits>;
  using function_type = void(*)( task_base_type * , void * );
  using destroy_type = void(*)( task_base_type * );

private:

  function_type m_apply;
  task_base_type* m_predecessor = nullptr;
  bool m_is_respawning = false;

public:

  template <class... Args>
    // requires std::is_constructible_v<base_t, Args&&...>
  KOKKOS_INLINE_FUNCTION
  constexpr explicit
  RunnableTaskBase(
    function_type apply_function_ptr,
    Args&&... args
  ) : base_t(std::forward<Args>(args)...),
      m_apply(apply_function_ptr)
  { }

  KOKKOS_INLINE_FUNCTION
  bool get_respawn_flag() const { return m_is_respawning; }

  KOKKOS_INLINE_FUNCTION
  void set_respawn_flag(bool value = true) { m_is_respawning = value; }

  KOKKOS_INLINE_FUNCTION
  bool has_predecessor() const { return m_predecessor != nullptr; }

  KOKKOS_INLINE_FUNCTION
  void clear_predecessor() { m_predecessor = nullptr; }

  KOKKOS_INLINE_FUNCTION
  task_base_type& get_predecessor() const {
    KOKKOS_EXPECTS(m_predecessor != nullptr);
    return *m_predecessor;
  }

  KOKKOS_INLINE_FUNCTION
  void set_predecessor(task_base_type& predecessor)
  {
    KOKKOS_EXPECTS(m_predecessor == nullptr);
    // Increment the reference count so that predecessor doesn't go away
    // before this task is enqueued.
    // (should be memory order acquire)
    predecessor.increment_reference_count();
    m_predecessor = &predecessor;
  }

  template <class TeamMember>
  void run(TeamMember& member) {
    (*m_apply)(this, &member);
  }
};

template <class ResultType>
struct TaskResultStorage {
  ResultType m_value;
  ResultType& reference() { return m_value; }
};

template <>
struct TaskResultStorage<void> { };

template <
  class TaskQueueTraits,
  class Scheduler,
  class ResultType,
  class FunctorType
>
class RunnableTask
  : public RunnableTaskBase<TaskQueueTraits>, // must be first base class for allocation reasons!!!
    public FunctorType,
    public TaskResultStorage<ResultType>
{
private:

  using base_t = RunnableTaskBase<TaskQueueTraits>;
  using task_base_type = TaskNode<TaskQueueTraits>;
  using specialization = TaskQueueSpecialization<Scheduler>;
  using member_type = typename specialization::member_type;
  using result_type = ResultType;
  using functor_type = FunctorType;

public:

  template <class... Args>
    // requires std::is_constructible_v<base_t, Args&&...>
  KOKKOS_INLINE_FUNCTION
  constexpr explicit
  RunnableTask(
    FunctorType&& functor,
    Args&&... args
  ) : base_t(
        std::forward<Args>(args)...
      ),
      functor_type(std::move(functor))
  { }

  KOKKOS_INLINE_FUNCTION
  ~RunnableTask() = delete;


  KOKKOS_INLINE_FUNCTION
  void apply_functor(member_type* member, void*)
  {
    this->functor_type::operator()(*member);
  }

  template< typename T >
  KOKKOS_INLINE_FUNCTION
  void apply_functor(member_type* member, T* const result)
  {
    this->functor_type::operator()(*member, *result);
  }

  KOKKOS_FUNCTION static
  void destroy( task_base_type * root )
  {
    TaskResult<result_type>::destroy(root);
  }

  KOKKOS_FUNCTION static
  void apply(task_base_type* self, void* member_as_void)
  {
    RunnableTask* const task = static_cast<RunnableTask*>(self);
    member_type* const member = reinterpret_cast<member_type*>(member_as_void);
    result_type* const result = TaskResult< result_type >::ptr( task );

    // Task may be serial or team.
    // If team then must synchronize before querying if respawn was requested.
    // If team then only one thread calls destructor.

    const bool only_one_thread =
#if defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_CUDA)
      0 == threadIdx.x && 0 == threadIdx.y ;
#else
      0 == member->team_rank();
#endif

    task->apply_functor(member, result);

    member->team_barrier();

    if ( only_one_thread && !(task->get_respawn_flag()) ) {
      // Did not respawn, destroy the functor to free memory.
      task->functor_type::~functor_type();
      // Cannot destroy and deallocate the task until its dependences
      // have been processed.
    }
  }

};





////////////////////////////////////////////////////////////////////////////////
// BEGIN OLD CODE
////////////////////////////////////////////////////////////////////////////////

/** \brief  Base class for task management, access, and execution.
 *
 *  Inheritance structure to allow static_cast from the task root type
 *  and a task's FunctorType.
 *
 *    // Enable a functor to access the base class
 *    // and provide memory for result value.
 *    TaskBase< Space , ResultType , FunctorType >
 *      : TaskBase< void , void , void >
 *      , FunctorType
 *      { ... };
 *    Followed by memory allocated for result value.
 *
 *
 *  States of a task:
 *
 *    Constructing State, NOT IN a linked list
 *      m_wait == 0
 *      m_next == 0
 *
 *    Scheduling transition : Constructing -> Waiting
 *      before:
 *        m_wait == 0
 *        m_next == this task's initial dependence, 0 if none
 *      after:
 *        m_wait == EndTag
 *        m_next == EndTag
 *
 *    Waiting State, IN a linked list
 *      m_apply != 0
 *      m_queue != 0
 *      m_ref_count > 0
 *      m_wait == head of linked list of tasks waiting on this task
 *      m_next == next of linked list of tasks
 *
 *    transition : Waiting -> Executing
 *      before:
 *        m_next == EndTag
 *      after::
 *        m_next == LockTag
 *
 *    Executing State, NOT IN a linked list
 *      m_apply != 0
 *      m_queue != 0
 *      m_ref_count > 0
 *      m_wait == head of linked list of tasks waiting on this task
 *      m_next == LockTag
 *
 *    Respawn transition : Executing -> Executing-Respawn
 *      before:
 *        m_next == LockTag
 *      after:
 *        m_next == this task's updated dependence, 0 if none
 *
 *    Executing-Respawn State, NOT IN a linked list
 *      m_apply != 0
 *      m_queue != 0
 *      m_ref_count > 0
 *      m_wait == head of linked list of tasks waiting on this task
 *      m_next == this task's updated dependence, 0 if none
 *
 *    transition : Executing -> Complete
 *      before:
 *        m_wait == head of linked list
 *      after:
 *        m_wait == LockTag
 *
 *    Complete State, NOT IN a linked list
 *      m_wait == LockTag: cannot add dependence (<=> complete)
 *      m_next == LockTag: not a member of a wait queue
 *
 */
class TaskBase
{
public:

  enum : int16_t   { TaskTeam = 0 , TaskSingle = 1 , Aggregate = 2 };
  enum : uintptr_t { LockTag = ~uintptr_t(0) , EndTag = ~uintptr_t(1) };

  template<typename, typename> friend class Kokkos::BasicTaskScheduler ;

  using queue_type = TaskQueueBase;

  using function_type = void(*)( TaskBase * , void * );
  typedef void (* destroy_type) ( TaskBase * );

  // sizeof(TaskBase) == 48

  function_type m_apply = nullptr;         ///< Apply function pointer
  queue_type* m_queue = nullptr;          ///< Pointer to the scheduler
  TaskBase* m_next = nullptr; ///< next in linked list of ready tasks
  TaskBase* m_wait = nullptr; ///< Queue of tasks waiting on this
  int32_t m_ref_count = 0;
  int32_t m_alloc_size = 0;
  int32_t m_dep_count ;                    ///< Aggregate's number of dependences
  int16_t        m_task_type ;   ///< Type of task
  int16_t        m_priority ;    ///< Priority of runnable task

  TaskBase( TaskBase && ) = delete ;
  TaskBase( const TaskBase & ) = delete ;
  TaskBase & operator = ( TaskBase && ) = delete ;
  TaskBase & operator = ( const TaskBase & ) = delete ;

#ifdef KOKKOS_CUDA_9_DEFAULTED_BUG_WORKAROUND
  KOKKOS_INLINE_FUNCTION ~TaskBase() {};
#else
  KOKKOS_INLINE_FUNCTION ~TaskBase() = default;
#endif

  KOKKOS_INLINE_FUNCTION constexpr
  TaskBase()
    : m_apply( nullptr )
    , m_queue( nullptr )
    , m_wait( nullptr )
    , m_next( nullptr )
    , m_ref_count( 0 )
    , m_alloc_size( 0 )
    , m_dep_count( 0 )
    , m_task_type( 0 )
    , m_priority( 0 )
    {}

  //----------------------------------------

  KOKKOS_INLINE_FUNCTION
  TaskBase * volatile * aggregate_dependences() volatile
    { return reinterpret_cast<TaskBase*volatile*>( this + 1 ); }

  KOKKOS_INLINE_FUNCTION
  bool requested_respawn()
    {
      // This should only be called when a task has finished executing and is
      // in the transition to either the complete or executing-respawn state.
      TaskBase * const lock = reinterpret_cast< TaskBase * >( LockTag );
      return lock != m_next;
    }

  KOKKOS_INLINE_FUNCTION
  void add_dependence( TaskBase* dep )
    {
      // Precondition: lock == m_next

      TaskBase * const lock = (TaskBase *) LockTag ;

      // Assign dependence to m_next.  It will be processed in the subsequent
      // call to schedule.  Error if the dependence is reset.
      if ( lock != Kokkos::atomic_exchange( & m_next, dep ) ) {
        Kokkos::abort("TaskScheduler ERROR: resetting task dependence");
      }

      if ( 0 != dep ) {
        // The future may be destroyed upon returning from this call
        // so increment reference count to track this assignment.
        Kokkos::atomic_increment( &(dep->m_ref_count) );
      }
    }

  //----------------------------------------

  KOKKOS_INLINE_FUNCTION
  int32_t reference_count() const
    { return *((int32_t volatile *)( & m_ref_count )); }

};

static_assert( sizeof(TaskBase) == 48
             , "Verifying expected sizeof(TaskBase)" );

} /* namespace Impl */
} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template< class Scheduler, typename ResultType , class FunctorType >
class Task
  : public TaskBase,
    public FunctorType
{
public:

  Task() = delete ;
  Task( Task && ) = delete ;
  Task( const Task & ) = delete ;
  Task & operator = ( Task && ) = delete ;
  Task & operator = ( const Task & ) = delete ;


  using root_type = TaskBase;
  using functor_type = FunctorType ;
  using result_type = ResultType ;

  using specialization = TaskQueueSpecialization<Scheduler> ;
  using member_type = typename specialization::member_type ;

  KOKKOS_INLINE_FUNCTION
  void apply_functor( member_type * const member , void * )
    { this->functor_type::operator()( *member ); }

  template< typename T >
  KOKKOS_INLINE_FUNCTION
  void apply_functor( member_type * const member
                    , T           * const result )
    { this->functor_type::operator()( *member , *result ); }

  KOKKOS_FUNCTION static
  void destroy( root_type * root )
  {
    TaskResult<result_type>::destroy(root);
  }

  KOKKOS_FUNCTION static
  void apply( root_type * root , void * exec )
    {
      Task* const task = static_cast< Task * >( root );
      member_type * const member = reinterpret_cast< member_type * >( exec );
      result_type * const result = TaskResult< result_type >::ptr( task );

      // Task may be serial or team.
      // If team then must synchronize before querying if respawn was requested.
      // If team then only one thread calls destructor.

      const bool only_one_thread =
#if defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_CUDA)
        0 == threadIdx.x && 0 == threadIdx.y ;
#else
        0 == member->team_rank();
#endif

      task->apply_functor( member , result );

      member->team_barrier();

      if ( only_one_thread && !(task->requested_respawn()) ) {
        // Did not respawn, destroy the functor to free memory.
        task->functor_type::~functor_type();
        // Cannot destroy and deallocate the task until its dependences
        // have been processed.
      }
    }

  // Constructor for runnable task
  KOKKOS_INLINE_FUNCTION constexpr
  Task( FunctorType && arg_functor )
    : root_type() , functor_type( std::move(arg_functor) )
  { }

  KOKKOS_INLINE_FUNCTION
  ~Task() = delete;
};

} /* namespace Impl */
} /* namespace Kokkos */

////////////////////////////////////////////////////////////////////////////////
// END OLD CODE
////////////////////////////////////////////////////////////////////////////////

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #if defined( KOKKOS_ENABLE_TASKDAG ) */
#endif /* #ifndef KOKKOS_IMPL_TASKBASE_HPP */

