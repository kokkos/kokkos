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
// Questions? Contact  H. Carter Edwards (hcedwar@sandia.gov)
// 
// ************************************************************************
//@HEADER
*/

// Experimental unified task-data parallel manycore LDRD

#ifndef KOKKOS_IMPL_TASKQUEUE_HPP
#define KOKKOS_IMPL_TASKQUEUE_HPP

#include <string>
#include <typeinfo>
#include <stdexcept>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

template< typename > class TaskPolicy ;

template< typename Arg1 = void , typename Arg2 = void > class Future ;

} /* namespace Kokkos */

namespace Kokkos {
namespace Impl {

template< typename , typename , typename > class TaskBase ;
template< typename > class TaskExec ;

} /* namespace Impl */
} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template< typename Space >
class TaskQueueSpecialization ;

/**
 *
 *  Required: TaskQueue<Space> : public TaskQueue<void>
 *
 */
template< typename ExecSpace >
class TaskQueue {
private:

  friend class TaskQueueSpecialization< ExecSpace > ;
  friend class Kokkos::TaskPolicy< ExecSpace > ;

  using execution_space = ExecSpace ;
  using specialization  = TaskQueueSpecialization< execution_space > ;
  using memory_space    = typename specialization::memory_space ;
  using device_type     = Kokkos::Device< execution_space , memory_space > ;
  using memory_pool     = Kokkos::Experimental::MemoryPool< device_type > ;
  using task_root_type  = Kokkos::Impl::TaskBase<execution_space,void,void> ;

  struct Destroy {
    TaskQueue * m_queue ;
    void destroy_shared_allocation();
  };

  //----------------------------------------

  enum : int { NumQueue = 3 };

  // Queue is organized as [ priority ][ type ]

  memory_pool               m_memory ;
  task_root_type * volatile m_ready[ NumQueue ][ 2 ];
  long                      m_accum_alloc ;
  int                       m_count_alloc ;
  int                       m_max_alloc ;
  int                       m_ready_count ;

  //----------------------------------------

  ~TaskQueue();
  TaskQueue() = delete ;
  TaskQueue( TaskQueue && ) = delete ;
  TaskQueue( TaskQueue const & ) = delete ;
  TaskQueue & operator = ( TaskQueue && ) = delete ;
  TaskQueue & operator = ( TaskQueue const & ) = delete ;

  TaskQueue
    ( const memory_space & arg_space
    , unsigned const arg_memory_pool_capacity
    , unsigned const arg_memory_pool_superblock_capacity_log2
    );

  KOKKOS_FUNCTION
  void schedule( task_root_type * const , task_root_type * );

  KOKKOS_FUNCTION
  void complete( task_root_type * );

  KOKKOS_FUNCTION
  static bool push_task( task_root_type * volatile * const
                       , task_root_type * const );

  KOKKOS_FUNCTION
  static task_root_type * pop_task( task_root_type * volatile * const );

  KOKKOS_FUNCTION static
  void decrement( task_root_type * task );

public:

  void execute() { specialization::execute( this ); }

  template< typename LV , typename RV >
  KOKKOS_FUNCTION static
  void assign( TaskBase< execution_space,LV,void> ** const lhs
             , TaskBase< execution_space,RV,void> *  const rhs )
    {
      using task_lhs = TaskBase< execution_space,LV,void> ;
#if 0
  {
    printf( "assign( 0x%lx { 0x%lx %d %d } , 0x%lx { 0x%lx %d %d } )\n"
          , uintptr_t( lhs ? *lhs : 0 )
          , uintptr_t( lhs && *lhs ? (*lhs)->m_next : 0 )
          , int( lhs && *lhs ? (*lhs)->m_task_type : 0 )
          , int( lhs && *lhs ? (*lhs)->m_ref_count : 0 )
          , uintptr_t(rhs)
          , uintptr_t( rhs ? rhs->m_next : 0 )
          , int( rhs ? rhs->m_task_type : 0 )
          , int( rhs ? rhs->m_ref_count : 0 )
          );
    fflush( stdout );
  }
#endif

      if ( *lhs ) decrement( *lhs );
      if ( rhs ) { Kokkos::atomic_fetch_add( &(rhs->m_ref_count) , 1 ); }

      // Force write of *lhs

      *static_cast< task_lhs * volatile * >(lhs) = rhs ;

      Kokkos::memory_fence();
    }

  KOKKOS_FUNCTION
  void * allocate( size_t n ); ///< Allocate from the memory pool

  KOKKOS_FUNCTION
  void deallocate( void * p , size_t n ); ///< Deallocate to the memory pool
};

} /* namespace Impl */
} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

/** \brief  Base class for task management, access, and execution.
 *
 *  Inheritance structure to allow static_cast from the task root type
 *  and a task's FunctorType.
 *
 *    // Enable a Future to access result data
 *    TaskBase< Space , ResultType , void >
 *      : TaskBase< void , void , void >
 *      { ... };
 *
 *    // Enable a functor to access the base class
 *    TaskBase< Space , ResultType , FunctorType >
 *      : TaskBase< Space , ResultType , void >
 *      , FunctorType
 *      { ... };
 *
 *
 *  States of a task:
 *
 *    Constructing State:
 *      m_apply == 0
 *      m_queue == 0
 *      m_ref_count == 0
 *
 *    Waiting State:
 *      m_apply != 0
 *      m_queue != 0
 *      m_ref_count > 0
 *      m_wait == EndTag OR valid task
 *      m_next == EndTag OR valid task
 *
 *    Executing State:
 *      m_apply != 0
 *      m_queue != 0
 *      m_ref_count > 0
 *      m_wait == EndTag OR valid task
 *      m_next == LockTag
 *
 *    Executing-Respawn State:
 *      m_apply != 0
 *      m_queue != 0
 *      m_ref_count > 0
 *      m_wait == EndTag OR valid task
 *      m_next == 0 OR valid task
 *
 *    Complete State:
 *      m_wait == LockTag, cannot add dependence
 *      m_next == LockTag, not a member of a wait queue
 *
 *
 *  Invariants / conditions:
 *
 *    when m_next == LockTag then NOT a member of a queue
 *    when m_wait == LockTag then Complete 
 */
template< typename ExecSpace >
class TaskBase< ExecSpace , void , void >
{
public:

  enum : int16_t   { TaskTeam = 0 , TaskSingle = 1 , Aggregate = 2 };
  enum : uintptr_t { LockTag = ~uintptr_t(0) , EndTag = ~uintptr_t(1) };

  using execution_space = ExecSpace ;
  using queue_type      = TaskQueue< execution_space > ;

  template< typename > friend class Kokkos::TaskPolicy ;

  typedef void (* function_type) ( TaskBase * , void * );

  // sizeof(TaskBase) == 48

  function_type  m_apply ;     ///< Apply function pointer
  queue_type   * m_queue ;     ///< Queue in which this task resides
  uintptr_t      m_wait ;      ///< Linked list of tasks waiting on this
  uintptr_t      m_next ;      ///< Waiting linked-list next
  int32_t        m_ref_count ; ///< Reference count
  int32_t        m_alloc_size ;///< Allocation size
  int32_t        m_dep_count ; ///< Aggregate's number of dependences
  int16_t        m_task_type ; ///< Type of task
  int16_t        m_priority ;  ///< Priority of runnable task

  TaskBase( TaskBase && ) = delete ;
  TaskBase( const TaskBase & ) = delete ;
  TaskBase & operator = ( TaskBase && ) = delete ;
  TaskBase & operator = ( const TaskBase & ) = delete ;

  KOKKOS_INLINE_FUNCTION ~TaskBase() = default ;

  KOKKOS_INLINE_FUNCTION
  constexpr TaskBase() noexcept
    : m_apply(0)
    , m_queue(0)
    , m_wait( EndTag  /* head of empty wait list */ )
    , m_next( LockTag /* not a member of a queue */ )
    , m_ref_count(0)
    , m_alloc_size(0)
    , m_dep_count(0)
    , m_task_type(0)
    , m_priority(0)
    {}

  //----------------------------------------

  KOKKOS_INLINE_FUNCTION
  TaskBase ** aggregate_dependences()
    { return reinterpret_cast<TaskBase**>( this + 1 ); }

  using get_return_type = void ;

  KOKKOS_INLINE_FUNCTION
  get_return_type get() const {}
};

template < typename ExecSpace , typename ResultType >
class TaskBase< ExecSpace , ResultType , void >
  : public TaskBase< ExecSpace , void , void >
{
private:

  static_assert( sizeof(TaskBase<ExecSpace,void,void>) == 48 , "" );

  TaskBase( TaskBase && ) = delete ;
  TaskBase( const TaskBase & ) = delete ;
  TaskBase & operator = ( TaskBase && ) = delete ;
  TaskBase & operator = ( const TaskBase & ) = delete ;

public:

  ResultType   m_result ;

  KOKKOS_INLINE_FUNCTION ~TaskBase() = default ;

  KOKKOS_INLINE_FUNCTION
  TaskBase()
    : TaskBase< ExecSpace , void , void >()
    , m_result()
    {}

  using get_return_type = ResultType const & ;

  KOKKOS_INLINE_FUNCTION
  get_return_type get() const { return m_result ; }
};


template< typename ExecSpace , typename ResultType , typename FunctorType >
class TaskBase
  : public TaskBase< ExecSpace , ResultType , void >
  , public FunctorType
{
private:

  TaskBase() = delete ;
  TaskBase( TaskBase && ) = delete ;
  TaskBase( const TaskBase & ) = delete ;
  TaskBase & operator = ( TaskBase && ) = delete ;
  TaskBase & operator = ( const TaskBase & ) = delete ;

public:

  using root_type    = TaskBase< ExecSpace , void , void > ;
  using base_type    = TaskBase< ExecSpace , ResultType , void > ;
  using member_type  = TaskExec< ExecSpace > ;
  using functor_type = FunctorType ;
  using result_type  = ResultType ;

  template< typename Type >
  KOKKOS_INLINE_FUNCTION static
  void apply_functor
    ( Type * const task
    , typename std::enable_if
        < std::is_same< typename Type::result_type , void >::value
        , member_type * const 
        >::type member
    )
    {
      using fType = typename Type::functor_type ;
      static_cast<fType*>(task)->operator()( *member );
    }

  template< typename Type >
  KOKKOS_INLINE_FUNCTION static
  void apply_functor
    ( Type * const task
    , typename std::enable_if
        < ! std::is_same< typename Type::result_type , void >::value
        , member_type * const 
        >::type member
    )
    {
      using fType = typename Type::functor_type ;
      static_cast<fType*>(task)->operator()( *member , task->m_result );
    }

  KOKKOS_FUNCTION static
  void apply( root_type * root , void * exec )
    {
      member_type * const member = reinterpret_cast< member_type * >( exec );
      TaskBase    * const task = static_cast< TaskBase * >( root );

      TaskBase::template apply_functor( task , member );

      // Task may be serial or team.
      // If team then must synchronize before querying task->m_next.
      // If team then only one thread calls destructor.

      member->team_barrier();

      if ( 0 == member->team_rank() &&
           root_type::LockTag == task->m_next ) {
        // Did not respawn, destroy the functor to free memory
        static_cast<functor_type*>(task)->~functor_type();
      }
    }

  KOKKOS_INLINE_FUNCTION
  TaskBase( FunctorType const & arg_functor )
    : base_type()
    , FunctorType( arg_functor )
    {}

  KOKKOS_INLINE_FUNCTION
  ~TaskBase() {}
};

} /* namespace Impl */
} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #ifndef KOKKOS_IMPL_TASKQUEUE_HPP */

