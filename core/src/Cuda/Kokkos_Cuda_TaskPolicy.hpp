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

#ifndef KOKKOS_CUDA_TASKPOLICY_HPP
#define KOKKOS_CUDA_TASKPOLICY_HPP

#include <Kokkos_Core_fwd.hpp>

#if defined( KOKKOS_HAVE_CUDA )

#include <Kokkos_Cuda.hpp>
#include <Kokkos_TaskPolicy.hpp>

//----------------------------------------------------------------------------

namespace Kokkos {
namespace Experimental {
namespace Impl {

struct CudaTaskPolicyQueue {

  typedef Kokkos::Experimental::MemoryPool< Kokkos::CudaUVMSpace >
    memory_space ;

  typedef Kokkos::Experimental::Impl::TaskMember< Kokkos::Cuda , void , void >
    task_root_type ;

  memory_space     m_space ;
  task_root_type * m_ready_team ;
  task_root_type * m_ready_serial ;
  task_root_type * m_garbage ;
  int              m_team_size ;
  int              m_default_dependence_capacity ;
  int              m_task_count ;

  __device__
  void driver();

  __device__
  void complete_executed_task( task_root_type * );

  __device__ static
  task_root_type * pop_ready_task( task_root_type * volatile * const queue );

  CudaTaskPolicyQueue( const unsigned arg_memory_pool_chunk
                     , const unsigned arg_memory_pool_size
                     , const unsigned arg_default_dependence_capacity
                     );

  struct Destroy {
    CudaTaskPolicyQueue * m_queue ;
    void destroy_shared_allocation();
  };
};

} /* namespace Impl */
} /* namespace Experimental */
} /* namespace Kokkos */


namespace Kokkos {
namespace Experimental {
namespace Impl {

/** \brief  Base class for all Kokkos::Cuda tasks */
template<>
class TaskMember< Kokkos::Cuda , void , void > {
public:

  typedef void         (* function_single_type) ( TaskMember * );
  typedef void         (* function_team_type)   ( TaskMember * , Kokkos::Impl::CudaTeamMember & );

private:

  friend struct CudaTaskPolicyQueue ;

  // Needed to disambiguate references to base class variables
  // without triggering a false-positive on Intel compiler warning #955.
  typedef TaskMember< Kokkos::Cuda , void , void > SelfType ;

  CudaTaskPolicyQueue  * m_policy_queue ;
  function_team_type     m_cuda_team ;    ///< Apply function
  function_single_type   m_cuda_serial ;  ///< Apply function
  TaskMember **          m_dep ;          ///< Dependences
  TaskMember *           m_wait ;         ///< Linked list of tasks waiting on this task
  TaskMember *           m_next ;         ///< Linked list of tasks waiting on a different task
  int                    m_size_alloc ;
  int                    m_dep_capacity ; ///< Capacity of dependences
  int                    m_dep_size ;     ///< Actual count of dependences
  int                    m_shmem_size ;
  int                    m_ref_count ;    ///< Reference count
  int                    m_state ;        ///< State of the task

  // 6 pointers + 6 integers = 9 words = 72 bytes

  TaskMember( TaskMember && ) = delete ;
  TaskMember( const TaskMember & ) = delete ;
  TaskMember & operator = ( TaskMember && ) = delete ;
  TaskMember & operator = ( const TaskMember & ) = delete ;

protected:

  TaskMember()
    : m_policy_queue(0)
    , m_cuda_team(0)
    , m_cuda_serial(0)
    , m_dep(0)
    , m_wait(0)
    , m_next(0)
    , m_size_alloc(0)
    , m_dep_capacity(0)
    , m_dep_size(0)
    , m_shmem_size(0)
    , m_ref_count(0)
    , m_state( TASK_STATE_CONSTRUCTING )
    {}

public:

  ~TaskMember();

  //----------------------------------------
  /*  Inheritence Requirements on task types:
   *
   *    class DerivedTaskType
   *      : public TaskMember< Cuda , DerivedType::value_type , FunctorType >
   *      { ... };
   *
   *    class TaskMember< Cuda , DerivedType::value_type , FunctorType >
   *      : public TaskMember< Cuda , DerivedType::value_type , void >
   *      , public Functor
   *      { ... };
   *
   *  If value_type != void
   *    class TaskMember< Cuda , value_type , void >
   *      : public TaskMember< Cuda , void , void >
   *
   *  Allocate space for DerivedTaskType followed by TaskMember*[ dependence_capacity ]
   *
   */
  //----------------------------------------
  // If after the 'apply' the task's state is waiting 
  // then it will be rescheduled and called again.
  // Otherwise the functor must be destroyed.

  template< class DerivedTaskType , class Tag >
  __device__ static
  void apply_single(
    typename std::enable_if
      <( std::is_same< Tag , void >::value &&
        std::is_same< typename DerivedTaskType::result_type , void >::value
       ), TaskMember * >::type t )
    {
      typedef typename DerivedTaskType::functor_type  functor_type ;

      DerivedTaskType * const self = static_cast< DerivedTaskType * >(t);

      static_cast<functor_type*>(self)->apply();

      if ( self->m_state == int(Kokkos::Experimental::TASK_STATE_EXECUTING) ) {
        static_cast<functor_type*>(self)->~functor_type();
      }

      t->m_policy_queue->complete_executed_task( t );
    }

  template< class DerivedTaskType , class Tag >
  __device__ static
  void apply_single(
    typename std::enable_if
      <( std::is_same< Tag , void >::value &&
        ! std::is_same< typename DerivedTaskType::result_type , void >::value
       ), TaskMember * >::type t )
    {
      typedef typename DerivedTaskType::functor_type  functor_type ;
      typedef typename DerivedTaskType::result_type   result_type ;

      DerivedTaskType * const self = static_cast< DerivedTaskType * >(t);

      static_cast<functor_type*>(self)->apply( self->m_result );

      if ( self->m_state == int(Kokkos::Experimental::TASK_STATE_EXECUTING) ) {
        static_cast<functor_type*>(self)->~functor_type();
      }

      t->m_policy_queue->complete_executed_task( t );
    }

  //----------------------------------------

  template< class DerivedTaskType , class Tag >
  __device__ static
  void apply_team(
    typename std::enable_if
      <( std::is_same<Tag,void>::value &&
         std::is_same<typename DerivedTaskType::result_type,void>::value
       ), TaskMember * >::type t
    , Kokkos::Impl::CudaTeamMember & member
    )
    {
      typedef typename DerivedTaskType::functor_type functor_type ;

      DerivedTaskType * const self = static_cast< DerivedTaskType * >(t);

      static_cast<functor_type*>(self)->apply( member );

      member.team_barrier();

      if ( member.team_rank() == 0 &&
           self->m_state == int(Kokkos::Experimental::TASK_STATE_EXECUTING) ) {

        static_cast<functor_type*>(self)->~functor_type();

        t->m_policy_queue->complete_executed_task( t );
      }
    }

  template< class DerivedTaskType , class Tag >
  __device__ static
  void apply_team(
    typename std::enable_if
      <( std::is_same<Tag,void>::value &&
         ! std::is_same<typename DerivedTaskType::result_type,void>::value
       ), TaskMember * >::type t
    , Kokkos::Impl::CudaTeamMember & member
    )
    {
      typedef typename DerivedTaskType::functor_type functor_type ;

      DerivedTaskType * const self = static_cast< DerivedTaskType * >(t);

      static_cast<functor_type*>(self)->apply( member , self->m_result );

      member.team_barrier();

      if ( member.team_rank() == 0 &&
           self->m_state == int(Kokkos::Experimental::TASK_STATE_EXECUTING) ) {

        static_cast<functor_type*>(self)->~functor_type();

        t->m_policy_queue->complete_executed_task( t );
      }
    }

  //----------------------------------------

  __device__ void reschedule();
  __device__ void schedule();

  //----------------------------------------

  KOKKOS_FUNCTION static
  void assign( TaskMember ** const lhs , TaskMember * const rhs );

  TaskMember * get_dependence( int i ) const ;

  KOKKOS_INLINE_FUNCTION
  int get_dependence() const
    { return m_dep_size ; }

  KOKKOS_FUNCTION void clear_dependence();
  KOKKOS_FUNCTION void add_dependence( TaskMember * before );

  //----------------------------------------

  typedef FutureValueTypeIsVoidError get_result_type ;

  KOKKOS_INLINE_FUNCTION
  get_result_type get() const { return get_result_type() ; }

  KOKKOS_INLINE_FUNCTION
  Kokkos::Experimental::TaskState get_state() const { return Kokkos::Experimental::TaskState( m_state ); }

};

/** \brief  A Future< Kokkos::Cuda , ResultType > will cast
 *          from  TaskMember< Kokkos::Cuda , void , void >
 *          to    TaskMember< Kokkos::Cuda , ResultType , void >
 *          to query the result.
 */
template< class ResultType >
class TaskMember< Kokkos::Cuda , ResultType , void >
  : public TaskMember< Kokkos::Cuda , void , void >
{
public:

  typedef ResultType result_type ;

  result_type  m_result ;

  typedef const result_type & get_result_type ;

  KOKKOS_INLINE_FUNCTION
  get_result_type get() const { return m_result ; }

  inline
  TaskMember() : TaskMember< Kokkos::Cuda , void , void >(), m_result() {}

#if defined( KOKKOS_HAVE_CXX11 )
  TaskMember( const TaskMember & ) = delete ;
  TaskMember & operator = ( const TaskMember & ) = delete ;
#else
private:
  TaskMember( const TaskMember & );
  TaskMember & operator = ( const TaskMember & );
#endif
};

/** \brief  Callback functions will cast
 *          from  TaskMember< Kokkos::Cuda , void , void >
 *          to    TaskMember< Kokkos::Cuda , ResultType , FunctorType >
 *          to execute work functions.
 */
template< class ResultType , class FunctorType >
class TaskMember< Kokkos::Cuda , ResultType , FunctorType >
  : public TaskMember< Kokkos::Cuda , ResultType , void >
  , public FunctorType
{
public:
  typedef ResultType   result_type ;
  typedef FunctorType  functor_type ;

  inline
  TaskMember( const functor_type & arg_functor )
    : TaskMember< Kokkos::Cuda , ResultType , void >()
    , functor_type( arg_functor )
    {}
};

} /* namespace Impl */
} /* namespace Experimental */
} /* namespace Kokkos */

template< class DerivedTaskType , class Tag >
__global__
void kokkos_cuda_task_policy_set_apply
  ( Kokkos::Experimental::Impl::TaskMember< Kokkos::Cuda , void , void > * self
  , int set_team )
{
  typedef Kokkos::Experimental::Impl::TaskMember< Kokkos::Cuda , void , void > Task ;

  if ( set_team ) {
    self->m_cuda_team =
      Task::template apply_team< DerivedTaskType , Tag > ;
  }
  else {
    self->m_cuda_serial =
      Task::template apply_single< DerivedTaskType , Tag > ;
  }
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Experimental {

void wait( TaskPolicy< Kokkos::Cuda > & );

template<>
class TaskPolicy< Kokkos::Cuda >
{
public:

  typedef Kokkos::Cuda                  execution_space ;
  typedef TaskPolicy                    execution_policy ;
  typedef Kokkos::Impl::CudaTeamMember  member_type ;

private:

  typedef Impl::TaskMember< Kokkos::Cuda , void , void >  task_root_type ;
  typedef Kokkos::Experimental::MemoryPool< Kokkos::CudaUVMSpace > memory_space ;
  typedef Kokkos::Experimental::Impl::SharedAllocationTracker track_type ;

  track_type                   m_track ;
  Impl::CudaTaskPolicyQueue  * m_cuda_uvm_queue ;

  template< class FunctorType >
  static inline
  const task_root_type * get_task_root( const FunctorType * f )
    {
      typedef Impl::TaskMember< execution_space , typename FunctorType::value_type , FunctorType > task_type ;
      return static_cast< const task_root_type * >( static_cast< const task_type * >(f) );
    }

  template< class FunctorType >
  static inline
  task_root_type * get_task_root( FunctorType * f )
    {
      typedef Impl::TaskMember< execution_space , typename FunctorType::value_type , FunctorType > task_type ;
      return static_cast< task_root_type * >( static_cast< task_type * >(f) );
    }


  /** \brief  Allocate and construct a task.
   *
   *  Allocate space for DerivedTaskType followed
   *  by TaskMember*[ dependence_capacity ]
   */
  template< class DerivedTaskType , class Tag >
  KOKKOS_INLINE_FUNCTION
  task_root_type *
  create( const typename DerivedTaskType::functor_type &  arg_functor
        , const int                                       arg_is_team
        , const unsigned                                  arg_team_shmem
        , const unsigned                                  arg_dependence_capacity
        )
    {
      typedef typename DerivedTaskType::functor_type functor_type ;

      enum { padding_size = sizeof(DerivedTaskType) % sizeof(task_root_type*)
                          ? sizeof(task_root_type*) - sizeof(DerivedTaskType) % sizeof(task_root_type*) : 0 };
      enum { derived_size = sizeof(DerivedTaskType) + padding_size };

      const unsigned dep_capacity
        = ~0u == arg_dependence_capacity
        ? m_cuda_uvm_queue->m_default_dependence_capacity
        : arg_dependence_capacity ;

      const unsigned size_alloc =
         derived_size + sizeof(task_root_type*) * dep_capacity ;

      DerivedTaskType * const task =
        reinterpret_cast<DerivedTaskType*>(
          m_cuda_uvm_queue->m_space.allocate( size_alloc ) );

      // Copy construct the functor into the task's UVM memory.
      // Destructor will called on the device when the task is complete.
      new( task ) DerivedTaskType( arg_functor );

      // Set values

      task->task_root_type::m_policy_queue = m_cuda_uvm_queue ;
      task->task_root_type::m_dep          = (task_root_type**)( ((unsigned char *)task) + derived_size );
      task->task_root_type::m_dep_capacity = dep_capacity ;
      task->task_root_type::m_size_alloc   = size_alloc ;
      task->task_root_type::m_shmem_size   = arg_team_shmem ;

      for ( unsigned i = 0 ; i < arg_dependence_capacity ; ++i ) task->task_root_type::m_dep[i] = 0 ;

      // Set the apply pointer on the device via kernel launch.
      // This will cause the allocated UVM memory to be copied to the device.

      ::kokkos_cuda_task_policy_set_apply<DerivedTaskType,Tag><<<1,1>>>( task , arg_is_team );

      // Synchronize to guarantee non-concurrent access
      // between host and device.

      cudaDeviceSynchronize();

      return static_cast< task_root_type * >( task );
    }

public:

  TaskPolicy( const unsigned  arg_memory_pool_chunk
            , const unsigned  arg_memory_pool_size
            , const unsigned  arg_default_dependence_capacity = 4
            );

  KOKKOS_INLINE_FUNCTION
  TaskPolicy() : m_track(), m_cuda_uvm_queue(0) {}

  KOKKOS_INLINE_FUNCTION
  TaskPolicy( TaskPolicy && ) = default ;

  KOKKOS_INLINE_FUNCTION
  TaskPolicy( const TaskPolicy & ) = default ;

  KOKKOS_INLINE_FUNCTION
  TaskPolicy & operator = ( TaskPolicy && ) = default ;

  KOKKOS_INLINE_FUNCTION
  TaskPolicy & operator = ( const TaskPolicy & ) = default ;

  //----------------------------------------
  // Create serial-thread task

  template< class FunctorType >
  KOKKOS_INLINE_FUNCTION
  Future< typename FunctorType::value_type , execution_space >
  create( const FunctorType & functor
        , const unsigned dependence_capacity = ~0u ) const
    {
      typedef typename FunctorType::value_type  value_type ;

      typedef Impl::TaskMember< execution_space , value_type , FunctorType >
        task_type ;

      return Future< value_type , execution_space >(
        TaskPolicy::create< task_type , void >
          ( functor
          , 0 /* arg_is_team */
          , 0 /* team shmem */
          , dependence_capacity
          )
        );
    }

  //----------------------------------------
  // Create thread-team task

  template< class FunctorType >
  KOKKOS_INLINE_FUNCTION
  Future< typename FunctorType::value_type , execution_space >
  create_team( const FunctorType & functor
             , const unsigned dependence_capacity = ~0u ) const
    {
      typedef typename FunctorType::value_type  value_type ;

      typedef Impl::TaskMember< execution_space , value_type , FunctorType >
        task_type ;

      return Future< value_type , execution_space >(
        TaskPolicy::create< task_type , void >
          ( functor
          , 1 /* arg_set_team */
          , Kokkos::Impl::FunctorTeamShmemSize< FunctorType >::value
              ( functor , m_cuda_uvm_queue->m_team_size )
          , dependence_capacity
          )
        );
    }

  //----------------------------------------

  template< class A1 , class A2 , class A3 , class A4 >
  KOKKOS_INLINE_FUNCTION
  void add_dependence( const Future<A1,A2> & after
                     , const Future<A3,A4> & before
                     , typename std::enable_if
                        < std::is_same< typename Future<A1,A2>::execution_space , execution_space >::value
                          &&
                          std::is_same< typename Future<A3,A4>::execution_space , execution_space >::value
                        >::type * = 0
                      ) const
    { after.m_task->add_dependence( before.m_task ); }

  template< class FunctorType , class A3 , class A4 >
  KOKKOS_INLINE_FUNCTION
  void add_dependence( FunctorType * task_functor
                     , const Future<A3,A4> & before
                     , typename std::enable_if
                        < std::is_same< typename Future<A3,A4>::execution_space , execution_space >::value
                        >::type * = 0
                      ) const
    { get_task_root(task_functor)->add_dependence( before.m_task ); }


  template< class ValueType >
  const Future< ValueType , execution_space > &
    spawn( const Future< ValueType , execution_space > & f ) const
      {
        f.m_task->schedule();
        return f ;
      }

  template< class FunctorType >
  KOKKOS_INLINE_FUNCTION
  void respawn( FunctorType * task_functor ) const
    { get_task_root(task_functor)->reschedule(); }

  //----------------------------------------
  // Functions for an executing task functor to query dependences,
  // set new dependences, and respawn itself.

  template< class FunctorType >
  KOKKOS_INLINE_FUNCTION
  Future< void , execution_space >
  get_dependence( const FunctorType * task_functor , int i ) const
    {
      return Future<void,execution_space>(
        get_task_root(task_functor)->get_dependence(i)
        );
    }

  template< class FunctorType >
  KOKKOS_INLINE_FUNCTION
  int get_dependence( const FunctorType * task_functor ) const
    { return get_task_root(task_functor)->get_dependence(); }

  template< class FunctorType >
  KOKKOS_INLINE_FUNCTION
  void clear_dependence( FunctorType * task_functor ) const
    { get_task_root(task_functor)->clear_dependence(); }

  //----------------------------------------

  static member_type & member_single();

  friend void wait( TaskPolicy< Kokkos::Cuda > & );
};

} /* namespace Experimental */
} /* namespace Kokkos */

#endif /* #if defined( KOKKOS_HAVE_CUDA ) */

//----------------------------------------------------------------------------

#endif /* #ifndef KOKKOS_CUDA_TASKPOLICY_HPP */


