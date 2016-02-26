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

#ifndef KOKKOS_SERIAL_TASKPOLICY_HPP
#define KOKKOS_SERIAL_TASKPOLICY_HPP

#include <Kokkos_Macros.hpp>
#if defined( KOKKOS_HAVE_SERIAL )

#include <string>
#include <typeinfo>
#include <stdexcept>

#include <Kokkos_Serial.hpp>
#include <Kokkos_TaskPolicy.hpp>
#include <Kokkos_View.hpp>

#include <impl/Kokkos_FunctorAdapter.hpp>

//----------------------------------------------------------------------------
/*  Inheritance structure to allow static_cast from the task root type
 *  and a task's FunctorType.
 *
 *    task_root_type == TaskMember< Space , void , void >
 *
 *    TaskMember< PolicyType , ResultType , FunctorType >
 *      : TaskMember< PolicyType::Space , ResultType , FunctorType >
 *      { ... };
 *
 *    TaskMember< Space , ResultType , FunctorType >
 *      : TaskMember< Space , ResultType , void >
 *      , FunctorType
 *      { ... };
 *
 *  when ResultType != void
 *
 *    TaskMember< Space , ResultType , void >
 *      : TaskMember< Space , void , void >
 *      { ... };
 *
 */
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Experimental {
namespace Impl {

/** \brief  Base class for all tasks in the Serial execution space */
template<>
class TaskMember< Kokkos::Serial , void , void >
{
public:

  typedef void         (* function_apply_type)  ( TaskMember * );
  typedef void         (* function_dealloc_type)( TaskMember * );
  typedef TaskMember * (* function_verify_type) ( TaskMember * );

private:

  const function_dealloc_type  m_dealloc ; ///< Deallocation
  const function_verify_type   m_verify ;  ///< Result type verification
  const function_apply_type    m_apply ;   ///< Apply function
  TaskMember ** const          m_dep ;     ///< Dependences
  TaskMember *                 m_wait ;    ///< Linked list of tasks waiting on this task
  TaskMember *                 m_next ;    ///< Linked list of tasks waiting on a different task
  const int                    m_dep_capacity ; ///< Capacity of dependences
  int                          m_dep_size ;     ///< Actual count of dependences
  int                          m_ref_count ;    ///< Reference count
  int                          m_state ;        ///< State of the task

  // size = 6 Pointers + 4 ints

  TaskMember() /* = delete */ ;
  TaskMember( const TaskMember & ) /* = delete */ ;
  TaskMember & operator = ( const TaskMember & ) /* = delete */ ;

  static void * allocate( const unsigned arg_sizeof_derived , const unsigned arg_dependence_capacity );
  static void   deallocate( void * );

  void throw_error_add_dependence() const ;
  static void throw_error_verify_type();

  template < class DerivedTaskType >
  static
  void deallocate( TaskMember * t )
    {
      DerivedTaskType * ptr = static_cast< DerivedTaskType * >(t);
      ptr->~DerivedTaskType();
      deallocate( (void *) ptr );
    }

protected :

  ~TaskMember();

  // Used by TaskMember< Serial , ResultType , void >
  TaskMember( const function_verify_type   arg_verify
            , const function_dealloc_type  arg_dealloc
            , const function_apply_type    arg_apply
            , const unsigned               arg_sizeof_derived
            , const unsigned               arg_dependence_capacity
            );

  // Used for TaskMember< Serial , void , void >
  TaskMember( const function_dealloc_type  arg_dealloc
            , const function_apply_type    arg_apply
            , const unsigned               arg_sizeof_derived
            , const unsigned               arg_dependence_capacity
            );

public:

  template< typename ResultType >
  KOKKOS_FUNCTION static
  TaskMember * verify_type( TaskMember * t )
    {
      enum { check_type = ! Kokkos::Impl::is_same< ResultType , void >::value };

      if ( check_type && t != 0 ) {

        // Verify that t->m_verify is this function
        const function_verify_type self = & TaskMember::template verify_type< ResultType > ;

        if ( t->m_verify != self ) {
          t = 0 ;
#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
          throw_error_verify_type();
#endif
        }
      }
      return t ;
    }

  //----------------------------------------
  /*  Inheritence Requirements on task types:
   *    typedef  FunctorType::value_type  value_type ;
   *    class DerivedTaskType
   *      : public TaskMember< Serial , value_type , FunctorType >
   *      { ... };
   *    class TaskMember< Serial , value_type , FunctorType >
   *      : public TaskMember< Serial , value_type , void >
   *      , public Functor
   *      { ... };
   *  If value_type != void
   *    class TaskMember< Serial , value_type , void >
   *      : public TaskMember< Serial , void , void >
   *
   *  Allocate space for DerivedTaskType followed by TaskMember*[ dependence_capacity ]
   *
   */

  /** \brief  Allocate and construct a single-thread task */
  template< class DerivedTaskType >
  static
  TaskMember * create( const typename DerivedTaskType::functor_type &  arg_functor
                     , const unsigned                                  arg_dependence_capacity
                     )
    {
      typedef typename DerivedTaskType::functor_type  functor_type ;
      typedef typename functor_type::value_type       value_type ;

      DerivedTaskType * const task =
        new( allocate( sizeof(DerivedTaskType) , arg_dependence_capacity ) )
          DerivedTaskType( & TaskMember::template deallocate< DerivedTaskType >
                         , & TaskMember::template apply_single< functor_type , value_type >
                         , sizeof(DerivedTaskType)
                         , arg_dependence_capacity
                         , arg_functor );

      return static_cast< TaskMember * >( task );
    }

  /** \brief  Allocate and construct a data parallel task */
  template< class DerivedTaskType >
  static
  TaskMember * create( const typename DerivedTaskType::policy_type &   arg_policy
                     , const typename DerivedTaskType::functor_type &  arg_functor
                     , const unsigned                                  arg_dependence_capacity
                     )
    {
      DerivedTaskType * const task =
        new( allocate( sizeof(DerivedTaskType) , arg_dependence_capacity ) )
          DerivedTaskType( & TaskMember::template deallocate< DerivedTaskType >
                         , sizeof(DerivedTaskType)
                         , arg_dependence_capacity
                         , arg_policy
                         , arg_functor
                         );

      return static_cast< TaskMember * >( task );
    }

  /** \brief  Allocate and construct a thread-team task */
  template< class DerivedTaskType >
  static
  TaskMember * create_team( const typename DerivedTaskType::functor_type &  arg_functor
                          , const unsigned                                  arg_dependence_capacity
                          )
    {
      typedef typename DerivedTaskType::functor_type  functor_type ;
      typedef typename functor_type::value_type       value_type ;

      DerivedTaskType * const task =
        new( allocate( sizeof(DerivedTaskType) , arg_dependence_capacity ) )
          DerivedTaskType( & TaskMember::template deallocate< DerivedTaskType >
                         , & TaskMember::template apply_team< functor_type , value_type >
                         , sizeof(DerivedTaskType)
                         , arg_dependence_capacity
                         , arg_functor );

      return static_cast< TaskMember * >( task );
    }

  void schedule();
  static void execute_ready_tasks();

  //----------------------------------------

  typedef FutureValueTypeIsVoidError get_result_type ;

  KOKKOS_INLINE_FUNCTION
  get_result_type get() const { return get_result_type() ; }

  KOKKOS_INLINE_FUNCTION
  Kokkos::Experimental::TaskState get_state() const { return Kokkos::Experimental::TaskState( m_state ); }

  //----------------------------------------

#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
  static
  void assign( TaskMember ** const lhs , TaskMember * const rhs , const bool no_throw = false );
#else
  KOKKOS_INLINE_FUNCTION static
  void assign( TaskMember ** const lhs , TaskMember * const rhs , const bool no_throw = false ) {}
#endif

  KOKKOS_INLINE_FUNCTION
  TaskMember * get_dependence( int i ) const
    { return ( Kokkos::Experimental::TASK_STATE_EXECUTING == m_state && 0 <= i && i < m_dep_size ) ? m_dep[i] : (TaskMember*) 0 ; }

  KOKKOS_INLINE_FUNCTION
  int get_dependence() const
    { return m_dep_size ; }

  KOKKOS_INLINE_FUNCTION
  void clear_dependence()
    {
      for ( int i = 0 ; i < m_dep_size ; ++i ) assign( m_dep + i , 0 );
      m_dep_size = 0 ;
    }

  KOKKOS_INLINE_FUNCTION
  void add_dependence( TaskMember * before )
    {
      if ( ( Kokkos::Experimental::TASK_STATE_CONSTRUCTING == m_state ||
             Kokkos::Experimental::TASK_STATE_EXECUTING    == m_state ) &&
           m_dep_size < m_dep_capacity ) {
        assign( m_dep + m_dep_size , before );
        ++m_dep_size ;
      }
      else {
        throw_error_add_dependence();
      }
    }

  //----------------------------------------

  template< class FunctorType , class ResultType >
  KOKKOS_INLINE_FUNCTION static
  void apply_single( typename Kokkos::Impl::enable_if< ! Kokkos::Impl::is_same< ResultType , void >::value , TaskMember * >::type t )
    {
      typedef TaskMember< Kokkos::Serial , ResultType , FunctorType > derived_type ;

      // TaskMember< Kokkos::Serial , ResultType , FunctorType >
      //   : public TaskMember< Kokkos::Serial , ResultType , void >
      //   , public FunctorType
      //   { ... };

      derived_type & m = * static_cast< derived_type * >( t );

      Kokkos::Impl::FunctorApply< FunctorType , void , ResultType & >::apply( (FunctorType &) m , & m.m_result );
    }

  template< class FunctorType , class ResultType >
  KOKKOS_INLINE_FUNCTION static
  void apply_single( typename Kokkos::Impl::enable_if< Kokkos::Impl::is_same< ResultType , void >::value , TaskMember * >::type t )
    {
      typedef TaskMember< Kokkos::Serial , ResultType , FunctorType > derived_type ;

      // TaskMember< Kokkos::Serial , ResultType , FunctorType >
      //   : public TaskMember< Kokkos::Serial , ResultType , void >
      //   , public FunctorType
      //   { ... };

      derived_type & m = * static_cast< derived_type * >( t );

      Kokkos::Impl::FunctorApply< FunctorType , void , void >::apply( (FunctorType &) m );
    }

  //----------------------------------------

  template< class FunctorType , class ResultType >
  static
  void apply_team( typename Kokkos::Impl::enable_if< ! Kokkos::Impl::is_same< ResultType , void >::value , TaskMember * >::type t )
    {
      typedef TaskMember< Kokkos::Serial , ResultType , FunctorType > derived_type ;
      typedef Kokkos::Impl::SerialTeamMember                          member_type ;

      // TaskMember< Kokkos::Serial , ResultType , FunctorType >
      //   : public TaskMember< Kokkos::Serial , ResultType , void >
      //   , public FunctorType
      //   { ... };

      derived_type & m = * static_cast< derived_type * >( t );

      m.FunctorType::apply( member_type(0,1,0) , m.m_result );
    }

  template< class FunctorType , class ResultType >
  static
  void apply_team( typename Kokkos::Impl::enable_if< Kokkos::Impl::is_same< ResultType , void >::value , TaskMember * >::type t )
    {
      typedef TaskMember< Kokkos::Serial , ResultType , FunctorType > derived_type ;
      typedef Kokkos::Impl::SerialTeamMember                          member_type ;

      // TaskMember< Kokkos::Serial , ResultType , FunctorType >
      //   : public TaskMember< Kokkos::Serial , ResultType , void >
      //   , public FunctorType
      //   { ... };

      derived_type & m = * static_cast< derived_type * >( t );

      m.FunctorType::apply( member_type(0,1,0) );
    }
};

//----------------------------------------------------------------------------
/** \brief  Base class for tasks with a result value in the Serial execution space.
 *
 *  The FunctorType must be void because this class is accessed by the
 *  Future class for the task and result value.
 *
 *  Must be derived from TaskMember<S,void,void> 'root class' so the Future class
 *  can correctly static_cast from the 'root class' to this class.
 */
template < class ResultType >
class TaskMember< Kokkos::Serial , ResultType , void >
  : public TaskMember< Kokkos::Serial , void , void >
{
public:

  ResultType  m_result ;

  typedef const ResultType & get_result_type ;

  KOKKOS_INLINE_FUNCTION
  get_result_type get() const { return m_result ; }

protected:

  typedef TaskMember< Kokkos::Serial , void , void >  task_root_type ;
  typedef task_root_type::function_dealloc_type       function_dealloc_type ;
  typedef task_root_type::function_apply_type         function_apply_type ;

  inline
  TaskMember( const function_dealloc_type  arg_dealloc
            , const function_apply_type    arg_apply
            , const unsigned               arg_sizeof_derived
            , const unsigned               arg_dependence_capacity
            )
    : task_root_type( & task_root_type::template verify_type< ResultType >
                    , arg_dealloc
                    , arg_apply
                    , arg_sizeof_derived
                    , arg_dependence_capacity )
    , m_result()
    {}
};

template< class ResultType , class FunctorType >
class TaskMember< Kokkos::Serial , ResultType , FunctorType >
  : public TaskMember< Kokkos::Serial , ResultType , void >
  , public FunctorType
{
public:

  typedef FunctorType  functor_type ;

  typedef TaskMember< Kokkos::Serial , void , void >        task_root_type ;
  typedef TaskMember< Kokkos::Serial , ResultType , void >  task_base_type ;
  typedef task_root_type::function_dealloc_type             function_dealloc_type ;
  typedef task_root_type::function_apply_type               function_apply_type ;

  inline
  TaskMember( const function_dealloc_type  arg_dealloc
            , const function_apply_type    arg_apply
            , const unsigned               arg_sizeof_derived
            , const unsigned               arg_dependence_capacity
            , const functor_type &         arg_functor
            )
    : task_base_type( arg_dealloc , arg_apply , arg_sizeof_derived , arg_dependence_capacity )
    , functor_type( arg_functor )
    {}
};

} /* namespace Impl */
} /* namespace Experimental */
} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Experimental {

template<>
class TaskPolicy< Kokkos::Serial >
{
public:

  typedef Kokkos::Serial                  execution_space ;
  typedef Kokkos::Impl::SerialTeamMember  member_type ;

private:

  typedef Impl::TaskMember< execution_space , void , void > task_root_type ;

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

  unsigned m_default_dependence_capacity ;

public:

  TaskPolicy
    ( const unsigned /* arg_task_max_count */
    , const unsigned /* arg_task_max_size */
    , const unsigned arg_task_default_dependence_capacity = 4
    , const unsigned /* arg_task_team_size */ = 0
    )
    : m_default_dependence_capacity( arg_task_default_dependence_capacity )
    {}

  TaskPolicy() = default ;
  TaskPolicy( TaskPolicy && rhs ) = default ;
  TaskPolicy( const TaskPolicy & rhs ) = default ;
  TaskPolicy & operator = ( TaskPolicy && rhs ) = default ;
  TaskPolicy & operator = ( const TaskPolicy & rhs ) = default ;

  //----------------------------------------

  template< class ValueType >
  KOKKOS_INLINE_FUNCTION
  const Future< ValueType , execution_space > &
    spawn( const Future< ValueType , execution_space > & f 
         , const bool priority = false ) const
      {
#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
        f.m_task->schedule();
#endif
        return f ;
      }

  // Create single-thread task

  template< class FunctorType >
  KOKKOS_INLINE_FUNCTION
  Future< typename FunctorType::value_type , execution_space >
  create( const FunctorType & functor
        , const unsigned dependence_capacity = ~0u ) const
    {
      typedef typename FunctorType::value_type value_type ;
      typedef Impl::TaskMember< execution_space , value_type , FunctorType >  task_type ;
      return Future< value_type , execution_space >(
#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
        task_root_type::create< task_type >(
          functor , ( ~0u == dependence_capacity ? m_default_dependence_capacity : dependence_capacity ) )
#endif
        );
    }

  template< class FunctorType >
  KOKKOS_INLINE_FUNCTION
  Future< typename FunctorType::value_type , execution_space >
  create_team( const FunctorType & functor
             , const unsigned dependence_capacity = ~0u ) const
    {
      typedef typename FunctorType::value_type value_type ;
      typedef Impl::TaskMember< execution_space , value_type , FunctorType >  task_type ;
      return Future< value_type , execution_space >(
#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
        task_root_type::create_team< task_type >(
          functor , ( ~0u == dependence_capacity ? m_default_dependence_capacity : dependence_capacity ) )
#endif
        );
    }

  // Add dependence
  template< class A1 , class A2 , class A3 , class A4 >
  KOKKOS_INLINE_FUNCTION
  void add_dependence( const Future<A1,A2> & after
                     , const Future<A3,A4> & before
                     , typename Kokkos::Impl::enable_if
                        < Kokkos::Impl::is_same< typename Future<A1,A2>::execution_space , execution_space >::value
                          &&
                          Kokkos::Impl::is_same< typename Future<A3,A4>::execution_space , execution_space >::value
                        >::type * = 0
                      ) const
    {
#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
      after.m_task->add_dependence( before.m_task );
#endif
    }

  //----------------------------------------
  // Functions for an executing task functor to query dependences,
  // set new dependences, and respawn itself.

  template< class FunctorType >
  KOKKOS_INLINE_FUNCTION
  Future< void , execution_space >
  get_dependence( const FunctorType * task_functor , int i ) const
    {
      return Future<void,execution_space>(
#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
        get_task_root(task_functor)->get_dependence(i)
#endif
        );
    }

  template< class FunctorType >
  KOKKOS_INLINE_FUNCTION
  int get_dependence( const FunctorType * task_functor ) const
#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
    { return get_task_root(task_functor)->get_dependence(); }
#else
    { return 0 ; }
#endif

  template< class FunctorType >
  KOKKOS_INLINE_FUNCTION
  void clear_dependence( FunctorType * task_functor ) const
#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
    { get_task_root(task_functor)->clear_dependence(); }
#else
    {}
#endif

  template< class FunctorType , class A3 , class A4 >
  KOKKOS_INLINE_FUNCTION
  void add_dependence( FunctorType * task_functor
                     , const Future<A3,A4> & before
                     , typename Kokkos::Impl::enable_if
                        < Kokkos::Impl::is_same< typename Future<A3,A4>::execution_space , execution_space >::value
                        >::type * = 0
                      ) const
#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
    { get_task_root(task_functor)->add_dependence( before.m_task ); }
#else
    {}
#endif

  template< class FunctorType >
  KOKKOS_INLINE_FUNCTION
  void respawn( FunctorType * task_functor 
              , const bool priority = false ) const
    {
#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
      get_task_root(task_functor)->schedule();
#endif
    }

  template< class FunctorType >
  KOKKOS_INLINE_FUNCTION
  void respawn_needing_memory( FunctorType * task_functor ) const
    {
#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
      get_task_root(task_functor)->schedule();
#endif
    }

  //----------------------------------------

  static member_type & member_single();
};

inline
void wait( TaskPolicy< Kokkos::Serial > & )
{ Impl::TaskMember< Kokkos::Serial , void , void >::execute_ready_tasks(); }

} /* namespace Experimental */
} // namespace Kokkos

//----------------------------------------------------------------------------

#endif /* defined( KOKKOS_HAVE_SERIAL ) */
#endif /* #define KOKKOS_SERIAL_TASK_HPP */

