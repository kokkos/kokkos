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

#ifndef KOKKOS_FUTURE_HPP
#define KOKKOS_FUTURE_HPP

//----------------------------------------------------------------------------

#include <Kokkos_Macros.hpp>
#if defined( KOKKOS_ENABLE_TASKDAG )

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_TaskScheduler_fwd.hpp>
//----------------------------------------------------------------------------

#include <impl/Kokkos_TaskQueue.hpp>

#include <Kokkos_Concepts.hpp> // is_space

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

/**
 *
 *  Future< space >  // value_type == void
 *  Future< value >  // space == Default
 *  Future< value , space >
 *
 */
template< typename Arg1 , typename Arg2 >
class Future {
private:

  template< typename > friend class TaskScheduler ;
  template< typename , typename > friend class Future ;
  template< typename , typename , typename > friend class Impl::TaskBase ;

  enum { Arg1_is_space  = Kokkos::is_space< Arg1 >::value };
  enum { Arg2_is_space  = Kokkos::is_space< Arg2 >::value };
  enum { Arg1_is_value  = ! Arg1_is_space &&
      ! std::is_same< Arg1 , void >::value };
  enum { Arg2_is_value  = ! Arg2_is_space &&
      ! std::is_same< Arg2 , void >::value };

  static_assert( ! ( Arg1_is_space && Arg2_is_space )
    , "Future cannot be given two spaces" );

  static_assert( ! ( Arg1_is_value && Arg2_is_value )
    , "Future cannot be given two value types" );

  using ValueType =
  typename std::conditional< Arg1_is_value , Arg1 ,
    typename std::conditional< Arg2_is_value , Arg2 , void
    >::type >::type ;

  using Space =
  typename std::conditional< Arg1_is_space , Arg1 ,
    typename std::conditional< Arg2_is_space , Arg2 , void
    >::type >::type ;

  using task_base  = Impl::TaskBase< void , void , void > ;
  using queue_type = Impl::TaskQueue< Space > ;

  task_base * m_task ;

  KOKKOS_INLINE_FUNCTION explicit
  Future( task_base * task ) : m_task(0)
  { if ( task ) queue_type::assign( & m_task , task ); }

  //----------------------------------------

public:

  using execution_space = typename Space::execution_space ;
  using value_type      = ValueType ;

  //----------------------------------------

  KOKKOS_INLINE_FUNCTION
  bool is_null() const { return 0 == m_task ; }

  KOKKOS_INLINE_FUNCTION
  int reference_count() const
  { return 0 != m_task ? m_task->reference_count() : 0 ; }

  //----------------------------------------

  KOKKOS_INLINE_FUNCTION
  void clear()
  { if ( m_task ) queue_type::assign( & m_task , (task_base*)0 ); }

  //----------------------------------------

  KOKKOS_INLINE_FUNCTION
  ~Future() { clear(); }

  //----------------------------------------

  KOKKOS_INLINE_FUNCTION
  constexpr Future() noexcept : m_task(0) {}

  KOKKOS_INLINE_FUNCTION
  Future( Future && rhs )
    : m_task( rhs.m_task ) { rhs.m_task = 0 ; }

  KOKKOS_INLINE_FUNCTION
  Future( const Future & rhs )
    : m_task(0)
  { if ( rhs.m_task ) queue_type::assign( & m_task , rhs.m_task ); }

  KOKKOS_INLINE_FUNCTION
  Future & operator = ( Future && rhs )
  {
    clear();
    m_task = rhs.m_task ;
    rhs.m_task = 0 ;
    return *this ;
  }

  KOKKOS_INLINE_FUNCTION
  Future & operator = ( const Future & rhs )
  {
    if ( m_task || rhs.m_task ) queue_type::assign( & m_task , rhs.m_task );
    return *this ;
  }

  //----------------------------------------

  template< class A1 , class A2 >
  KOKKOS_INLINE_FUNCTION
  Future( Future<A1,A2> && rhs )
    : m_task( rhs.m_task )
  {
    static_assert
      ( std::is_same< Space , void >::value ||
          std::is_same< Space , typename Future<A1,A2>::Space >::value
        , "Assigned Futures must have the same space" );

    static_assert
      ( std::is_same< value_type , void >::value ||
          std::is_same< value_type , typename Future<A1,A2>::value_type >::value
        , "Assigned Futures must have the same value_type" );

    rhs.m_task = 0 ;
  }

  template< class A1 , class A2 >
  KOKKOS_INLINE_FUNCTION
  Future( const Future<A1,A2> & rhs )
    : m_task(0)
  {
    static_assert
      ( std::is_same< Space , void >::value ||
          std::is_same< Space , typename Future<A1,A2>::Space >::value
        , "Assigned Futures must have the same space" );

    static_assert
      ( std::is_same< value_type , void >::value ||
          std::is_same< value_type , typename Future<A1,A2>::value_type >::value
        , "Assigned Futures must have the same value_type" );

    if ( rhs.m_task ) queue_type::assign( & m_task , rhs.m_task );
  }

  template< class A1 , class A2 >
  KOKKOS_INLINE_FUNCTION
  Future & operator = ( const Future<A1,A2> & rhs )
  {
    static_assert
      ( std::is_same< Space , void >::value ||
          std::is_same< Space , typename Future<A1,A2>::Space >::value
        , "Assigned Futures must have the same space" );

    static_assert
      ( std::is_same< value_type , void >::value ||
          std::is_same< value_type , typename Future<A1,A2>::value_type >::value
        , "Assigned Futures must have the same value_type" );

    if ( m_task || rhs.m_task ) queue_type::assign( & m_task , rhs.m_task );
    return *this ;
  }

  template< class A1 , class A2 >
  KOKKOS_INLINE_FUNCTION
  Future & operator = ( Future<A1,A2> && rhs )
  {
    static_assert
      ( std::is_same< Space , void >::value ||
          std::is_same< Space , typename Future<A1,A2>::Space >::value
        , "Assigned Futures must have the same space" );

    static_assert
      ( std::is_same< value_type , void >::value ||
          std::is_same< value_type , typename Future<A1,A2>::value_type >::value
        , "Assigned Futures must have the same value_type" );

    clear();
    m_task = rhs.m_task ;
    rhs.m_task = 0 ;
    return *this ;
  }

  //----------------------------------------

  KOKKOS_INLINE_FUNCTION
  int is_ready() const noexcept
  { return ( 0 == m_task ) || ( ((task_base*) task_base::LockTag) == m_task->m_wait ); }

  KOKKOS_INLINE_FUNCTION
  const typename Impl::TaskResult< ValueType >::reference_type
  get() const
  {
    if ( 0 == m_task ) {
      Kokkos::abort( "Kokkos:::Future::get ERROR: is_null()");
    }
    return Impl::TaskResult< ValueType >::get( m_task );
  }
};

// Is a Future with the given execution space
template< typename , typename ExecSpace = void >
struct is_future : public std::false_type {};

template< typename Arg1 , typename Arg2 , typename ExecSpace >
struct is_future< Future<Arg1,Arg2> , ExecSpace >
  : public std::integral_constant
    < bool ,
      ( std::is_same< ExecSpace , void >::value ||
        std::is_same< ExecSpace
          , typename Future<Arg1,Arg2>::execution_space >::value )
    > {};

} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #if defined( KOKKOS_ENABLE_TASKDAG ) */
#endif /* #ifndef KOKKOS_FUTURE */
