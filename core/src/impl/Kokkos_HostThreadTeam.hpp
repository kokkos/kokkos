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

#ifndef KOKKOS_IMPL_HOSTTHREADTEAM_HPP
#define KOKKOS_IMPL_HOSTTHREADTEAM_HPP

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Pair.hpp>
#include <Kokkos_MemoryTraits.hpp>
#include <Kokkos_Atomic.hpp>
#include <Kokkos_ExecPolicy.hpp>
#include <impl/Kokkos_Reducer.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

class HostThreadTeamMember {
private:

  // per-thread scratch memory buffer chunks:
  //   [ 0                  .. m_rendezvous_begin )
  //   [ m_rendezvous_begin .. m_collective_begin )
  //   [ m_collective_begin .. m_shared_begin )
  //   [ m_shared_begin     .. m_local_begin )
  //   [ m_local_begin      .. m_scratch_end )

  enum : int { m_rendezvous_begin = 64 };
  enum : int { m_collective_begin = m_rendezvous_begin + 16 };

  Kokkos::pair<int,int>  m_work_range ; // Work range, 64bit alignment

  int64_t   * m_scratch ;       // per-thread buffer
  int64_t   * m_shared ;        // == m_team[0]->m_scratch
  int         m_shared_begin ;
  int         m_local_begin ;
  int         m_scratch_end ;
  int         m_team_rank_steal ; // work stealing rank
  int         m_team_rank ;
  int         m_team_size ;
  int         m_league_rank ;
  int         m_league_size ;
  int mutable m_rendezvous_step ;  

  // Memory chunks:

  HostThreadTeamMember * member( int r ) const noexcept
    { return ((HostThreadTeamMember*const*)m_shared)[r]; }

  int64_t * rendezvous_memory() const noexcept
    { return m_shared + m_rendezvous_begin ; }

  int64_t * collective_memory() const noexcept
    { return m_shared + m_collective_begin ; }

  int64_t * shared_memory() const noexcept
    { return m_shared + m_shared_begin ; }

  int64_t * local_memory() const noexcept
    { return m_scratch + m_local_begin ; }

  // Rendezvous pattern:
  //   if ( team_rendezvous(root) ) {
  //     ... only root thread here while all others wait ...
  //     team_rendezvous_release();
  //   }
  //   else {
  //     ... all other threads release here ...
  //   }
  int  team_rendezvous( int const root = 0 ) const noexcept ;
  void team_rendezvous_release() const noexcept ;

public:

  //----------------------------------------

  constexpr HostThreadTeamMember() noexcept
    : m_work_range(-1,-1)
    , m_scratch(0)
    , m_shared(0)
    , m_shared_begin(0)
    , m_local_begin(0)
    , m_scratch_end(0)
    , m_team_rank_steal(0)
    , m_team_rank(0)
    , m_team_size(0)
    , m_league_rank(0)
    , m_league_size(0)
    , m_rendezvous_step(0)
    {}

  // Form active team from members[0..size)
  // Requires:
  //   1) each member is unique
  //   2) each member not alread a member of another team
  static void form_active_team( HostThreadTeamMember * members[]
                              , const int size );

  // The leader of an active team disbands that team
  void disband_active_team();

  //----------------------------------------

  template< class MemorySpace >
  void resize_scratch( int collect_size
                     , int shared_size
                     , int local_size )
    {
      enum : int { align = Kokkos::Impl::MEMORY_ALIGNMENT };

      collect_size += collect_size % align ? align - collect_size % align : 0 ;
      shared_size  += shared_size  % align ? align - shared_size  % align : 0 ;
      local_size   += local_size   % align ? align - local_size   % align : 0 ;

      const size_t total =
        m_collective_begin + collect_size + shared_size + local_size ;

      if ( m_scratch_end < total ) {
        MemorySpace space ; // default space

        space.deallocate( m_scratch , m_scratch_end );

        m_scratch = space.allocate( total );

        m_shared_begin = m_collective_begin + collect_size ;
        m_local_begin  = m_shared_begin     + shared_size ;
        m_scratch_end  = total ;
      }
      else {
        m_shared_begin = m_collective_begin + collect_size ;
        m_local_begin  = m_shared_begin     + shared_size ;
      }
    }

  //----------------------------------------

  KOKKOS_INLINE_FUNCTION
  int team_rank() const noexcept { return m_team_size ; }

  KOKKOS_INLINE_FUNCTION
  int team_size() const noexcept { return m_team_rank ; }

  KOKKOS_INLINE_FUNCTION
  int league_rank() const noexcept { return m_league_size ; }

  KOKKOS_INLINE_FUNCTION
  int league_size() const noexcept { return m_league_rank ; }

  //----------------------------------------
  // Team collectives

  KOKKOS_INLINE_FUNCTION void team_barrier() const noexcept
#if defined( KOKKOS_ACTIVE_EXECUTION_SPACE_HOST )
    { if ( team_rendezvous() ) team_rendezvous_release(); }
#else
    {}
#endif

  template< typename T >
  KOKKOS_INLINE_FUNCTION
  void team_broadcast( T & value , const int source_team_rank = 0 ) const noexcept
#if defined( KOKKOS_ACTIVE_EXECUTION_SPACE_HOST )
    {
      T volatile * const shared_value =
        (T*)(( (int64_t*) m_team_shared_internal) + m_shared_rendezvous_int64 );

      // Don't overwrite shared memory until all threads arrive

      if ( team_rendezvous( source_team_rank ) ) {
        // All threads have entered 'team_rendezvous'
        // only this thread returned from 'team_rendezvous'
        // with a return value of 'true'

        *shared_value = value ;

        team_rendezvous_release();
        // This thread released all other threads from 'team_rendezvous'
        // with a return value of 'false'
      }
      else {
        value = *shared_value ;
      }
    }
#else
    {}
#endif

  // team_reduce( Sum(result) );
  // team_reduce( Min(result) );
  // team_reduce( Max(result) );
  template< typename ReducerType >
  KOKKOS_INLINE_FUNCTION
  void team_reduce( ReducerType const & reducer ) const noexcept
#if defined( KOKKOS_ACTIVE_EXECUTION_SPACE_HOST )
    {
      static_assert( Kokkos::is_reducer< ReducerType >::value
                   , "team_reduce requires a reducer object" );

      using value_type = typename ReducerType::value_type ;

      value_type volatile * const shared_value =
        (value_type*) collective_memory();

      value_type * const member_value =
        shared_value + m_team_rank * reducer.length();

      // Don't overwrite shared memory until all threads arrive

      team_barrier();

      for ( int i = 0 ; i < reducer.length() ; ++i ) {
        member_value[i] = reducer[i];
      }

      // Wait for all team members to store data

      if ( team_rendezvous() ) {
        // All threads have entered 'team_rendezvous'
        // only this thread returned from 'team_rendezvous'
        // with a return value of 'true'
        //
        // This thread sums contributed values
        for ( int i = 1 ; i < m_team_size ; ++i ) {
          reducer.join( shared_value , shared_value + i * reducer.length() );
        }
        team_rendezvous_release();
        // This thread released all other threads from 'team_rendezvous'
        // with a return value of 'false'
      }

      for ( int i = 0 ; i < reducer.length() ; ++i ) {
        reducer[i] = shared_value[i] ;
      }
    }
#else
    {}
#endif

  template< typename T >
  KOKKOS_INLINE_FUNCTION
  void team_scan( T & value , T * const global = 0 ) const noexcept
#if defined( KOKKOS_ACTIVE_EXECUTION_SPACE_HOST )
    {
      T volatile * const shared_value =
        (T*)(( (int64_t*) m_team_shared_internal) + m_shared_rendezvous_int64 );

      // Don't overwrite shared memory until all threads arrive
      team_barrier();

      shared_value[ m_team_rank + 1 ] = value ;

      if ( team_rendezvous() ) {
        // All threads have entered 'team_rendezvous'
        // only this thread returned from 'team_rendezvous'
        // with a return value of 'true'
        //
        // This thread scans contributed values
        for ( int i = 1 ; i < m_team_size ; ++i ) {
          shared_value[i+1] += shared_value[i] ;
        }

        shared_value[0] = 0 ;

        // If adding to global value then atomic_fetch_add to that value
        // and sum previous value to every entry of the scan.
        if ( global ) {
          shared_value[0] =
            Kokkos::atomic_fetch_add( global , shared_value[m_team_size] );
          for ( int i = 1 ; i < m_team_size ; ++i ) {
            shared_value[i] += shared_value[0] ;
          }
        }

        team_rendezvous_release();
      }

      value = shared_value[ m_team_rank ];
    }
#else
    {}
#endif

  //----------------------------------------
  // Work stealing

  int get_work_local_begin() noexcept
    {
      using pair_t = Kokkos::pair<int,int> ;

      pair_t w( -1 , -1 );

      for ( int attempt = true ; attempt ; ) {

        // Query and attempt to update m_work_range
        //   from: [ w.first     , w.second )
        //   to:   [ w.first + 1 , w.second ) = w_new
        //
        // If w is invalid then is just a query.

        const pair_t w_new( w.first + 1 , w.second );

        w = Kokkos::atomic_compare_exchange( & m_work_range, w, w_new );

        if ( w.first < w.second ) { 
          // m_work_range is viable

          // If steal is successful then don't repeat attempt to steal
          attempt = ! ( w_new.first  == w.first + 1 &&
                        w_new.second == w.second );
        }
        else {
          // m_work_range is not viable
          w.first  = -1 ;
          w.second = -1 ;

          attempt = false ;
        }
      }

      return w.first ;
    }

  int get_work_steal_end() noexcept
    {
      using pair_t = Kokkos::pair<int,int> ;

      pair_t volatile * steal_work_range =
        & member(m_team_rank_steal)->m_work_range ;

      pair_t w( -1 , -1 );

      for ( int attempt = true ; attempt ; ) {

        // Query and attempt to update steal_work_range
        //   from: [ w.first , w.second )
        //   to:   [ w.first , w.second - 1 ) = w_new
        //
        // If w is invalid then is just a query.

        const pair_t w_new( w.first , w.second - 1 );

        w = Kokkos::atomic_compare_exchange( steal_work_range, w, w_new );

        if ( w.first < w.second ) { 
          // steal_work_range is viable

          // If steal is successful then don't repeat attempt to steal
          attempt = ! ( w_new.first  == w.first &&
                        w_new.second == w.second - 1 );
        }
        else {
          // steal_work_range is not viable, move to next member
          w.first  = -1 ;
          w.second = -1 ;

          m_team_rank_steal = ( m_team_rank_steal + 1 ) % m_team_size ;

          steal_work_range = & member(m_team_rank_steal)->m_work_range ;

          // If tried all other members then don't repeat attempt to steal
          attempt = m_team_rank_steal != m_team_rank ;
        }
      }

      // May exit because successfully stole work and w_old is good.
      // May exit because no work left to steal and w_old = (-1,-1).

      return w.second ;
    }
};

//----------------------------------------------------------------------------

}} /* namespace Kokkos::Impl */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

template<typename iType>
KOKKOS_INLINE_FUNCTION
Impl::TeamThreadRangeBoundariesStruct<iType,Impl::HostThreadTeamMember>
TeamThreadRange( Impl::HostThreadTeamMember const & member
               , iType const & count )
{
  return
    Impl::TeamThreadRangeBoundariesStruct
      <iType,Impl::HostThreadTeamMember>(member,0,count);
}

template<typename iType1, typename iType2>
KOKKOS_INLINE_FUNCTION
Impl::TeamThreadRangeBoundariesStruct
  < typename std::common_type< iType1, iType2 >::type
  , Impl::HostThreadTeamMember >
TeamThreadRange( Impl::HostThreadTeamMember const & member
               , iType1 const & begin , iType2 const & end )
{
  return
    Impl::TeamThreadRangeBoundariesStruct
      < typename std::common_type< iType1, iType2 >::type
      , Impl::HostThreadTeamMember >( member , begin , end );
}

template<typename iType>
KOKKOS_INLINE_FUNCTION
Impl::ThreadVectorRangeBoundariesStruct<iType,Impl::HostThreadTeamMember >
ThreadVectorRange
  ( Impl::HostThreadTeamMember & member
  , const iType & count )
{
  return Impl::ThreadVectorRangeBoundariesStruct<iType,Impl::HostThreadTeamMember >(member,count);
}

/** \brief  Inter-thread parallel_for. Executes lambda(iType i) for each i=0..N-1.
 *
 * The range i=0..N-1 is mapped to all threads of the the calling thread team.
 * This functionality requires C++11 support.
*/
template<typename iType, class Closure>
KOKKOS_INLINE_FUNCTION
void parallel_for
  ( Impl::TeamThreadRangeBoundariesStruct<iType,Impl::HostThreadTeamMember >
      const & loop_boundaries
  , Closure const & closure
  )
{
  for( iType i = loop_boundaries.start
     ; i <  loop_boundaries.end
     ; i += loop_boundaries.increment ) {
    closure (i);
  }
}

template<typename iType, class Lambda, typename ValueType>
KOKKOS_INLINE_FUNCTION
void parallel_reduce
  ( const Impl::TeamThreadRangeBoundariesStruct<iType,Impl::HostThreadTeamMember >& loop_boundaries
  , const Lambda& lambda
  , ValueType& initialized_result)
{
  int team_rank = loop_boundaries.thread.team_rank(); // member num within the team
  ValueType result = initialized_result;

  for( iType i = loop_boundaries.start; i < loop_boundaries.end; i+=loop_boundaries.increment) {
    lambda(i, result);
  }

  if ( 1 < loop_boundaries.thread.team_size() ) {

    ValueType *shared = (ValueType*) loop_boundaries.thread.team_shared();

    loop_boundaries.thread.team_barrier();
    shared[team_rank] = result;

    loop_boundaries.thread.team_barrier();

    // reduce across threads to thread 0
    if (team_rank == 0) {
      for (int i = 1; i < loop_boundaries.thread.team_size(); i++) {
        shared[0] += shared[i];
      }
    }

    loop_boundaries.thread.team_barrier();

    // broadcast result
    initialized_result = shared[0];
  }
  else {
    initialized_result = result ;
  }
}

template< typename iType, class Closure, class Reducer >
KOKKOS_INLINE_FUNCTION
typename std::enable_if< Kokkos::is_reducer< Reducer >::value >::type
parallel_reduce
  ( Impl::TeamThreadRangeBoundariesStruct<iType,Impl::HostThreadTeamMember >
             const & loop_boundaries
  , Closure  const & closure
  , Reducer  const & reducer
  )
{
  reducer.init( reducer.reference() );

  for( iType i = loop_boundaries.start
     ; i <  loop_boundaries.end
     ; i += loop_boundaries.increment ) {
    closure( i , reducer.reference() );
  }

  loop_boundaries.thread.team_reduce( reducer );
}

template< typename iType, typename Closure, typename ValueType >
KOKKOS_INLINE_FUNCTION
typename std::enable_if< std::is_trivial<ValueType>::value >::type
parallel_reduce
  ( Impl::TeamThreadRangeBoundariesStruct<iType,Impl::HostThreadTeamMember >
             const & loop_boundaries
  , Closure  const & closure
  , ValueType      & result
  )
{
  Impl::ReduceSum< ValueType > reducer( result );

  reducer.init( reducer.value() );

  for( iType i = loop_boundaries.start
     ; i <  loop_boundaries.end
     ; i += loop_boundaries.increment ) {
    closure( i , reducer.value() );
  }

  loop_boundaries.thread.team_reduce( reducer );
}

template< typename iType, class Lambda, typename ValueType >
KOKKOS_INLINE_FUNCTION
void parallel_reduce
  (const Impl::ThreadVectorRangeBoundariesStruct<iType,Impl::HostThreadTeamMember >& loop_boundaries,
   const Lambda & lambda,
   ValueType& initialized_result)
{
}

// placeholder for future function
template< typename iType, class Lambda, typename ValueType, class JoinType >
KOKKOS_INLINE_FUNCTION
void parallel_reduce
  (const Impl::ThreadVectorRangeBoundariesStruct<iType,Impl::HostThreadTeamMember >& loop_boundaries,
   const Lambda & lambda,
   const JoinType & join,
   ValueType& initialized_result)
{
}

#if 0

template< typename iType, class Closure >
KOKKOS_INLINE_FUNCTION
void parallel_scan
  ( Impl::TeamThreadRangeBoundariesStruct<iType,Impl::HostThreadTeamMember >
       const & loop_boundaries
  , Closure const & closure
  )
{
  // Extract ValueType from the closure

  ValueType accum = 0 ;

  // Intra-member scan
  for ( iType i = loop_boundaries.start
      ; i <  loop_boundaries.end
      ; i += loop_boundaries.increment ) {
    lambda(i,accum,false);
  }

  // 'accum' output is the exclusive prefix sum
  loop_boundaries.thread.team_scan(accum);

  for ( iType i = loop_boundaries.start
      ; i <  loop_boundaries.end
      ; i += loop_boundaries.increment ) {
    lambda(i,accum,true);
  }
}

// placeholder for future function
template< typename iType, class Lambda, typename ValueType >
KOKKOS_INLINE_FUNCTION
void parallel_scan
  (const Impl::ThreadVectorRangeBoundariesStruct<iType,Impl::HostThreadTeamMember >& loop_boundaries,
   const Lambda & lambda)
{
}

#endif

} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #ifndef KOKKOS_IMPL_HOSTTHREADTEAM_HPP */

