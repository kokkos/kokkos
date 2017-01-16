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
#include <Kokkos_Atomic.hpp>
#include <Kokkos_ExecPolicy.hpp>
#include <impl/Kokkos_FunctorAdapter.hpp>
#include <impl/Kokkos_Reducer.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template< class HostExecSpace >
class HostThreadTeamMember ;

class HostThreadTeamData {
public:

  template< class > friend class HostThreadTeamMember ;

  // Assume upper bounds on number of threads:
  //   pool size       <= 1024 threads
  //   pool rendezvous <= 2 * ( 1024 / 8 ) = 256
  //   team size       <= 64 threads
  //   team rendezvous <= 2 * ( 64 / 8 ) = 16

  enum : int { max_pool_members  = 1024 };
  enum : int { max_team_members  = 64 };
  enum : int { max_pool_rendezvous  = 2 * ( max_pool_members / 8 ) };
  enum : int { max_team_rendezvous  = 2 * ( max_team_members / 8 ) };

private:

  // per-thread scratch memory buffer chunks:
  //
  //   [ pool_members ]     = [ m_pool_members    .. m_pool_rendezvous )
  //   [ pool_rendezvous ]  = [ m_pool_rendezvous .. m_team_rendezvous )
  //   [ team_rendezvous ]  = [ m_team_rendezvous .. m_pool_reduce )
  //   [ pool_reduce ]      = [ m_pool_reduce     .. m_team_reduce )
  //   [ team_reduce ]      = [ m_team_reduce     .. m_team_shared )
  //   [ team_shared ]      = [ m_team_shared     .. m_thread_local )
  //   [ thread_local ]     = [ m_thread_local    .. m_scratch_size )

  enum : int { m_pool_members    = 0 };
  enum : int { m_pool_rendezvous = m_pool_members    + max_pool_members };
  enum : int { m_team_rendezvous = m_pool_rendezvous + max_pool_rendezvous };
  enum : int { m_pool_reduce     = m_team_rendezvous + max_team_rendezvous };

  using pair_int_t = Kokkos::pair<int,int> ;

  pair_int_t  m_work_range ;
  int64_t   * m_scratch ;       // per-thread buffer
  int64_t   * m_pool_scratch ;  // == pool[0]->m_scratch
  int64_t   * m_team_scratch ;  // == pool[ 0 + m_team_base ]->m_scratch
  int         m_pool_rank ;
  int         m_pool_size ;
  int         m_team_reduce ;
  int         m_team_shared ;
  int         m_thread_local ;
  int         m_scratch_size ;
  int         m_team_base ;
  int         m_team_rank ;
  int         m_team_size ;
  int         m_team_alloc ;
  int         m_league_rank ;
  int         m_league_size ;
  int         m_steal_rank ; // work stealing rank
  int mutable m_pool_rendezvous_step ;  
  int mutable m_team_rendezvous_step ;  

  HostThreadTeamData * pool_member( int r ) const noexcept
    { return ((HostThreadTeamData**)(m_pool_scratch+m_pool_members))[r]; }

  HostThreadTeamData * team_member( int r ) const noexcept
    { return ((HostThreadTeamData**)(m_pool_scratch+m_pool_members))[m_team_base+r]; }

  // Rendezvous pattern:
  //   if ( rendezvous(root) ) {
  //     ... only root thread here while all others wait ...
  //     rendezvous_release();
  //   }
  //   else {
  //     ... all other threads release here ...
  //   }
  static
  int rendezvous( int64_t * const buffer
                , int & rendezvous_step
                , int const size
                , int const rank ) noexcept ;

  static
  void rendezvous_release( int64_t * const buffer
                         , int const rendezvous_step ) noexcept ;

public:

  inline
  int team_rendezvous( int const root ) const noexcept
    {
      return m_team_rank < m_team_size ?
        rendezvous( m_team_scratch + m_team_rendezvous
                  , m_team_rendezvous_step
                  , m_team_size
                  , ( m_team_rank + m_team_size - root ) % m_team_size ) : 0 ;
    }

  inline
  int team_rendezvous() const noexcept
    {
      return m_team_rank < m_team_size ?
        rendezvous( m_team_scratch + m_team_rendezvous
                  , m_team_rendezvous_step
                  , m_team_size
                  , m_team_rank ) : 0 ;
    }

  inline
  void team_rendezvous_release() const noexcept
    {
      rendezvous_release( m_team_scratch + m_team_rendezvous
                        , m_team_rendezvous_step );
    }

  inline
  int pool_rendezvous() const noexcept
    {
      return rendezvous( m_pool_scratch + m_pool_rendezvous
                       , m_pool_rendezvous_step
                       , m_pool_size
                       , m_pool_rank );
    }

  inline
  void pool_rendezvous_release() const noexcept
    {
      rendezvous_release( m_pool_scratch + m_pool_rendezvous
                        , m_pool_rendezvous_step );
    }

  //----------------------------------------

  constexpr HostThreadTeamData() noexcept
    : m_work_range(-1,-1)
    , m_scratch(0)
    , m_pool_scratch(0)
    , m_team_scratch(0)
    , m_pool_rank(0)
    , m_pool_size(1)
    , m_team_reduce(0)
    , m_team_shared(0)
    , m_thread_local(0)
    , m_scratch_size(0)
    , m_team_base(0)
    , m_team_rank(0)
    , m_team_size(1)
    , m_team_alloc(1)
    , m_league_rank(0)
    , m_league_size(1)
    , m_steal_rank(0)
    , m_pool_rendezvous_step(0)
    , m_team_rendezvous_step(0)
    {}

  //----------------------------------------
  // Organize array of members into a pool.
  // The 0th member is the root of the pool.
  // Requires members are not already in a pool.
  // Pool members are ordered as "close" - sorted by NUMA and then CORE
  // Each thread is its own team with team_size == 1.
  static void organize_pool( HostThreadTeamData * members[]
                           , const int size );

  // Root of a pool disbands the pool.
  void disband_pool();

  // Each thread within a pool organizes itself into a team.
  // Must be called by all threads of the pool.
  // Organizing threads into a team performs a barrier across the
  // entire pool to insure proper initialization of the team
  // rendezvous mechanism before a team rendezvous can be performed.
  //
  // Return true  if a valid member of a team.
  // Return false if not a member and thread should be idled.
  int organize_team( const int team_size );

  // Each thread within a pool disbands itself from current team.
  // Each thread becomes its own team with team_size == 1.
  // Must be called by all threads of the pool.
  void disband_team();

  //----------------------------------------

  constexpr int pool_rank() const { return m_pool_rank ; }
  constexpr int pool_size() const { return m_pool_size ; }

  //----------------------------------------

  static constexpr int align_to_int64( int n )
    {
      enum : int { mask  = 0x0f }; // align to 16 bytes
      enum : int { shift = 3 };    // size to 8 bytes
      return ( ( n + mask ) & ~mask ) >> shift ;
    }

  constexpr int pool_reduce_bytes() const
    { return sizeof(int64_t) * ( m_team_reduce - m_pool_reduce ); }

  constexpr int team_reduce_bytes() const
    { return sizeof(int64_t) * ( m_team_shared - m_team_reduce ); }

  constexpr int team_shared_bytes() const
    { return sizeof(int64_t) * ( m_thread_local - m_team_shared ); }

  constexpr int thread_local_bytes() const
    { return sizeof(int64_t) * ( m_scratch_size - m_thread_local ); }

  constexpr int scratch_bytes() const
    { return sizeof(int64_t) * m_scratch_size ; }

  // Memory chunks:

  int64_t * scratch_buffer() const noexcept
    { return m_scratch ; }

  int64_t * pool_reduce() const noexcept
    { return m_pool_scratch + m_pool_reduce ; }

  int64_t * pool_reduce_local() const noexcept
    { return m_scratch + m_pool_reduce ; }

  int64_t * team_reduce() const noexcept
    { return m_team_scratch + m_team_reduce ; }

  int64_t * team_reduce_local() const noexcept
    { return m_scratch + m_team_reduce ; }

  int64_t * team_shared() const noexcept
    { return m_team_scratch + m_team_shared ; }

  int64_t * local_scratch() const noexcept
    { return m_scratch + m_thread_local ; }

  // Given:
  //   pool_reduce_size  = number bytes for pool reduce
  //   team_reduce_size  = number bytes for team reduce
  //   team_shared_size  = number bytes for team shared memory
  //   thread_local_size = number bytes for thread local memory
  // Return:
  //   total number of bytes that must be allocated
  static
  size_t scratch_size( int pool_reduce_size
                     , int team_reduce_size
                     , int team_shared_size
                     , int thread_local_size )
    {
      pool_reduce_size  = align_to_int64( pool_reduce_size );
      team_reduce_size  = align_to_int64( team_reduce_size );
      team_shared_size  = align_to_int64( team_shared_size );
      thread_local_size = align_to_int64( thread_local_size );
 
      const size_t total_bytes = (
        m_pool_reduce +
        pool_reduce_size +
        team_reduce_size +
        team_shared_size +
        thread_local_size ) * sizeof(int64_t);

      return total_bytes ;
    }

  // Given:
  //   alloc_ptr         = pointer to allocated memory
  //   alloc_size        = number bytes of allocated memory
  //   pool_reduce_size  = number bytes for pool reduce/scan operations
  //   team_reduce_size  = number bytes for team reduce/scan operations
  //   team_shared_size  = number bytes for team-shared memory
  //   thread_local_size = number bytes for thread-local memory
  // Return:
  //   total number of bytes that must be allocated
  void scratch_assign( void * const alloc_ptr
                     , size_t const alloc_size
                     , int pool_reduce_size
                     , int team_reduce_size
                     , int team_shared_size
                     , int /* thread_local_size */ )
    {
      pool_reduce_size  = align_to_int64( pool_reduce_size );
      team_reduce_size  = align_to_int64( team_reduce_size );
      team_shared_size  = align_to_int64( team_shared_size );
      // thread_local_size = align_to_int64( thread_local_size );

      m_scratch      = (int64_t *) alloc_ptr ;
      m_team_reduce  = m_pool_reduce + pool_reduce_size ;
      m_team_shared  = m_team_reduce + team_reduce_size ;
      m_thread_local = m_team_shared + team_shared_size ;
      m_scratch_size = align_to_int64( alloc_size );

#if 0
fprintf(stdout,"HostThreadTeamData::scratch_assign { %d %d %d %d %d %d %d }\n"
       , int(m_pool_members)
       , int(m_pool_rendezvous)
       , int(m_pool_reduce)
       , int(m_team_reduce)
       , int(m_team_shared)
       , int(m_thread_local)
       , int(m_scratch_size)
       );
fflush(stdout);
#endif

    }

  //----------------------------------------
  // Set the initial work partitioning of [ 0 .. length ) among the teams

  void set_work_partition( int const length ) noexcept
    {
      int const chunk = ( length + m_league_size - 1 ) / m_league_size ;

      m_work_range.first  = chunk * m_league_rank ;
      m_work_range.second = m_work_range.first + chunk < m_league_size 
                          ? m_work_range.first + chunk : m_league_size ;

      m_steal_rank = ( m_team_base + m_team_alloc ) % m_team_alloc ;
    }

  // Get one work index within the range
  int get_work_static() noexcept
    {
      return m_work_range.first < m_work_range.second ?
             m_work_range.first++ : -1 ;
    }

  // Get a work index within the range.
  // First try to steal from beginning of own teams's partition.
  // If that fails then try to steal from end of another teams' partition.
  int get_work_stealing() noexcept ;
};

//----------------------------------------------------------------------------

template< class HostExecSpace >
class HostThreadTeamMember {
public:

  using scratch_memory_space = typename HostExecSpace::scratch_memory_space ;

private:

  scratch_memory_space m_scratch ;
  HostThreadTeamData & m_data ;
  int const            m_league_rank ;
  int const            m_league_size ;

public:

  constexpr HostThreadTeamMember( HostThreadTeamData & arg_data ) noexcept
    : m_scratch( arg_data.team_shared() , arg_data.team_shared_bytes() )
    , m_data( arg_data )
    , m_league_rank(0)
    , m_league_size(1)
    {}

  constexpr HostThreadTeamMember( HostThreadTeamData & arg_data
                                , int const            arg_league_rank
                                , int const            arg_league_size
                                ) noexcept
    : m_scratch( arg_data.team_shared() , arg_data.team_shared_bytes() )
    , m_data( arg_data )
    , m_league_rank( arg_league_rank )
    , m_league_size( arg_league_size )
    {}

  ~HostThreadTeamMember() = default ;
  HostThreadTeamMember() = delete ;
  HostThreadTeamMember( HostThreadTeamMember && ) = default ;
  HostThreadTeamMember( HostThreadTeamMember const & ) = default ;
  HostThreadTeamMember & operator = ( HostThreadTeamMember && ) = default ;
  HostThreadTeamMember & operator = ( HostThreadTeamMember const & ) = default ;

  //----------------------------------------

  KOKKOS_INLINE_FUNCTION
  int team_rank() const noexcept { return m_data.m_team_rank ; }

  KOKKOS_INLINE_FUNCTION
  int team_size() const noexcept { return m_data.m_team_size ; }

  KOKKOS_INLINE_FUNCTION
  int league_rank() const noexcept { return m_league_rank ; }

  KOKKOS_INLINE_FUNCTION
  int league_size() const noexcept { return m_league_size ; }

  //----------------------------------------

  KOKKOS_INLINE_FUNCTION
  const scratch_memory_space & team_shmem() const
    { return m_scratch.set_team_thread_mode(0,1,0); }
  
  KOKKOS_INLINE_FUNCTION
  const scratch_memory_space & team_scratch(int) const
    { return m_scratch.set_team_thread_mode(0,1,0); }
  
  KOKKOS_INLINE_FUNCTION
  const scratch_memory_space & thread_scratch(int) const
    { return m_scratch.set_team_thread_mode(0,m_data.m_team_size,m_data.m_team_rank); }
  
  //----------------------------------------
  // Team collectives

  KOKKOS_INLINE_FUNCTION void team_barrier() const noexcept
#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
    {
      if ( 1 < m_data.m_team_size ) {
        if ( m_data.team_rendezvous() ) m_data.team_rendezvous_release();
      }
    }
#else
    {}
#endif

  template< typename T >
  KOKKOS_INLINE_FUNCTION
  void team_broadcast( T & value , const int source_team_rank = 0 ) const noexcept
#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
    {
      if ( 1 < m_data.m_team_size ) {
        T volatile * const shared_value = (T*) m_data.team_reduce();

        // Don't overwrite shared memory until all threads arrive

        if ( m_data.team_rendezvous( source_team_rank ) ) {
          // All threads have entered 'team_rendezvous'
          // only this thread returned from 'team_rendezvous'
          // with a return value of 'true'

          *shared_value = value ;

          m_data.team_rendezvous_release();
          // This thread released all other threads from 'team_rendezvous'
          // with a return value of 'false'
        }
        else {
          value = *shared_value ;
        }
      }
    }
#else
    { Kokkos::abort("HostThreadTeamMember team_broadcast\n"); }
#endif

  // team_reduce( Sum(result) );
  // team_reduce( Min(result) );
  // team_reduce( Max(result) );
  template< typename ReducerType >
  KOKKOS_INLINE_FUNCTION
  void team_reduce( ReducerType const & reducer ) const noexcept
#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
    {
      static_assert( Kokkos::is_reducer< ReducerType >::value
                   , "team_reduce requires a reducer object" );

      if ( 1 < m_data.m_team_size ) {

        using value_type = typename ReducerType::value_type ;

        value_type volatile * const shared_value =
          (value_type*) m_data.team_reduce();

        value_type volatile * const member_value =
          shared_value + m_data.m_team_rank * reducer.length();

        // Don't overwrite shared memory until all threads arrive

        team_barrier();

        for ( int i = 0 ; i < reducer.length() ; ++i ) {
          member_value[i] = reducer[i];
        }

        // Wait for all team members to store data

        if ( m_data.team_rendezvous() ) {
          // All threads have entered 'team_rendezvous'
          // only this thread returned from 'team_rendezvous'
          // with a return value of 'true'
          //
          // This thread sums contributed values
          for ( int i = 1 ; i < m_data.m_team_size ; ++i ) {
            reducer.join( shared_value , shared_value + i * reducer.length() );
          }
          m_data.team_rendezvous_release();
          // This thread released all other threads from 'team_rendezvous'
          // with a return value of 'false'
        }

        for ( int i = 0 ; i < reducer.length() ; ++i ) {
          reducer[i] = shared_value[i] ;
        }
      }
    }
#else
    { Kokkos::abort("HostThreadTeamMember team_reduce\n"); }
#endif

  template< typename T >
  KOKKOS_INLINE_FUNCTION
  void team_scan( T & value , T * const global = 0 ) const noexcept
#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
    {
      if ( 1 < m_data.m_team_size ) {

        T volatile * const shared_value = (T*) m_data.team_reduce();

        // Don't overwrite shared memory until all threads arrive
        team_barrier();

        shared_value[ m_data.m_team_rank + 1 ] = value ;

        if ( m_data.team_rendezvous() ) {
          // All threads have entered 'team_rendezvous'
          // only this thread returned from 'team_rendezvous'
          // with a return value of 'true'
          //
          // This thread scans contributed values
          for ( int i = 1 ; i < m_data.m_team_size ; ++i ) {
            shared_value[i+1] += shared_value[i] ;
          }

          shared_value[0] = 0 ;

          // If adding to global value then atomic_fetch_add to that value
          // and sum previous value to every entry of the scan.
          if ( global ) {
            shared_value[0] =
              Kokkos::atomic_fetch_add( global
                                      , shared_value[m_data.m_team_size] );
            for ( int i = 1 ; i < m_data.m_team_size ; ++i ) {
              shared_value[i] += shared_value[0] ;
            }
          }

          m_data.team_rendezvous_release();
        }

        value = shared_value[ m_data.m_team_rank ];
      }
      else if ( global ) {
        value = Kokkos::atomic_fetch_add( global , value );
      }
      else {
        value = 0 ;
      }
    }
#else
    { Kokkos::abort("HostThreadTeamMember team_scan\n"); }
#endif

};


}} /* namespace Kokkos::Impl */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

template<class Space,typename iType>
KOKKOS_INLINE_FUNCTION
Impl::TeamThreadRangeBoundariesStruct<iType,Impl::HostThreadTeamMember<Space> >
TeamThreadRange( Impl::HostThreadTeamMember<Space> const & member
               , iType const & count )
{
  return
    Impl::TeamThreadRangeBoundariesStruct
      <iType,Impl::HostThreadTeamMember<Space> >(member,0,count);
}

template<class Space, typename iType1, typename iType2>
KOKKOS_INLINE_FUNCTION
Impl::TeamThreadRangeBoundariesStruct
  < typename std::common_type< iType1, iType2 >::type
  , Impl::HostThreadTeamMember<Space> >
TeamThreadRange( Impl::HostThreadTeamMember<Space> const & member
               , iType1 const & begin , iType2 const & end )
{
  return
    Impl::TeamThreadRangeBoundariesStruct
      < typename std::common_type< iType1, iType2 >::type
      , Impl::HostThreadTeamMember<Space> >( member , begin , end );
}

template<class Space, typename iType>
KOKKOS_INLINE_FUNCTION
Impl::ThreadVectorRangeBoundariesStruct<iType,Impl::HostThreadTeamMember<Space> >
ThreadVectorRange
  ( Impl::HostThreadTeamMember<Space> & member
  , const iType & count )
{
  return Impl::ThreadVectorRangeBoundariesStruct<iType,Impl::HostThreadTeamMember<Space> >(member,count);
}

/** \brief  Inter-thread parallel_for. Executes lambda(iType i) for each i=0..N-1.
 *
 * The range i=0..N-1 is mapped to all threads of the the calling thread team.
 * This functionality requires C++11 support.
*/
template<typename iType, class Space, class Closure>
KOKKOS_INLINE_FUNCTION
void parallel_for
  ( Impl::TeamThreadRangeBoundariesStruct<iType,Impl::HostThreadTeamMember<Space> >
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

template< typename iType, class Space, class Closure, class Reducer >
KOKKOS_INLINE_FUNCTION
typename std::enable_if< Kokkos::is_reducer< Reducer >::value >::type
parallel_reduce
  ( Impl::TeamThreadRangeBoundariesStruct<iType,Impl::HostThreadTeamMember<Space> >
             const & loop_boundaries
  , Closure  const & closure
  , Reducer  const & reducer
  )
{
  reducer.init( reducer.data() );

  for( iType i = loop_boundaries.start
     ; i <  loop_boundaries.end
     ; i += loop_boundaries.increment ) {
    closure( i , reducer.reference() );
  }

  loop_boundaries.thread.team_reduce( reducer );
}

template< typename iType, class Space, typename Closure, typename ValueType >
KOKKOS_INLINE_FUNCTION
typename std::enable_if< std::is_trivial<ValueType>::value &&
                         ! Kokkos::is_reducer<ValueType>::value >::type
parallel_reduce
  ( Impl::TeamThreadRangeBoundariesStruct<iType,Impl::HostThreadTeamMember<Space> >
             const & loop_boundaries
  , Closure  const & closure
  , ValueType      & result
  )
{
  Impl::Reducer< ValueType , Impl::ReduceSum > reducer( & result );

  reducer.init( reducer.data() );

  for( iType i = loop_boundaries.start
     ; i <  loop_boundaries.end
     ; i += loop_boundaries.increment ) {
    closure( i , reducer.reference() );
  }

  loop_boundaries.thread.team_reduce( reducer );
}

#if 0
// placeholders for future functions

template< typename iType, class Lambda, typename ValueType >
KOKKOS_INLINE_FUNCTION
void parallel_reduce
  (const Impl::ThreadVectorRangeBoundariesStruct<iType,Impl::HostThreadTeamMember >& loop_boundaries,
   const Lambda & lambda,
   ValueType& initialized_result)
{
}

template< typename iType, class Lambda, typename ValueType, class JoinType >
KOKKOS_INLINE_FUNCTION
void parallel_reduce
  (const Impl::ThreadVectorRangeBoundariesStruct<iType,Impl::HostThreadTeamMember >& loop_boundaries,
   const Lambda & lambda,
   const JoinType & join,
   ValueType& initialized_result)
{
}
#endif


template< typename iType, class Space, class Closure >
KOKKOS_INLINE_FUNCTION
void parallel_scan
  ( Impl::TeamThreadRangeBoundariesStruct<iType,Impl::HostThreadTeamMember<Space> >
       const & loop_boundaries
  , Closure const & closure
  )
{
  // Extract ValueType from the closure

  using value_type =
    typename Kokkos::Impl::FunctorValueTraits< Closure , void >::value_type ;

  value_type accum = 0 ;

  // Intra-member scan
  for ( iType i = loop_boundaries.start
      ; i <  loop_boundaries.end
      ; i += loop_boundaries.increment ) {
    closure(i,accum,false);
  }

  // 'accum' output is the exclusive prefix sum
  loop_boundaries.thread.team_scan(accum);

  for ( iType i = loop_boundaries.start
      ; i <  loop_boundaries.end
      ; i += loop_boundaries.increment ) {
    closure(i,accum,true);
  }
}

#if 0

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

