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

#include <limits>
#include <Kokkos_Macros.hpp>
#include <impl/Kokkos_HostThreadTeam.hpp>
#include <impl/Kokkos_Error.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#if ( KOKKOS_ENABLE_ASM )
  #if defined( __arm__ ) || defined( __aarch64__ )
    /* No-operation instruction to idle the thread. */
    #define YIELD   asm volatile("nop")
  #else
    /* Pause instruction to prevent excess processor bus usage */
    #define YIELD   asm volatile("pause\n":::"memory")
  #endif
#elif defined ( KOKKOS_ENABLE_WINTHREAD )
  #include <process.h>
  #define YIELD  Sleep(0)
#elif defined ( _WIN32)  && defined (_MSC_VER)
  /* Windows w/ Visual Studio */
  #define NOMINMAX
  #include <winsock2.h>
  #include <windows.h>
#define YIELD YieldProcessor();
#elif defined ( _WIN32 )
  /* Windows w/ Intel*/
  #define YIELD __asm__ __volatile__("pause\n":::"memory")
#else
  #include <sched.h>
  #define YIELD  sched_yield()
#endif

namespace Kokkos {
namespace Impl {
namespace {

void wait_until_equal( int64_t const value , int64_t volatile * const sync )
{
  while ( value != *sync ) {
    YIELD ;
  }
}

}
} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

void HostThreadTeamData::organize_pool
  ( HostThreadTeamData * members[] , const int size )
{
  bool ok = true ;

  // Verify not already a member of a pool:
  for ( int rank = 0 ; rank < size && ok ; ++rank ) {
    ok = ( 0 != members[rank] ) && ( 0 == members[rank]->m_pool_scratch );
  }

  if ( ok ) {

    int64_t * const root_scratch = members[0]->m_scratch ;

    for ( int i = m_pool_rendezvous ; i < m_pool_reduce ; ++i ) {
      root_scratch[i] = 0 ;
    }

    {
      HostThreadTeamData ** const pool =
        (HostThreadTeamData **) (root_scratch + m_pool_members);

      // team size == 1, league size == pool_size

      for ( int rank = 0 ; rank < size ; ++rank ) {
        HostThreadTeamData * const mem = members[ rank ] ;
        mem->m_pool_scratch = root_scratch ;
        mem->m_team_scratch = mem->m_scratch ;
        mem->m_pool_rank    = rank ;
        mem->m_pool_size    = size ;
        mem->m_team_base    = rank ;
        mem->m_team_rank    = 0 ;
        mem->m_team_size    = 1 ;
        mem->m_team_alloc   = 1 ;
        mem->m_league_rank  = rank ;
        mem->m_league_size  = size ;
        mem->m_pool_rendezvous_step = 0 ;
        mem->m_team_rendezvous_step = 0 ;
        pool[ rank ] = mem ;
      }
    }
  }
  else {
    Kokkos::Impl::throw_runtime_exception("Kokkos::Impl::HostThreadTeamData::organize_pool ERROR pool already exists");
  }
}

void HostThreadTeamData::disband_pool()
{
   m_work_range.first  = -1 ;
   m_work_range.second = -1 ;
   m_pool_scratch = 0 ;
   m_team_scratch = 0 ;
   m_pool_rank    = 0 ;
   m_pool_size    = 1 ;
   m_team_base    = 0 ;
   m_team_rank    = 0 ;
   m_team_size    = 1 ;
   m_team_alloc   = 1 ;
   m_league_rank  = 0 ;
   m_league_size  = 1 ;
   m_pool_rendezvous_step = 0 ;
   m_team_rendezvous_step = 0 ;
}

int HostThreadTeamData::organize_team( const int team_size )
{
  // Pool is initialized
  const bool ok_pool = 0 != m_pool_scratch ;

  // Team is not set
  const bool ok_team =
    m_team_scratch == m_scratch &&
    m_team_base    == m_pool_rank &&
    m_team_rank    == 0 &&
    m_team_size    == 1 &&
    m_team_alloc   == 1 &&
    m_league_rank  == m_pool_rank &&
    m_league_size  == m_pool_size ;

  // Pool and be symmetrically partitioned per team_size
  const bool ok_size =
    0 <  team_size &&
    0 == m_pool_size % ( ( m_pool_size + team_size - 1 ) / team_size );

  if ( ok_pool && ok_team && ok_size ) {

    HostThreadTeamData * const * const pool =
      (HostThreadTeamData **) (m_pool_scratch + m_pool_members);

    const int league_size     = ( m_pool_size + team_size - 1 ) / team_size ;
    const int team_alloc_size = m_pool_size / league_size ;
    const int team_alloc_rank = m_pool_rank % team_alloc_size ;
    const int league_rank     = m_pool_rank / team_alloc_size ;
    const int team_base_rank  = league_rank * team_alloc_size ;

    m_team_scratch = pool[ team_base_rank ]->m_scratch ;
    m_team_base    = team_base_rank ;
    m_team_rank    = team_alloc_rank < team_size ? team_alloc_rank : -1 ;
    m_team_size    = team_size ;
    m_team_alloc   = team_alloc_size ;
    m_league_rank  = league_rank ;
    m_league_size  = league_size ;
    m_team_rendezvous_step = 0 ;

    if ( team_base_rank == m_pool_rank ) {
      for ( int i = m_team_rendezvous ; i < m_pool_reduce ; ++i ) {
        m_scratch[i] = 0 ;
      }
    }

    // Organizing threads into a team performs a barrier across the
    // entire pool to insure proper initialization of the team
    // rendezvous mechanism before a team rendezvous can be performed.

    if ( pool_rendezvous() ) {
      pool_rendezvous_release();
    }
  }
  else {
    Kokkos::Impl::throw_runtime_exception("Kokkos::Impl::HostThreadTeamData::organize_team ERROR");
  }

  return 0 <= m_team_rank ;
}

void HostThreadTeamData::disband_team()
{
  m_team_scratch = m_scratch ;
  m_team_base    = m_pool_rank ;
  m_team_rank    = 0 ;
  m_team_size    = 1 ;
  m_team_alloc   = 1 ;
  m_league_rank  = m_pool_rank ;
  m_league_size  = m_pool_size ;
  m_team_rendezvous_step = 0 ;
}

//----------------------------------------------------------------------------
/* pattern for rendezvous
 *
 *  if ( rendezvous() ) {
 *     ... all other threads are still in team_rendezvous() ...
 *     rendezvous_release();
 *     ... all other threads are released from team_rendezvous() ...
 *  }
 */

int HostThreadTeamData::rendezvous( int64_t * const buffer
                                    , int & rendezvous_step
                                    , int const size
                                    , int const rank ) noexcept
{
  // Requires:
  //   Called by rank = [ 0 .. size )

  // A sequence of rendezvous uses alternating locations in memory
  // and alternating synchronization values to prevent rendezvous
  // from overtaking one another.

  // Each member has a designated byte to set in the span

  // 1 <= step <= 4

  const int step = ( rendezvous_step & 03 ) + 1 ;

  rendezvous_step = step ;

  // For an upper bound of 64 threads per team the shared array is uint64_t[16].
  // For this step the interval begins at ( step & 01 ) * 8

  int64_t volatile * const sync_base = buffer + (( step & 01 ) << 3 );

  union {
    int64_t full ;
    int8_t  byte[8] ;
  } value ;

  for ( int shift = 6 ; shift ; ) {
    const int group = rank << shift ; shift -= 3 ;
    const int start = 1    << shift ;

    if ( start <= rank && group < size ) {
      // "Outer" rendezvous for ranks
      //   Iteration #0: [ 8 .. 64 ) waits for [ 64 .. 512 )
      //   Iteration #1: [ 1 .. 8 )  waits for [  8 ..  64 )
      //
      // Requires:
      //   size <= 512
      //
      // Effects:
      //   rank waits for [ rank*8 .. (rank+1)*8 )

      value.full = 0 ;
      const int n = ( size - group ) < 8 ? size - group : 8 ;
      for ( int i = 0 ; i < n ; ++i ) value.byte[i] = step ;
      wait_until_equal( value.full , sync_base + rank );
    }
  }

  if ( rank ) {
    // All previous memory stores must be complete.
    // Then store value at this thread's designated byte in the shared array.

    Kokkos::memory_fence();

    ((volatile int8_t*) sync_base)[rank] = int8_t( step );
  }

  { // "Inner" rendezvous for ranks [ 0 .. 7 ]
    // Effects:
    //   0  < rank < size  wait for [0..7]
    //   0 == rank         wait for [1..7]

    value.full = 0 ;
    const int n = size < 8 ? size : 8 ;
    for ( int i = 1 ; i < n ; ++i ) value.byte[i] = step ;
    value.byte[0] = rank ? step : ((volatile int8_t*) sync_base)[0];

    wait_until_equal( value.full , sync_base );
  }

  return rank ? 0 : 1 ;
}

void HostThreadTeamData::
  rendezvous_release( int64_t * const buffer
                    , int const rendezvous_step ) noexcept
{
  // Requires:
  //   Called after team_rendezvous
  //   Called only by true == team_rendezvous(root)

  int64_t volatile * const sync_base =
    buffer + (( rendezvous_step & 01 ) << 3 );

  // Memory fence to be sure all prevous writes are complete:
  Kokkos::memory_fence();

  ((volatile int8_t*) sync_base)[0] = int8_t( rendezvous_step );
}

//----------------------------------------------------------------------------

int HostThreadTeamData::get_work_stealing() noexcept
{
  pair_int_t w( -1 , -1 );

  if ( 1 == m_team_size || team_rendezvous() ) {

    // Attempt first from beginning of my work range
    for ( int attempt = m_work_range.first < m_work_range.second ; attempt ; ) {

      // Query and attempt to update m_work_range
      //   from: [ w.first     , w.second )
      //   to:   [ w.first + 1 , w.second ) = w_new
      //
      // If w is invalid then is just a query.

      const pair_int_t w_new( w.first + 1 , w.second );

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

        attempt = 0 ;
      }
    }

    if ( w.first == -1 && m_steal_rank != m_pool_rank ) {

      HostThreadTeamData * const * const pool =
        (HostThreadTeamData**)( m_pool_scratch + m_pool_members );

      // Attempt from begining failed, try to steal from end of neighbor

      pair_int_t volatile * steal_range =
        & ( pool[ m_steal_rank ]->m_work_range );

      for ( int attempt = true ; attempt ; ) {

        // Query and attempt to update steal_work_range
        //   from: [ w.first , w.second )
        //   to:   [ w.first , w.second - 1 ) = w_new
        //
        // If w is invalid then is just a query.

        const pair_int_t w_new( w.first , w.second - 1 );

        w = Kokkos::atomic_compare_exchange( steal_range, w, w_new );

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

          m_steal_rank = ( m_steal_rank + m_team_alloc ) % m_pool_size ;

          steal_range = & ( pool[ m_steal_rank ]->m_work_range );

          // If tried all other members then don't repeat attempt to steal
          attempt = m_steal_rank != m_pool_rank ;
        }
      }

      if ( w.first != -1 ) w.first = w.second - 1 ;
    }

    if ( 1 < m_team_size ) {
      // Must share the work index
      *((int volatile *) team_reduce()) = w.first ;

      team_rendezvous_release();
    }
  }
  else if ( 1 < m_team_size ) {
    w.first = *((int volatile *) team_reduce());
  }

  // May exit because successfully stole work and w is good.
  // May exit because no work left to steal and w = (-1,-1).

#if 0
fprintf(stdout,"HostThreadTeamData::get_work_stealing() pool(%d of %d) %d\n"
       , m_pool_rank , m_pool_size , w.first );
fflush(stdout);
#endif

  return w.first ;
}

} // namespace Impl
} // namespace Kokkos

