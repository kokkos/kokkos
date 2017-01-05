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

#include <impl/Kokkos_HostThreadTeam.hpp>

namespace Kokkos {
namespace Impl {

//----------------------------------------------------------------------------

void HostThreadTeamMember::form_active_team
  ( HostThreadTeamMember * members[] , const int size )
{
  // Verify not already a member of another team
  for ( int i = 0 ; i < size ; ++i ) {

  }

  HostThreadTeamMember & leader = * members[0];

  for ( int i = 0 ; i < size ; ++i ) {

    ((HostThreadTeamMember **)( leader.m_scratch ))[i] = members[i] ;

    HostThreadTeamMember & mem = * members[i] ;

    mem.m_shared          = leader.m_scratch ;
    mem.m_team_rank_steal = ( i + 1 ) % size ;
    mem.m_team_rank       = i ;
    mem.m_team_size       = size ;
    mem.m_league_rank     = 0 ;
    mem.m_league_size     = 1 ;
    mem.m_rendezvous_step = 0 ;
  }
}

// Team leader disbands the team
void HostThreadTeamMember::disband_active_team()
{
  // Precondition:
  //   this == ((HostThreadTeamMember**)m_shared)[0]

  for ( int r = m_team_size - 1 ; 0 <= r ; --r ) {

    HostThreadTeamMember & mem =
      *((HostThreadTeamMember**)m_shared)[r] ;

    ((HostThreadTeamMember**)m_shared)[r] = 0 ;

    mem.m_shared          = 0 ;
    mem.m_team_rank_steal = 0 ;
    mem.m_team_rank       = 0 ;
    mem.m_team_size       = 0 ;
    mem.m_league_rank     = 0 ;
    mem.m_league_size     = 0 ;
    mem.m_rendezvous_step = 0 ;
  }
}

//----------------------------------------------------------------------------
/* pattern for team_rendezvous
 *
 *  if ( team_rendezvous() ) {
 *     ... all other threads are still in team_rendezvous() ...
 *     team_rendezvous_release();
 *     ... all other threads are released from team_rendezvous() ...
 *  }
 */

int HostThreadTeamMember::team_rendezvous( int const root ) const noexcept
{
  // Requires:
  //   Called by all team members

  // A sequence of rendezvous uses alternating locations in memory
  // and alternating synchronization values to prevent rendezvous
  // from overtaking one another.

  // Each team member has a designated byte to set in the span

  // 1 <= step <= 4

  const int size  = m_team_size ;
  const int rank  = root ? ( m_team_rank + root ) % size : m_team_rank ;
  const int base  = rank ? 0 : 1 ;
  const int step  = ( m_rendezvous_step & 03 ) + 1 ; m_rendezvous_step = step ;

  // For an upper bound of 64 threads per team the shared array is uint64_t[16].
  // For this step the interval begins at ( step & 01 ) * 8

  int64_t volatile * const sync_base = rendezvous_memory() + (( step & 01 ) << 3 );

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
      int64_t volatile * const sync = sync_base + rank ;
      while ( value.full != *sync );
    }
  }

  if ( rank ) {
    // All previous memory stores must be complete.
    // Then store value at this thread's designated byte in the shared array.

    Kokkos::memory_fence();

    *(((volatile int8_t*) sync_base) + rank ) = int8_t( step );
  }

  { // "Inner" rendezvous for ranks [ 0 .. 8 )
    // Effects:
    //   0  < rank < size  wait for [0..7]
    //   0 == rank         wait for [1..7]

    value.full = 0 ;
    const int n = size < 8 ? size : 8 ;
    for ( int i = base ; i < n ; ++i ) value.byte[i] = step ;
    while ( value.full != *sync_base );
  }

  return base ; // rank == 0
}

void HostThreadTeamMember::team_rendezvous_release() const noexcept
{
  // Requires:
  //   Called after team_rendezvous
  //   Called only by true == team_rendezvous(root)

  int64_t volatile * const sync_base =
    rendezvous_memory() + (( m_rendezvous_step & 01 ) << 3 );

  Kokkos::memory_fence();

  *((volatile int8_t*) sync_base) = int8_t( m_rendezvous_step );
}

} // namespace Impl
} // namespace Kokkos

