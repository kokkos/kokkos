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

#ifndef KOKKOS_OPENMPEXEC_HPP
#define KOKKOS_OPENMPEXEC_HPP

#include <impl/Kokkos_Traits.hpp>
#include <impl/Kokkos_spinwait.hpp>
#include <impl/Kokkos_HostThreadTeam.hpp>

#include <Kokkos_Atomic.hpp>

#include <iostream>
#include <sstream>
#include <fstream>

//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

// Resize thread team data scratch memory
void openmp_resize_thread_team_data( size_t pool_reduce_bytes
                                   , size_t team_reduce_bytes
                                   , size_t team_shared_bytes
                                   , size_t thread_local_bytes );

// Get thread team data structure for omp_get_thread_num()
HostThreadTeamData * openmp_get_thread_team_data();

// Get thread team data structure for rank
HostThreadTeamData * openmp_get_thread_team_data( int );

} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

//----------------------------------------------------------------------------
/** \brief  Data for OpenMP thread execution */

class OpenMPexec {
public:

  enum { MAX_THREAD_COUNT = 4096 };

private:

  static OpenMPexec * m_pool[ MAX_THREAD_COUNT ]; // Indexed by: m_pool_rank_rev

  static int          m_pool_topo[ 4 ];
  static int          m_map_rank[ MAX_THREAD_COUNT ];

  friend class Kokkos::OpenMP ;

  int const  m_pool_rank ;
  int const  m_pool_rank_rev ;
  int const  m_scratch_exec_end ;
  int const  m_scratch_reduce_end ;
  int const  m_scratch_thread_end ;

  int volatile  m_barrier_state ;

  // Members for dynamic scheduling
  // Which thread am I stealing from currently
  int m_current_steal_target;
  // This thread's owned work_range
  Kokkos::pair<long,long> m_work_range KOKKOS_ALIGN_16;
  // Team Offset if one thread determines work_range for others
  long m_team_work_index;

  // Is this thread stealing (i.e. its owned work_range is exhausted
  bool m_stealing;

  OpenMPexec();
  OpenMPexec( const OpenMPexec & );
  OpenMPexec & operator = ( const OpenMPexec & );

  static void clear_scratch();

public:

  // Topology of a cache coherent thread pool:
  //   TOTAL = NUMA x GRAIN
  //   pool_size( depth = 0 )
  //   pool_size(0) = total number of threads
  //   pool_size(1) = number of threads per NUMA
  //   pool_size(2) = number of threads sharing finest grain memory hierarchy

  inline static
  int pool_size( int depth = 0 ) { return m_pool_topo[ depth ]; }

  inline static
  OpenMPexec * pool_rev( int pool_rank_rev ) { return m_pool[ pool_rank_rev ]; }

  inline int pool_rank() const { return m_pool_rank ; }
  inline int pool_rank_rev() const { return m_pool_rank_rev ; }

  inline long team_work_index() const { return m_team_work_index ; }

  inline int scratch_reduce_size() const
    { return m_scratch_reduce_end - m_scratch_exec_end ; }

  inline int scratch_thread_size() const
    { return m_scratch_thread_end - m_scratch_reduce_end ; }

  inline void * scratch_reduce() const { return ((char *) this) + m_scratch_exec_end ; }
  inline void * scratch_thread() const { return ((char *) this) + m_scratch_reduce_end ; }

  inline
  void state_wait( int state )
    { Impl::spinwait( m_barrier_state , state ); }

  inline
  void state_set( int state ) { m_barrier_state = state ; }

  ~OpenMPexec() {}

  OpenMPexec( const int arg_poolRank
            , const int arg_scratch_exec_size
            , const int arg_scratch_reduce_size
            , const int arg_scratch_thread_size )
    : m_pool_rank( arg_poolRank )
    , m_pool_rank_rev( pool_size() - ( arg_poolRank + 1 ) )
    , m_scratch_exec_end( arg_scratch_exec_size )
    , m_scratch_reduce_end( m_scratch_exec_end   + arg_scratch_reduce_size )
    , m_scratch_thread_end( m_scratch_reduce_end + arg_scratch_thread_size )
    , m_barrier_state(0)
    {}

  static void finalize();

  static void initialize( const unsigned  team_count ,
                          const unsigned threads_per_team ,
                          const unsigned numa_count ,
                          const unsigned cores_per_numa );

  static void verify_is_process( const char * const );
  static void verify_initialized( const char * const );

  static void resize_scratch( size_t reduce_size , size_t thread_size );

  inline static
  OpenMPexec * get_thread_omp() { return m_pool[ m_map_rank[ omp_get_thread_num() ] ]; }

  /* Dynamic Scheduling related functionality */
  // Initialize the work range for this thread
  inline void set_work_range(const long& begin, const long& end, const long& chunk_size) {
    m_work_range.first = (begin+chunk_size-1)/chunk_size;
    m_work_range.second = end>0?(end+chunk_size-1)/chunk_size:m_work_range.first;
  }

  // Claim and index from this thread's range from the beginning
  inline long get_work_index_begin () {
    Kokkos::pair<long,long> work_range_new = m_work_range;
    Kokkos::pair<long,long> work_range_old = work_range_new;
    if(work_range_old.first>=work_range_old.second)
      return -1;

    work_range_new.first+=1;

    bool success = false;
    while(!success) {
      work_range_new = Kokkos::atomic_compare_exchange(&m_work_range,work_range_old,work_range_new);
      success = ( (work_range_new == work_range_old) ||
                  (work_range_new.first>=work_range_new.second));
      work_range_old = work_range_new;
      work_range_new.first+=1;
    }
    if(work_range_old.first<work_range_old.second)
      return work_range_old.first;
    else
      return -1;
  }

  // Claim and index from this thread's range from the end
  inline long get_work_index_end () {
    Kokkos::pair<long,long> work_range_new = m_work_range;
    Kokkos::pair<long,long> work_range_old = work_range_new;
    if(work_range_old.first>=work_range_old.second)
      return -1;
    work_range_new.second-=1;
    bool success = false;
    while(!success) {
      work_range_new = Kokkos::atomic_compare_exchange(&m_work_range,work_range_old,work_range_new);
      success = ( (work_range_new == work_range_old) ||
                  (work_range_new.first>=work_range_new.second) );
      work_range_old = work_range_new;
      work_range_new.second-=1;
    }
    if(work_range_old.first<work_range_old.second)
      return work_range_old.second-1;
    else
      return -1;
  }

  // Reset the steal target
  inline void reset_steal_target() {
    m_current_steal_target = (m_pool_rank+1)%m_pool_topo[0];
    m_stealing = false;
  }

  // Reset the steal target
  inline void reset_steal_target(int team_size) {
    m_current_steal_target = (m_pool_rank_rev+team_size);
    if(m_current_steal_target>=m_pool_topo[0])
      m_current_steal_target = 0;//m_pool_topo[0]-1;
    m_stealing = false;
  }

  // Get a steal target; start with my-rank + 1 and go round robin, until arriving at this threads rank
  // Returns -1 fi no active steal target available
  inline int get_steal_target() {
    while(( m_pool[m_current_steal_target]->m_work_range.second <=
            m_pool[m_current_steal_target]->m_work_range.first  ) &&
          (m_current_steal_target!=m_pool_rank) ) {
      m_current_steal_target = (m_current_steal_target+1)%m_pool_topo[0];
    }
    if(m_current_steal_target == m_pool_rank)
      return -1;
    else
      return m_current_steal_target;
  }

  inline int get_steal_target(int team_size) {

    while(( m_pool[m_current_steal_target]->m_work_range.second <=
            m_pool[m_current_steal_target]->m_work_range.first  ) &&
          (m_current_steal_target!=m_pool_rank_rev) ) {
      if(m_current_steal_target + team_size < m_pool_topo[0])
        m_current_steal_target = (m_current_steal_target+team_size);
      else
        m_current_steal_target = 0;
    }

    if(m_current_steal_target == m_pool_rank_rev)
      return -1;
    else
      return m_current_steal_target;
  }

  inline long steal_work_index (int team_size = 0) {
    long index = -1;
    int steal_target = team_size>0?get_steal_target(team_size):get_steal_target();
    while ( (steal_target != -1) && (index == -1)) {
      index = m_pool[steal_target]->get_work_index_end();
      if(index == -1)
        steal_target = team_size>0?get_steal_target(team_size):get_steal_target();
    }
    return index;
  }

  // Get a work index. Claim from owned range until its exhausted, then steal from other thread
  inline long get_work_index (int team_size = 0) {
    long work_index = -1;
    if(!m_stealing) work_index = get_work_index_begin();

    if( work_index == -1) {
      memory_fence();
      m_stealing = true;
      work_index = steal_work_index(team_size);
    }
    m_team_work_index = work_index;
    memory_fence();
    return work_index;
  }

};

} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template< class ... Properties >
class TeamPolicyInternal< Kokkos::OpenMP, Properties ... >: public PolicyTraits<Properties ...>
{
public:

  //! Tag this class as a kokkos execution policy
  typedef TeamPolicyInternal      execution_policy ;

  typedef PolicyTraits<Properties ... > traits;

  TeamPolicyInternal& operator = (const TeamPolicyInternal& p) {
    m_league_size = p.m_league_size;
    m_team_size = p.m_team_size;
    m_team_alloc = p.m_team_alloc;
    m_team_iter = p.m_team_iter;
    m_team_scratch_size[0] = p.m_team_scratch_size[0];
    m_thread_scratch_size[0] = p.m_thread_scratch_size[0];
    m_team_scratch_size[1] = p.m_team_scratch_size[1];
    m_thread_scratch_size[1] = p.m_thread_scratch_size[1];
    m_chunk_size = p.m_chunk_size;
    return *this;
  }

  //----------------------------------------

  template< class FunctorType >
  inline static
  int team_size_max( const FunctorType & )
    { return traits::execution_space::thread_pool_size(1); }

  template< class FunctorType >
  inline static
  int team_size_recommended( const FunctorType & )
    { return traits::execution_space::thread_pool_size(2); }

  template< class FunctorType >
  inline static
  int team_size_recommended( const FunctorType &, const int& )
    { return traits::execution_space::thread_pool_size(2); }

  //----------------------------------------

private:

  int m_league_size ;
  int m_team_size ;
  int m_team_alloc ;
  int m_team_iter ;

  size_t m_team_scratch_size[2];
  size_t m_thread_scratch_size[2];

  int m_chunk_size;

  inline void init( const int league_size_request
                  , const int team_size_request )
    {
      const int pool_size  = traits::execution_space::thread_pool_size(0);
      const int team_max   = traits::execution_space::thread_pool_size(1);
      const int team_grain = traits::execution_space::thread_pool_size(2);

      m_league_size = league_size_request ;

      m_team_size = team_size_request < team_max ?
                    team_size_request : team_max ;

      // Round team size up to a multiple of 'team_gain'
      const int team_size_grain = team_grain * ( ( m_team_size + team_grain - 1 ) / team_grain );
      const int team_count      = pool_size / team_size_grain ;

      // Constraint : pool_size = m_team_alloc * team_count
      m_team_alloc = pool_size / team_count ;

      // Maxumum number of iterations each team will take:
      m_team_iter  = ( m_league_size + team_count - 1 ) / team_count ;

      set_auto_chunk_size();
    }

public:

  inline int team_size()   const { return m_team_size ; }
  inline int league_size() const { return m_league_size ; }

  inline size_t scratch_size(const int& level, int team_size_ = -1) const {
    if(team_size_ < 0) team_size_ = m_team_size;
    return m_team_scratch_size[level] + team_size_*m_thread_scratch_size[level] ;
  }

  /** \brief  Specify league size, request team size */
  TeamPolicyInternal( typename traits::execution_space &
            , int league_size_request
            , int team_size_request
            , int /* vector_length_request */ = 1 )
            : m_team_scratch_size { 0 , 0 }
            , m_thread_scratch_size { 0 , 0 }
            , m_chunk_size(0)
    { init( league_size_request , team_size_request ); }

  TeamPolicyInternal( typename traits::execution_space &
            , int league_size_request
            , const Kokkos::AUTO_t & /* team_size_request */
            , int /* vector_length_request */ = 1)
            : m_team_scratch_size { 0 , 0 }
            , m_thread_scratch_size { 0 , 0 }
            , m_chunk_size(0)
    { init( league_size_request , traits::execution_space::thread_pool_size(2) ); }

  TeamPolicyInternal( int league_size_request
            , int team_size_request
            , int /* vector_length_request */ = 1 )
            : m_team_scratch_size { 0 , 0 }
            , m_thread_scratch_size { 0 , 0 }
            , m_chunk_size(0)
    { init( league_size_request , team_size_request ); }

  TeamPolicyInternal( int league_size_request
            , const Kokkos::AUTO_t & /* team_size_request */
            , int /* vector_length_request */ = 1 )
            : m_team_scratch_size { 0 , 0 }
            , m_thread_scratch_size { 0 , 0 }
            , m_chunk_size(0)
    { init( league_size_request , traits::execution_space::thread_pool_size(2) ); }

  inline int team_alloc() const { return m_team_alloc ; }
  inline int team_iter()  const { return m_team_iter ; }

  inline int chunk_size() const { return m_chunk_size ; }

  /** \brief set chunk_size to a discrete value*/
  inline TeamPolicyInternal set_chunk_size(typename traits::index_type chunk_size_) const {
    TeamPolicyInternal p = *this;
    p.m_chunk_size = chunk_size_;
    return p;
  }

  inline TeamPolicyInternal set_scratch_size(const int& level, const PerTeamValue& per_team) const {
    TeamPolicyInternal p = *this;
    p.m_team_scratch_size[level] = per_team.value;
    return p;
  };

  inline TeamPolicyInternal set_scratch_size(const int& level, const PerThreadValue& per_thread) const {
    TeamPolicyInternal p = *this;
    p.m_thread_scratch_size[level] = per_thread.value;
    return p;
  };

  inline TeamPolicyInternal set_scratch_size(const int& level, const PerTeamValue& per_team, const PerThreadValue& per_thread) const {
    TeamPolicyInternal p = *this;
    p.m_team_scratch_size[level] = per_team.value;
    p.m_thread_scratch_size[level] = per_thread.value;
    return p;
  };

private:
  /** \brief finalize chunk_size if it was set to AUTO*/
  inline void set_auto_chunk_size() {

    int concurrency = traits::execution_space::thread_pool_size(0)/m_team_alloc;
    if( concurrency==0 ) concurrency=1;

    if(m_chunk_size > 0) {
      if(!Impl::is_integral_power_of_two( m_chunk_size ))
        Kokkos::abort("TeamPolicy blocking granularity must be power of two" );
    }

    int new_chunk_size = 1;
    while(new_chunk_size*100*concurrency < m_league_size)
      new_chunk_size *= 2;
    if(new_chunk_size < 128) {
      new_chunk_size = 1;
      while( (new_chunk_size*40*concurrency < m_league_size ) && (new_chunk_size<128) )
        new_chunk_size*=2;
    }
    m_chunk_size = new_chunk_size;
  }

public:
  typedef Impl::HostThreadTeamMember< Kokkos::OpenMP > member_type ;
};
} // namespace Impl

} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

inline
int OpenMP::thread_pool_size( int depth )
{
  return Impl::OpenMPexec::pool_size(depth);
}

KOKKOS_INLINE_FUNCTION
int OpenMP::thread_pool_rank()
{
#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
  return Impl::OpenMPexec::m_map_rank[ omp_get_thread_num() ];
#else
  return -1 ;
#endif
}

} // namespace Kokkos

#endif /* #ifndef KOKKOS_OPENMPEXEC_HPP */
