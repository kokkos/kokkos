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

#include <Kokkos_Macros.hpp>
#if defined( KOKKOS_ENABLE_OPENMP )

#if !defined(_OPENMP)
#error "You enabled Kokkos OpenMP support without enabling OpenMP in the compiler!"
#endif

#include <Kokkos_OpenMP.hpp>

#include <impl/Kokkos_Traits.hpp>
#include <impl/Kokkos_HostThreadTeam.hpp>

#include <Kokkos_Atomic.hpp>

#include <iostream>
#include <sstream>
#include <fstream>

#include <omp.h>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

class OpenMPExec;

extern int g_openmp_initialized;
extern int g_openmp_hardware_max_threads;

extern __thread int t_openmp_pool_rank;
extern __thread int t_openmp_pool_size;
extern __thread int t_openmp_hardware_id;
extern __thread OpenMPExec * t_openmp_instance;

//----------------------------------------------------------------------------
/** \brief  Data for OpenMP thread execution */

class OpenMPExec {
public:

  friend class Kokkos::OpenMP ;

  enum { MAX_THREAD_COUNT = 512 };

  void clear_thread_data();

  void validate_partition( int & num_partitions, int & partition_size );

private:
  OpenMPExec( int control_id )
    : m_control_id{ control_id }
    , m_in_parallel( false )
    , m_pool()
  {}

  ~OpenMPExec()
  {
    clear_thread_data();
  }

  int                  m_control_id ;
  int                  m_in_parallel;
  HostThreadTeamData * m_pool[ MAX_THREAD_COUNT ];

public:

  // Topology of a cache coherent thread pool:
  //   TOTAL = NUMA x GRAIN
  //   pool_size( depth = 0 )
  //   pool_size(0) = total number of threads
  //   pool_size(1) = number of threads per NUMA
  //   pool_size(2) = number of threads sharing finest grain memory hierarchy

  inline bool in_parallel() { return m_in_parallel; }
  inline void set_in_parallel() { m_in_parallel = true; }
  inline void unset_in_parallel() { m_in_parallel = false; }

  static void verify_is_process( const char * const );
  static void verify_initialized( const char * const );

  void resize_thread_data( size_t pool_reduce_bytes
                         , size_t team_reduce_bytes
                         , size_t team_shared_bytes
                         , size_t thread_local_bytes );

  inline
  HostThreadTeamData * get_thread_data() const noexcept
  { return m_pool[ t_openmp_pool_rank ]; }

  inline
  HostThreadTeamData * get_thread_data( int i ) const noexcept
  { return m_pool[i]; }
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
  int team_size_max( const FunctorType & ) {
      int pool_size = traits::execution_space::thread_pool_size(1);
      int max_host_team_size =  Impl::HostThreadTeamData::max_team_members;
      return pool_size<max_host_team_size?pool_size:max_host_team_size;
    }

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
      const int max_host_team_size =  Impl::HostThreadTeamData::max_team_members;
      const int team_max   = pool_size<max_host_team_size?pool_size:max_host_team_size;
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
bool OpenMP::is_initialized() { return Impl::t_openmp_instance != nullptr; }

inline
int OpenMP::thread_pool_size( int depth )
{
  return depth < 2 ? Impl::t_openmp_pool_size : 1;
}

KOKKOS_INLINE_FUNCTION
int OpenMP::thread_pool_rank()
{
#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
  return Impl::t_openmp_pool_rank;
#else
  return -1 ;
#endif
}

KOKKOS_INLINE_FUNCTION
int OpenMP::hardware_thread_id()
{
#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
  return Impl::t_openmp_hardware_id;
#else
  return -1 ;
#endif
}

inline
int OpenMP::max_hardware_threads()
{
  return Impl::g_openmp_hardware_max_threads;
}


template <typename F>
void OpenMP::partition( F const& f
                       , int num_partitions
                       , int partition_size
                       )
{
  using Exec = Impl::OpenMPExec;

  const int prev_pool_size = Impl::t_openmp_pool_size;
  Exec * prev_instance     = Impl::t_openmp_instance;

  prev_instance->validate_partition( num_partitions, partition_size );

  OpenMP::memory_space space;

#ifdef KOKKOS_ENABLE_PROC_BIND
  #pragma omp parallel num_threads(num_partitions) proc_bind(spread)
#else
  #pragma omp parallel num_threads(num_partitions)
#endif
  {
    void * const ptr = space.allocate( sizeof(Exec) );

    Exec * new_instance = new (ptr) Exec( Impl::t_openmp_hardware_id );

    #pragma omp parallel num_threads(partition_size) proc_bind(spread)
    {
      Impl::t_openmp_instance  = new_instance;
      Impl::t_openmp_pool_rank = omp_get_thread_num();
      Impl::t_openmp_pool_size = partition_size;
    }

    f( omp_get_thread_num(), omp_get_num_threads() );

    new_instance->~Exec();
    space.deallocate( new_instance, sizeof(Exec) );
  }

  // reset pool_rank and instance
#ifdef KOKKOS_ENABLE_PROC_BIND
  #pragma omp parallel num_threads( prev_pool_size ) proc_bind(spread)
#else
  #pragma omp parallel num_threads( prev_pool_size )
#endif
  {
    Impl::t_openmp_instance  = prev_instance;
    Impl::t_openmp_pool_rank = omp_get_thread_num();
    Impl::t_openmp_pool_size = prev_pool_size;
  }
}

inline
bool OpenMP::in_parallel()
{
  return Impl::t_openmp_instance->in_parallel();
}

class OpenMP::Lock
{
public:
  void lock()     { omp_set_lock( &m_lock );   }
  void unlock()   { omp_unset_lock( &m_lock ); }
  bool try_lock() { return static_cast<bool>(omp_test_lock( &m_lock )); }

  Lock()  { omp_init_lock( &m_lock ); }
  ~Lock() { omp_destroy_lock( &m_lock ); }

  Lock( Lock const& ) = delete;
  Lock( Lock && )     = delete;
  Lock & operator=( Lock const& ) = delete;
  Lock & operator=( Lock && )     = delete;

private:
  omp_lock_t m_lock;
};


} // namespace Kokkos

#endif
#endif /* #ifndef KOKKOS_OPENMPEXEC_HPP */

