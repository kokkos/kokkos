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

namespace Kokkos { namespace Impl {

class OpenMPExec;

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

  static void validate_partition( int & num_partitions, int & partition_size );

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

  inline bool in_parallel() const noexcept
  { return m_in_parallel; }

  inline void set_in_parallel()   { m_in_parallel = true; }
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

}} // namespace Kokkos::Impl

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

inline OpenMP::OpenMP() noexcept
{}

inline
bool OpenMP::is_initialized() noexcept
{ return Impl::t_openmp_instance != nullptr; }

inline
bool OpenMP::in_parallel( OpenMP const& ) noexcept
{
  return Impl::t_openmp_instance->in_parallel();
}

inline
void OpenMP::fence( OpenMP const& instance ) noexcept {}

inline
bool OpenMP::is_asynchronous( OpenMP const& instance ) noexcept
{ return false; }



template <typename F>
void OpenMP::partition_master( F const& f
                             , int num_partitions
                             , int partition_size
                             )
{
  using Exec = Impl::OpenMPExec;

  const int prev_pool_size = Impl::t_openmp_pool_size;
  Exec * prev_instance     = Impl::t_openmp_instance;

  Exec::validate_partition( num_partitions, partition_size );

  OpenMP::memory_space space;

#ifdef KOKKOS_ENABLE_PROC_BIND
  #pragma omp parallel num_threads(num_partitions) proc_bind(spread)
#else
  #pragma omp parallel num_threads(num_partitions)
#endif
  {
    void * const ptr = space.allocate( sizeof(Exec) );

    Exec * new_instance = new (ptr) Exec( Impl::t_openmp_hardware_id );

#ifdef KOKKOS_ENABLE_PROC_BIND
    #pragma omp parallel num_threads(partition_size) proc_bind(spread)
#else
    #pragma omp parallel num_threads(partition_size)
#endif
    {
      Impl::t_openmp_instance  = new_instance;
      Impl::t_openmp_pool_rank = omp_get_thread_num();
      Impl::t_openmp_pool_size = partition_size;
    }

    {
      memory_fence();

      size_t pool_reduce_bytes  =   32 * partition_size ;
      size_t team_reduce_bytes  =   32 * partition_size ;
      size_t team_shared_bytes  = 1024 * partition_size ;
      size_t thread_local_bytes = 1024 ;

      new_instance->resize_thread_data( pool_reduce_bytes
                                      , team_reduce_bytes
                                      , team_shared_bytes
                                      , thread_local_bytes
                                      );
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
  memory_fence();
}


inline
int OpenMP::thread_pool_size() noexcept
{
  return Impl::t_openmp_pool_size;
}

KOKKOS_INLINE_FUNCTION
int OpenMP::thread_pool_rank() noexcept
{
#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
  return Impl::t_openmp_pool_rank;
#else
  return -1 ;
#endif
}

namespace Experimental {

template<>
class MasterLock<OpenMP>
{
public:
  void lock()     { omp_set_lock( &m_lock );   }
  void unlock()   { omp_unset_lock( &m_lock ); }
  bool try_lock() { return static_cast<bool>(omp_test_lock( &m_lock )); }

  MasterLock()  { omp_init_lock( &m_lock ); }
  ~MasterLock() { omp_destroy_lock( &m_lock ); }

  MasterLock( MasterLock const& ) = delete;
  MasterLock( MasterLock && )     = delete;
  MasterLock & operator=( MasterLock const& ) = delete;
  MasterLock & operator=( MasterLock && )     = delete;

private:
  omp_lock_t m_lock;

};

template<>
class UniqueToken< OpenMP, UniqueTokenScope::Instance>
{
public:
  using execution_space = OpenMP;
  using size_type       = int;

  /// \brief create object size for concurrency on the given instance
  ///
  /// This object should not be shared between instances
  UniqueToken( execution_space const& = execution_space() ) noexcept {}

  /// \brief upper bound for acquired values, i.e. 0 <= value < size()
  inline
  int size() const noexcept { return Kokkos::Impl::t_openmp_pool_size; }

  /// \brief acquire value such that 0 <= value < size()
  inline
  int acquire() const  noexcept { return Kokkos::Impl::t_openmp_pool_rank; }

  /// \brief release a value acquired by generate
  inline
  void release( int ) const noexcept {}
};

template<>
class UniqueToken< OpenMP, UniqueTokenScope::Global>
{
public:
  using execution_space = OpenMP;
  using size_type       = int;

  /// \brief create object size for concurrency on the given instance
  ///
  /// This object should not be shared between instances
  UniqueToken( execution_space const& = execution_space() ) noexcept {}

  /// \brief upper bound for acquired values, i.e. 0 <= value < size()
  inline
  int size() const noexcept { return Kokkos::Impl::g_openmp_hardware_max_threads; }

  /// \brief acquire value such that 0 <= value < size()
  inline
  int acquire() const noexcept { return Kokkos::Impl::t_openmp_hardware_id; }

  /// \brief release a value acquired by generate
  inline
  void release( int ) const noexcept {}
};

} // namespace Experimental


#if !defined( KOKKOS_DISABLE_DEPRECATED )

inline
int OpenMP::thread_pool_size( int depth )
{
  return depth < 2 ? Impl::t_openmp_pool_size : 1;
}

KOKKOS_INLINE_FUNCTION
int OpenMP::hardware_thread_id() noexcept
{
#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
  return Impl::t_openmp_hardware_id;
#else
  return -1 ;
#endif
}

inline
int OpenMP::max_hardware_threads() noexcept
{
  return Impl::g_openmp_hardware_max_threads;
}

#endif // KOKKOS_DISABLE_DEPRECATED

} // namespace Kokkos

#endif
#endif /* #ifndef KOKKOS_OPENMPEXEC_HPP */

