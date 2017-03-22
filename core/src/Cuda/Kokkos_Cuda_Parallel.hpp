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

#ifndef KOKKOS_CUDA_PARALLEL_HPP
#define KOKKOS_CUDA_PARALLEL_HPP

#include <iostream>
#include <algorithm>
#include <stdio.h>

#include <Kokkos_Macros.hpp>

/* only compile this file if CUDA is enabled for Kokkos */
#if defined( __CUDACC__ ) && defined( KOKKOS_ENABLE_CUDA )

#include <utility>
#include <Kokkos_Parallel.hpp>

#include <Cuda/Kokkos_CudaExec.hpp>
#include <Cuda/Kokkos_Cuda_ReduceScan.hpp>
#include <Cuda/Kokkos_Cuda_Internal.hpp>
#include <Kokkos_Vectorization.hpp>

#if defined(KOKKOS_ENABLE_PROFILING)
#include <impl/Kokkos_Profiling_Interface.hpp>
#include <typeinfo>
#endif

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template< typename Type >
struct CudaJoinFunctor {
  typedef Type value_type ;

  KOKKOS_INLINE_FUNCTION
  static void join( volatile value_type & update ,
                    volatile const value_type & input )
    { update += input ; }
};

class CudaTeamMember {
private:

  typedef Kokkos::Cuda                           execution_space ;
  typedef execution_space::scratch_memory_space  scratch_memory_space ;

  void                * m_team_reduce ;
  scratch_memory_space  m_team_shared ;
  int                   m_league_rank ;
  int                   m_league_size ;

public:

  KOKKOS_INLINE_FUNCTION
  const execution_space::scratch_memory_space & team_shmem() const
    { return m_team_shared.set_team_thread_mode(0,1,0) ; }
  KOKKOS_INLINE_FUNCTION
  const execution_space::scratch_memory_space & team_scratch(const int& level) const
    { return m_team_shared.set_team_thread_mode(level,1,0) ; }
  KOKKOS_INLINE_FUNCTION
  const execution_space::scratch_memory_space & thread_scratch(const int& level) const
    { return m_team_shared.set_team_thread_mode(level,team_size(),team_rank()) ; }

  KOKKOS_INLINE_FUNCTION int league_rank() const { return m_league_rank ; }
  KOKKOS_INLINE_FUNCTION int league_size() const { return m_league_size ; }
  KOKKOS_INLINE_FUNCTION int team_rank() const {
    #ifdef __CUDA_ARCH__
    return threadIdx.y ;
    #else
    return 1;
    #endif
  }
  KOKKOS_INLINE_FUNCTION int team_size() const {
    #ifdef __CUDA_ARCH__
    return blockDim.y ;
    #else
    return 1;
    #endif
  }

  KOKKOS_INLINE_FUNCTION void team_barrier() const {
    #ifdef __CUDA_ARCH__
    __syncthreads();
    #endif
  }

  template<class ValueType>
  KOKKOS_INLINE_FUNCTION void team_broadcast(ValueType& value, const int& thread_id) const {
    #ifdef __CUDA_ARCH__
    __shared__ ValueType sh_val;
    if(threadIdx.x == 0 && threadIdx.y == thread_id) {
      sh_val = value;
    }
    team_barrier();
    value = sh_val;
    team_barrier();
    #endif
  }

  template< class ValueType, class JoinOp >
  KOKKOS_INLINE_FUNCTION
  typename JoinOp::value_type team_reduce( const ValueType & value
                                         , const JoinOp & op_in ) const {
      #ifdef __CUDA_ARCH__
      typedef JoinLambdaAdapter<ValueType,JoinOp> JoinOpFunctor ;
      const JoinOpFunctor op(op_in);
      ValueType * const base_data = (ValueType *) m_team_reduce ;

      __syncthreads(); // Don't write in to shared data until all threads have entered this function

      if ( 0 == threadIdx.y ) { base_data[0] = 0 ; }

      base_data[ threadIdx.y ] = value ;

      Impl::cuda_intra_block_reduce_scan<false,JoinOpFunctor,void>( op , base_data );

      return base_data[ blockDim.y - 1 ];
      #else
      return typename JoinOp::value_type();
      #endif
    }

  /** \brief  Intra-team exclusive prefix sum with team_rank() ordering
   *          with intra-team non-deterministic ordering accumulation.
   *
   *  The global inter-team accumulation value will, at the end of the
   *  league's parallel execution, be the scan's total.
   *  Parallel execution ordering of the league's teams is non-deterministic.
   *  As such the base value for each team's scan operation is similarly
   *  non-deterministic.
   */
  template< typename Type >
  KOKKOS_INLINE_FUNCTION Type team_scan( const Type & value , Type * const global_accum ) const {
      #ifdef __CUDA_ARCH__
      Type * const base_data = (Type *) m_team_reduce ;

      __syncthreads(); // Don't write in to shared data until all threads have entered this function

      if ( 0 == threadIdx.y ) { base_data[0] = 0 ; }

      base_data[ threadIdx.y + 1 ] = value ;

      Impl::cuda_intra_block_reduce_scan<true,Impl::CudaJoinFunctor<Type>,void>( Impl::CudaJoinFunctor<Type>() , base_data + 1 );

      if ( global_accum ) {
        if ( blockDim.y == threadIdx.y + 1 ) {
          base_data[ blockDim.y ] = atomic_fetch_add( global_accum , base_data[ blockDim.y ] );
        }
        __syncthreads(); // Wait for atomic
        base_data[ threadIdx.y ] += base_data[ blockDim.y ] ;
      }

      return base_data[ threadIdx.y ];
      #else
      return Type();
      #endif
    }

  /** \brief  Intra-team exclusive prefix sum with team_rank() ordering.
   *
   *  The highest rank thread can compute the reduction total as
   *    reduction_total = dev.team_scan( value ) + value ;
   */
  template< typename Type >
  KOKKOS_INLINE_FUNCTION Type team_scan( const Type & value ) const {
    return this->template team_scan<Type>( value , 0 );
  }

  //----------------------------------------
  // Private for the driver

  KOKKOS_INLINE_FUNCTION
  CudaTeamMember( void * shared
                , const int shared_begin
                , const int shared_size
                , void*     scratch_level_1_ptr
                , const int scratch_level_1_size
                , const int arg_league_rank
                , const int arg_league_size )
    : m_team_reduce( shared )
    , m_team_shared( ((char *)shared) + shared_begin , shared_size,  scratch_level_1_ptr, scratch_level_1_size)
    , m_league_rank( arg_league_rank )
    , m_league_size( arg_league_size )
    {}

};

} // namespace Impl

namespace Impl {
template< class ... Properties >
class TeamPolicyInternal< Kokkos::Cuda , Properties ... >: public PolicyTraits<Properties ... >
{
public:

  //! Tag this class as a kokkos execution policy
  typedef TeamPolicyInternal      execution_policy ;

  typedef PolicyTraits<Properties ... > traits;

private:

  enum { MAX_WARP = 8 };

  int m_league_size ;
  int m_team_size ;
  int m_vector_length ;
  int m_team_scratch_size[2] ;
  int m_thread_scratch_size[2] ;
  int m_chunk_size;

public:

  //! Execution space of this execution policy
  typedef Kokkos::Cuda  execution_space ;

  TeamPolicyInternal& operator = (const TeamPolicyInternal& p) {
    m_league_size = p.m_league_size;
    m_team_size = p.m_team_size;
    m_vector_length = p.m_vector_length;
    m_team_scratch_size[0] = p.m_team_scratch_size[0];
    m_team_scratch_size[1] = p.m_team_scratch_size[1];
    m_thread_scratch_size[0] = p.m_thread_scratch_size[0];
    m_thread_scratch_size[1] = p.m_thread_scratch_size[1];
    m_chunk_size = p.m_chunk_size;
    return *this;
  }

  //----------------------------------------

  template< class FunctorType >
  inline static
  int team_size_max( const FunctorType & functor )
    {
      int n = MAX_WARP * Impl::CudaTraits::WarpSize ;

      for ( ; n ; n >>= 1 ) {
        const int shmem_size =
          /* for global reduce */ Impl::cuda_single_inter_block_reduce_scan_shmem<false,FunctorType,typename traits::work_tag>( functor , n )
          /* for team   reduce */ + ( n + 2 ) * sizeof(double)
          /* for team   shared */ + Impl::FunctorTeamShmemSize< FunctorType >::value( functor , n );

        if ( shmem_size < Impl::CudaTraits::SharedMemoryCapacity ) break ;
      }

      return n ;
    }

  template< class FunctorType >
  static int team_size_recommended( const FunctorType & functor )
    { return team_size_max( functor ); }

  template< class FunctorType >
  static int team_size_recommended( const FunctorType & functor , const int vector_length)
    {
      int max = team_size_max( functor )/vector_length;
      if(max<1) max = 1;
      return max;
    }

  inline static
  int vector_length_max()
    { return Impl::CudaTraits::WarpSize; }

  //----------------------------------------

  inline int vector_length()   const { return m_vector_length ; }
  inline int team_size()   const { return m_team_size ; }
  inline int league_size() const { return m_league_size ; }
  inline int scratch_size(int level, int team_size_ = -1) const {
    if(team_size_<0) team_size_ = m_team_size;
    return m_team_scratch_size[level] + team_size_*m_thread_scratch_size[level];
  }
  inline size_t team_scratch_size(int level) const {
    return m_team_scratch_size[level];
  }
  inline size_t thread_scratch_size(int level) const {
    return m_thread_scratch_size[level];
  }

  TeamPolicyInternal()
    : m_league_size( 0 )
    , m_team_size( 0 )
    , m_vector_length( 0 )
    , m_team_scratch_size {0,0}
    , m_thread_scratch_size {0,0}
    , m_chunk_size ( 32 )
   {}

  /** \brief  Specify league size, request team size */
  TeamPolicyInternal( execution_space &
            , int league_size_
            , int team_size_request
            , int vector_length_request = 1 )
    : m_league_size( league_size_ )
    , m_team_size( team_size_request )
    , m_vector_length( vector_length_request )
    , m_team_scratch_size {0,0}
    , m_thread_scratch_size {0,0}
    , m_chunk_size ( 32 )
    {
      // Allow only power-of-two vector_length
      if ( ! Kokkos::Impl::is_integral_power_of_two( vector_length_request ) ) {
        Impl::throw_runtime_exception( "Requested non-power-of-two vector length for TeamPolicy.");
      }

      // Make sure league size is permissable
      if(league_size_ >= int(Impl::cuda_internal_maximum_grid_count()))
        Impl::throw_runtime_exception( "Requested too large league_size for TeamPolicy on Cuda execution space.");

      // Make sure total block size is permissable
      if ( m_team_size * m_vector_length > 1024 ) {
        Impl::throw_runtime_exception(std::string("Kokkos::TeamPolicy< Cuda > the team size is too large. Team size x vector length must be smaller than 1024."));
      }
    }

  /** \brief  Specify league size, request team size */
  TeamPolicyInternal( execution_space &
            , int league_size_
            , const Kokkos::AUTO_t & /* team_size_request */
            , int vector_length_request = 1 )
    : m_league_size( league_size_ )
    , m_team_size( -1 )
    , m_vector_length( vector_length_request )
    , m_team_scratch_size {0,0}
    , m_thread_scratch_size {0,0}
    , m_chunk_size ( 32 )
    {
      // Allow only power-of-two vector_length
      if ( ! Kokkos::Impl::is_integral_power_of_two( vector_length_request ) ) {
        Impl::throw_runtime_exception( "Requested non-power-of-two vector length for TeamPolicy.");
      }

      // Make sure league size is permissable
      if(league_size_ >= int(Impl::cuda_internal_maximum_grid_count()))
        Impl::throw_runtime_exception( "Requested too large league_size for TeamPolicy on Cuda execution space.");
    }

  TeamPolicyInternal( int league_size_
            , int team_size_request
            , int vector_length_request = 1 )
    : m_league_size( league_size_ )
    , m_team_size( team_size_request )
    , m_vector_length ( vector_length_request )
    , m_team_scratch_size {0,0}
    , m_thread_scratch_size {0,0}
    , m_chunk_size ( 32 )
    {
      // Allow only power-of-two vector_length
      if ( ! Kokkos::Impl::is_integral_power_of_two( vector_length_request ) ) {
        Impl::throw_runtime_exception( "Requested non-power-of-two vector length for TeamPolicy.");
      }

      // Make sure league size is permissable
      if(league_size_ >= int(Impl::cuda_internal_maximum_grid_count()))
        Impl::throw_runtime_exception( "Requested too large league_size for TeamPolicy on Cuda execution space.");

      // Make sure total block size is permissable
      if ( m_team_size * m_vector_length > 1024 ) {
        Impl::throw_runtime_exception(std::string("Kokkos::TeamPolicy< Cuda > the team size is too large. Team size x vector length must be smaller than 1024."));
      }
    }

  TeamPolicyInternal( int league_size_
            , const Kokkos::AUTO_t & /* team_size_request */
            , int vector_length_request = 1 )
    : m_league_size( league_size_ )
    , m_team_size( -1 )
    , m_vector_length ( vector_length_request )
    , m_team_scratch_size {0,0}
    , m_thread_scratch_size {0,0}
    , m_chunk_size ( 32 )
    {
      // Allow only power-of-two vector_length
      if ( ! Kokkos::Impl::is_integral_power_of_two( vector_length_request ) ) {
        Impl::throw_runtime_exception( "Requested non-power-of-two vector length for TeamPolicy.");
      }

      // Make sure league size is permissable
      if(league_size_ >= int(Impl::cuda_internal_maximum_grid_count()))
        Impl::throw_runtime_exception( "Requested too large league_size for TeamPolicy on Cuda execution space.");
    }

  inline int chunk_size() const { return m_chunk_size ; }

  /** \brief set chunk_size to a discrete value*/
  inline TeamPolicyInternal set_chunk_size(typename traits::index_type chunk_size_) const {
    TeamPolicyInternal p = *this;
    p.m_chunk_size = chunk_size_;
    return p;
  }

  /** \brief set per team scratch size for a specific level of the scratch hierarchy */
  inline TeamPolicyInternal set_scratch_size(const int& level, const PerTeamValue& per_team) const {
    TeamPolicyInternal p = *this;
    p.m_team_scratch_size[level] = per_team.value;
    return p;
  };

  /** \brief set per thread scratch size for a specific level of the scratch hierarchy */
  inline TeamPolicyInternal set_scratch_size(const int& level, const PerThreadValue& per_thread) const {
    TeamPolicyInternal p = *this;
    p.m_thread_scratch_size[level] = per_thread.value;
    return p;
  };

  /** \brief set per thread and per team scratch size for a specific level of the scratch hierarchy */
  inline TeamPolicyInternal set_scratch_size(const int& level, const PerTeamValue& per_team, const PerThreadValue& per_thread) const {
    TeamPolicyInternal p = *this;
    p.m_team_scratch_size[level] = per_team.value;
    p.m_thread_scratch_size[level] = per_thread.value;
    return p;
  };

  typedef Kokkos::Impl::CudaTeamMember member_type ;
};
} // namspace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template< class FunctorType , class ... Traits >
class ParallelFor< FunctorType
                 , Kokkos::RangePolicy< Traits ... >
                 , Kokkos::Cuda
                 >
{
private:

  typedef Kokkos::RangePolicy< Traits ... > Policy;
  typedef typename Policy::member_type  Member ;
  typedef typename Policy::work_tag     WorkTag ;

  const FunctorType  m_functor ;
  const Policy       m_policy ;

  ParallelFor() = delete ;
  ParallelFor & operator = ( const ParallelFor & ) = delete ;

  template< class TagType >
  inline __device__
  typename std::enable_if< std::is_same< TagType , void >::value >::type
  exec_range( const Member i ) const
    { m_functor( i ); }

  template< class TagType >
  inline __device__
  typename std::enable_if< ! std::is_same< TagType , void >::value >::type
  exec_range( const Member i ) const
    { m_functor( TagType() , i ); }

public:

  typedef FunctorType functor_type ;

  inline
  __device__
  void operator()(void) const
    {
      const Member work_stride = blockDim.y * gridDim.x ;
      const Member work_end    = m_policy.end();

      for ( Member
              iwork =  m_policy.begin() + threadIdx.y + blockDim.y * blockIdx.x ;
              iwork <  work_end ;
              iwork += work_stride ) {
        this-> template exec_range< WorkTag >( iwork );
      }
    }

  inline
  void execute() const
    {
      const int nwork = m_policy.end() - m_policy.begin();
      const dim3 block(  1 , CudaTraits::WarpSize * cuda_internal_maximum_warp_count(), 1);
      const dim3 grid( std::min( ( nwork + block.y - 1 ) / block.y , cuda_internal_maximum_grid_count() ) , 1 , 1);

      CudaParallelLaunch< ParallelFor >( *this , grid , block , 0 );
    }

  ParallelFor( const FunctorType  & arg_functor ,
               const Policy       & arg_policy )
    : m_functor( arg_functor )
    , m_policy(  arg_policy )
    { }
};

template< class FunctorType , class ... Properties >
class ParallelFor< FunctorType
                 , Kokkos::TeamPolicy< Properties ... >
                 , Kokkos::Cuda
                 >
{
private:

  typedef TeamPolicyInternal< Kokkos::Cuda , Properties ... >   Policy ;
  typedef typename Policy::member_type  Member ;
  typedef typename Policy::work_tag     WorkTag ;

public:

  typedef FunctorType      functor_type ;
  typedef Cuda::size_type  size_type ;

private:

  // Algorithmic constraints: blockDim.y is a power of two AND blockDim.y == blockDim.z == 1
  // shared memory utilization:
  //
  //  [ team   reduce space ]
  //  [ team   shared space ]
  //

  const FunctorType m_functor ;
  const size_type   m_league_size ;
  const size_type   m_team_size ;
  const size_type   m_vector_size ;
  const size_type   m_shmem_begin ;
  const size_type   m_shmem_size ;
  void*             m_scratch_ptr[2] ;
  const int         m_scratch_size[2] ;

  template< class TagType >
  __device__ inline
  typename std::enable_if< std::is_same< TagType , void >::value >::type
  exec_team( const Member & member ) const
    { m_functor( member ); }

  template< class TagType >
  __device__ inline
  typename std::enable_if< ! std::is_same< TagType , void >::value >::type
  exec_team( const Member & member ) const
    { m_functor( TagType() , member ); }

public:

  __device__ inline
  void operator()(void) const
  {
    // Iterate this block through the league
    int threadid = 0;
    if ( m_scratch_size[1]>0 ) {
      __shared__ int base_thread_id;
      if (threadIdx.x==0 && threadIdx.y==0 ) {
        threadid = ((blockIdx.x*blockDim.z + threadIdx.z) * blockDim.x * blockDim.y) % kokkos_impl_cuda_lock_arrays.n;
        threadid = ((threadid + blockDim.x * blockDim.y-1)/(blockDim.x * blockDim.y)) * blockDim.x * blockDim.y;
        if(threadid > kokkos_impl_cuda_lock_arrays.n) threadid-=blockDim.x * blockDim.y;
        int done = 0;
        while (!done) {
          done = (0 == atomicCAS(&kokkos_impl_cuda_lock_arrays.atomic[threadid],0,1));
          if(!done) {
            threadid += blockDim.x * blockDim.y;
            if(threadid > kokkos_impl_cuda_lock_arrays.n) threadid = 0;
          }
        }
        base_thread_id = threadid;
      }
      __syncthreads();
      threadid = base_thread_id;
    }


    for ( int league_rank = blockIdx.x ; league_rank < m_league_size ; league_rank += gridDim.x ) {

      this-> template exec_team< WorkTag >(
        typename Policy::member_type( kokkos_impl_cuda_shared_memory<void>()
                                    , m_shmem_begin
                                    , m_shmem_size
                                    , (void*) ( ((char*)m_scratch_ptr[1]) + threadid/(blockDim.x*blockDim.y) * m_scratch_size[1])
                                    , m_scratch_size[1]
                                    , league_rank
                                    , m_league_size ) );
    }
  }

  inline
  void execute() const
    {
      const int shmem_size_total = m_shmem_begin + m_shmem_size ;
      const dim3 grid( int(m_league_size) , 1 , 1 );
      const dim3 block( int(m_vector_size) , int(m_team_size) , 1 );

      CudaParallelLaunch< ParallelFor >( *this, grid, block, shmem_size_total ); // copy to device and execute

    }

  ParallelFor( const FunctorType  & arg_functor
             , const Policy       & arg_policy
             )
    : m_functor( arg_functor )
    , m_league_size( arg_policy.league_size() )
    , m_team_size( 0 <= arg_policy.team_size() ? arg_policy.team_size() :
        Kokkos::Impl::cuda_get_opt_block_size< ParallelFor >( arg_functor , arg_policy.vector_length(), arg_policy.team_scratch_size(0),arg_policy.thread_scratch_size(0) ) / arg_policy.vector_length() )
    , m_vector_size( arg_policy.vector_length() )
    , m_shmem_begin( sizeof(double) * ( m_team_size + 2 ) )
    , m_shmem_size( arg_policy.scratch_size(0,m_team_size) + FunctorTeamShmemSize< FunctorType >::value( m_functor , m_team_size ) )
    , m_scratch_ptr{NULL,NULL}
    , m_scratch_size{arg_policy.scratch_size(0,m_team_size),arg_policy.scratch_size(1,m_team_size)}
    {
      // Functor's reduce memory, team scan memory, and team shared memory depend upon team size.
      m_scratch_ptr[1] = cuda_resize_scratch_space(m_scratch_size[1]*(Cuda::concurrency()/(m_team_size*m_vector_size)));

      const int shmem_size_total = m_shmem_begin + m_shmem_size ;
      if ( CudaTraits::SharedMemoryCapacity < shmem_size_total ) {
        Kokkos::Impl::throw_runtime_exception(std::string("Kokkos::Impl::ParallelFor< Cuda > insufficient shared memory"));
      }

      if ( int(m_team_size) >
           int(Kokkos::Impl::cuda_get_max_block_size< ParallelFor >
                 ( arg_functor , arg_policy.vector_length(), arg_policy.team_scratch_size(0),arg_policy.thread_scratch_size(0) ) / arg_policy.vector_length())) {
        Kokkos::Impl::throw_runtime_exception(std::string("Kokkos::Impl::ParallelFor< Cuda > requested too large team size."));
      }
    }
};

} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template< class FunctorType , class ReducerType, class ... Traits >
class ParallelReduce< FunctorType
                    , Kokkos::RangePolicy< Traits ... >
                    , ReducerType
                    , Kokkos::Cuda
                    >
{
private:

  typedef Kokkos::RangePolicy< Traits ... >         Policy ;

  typedef typename Policy::WorkRange    WorkRange ;
  typedef typename Policy::work_tag     WorkTag ;
  typedef typename Policy::member_type  Member ;

  typedef Kokkos::Impl::if_c< std::is_same<InvalidType,ReducerType>::value, FunctorType, ReducerType> ReducerConditional;
  typedef typename ReducerConditional::type ReducerTypeFwd;

  typedef Kokkos::Impl::FunctorValueTraits< ReducerTypeFwd, WorkTag > ValueTraits ;
  typedef Kokkos::Impl::FunctorValueInit<   ReducerTypeFwd, WorkTag > ValueInit ;
  typedef Kokkos::Impl::FunctorValueJoin<   ReducerTypeFwd, WorkTag > ValueJoin ;

public:

  typedef typename ValueTraits::pointer_type    pointer_type ;
  typedef typename ValueTraits::value_type      value_type ;
  typedef typename ValueTraits::reference_type  reference_type ;
  typedef FunctorType                           functor_type ;
  typedef Cuda::size_type                       size_type ;

  // Algorithmic constraints: blockSize is a power of two AND blockDim.y == blockDim.z == 1

  const FunctorType   m_functor ;
  const Policy        m_policy ;
  const ReducerType   m_reducer ;
  const pointer_type  m_result_ptr ;
  size_type *         m_scratch_space ;
  size_type *         m_scratch_flags ;
  size_type *         m_unified_space ;

  // Shall we use the shfl based reduction or not (only use it for static sized types of more than 128bit
  enum { UseShflReduction = ((sizeof(value_type)>2*sizeof(double)) && ValueTraits::StaticValueSize) };
  // Some crutch to do function overloading
private:
  typedef double DummyShflReductionType;
  typedef int DummySHMEMReductionType;

public:
  template< class TagType >
  __device__ inline
  typename std::enable_if< std::is_same< TagType , void >::value >::type
  exec_range( const Member & i , reference_type update ) const
    { m_functor( i , update ); }

  template< class TagType >
  __device__ inline
  typename std::enable_if< ! std::is_same< TagType , void >::value >::type
  exec_range( const Member & i , reference_type update ) const
    { m_functor( TagType() , i , update ); }

  __device__ inline
  void operator() () const {
    run(Kokkos::Impl::if_c<UseShflReduction, DummyShflReductionType, DummySHMEMReductionType>::select(1,1.0) );
  }

  __device__ inline
  void run(const DummySHMEMReductionType& ) const
  {
    const integral_nonzero_constant< size_type , ValueTraits::StaticValueSize / sizeof(size_type) >
      word_count( ValueTraits::value_size( ReducerConditional::select(m_functor , m_reducer) ) / sizeof(size_type) );

    {
      reference_type value =
        ValueInit::init( ReducerConditional::select(m_functor , m_reducer) , kokkos_impl_cuda_shared_memory<size_type>() + threadIdx.y * word_count.value );

      // Number of blocks is bounded so that the reduction can be limited to two passes.
      // Each thread block is given an approximately equal amount of work to perform.
      // Accumulate the values for this block.
      // The accumulation ordering does not match the final pass, but is arithmatically equivalent.

      const WorkRange range( m_policy , blockIdx.x , gridDim.x );

      for ( Member iwork = range.begin() + threadIdx.y , iwork_end = range.end() ;
            iwork < iwork_end ; iwork += blockDim.y ) {
        this-> template exec_range< WorkTag >( iwork , value );
      }
    }

    // Reduce with final value at blockDim.y - 1 location.
    if ( cuda_single_inter_block_reduce_scan<false,ReducerTypeFwd,WorkTag>(
           ReducerConditional::select(m_functor , m_reducer) , blockIdx.x , gridDim.x ,
           kokkos_impl_cuda_shared_memory<size_type>() , m_scratch_space , m_scratch_flags ) ) {

      // This is the final block with the final result at the final threads' location

      size_type * const shared = kokkos_impl_cuda_shared_memory<size_type>() + ( blockDim.y - 1 ) * word_count.value ;
      size_type * const global = m_unified_space ? m_unified_space : m_scratch_space ;

      if ( threadIdx.y == 0 ) {
        Kokkos::Impl::FunctorFinal< ReducerTypeFwd , WorkTag >::final( ReducerConditional::select(m_functor , m_reducer) , shared );
      }

      if ( CudaTraits::WarpSize < word_count.value ) { __syncthreads(); }

      for ( unsigned i = threadIdx.y ; i < word_count.value ; i += blockDim.y ) { global[i] = shared[i]; }
    }
  }

  __device__ inline
   void run(const DummyShflReductionType&) const
   {

     value_type value;
     ValueInit::init( ReducerConditional::select(m_functor , m_reducer) , &value);
     // Number of blocks is bounded so that the reduction can be limited to two passes.
     // Each thread block is given an approximately equal amount of work to perform.
     // Accumulate the values for this block.
     // The accumulation ordering does not match the final pass, but is arithmatically equivalent.

     const WorkRange range( m_policy , blockIdx.x , gridDim.x );

     for ( Member iwork = range.begin() + threadIdx.y , iwork_end = range.end() ;
           iwork < iwork_end ; iwork += blockDim.y ) {
       this-> template exec_range< WorkTag >( iwork , value );
     }

     pointer_type const result = (pointer_type) (m_unified_space ? m_unified_space : m_scratch_space) ;

     int max_active_thread = range.end()-range.begin() < blockDim.y ? range.end() - range.begin():blockDim.y;

     max_active_thread = (max_active_thread == 0)?blockDim.y:max_active_thread;

    value_type init;
    ValueInit::init( ReducerConditional::select(m_functor , m_reducer) , &init);
     if(Impl::cuda_inter_block_reduction<ReducerTypeFwd,ValueJoin,WorkTag>
            (value,init,ValueJoin(ReducerConditional::select(m_functor , m_reducer)),m_scratch_space,result,m_scratch_flags,max_active_thread)) {
       const unsigned id = threadIdx.y*blockDim.x + threadIdx.x;
       if(id==0) {
         Kokkos::Impl::FunctorFinal< ReducerTypeFwd , WorkTag >::final( ReducerConditional::select(m_functor , m_reducer) , (void*) &value );
         *result = value;
       }
     }
   }

  // Determine block size constrained by shared memory:
  static inline
  unsigned local_block_size( const FunctorType & f )
    {
      unsigned n = CudaTraits::WarpSize * 8 ;
      while ( n && CudaTraits::SharedMemoryCapacity < cuda_single_inter_block_reduce_scan_shmem<false,FunctorType,WorkTag>( f , n ) ) { n >>= 1 ; }
      return n ;
    }

  inline
  void execute()
    {
      const int nwork = m_policy.end() - m_policy.begin();
      if ( nwork ) {
        const int block_size = local_block_size( m_functor );

        m_scratch_space = cuda_internal_scratch_space( ValueTraits::value_size( ReducerConditional::select(m_functor , m_reducer) ) * block_size /* block_size == max block_count */ );
        m_scratch_flags = cuda_internal_scratch_flags( sizeof(size_type) );
        m_unified_space = cuda_internal_scratch_unified( ValueTraits::value_size( ReducerConditional::select(m_functor , m_reducer) ) );

        // REQUIRED ( 1 , N , 1 )
        const dim3 block( 1 , block_size , 1 );
        // Required grid.x <= block.y
        const dim3 grid( std::min( int(block.y) , int( ( nwork + block.y - 1 ) / block.y ) ) , 1 , 1 );

      const int shmem = UseShflReduction?0:cuda_single_inter_block_reduce_scan_shmem<false,FunctorType,WorkTag>( m_functor , block.y );

      CudaParallelLaunch< ParallelReduce >( *this, grid, block, shmem ); // copy to device and execute

      Cuda::fence();

      if ( m_result_ptr ) {
        if ( m_unified_space ) {
          const int count = ValueTraits::value_count( ReducerConditional::select(m_functor , m_reducer)  );
          for ( int i = 0 ; i < count ; ++i ) { m_result_ptr[i] = pointer_type(m_unified_space)[i] ; }
        }
        else {
          const int size = ValueTraits::value_size( ReducerConditional::select(m_functor , m_reducer)  );
          DeepCopy<HostSpace,CudaSpace>( m_result_ptr , m_scratch_space , size );
        }
      }
    }
    else {
      if (m_result_ptr) {
        ValueInit::init( ReducerConditional::select(m_functor , m_reducer) , m_result_ptr );
      }
    }
  }

  template< class HostViewType >
  ParallelReduce( const FunctorType  & arg_functor
                , const Policy       & arg_policy
                , const HostViewType & arg_result
                , typename std::enable_if<
                   Kokkos::is_view< HostViewType >::value
                ,void*>::type = NULL)
  : m_functor( arg_functor )
  , m_policy(  arg_policy )
  , m_reducer( InvalidType() )
  , m_result_ptr( arg_result.ptr_on_device() )
  , m_scratch_space( 0 )
  , m_scratch_flags( 0 )
  , m_unified_space( 0 )
  { }

  ParallelReduce( const FunctorType  & arg_functor
                , const Policy       & arg_policy
                , const ReducerType & reducer)
  : m_functor( arg_functor )
  , m_policy(  arg_policy )
  , m_reducer( reducer )
  , m_result_ptr( reducer.result_view().ptr_on_device() )
  , m_scratch_space( 0 )
  , m_scratch_flags( 0 )
  , m_unified_space( 0 )
  { }
};

//----------------------------------------------------------------------------

template< class FunctorType , class ReducerType, class ... Properties >
class ParallelReduce< FunctorType
                    , Kokkos::TeamPolicy< Properties ... >
                    , ReducerType
                    , Kokkos::Cuda
                    >
{
private:

  typedef TeamPolicyInternal< Kokkos::Cuda, Properties ... >  Policy ;
  typedef typename Policy::member_type  Member ;
  typedef typename Policy::work_tag     WorkTag ;

  typedef Kokkos::Impl::if_c< std::is_same<InvalidType,ReducerType>::value, FunctorType, ReducerType> ReducerConditional;
  typedef typename ReducerConditional::type ReducerTypeFwd;

  typedef Kokkos::Impl::FunctorValueTraits< ReducerTypeFwd, WorkTag > ValueTraits ;
  typedef Kokkos::Impl::FunctorValueInit<   ReducerTypeFwd, WorkTag > ValueInit ;
  typedef Kokkos::Impl::FunctorValueJoin<   ReducerTypeFwd, WorkTag > ValueJoin ;

  typedef typename ValueTraits::pointer_type    pointer_type ;
  typedef typename ValueTraits::reference_type  reference_type ;
  typedef typename ValueTraits::value_type      value_type ;

public:

  typedef FunctorType      functor_type ;
  typedef Cuda::size_type  size_type ;

  enum { UseShflReduction = (true && ValueTraits::StaticValueSize) };

private:
  typedef double DummyShflReductionType;
  typedef int DummySHMEMReductionType;

  // Algorithmic constraints: blockDim.y is a power of two AND blockDim.y == blockDim.z == 1
  // shared memory utilization:
  //
  //  [ global reduce space ]
  //  [ team   reduce space ]
  //  [ team   shared space ]
  //

  const FunctorType   m_functor ;
  const ReducerType   m_reducer ;
  const pointer_type  m_result_ptr ;
  size_type *         m_scratch_space ;
  size_type *         m_scratch_flags ;
  size_type *         m_unified_space ;
  size_type           m_team_begin ;
  size_type           m_shmem_begin ;
  size_type           m_shmem_size ;
  void*               m_scratch_ptr[2] ;
  int                 m_scratch_size[2] ;
  const size_type     m_league_size ;
  const size_type     m_team_size ;
  const size_type     m_vector_size ;

  template< class TagType >
  __device__ inline
  typename std::enable_if< std::is_same< TagType , void >::value >::type
  exec_team( const Member & member , reference_type update ) const
    { m_functor( member , update ); }

  template< class TagType >
  __device__ inline
  typename std::enable_if< ! std::is_same< TagType , void >::value >::type
  exec_team( const Member & member , reference_type update ) const
    { m_functor( TagType() , member , update ); }

public:

  __device__ inline
  void operator() () const {
    int threadid = 0;
    if ( m_scratch_size[1]>0 ) {
      __shared__ int base_thread_id;
      if (threadIdx.x==0 && threadIdx.y==0 ) {
        threadid = ((blockIdx.x*blockDim.z + threadIdx.z) * blockDim.x * blockDim.y) % kokkos_impl_cuda_lock_arrays.n;
        threadid = ((threadid + blockDim.x * blockDim.y-1)/(blockDim.x * blockDim.y)) * blockDim.x * blockDim.y;
        if(threadid > kokkos_impl_cuda_lock_arrays.n) threadid-=blockDim.x * blockDim.y;
        int done = 0;
        while (!done) {
          done = (0 == atomicCAS(&kokkos_impl_cuda_lock_arrays.atomic[threadid],0,1));
          if(!done) {
            threadid += blockDim.x * blockDim.y;
            if(threadid > kokkos_impl_cuda_lock_arrays.n) threadid = 0;
          }
        }
        base_thread_id = threadid;
      }
      __syncthreads();
      threadid = base_thread_id;
    }

    run(Kokkos::Impl::if_c<UseShflReduction, DummyShflReductionType, DummySHMEMReductionType>::select(1,1.0), threadid );
  }

  __device__ inline
  void run(const DummySHMEMReductionType&, const int& threadid) const
  {
    const integral_nonzero_constant< size_type , ValueTraits::StaticValueSize / sizeof(size_type) >
      word_count( ValueTraits::value_size( ReducerConditional::select(m_functor , m_reducer) ) / sizeof(size_type) );

    reference_type value =
      ValueInit::init( ReducerConditional::select(m_functor , m_reducer) , kokkos_impl_cuda_shared_memory<size_type>() + threadIdx.y * word_count.value );

    // Iterate this block through the league
    for ( int league_rank = blockIdx.x ; league_rank < m_league_size ; league_rank += gridDim.x ) {
      this-> template exec_team< WorkTag >
        ( Member( kokkos_impl_cuda_shared_memory<char>() + m_team_begin
                                        , m_shmem_begin
                                        , m_shmem_size
                                        , (void*) ( ((char*)m_scratch_ptr[1]) + threadid/(blockDim.x*blockDim.y) * m_scratch_size[1])
                                        , m_scratch_size[1]
                                        , league_rank
                                        , m_league_size )
        , value );
    }

    // Reduce with final value at blockDim.y - 1 location.
    if ( cuda_single_inter_block_reduce_scan<false,FunctorType,WorkTag>(
           ReducerConditional::select(m_functor , m_reducer) , blockIdx.x , gridDim.x ,
           kokkos_impl_cuda_shared_memory<size_type>() , m_scratch_space , m_scratch_flags ) ) {

      // This is the final block with the final result at the final threads' location

      size_type * const shared = kokkos_impl_cuda_shared_memory<size_type>() + ( blockDim.y - 1 ) * word_count.value ;
      size_type * const global = m_unified_space ? m_unified_space : m_scratch_space ;

      if ( threadIdx.y == 0 ) {
        Kokkos::Impl::FunctorFinal< ReducerTypeFwd , WorkTag >::final( ReducerConditional::select(m_functor , m_reducer) , shared );
      }

      if ( CudaTraits::WarpSize < word_count.value ) { __syncthreads(); }

      for ( unsigned i = threadIdx.y ; i < word_count.value ; i += blockDim.y ) { global[i] = shared[i]; }
    }
  }

  __device__ inline
  void run(const DummyShflReductionType&, const int& threadid) const
  {
    value_type value;
    ValueInit::init( ReducerConditional::select(m_functor , m_reducer) , &value);

    // Iterate this block through the league
    for ( int league_rank = blockIdx.x ; league_rank < m_league_size ; league_rank += gridDim.x ) {
      this-> template exec_team< WorkTag >
        ( Member( kokkos_impl_cuda_shared_memory<char>() + m_team_begin
                                        , m_shmem_begin
                                        , m_shmem_size
                                        , (void*) ( ((char*)m_scratch_ptr[1]) + threadid/(blockDim.x*blockDim.y) * m_scratch_size[1])
                                        , m_scratch_size[1]
                                        , league_rank
                                        , m_league_size )
        , value );
    }

    pointer_type const result = (pointer_type) (m_unified_space ? m_unified_space : m_scratch_space) ;

    value_type init;
    ValueInit::init( ReducerConditional::select(m_functor , m_reducer) , &init);
    if(Impl::cuda_inter_block_reduction<FunctorType,ValueJoin,WorkTag>
           (value,init,ValueJoin(ReducerConditional::select(m_functor , m_reducer)),m_scratch_space,result,m_scratch_flags,blockDim.y)) {
      const unsigned id = threadIdx.y*blockDim.x + threadIdx.x;
      if(id==0) {
        Kokkos::Impl::FunctorFinal< ReducerTypeFwd , WorkTag >::final( ReducerConditional::select(m_functor , m_reducer) , (void*) &value );
        *result = value;
      }
    }
  }

  inline
  void execute()
    {
      const int nwork = m_league_size * m_team_size ;
      if ( nwork ) {
        const int block_count = UseShflReduction? std::min( m_league_size , size_type(1024) )
          :std::min( m_league_size , m_team_size );

        m_scratch_space = cuda_internal_scratch_space( ValueTraits::value_size( ReducerConditional::select(m_functor , m_reducer) ) * block_count );
        m_scratch_flags = cuda_internal_scratch_flags( sizeof(size_type) );
        m_unified_space = cuda_internal_scratch_unified( ValueTraits::value_size( ReducerConditional::select(m_functor , m_reducer) ) );

        const dim3 block( m_vector_size , m_team_size , 1 );
        const dim3 grid( block_count , 1 , 1 );
        const int shmem_size_total = m_team_begin + m_shmem_begin + m_shmem_size ;

        CudaParallelLaunch< ParallelReduce >( *this, grid, block, shmem_size_total ); // copy to device and execute

        Cuda::fence();

        if ( m_result_ptr ) {
          if ( m_unified_space ) {
            const int count = ValueTraits::value_count( ReducerConditional::select(m_functor , m_reducer) );
            for ( int i = 0 ; i < count ; ++i ) { m_result_ptr[i] = pointer_type(m_unified_space)[i] ; }
          }
          else {
            const int size = ValueTraits::value_size( ReducerConditional::select(m_functor , m_reducer) );
            DeepCopy<HostSpace,CudaSpace>( m_result_ptr, m_scratch_space, size );
          }
        }
      }
      else {
        if (m_result_ptr) {
          ValueInit::init( ReducerConditional::select(m_functor , m_reducer) , m_result_ptr );
        }
      }
    }

  template< class HostViewType >
  ParallelReduce( const FunctorType  & arg_functor
                , const Policy       & arg_policy
                , const HostViewType & arg_result
                , typename std::enable_if<
                                   Kokkos::is_view< HostViewType >::value
                                ,void*>::type = NULL)
  : m_functor( arg_functor )
  , m_reducer( InvalidType() )
  , m_result_ptr( arg_result.ptr_on_device() )
  , m_scratch_space( 0 )
  , m_scratch_flags( 0 )
  , m_unified_space( 0 )
  , m_team_begin( 0 )
  , m_shmem_begin( 0 )
  , m_shmem_size( 0 )
  , m_scratch_ptr{NULL,NULL}
  , m_league_size( arg_policy.league_size() )
  , m_team_size( 0 <= arg_policy.team_size() ? arg_policy.team_size() :
      Kokkos::Impl::cuda_get_opt_block_size< ParallelReduce >( arg_functor , arg_policy.vector_length(),
                                                               arg_policy.team_scratch_size(0),arg_policy.thread_scratch_size(0) ) /
                                                               arg_policy.vector_length() )
  , m_vector_size( arg_policy.vector_length() )
  , m_scratch_size{
    arg_policy.scratch_size(0,( 0 <= arg_policy.team_size() ? arg_policy.team_size() :
        Kokkos::Impl::cuda_get_opt_block_size< ParallelReduce >( arg_functor , arg_policy.vector_length(),
                                                                 arg_policy.team_scratch_size(0),arg_policy.thread_scratch_size(0) ) /
                                                                 arg_policy.vector_length() )
    ), arg_policy.scratch_size(1,( 0 <= arg_policy.team_size() ? arg_policy.team_size() :
        Kokkos::Impl::cuda_get_opt_block_size< ParallelReduce >( arg_functor , arg_policy.vector_length(),
                                                                 arg_policy.team_scratch_size(0),arg_policy.thread_scratch_size(0) ) /
                                                                 arg_policy.vector_length() )
        )}
  {
    // Return Init value if the number of worksets is zero
    if( arg_policy.league_size() == 0) {
      ValueInit::init( ReducerConditional::select(m_functor , m_reducer) , arg_result.ptr_on_device() );
      return ;
    }

    m_team_begin = UseShflReduction?0:cuda_single_inter_block_reduce_scan_shmem<false,FunctorType,WorkTag>( arg_functor , m_team_size );
    m_shmem_begin = sizeof(double) * ( m_team_size + 2 );
    m_shmem_size = arg_policy.scratch_size(0,m_team_size) + FunctorTeamShmemSize< FunctorType >::value( arg_functor , m_team_size );
    m_scratch_ptr[1] = cuda_resize_scratch_space(m_scratch_size[1]*(Cuda::concurrency()/(m_team_size*m_vector_size)));
    m_scratch_size[0] = m_shmem_size;
    m_scratch_size[1] = arg_policy.scratch_size(1,m_team_size);

    // The global parallel_reduce does not support vector_length other than 1 at the moment
    if( (arg_policy.vector_length() > 1) && !UseShflReduction )
      Impl::throw_runtime_exception( "Kokkos::parallel_reduce with a TeamPolicy using a vector length of greater than 1 is not currently supported for CUDA for dynamic sized reduction types.");

    if( (m_team_size < 32) && !UseShflReduction )
      Impl::throw_runtime_exception( "Kokkos::parallel_reduce with a TeamPolicy using a team_size smaller than 32 is not currently supported with CUDA for dynamic sized reduction types.");

    // Functor's reduce memory, team scan memory, and team shared memory depend upon team size.

    const int shmem_size_total = m_team_begin + m_shmem_begin + m_shmem_size ;

    if (! Kokkos::Impl::is_integral_power_of_two( m_team_size )  && !UseShflReduction ) {
      Kokkos::Impl::throw_runtime_exception(std::string("Kokkos::Impl::ParallelReduce< Cuda > bad team size"));
    }

    if ( CudaTraits::SharedMemoryCapacity < shmem_size_total ) {
      Kokkos::Impl::throw_runtime_exception(std::string("Kokkos::Impl::ParallelReduce< Cuda > requested too much L0 scratch memory"));
    }

    if ( unsigned(m_team_size) >
         unsigned(Kokkos::Impl::cuda_get_max_block_size< ParallelReduce >
               ( arg_functor , arg_policy.vector_length(), arg_policy.team_scratch_size(0),arg_policy.thread_scratch_size(0) ) / arg_policy.vector_length())) {
      Kokkos::Impl::throw_runtime_exception(std::string("Kokkos::Impl::ParallelReduce< Cuda > requested too large team size."));
    }

  }

  ParallelReduce( const FunctorType  & arg_functor
                , const Policy       & arg_policy
                , const ReducerType & reducer)
  : m_functor( arg_functor )
  , m_reducer( reducer )
  , m_result_ptr( reducer.result_view().ptr_on_device() )
  , m_scratch_space( 0 )
  , m_scratch_flags( 0 )
  , m_unified_space( 0 )
  , m_team_begin( 0 )
  , m_shmem_begin( 0 )
  , m_shmem_size( 0 )
  , m_scratch_ptr{NULL,NULL}
  , m_league_size( arg_policy.league_size() )
  , m_team_size( 0 <= arg_policy.team_size() ? arg_policy.team_size() :
      Kokkos::Impl::cuda_get_opt_block_size< ParallelReduce >( arg_functor , arg_policy.vector_length(),
                                                               arg_policy.team_scratch_size(0),arg_policy.thread_scratch_size(0) ) /
      arg_policy.vector_length() )
  , m_vector_size( arg_policy.vector_length() )
  {
    // Return Init value if the number of worksets is zero
    if( arg_policy.league_size() == 0) {
      ValueInit::init( ReducerConditional::select(m_functor , m_reducer) , m_result_ptr );
      return ;
    }

    m_team_begin = UseShflReduction?0:cuda_single_inter_block_reduce_scan_shmem<false,FunctorType,WorkTag>( arg_functor , m_team_size );
    m_shmem_begin = sizeof(double) * ( m_team_size + 2 );
    m_shmem_size = arg_policy.scratch_size(0,m_team_size) + FunctorTeamShmemSize< FunctorType >::value( arg_functor , m_team_size );
    m_scratch_ptr[1] = cuda_resize_scratch_space(m_scratch_size[1]*(Cuda::concurrency()/(m_team_size*m_vector_size)));
    m_scratch_size[0] = m_shmem_size;
    m_scratch_size[1] = arg_policy.scratch_size(1,m_team_size);

    // The global parallel_reduce does not support vector_length other than 1 at the moment
    if( (arg_policy.vector_length() > 1) && !UseShflReduction )
      Impl::throw_runtime_exception( "Kokkos::parallel_reduce with a TeamPolicy using a vector length of greater than 1 is not currently supported for CUDA for dynamic sized reduction types.");

    if( (m_team_size < 32) && !UseShflReduction )
      Impl::throw_runtime_exception( "Kokkos::parallel_reduce with a TeamPolicy using a team_size smaller than 32 is not currently supported with CUDA for dynamic sized reduction types.");

    // Functor's reduce memory, team scan memory, and team shared memory depend upon team size.

    const int shmem_size_total = m_team_begin + m_shmem_begin + m_shmem_size ;

    if ( (! Kokkos::Impl::is_integral_power_of_two( m_team_size )  && !UseShflReduction ) ||
         CudaTraits::SharedMemoryCapacity < shmem_size_total ) {
      Kokkos::Impl::throw_runtime_exception(std::string("Kokkos::Impl::ParallelReduce< Cuda > bad team size"));
    }

    if ( int(m_team_size) >
         int(Kokkos::Impl::cuda_get_max_block_size< ParallelReduce >
               ( arg_functor , arg_policy.vector_length(), arg_policy.team_scratch_size(0),arg_policy.thread_scratch_size(0) ) / arg_policy.vector_length())) {
      Kokkos::Impl::throw_runtime_exception(std::string("Kokkos::Impl::ParallelReduce< Cuda > requested too large team size."));
    }

  }
};

} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template< class FunctorType , class ... Traits >
class ParallelScan< FunctorType
                  , Kokkos::RangePolicy< Traits ... >
                  , Kokkos::Cuda
                  >
{
private:

  typedef Kokkos::RangePolicy< Traits ... >  Policy ;
  typedef typename Policy::member_type  Member ;
  typedef typename Policy::work_tag     WorkTag ;
  typedef typename Policy::WorkRange    WorkRange ;

  typedef Kokkos::Impl::FunctorValueTraits< FunctorType, WorkTag > ValueTraits ;
  typedef Kokkos::Impl::FunctorValueInit<   FunctorType, WorkTag > ValueInit ;
  typedef Kokkos::Impl::FunctorValueOps<    FunctorType, WorkTag > ValueOps ;

public:

  typedef typename ValueTraits::pointer_type    pointer_type ;
  typedef typename ValueTraits::reference_type  reference_type ;
  typedef FunctorType                           functor_type ;
  typedef Cuda::size_type                       size_type ;

private:

  // Algorithmic constraints:
  //  (a) blockDim.y is a power of two
  //  (b) blockDim.y == blockDim.z == 1
  //  (c) gridDim.x  <= blockDim.y * blockDim.y
  //  (d) gridDim.y  == gridDim.z == 1

  const FunctorType m_functor ;
  const Policy      m_policy ;
  size_type *       m_scratch_space ;
  size_type *       m_scratch_flags ;
  size_type         m_final ;

  template< class TagType >
  __device__ inline
  typename std::enable_if< std::is_same< TagType , void >::value >::type
  exec_range( const Member & i , reference_type update , const bool final_result ) const
    { m_functor( i , update , final_result ); }

  template< class TagType >
  __device__ inline
  typename std::enable_if< ! std::is_same< TagType , void >::value >::type
  exec_range( const Member & i , reference_type update , const bool final_result ) const
    { m_functor( TagType() , i , update , final_result ); }

  //----------------------------------------

  __device__ inline
  void initial(void) const
  {
    const integral_nonzero_constant< size_type , ValueTraits::StaticValueSize / sizeof(size_type) >
      word_count( ValueTraits::value_size( m_functor ) / sizeof(size_type) );

    size_type * const shared_value = kokkos_impl_cuda_shared_memory<size_type>() + word_count.value * threadIdx.y ;

    ValueInit::init( m_functor , shared_value );

    // Number of blocks is bounded so that the reduction can be limited to two passes.
    // Each thread block is given an approximately equal amount of work to perform.
    // Accumulate the values for this block.
    // The accumulation ordering does not match the final pass, but is arithmatically equivalent.

    const WorkRange range( m_policy , blockIdx.x , gridDim.x );

    for ( Member iwork = range.begin() + threadIdx.y , iwork_end = range.end() ;
          iwork < iwork_end ; iwork += blockDim.y ) {
      this-> template exec_range< WorkTag >( iwork , ValueOps::reference( shared_value ) , false );
    }

    // Reduce and scan, writing out scan of blocks' totals and block-groups' totals.
    // Blocks' scan values are written to 'blockIdx.x' location.
    // Block-groups' scan values are at: i = ( j * blockDim.y - 1 ) for i < gridDim.x
    cuda_single_inter_block_reduce_scan<true,FunctorType,WorkTag>( m_functor , blockIdx.x , gridDim.x , kokkos_impl_cuda_shared_memory<size_type>() , m_scratch_space , m_scratch_flags );
  }

  //----------------------------------------

  __device__ inline
  void final(void) const
  {
    const integral_nonzero_constant< size_type , ValueTraits::StaticValueSize / sizeof(size_type) >
      word_count( ValueTraits::value_size( m_functor ) / sizeof(size_type) );

    // Use shared memory as an exclusive scan: { 0 , value[0] , value[1] , value[2] , ... }
    size_type * const shared_data   = kokkos_impl_cuda_shared_memory<size_type>();
    size_type * const shared_prefix = shared_data + word_count.value * threadIdx.y ;
    size_type * const shared_accum  = shared_data + word_count.value * ( blockDim.y + 1 );

    // Starting value for this thread block is the previous block's total.
    if ( blockIdx.x ) {
      size_type * const block_total = m_scratch_space + word_count.value * ( blockIdx.x - 1 );
      for ( unsigned i = threadIdx.y ; i < word_count.value ; ++i ) { shared_accum[i] = block_total[i] ; }
    }
    else if ( 0 == threadIdx.y ) {
      ValueInit::init( m_functor , shared_accum );
    }

    const WorkRange range( m_policy , blockIdx.x , gridDim.x );

    for ( typename Policy::member_type iwork_base = range.begin(); iwork_base < range.end() ; iwork_base += blockDim.y ) {

      const typename Policy::member_type iwork = iwork_base + threadIdx.y ;

      __syncthreads(); // Don't overwrite previous iteration values until they are used

      ValueInit::init( m_functor , shared_prefix + word_count.value );

      // Copy previous block's accumulation total into thread[0] prefix and inclusive scan value of this block
      for ( unsigned i = threadIdx.y ; i < word_count.value ; ++i ) {
        shared_data[i + word_count.value] = shared_data[i] = shared_accum[i] ;
      }

      if ( CudaTraits::WarpSize < word_count.value ) { __syncthreads(); } // Protect against large scan values.

      // Call functor to accumulate inclusive scan value for this work item
      if ( iwork < range.end() ) {
        this-> template exec_range< WorkTag >( iwork , ValueOps::reference( shared_prefix + word_count.value ) , false );
      }

      // Scan block values into locations shared_data[1..blockDim.y]
      cuda_intra_block_reduce_scan<true,FunctorType,WorkTag>( m_functor , typename ValueTraits::pointer_type(shared_data+word_count.value) );

      {
        size_type * const block_total = shared_data + word_count.value * blockDim.y ;
        for ( unsigned i = threadIdx.y ; i < word_count.value ; ++i ) { shared_accum[i] = block_total[i]; }
      }

      // Call functor with exclusive scan value
      if ( iwork < range.end() ) {
        this-> template exec_range< WorkTag >( iwork , ValueOps::reference( shared_prefix ) , true );
      }
    }
  }

public:

  //----------------------------------------

  __device__ inline
  void operator()(void) const
  {
    if ( ! m_final ) {
      initial();
    }
    else {
      final();
    }
  }

  // Determine block size constrained by shared memory:
  static inline
  unsigned local_block_size( const FunctorType & f )
    {
      // blockDim.y must be power of two = 128 (4 warps) or 256 (8 warps) or 512 (16 warps)
      // gridDim.x <= blockDim.y * blockDim.y
      //
      // 4 warps was 10% faster than 8 warps and 20% faster than 16 warps in unit testing

      unsigned n = CudaTraits::WarpSize * 4 ;
      while ( n && CudaTraits::SharedMemoryCapacity < cuda_single_inter_block_reduce_scan_shmem<false,FunctorType,WorkTag>( f , n ) ) { n >>= 1 ; }
      return n ;
    }

  inline
  void execute()
    {
      const int nwork    = m_policy.end() - m_policy.begin();
      if ( nwork ) {
        enum { GridMaxComputeCapability_2x = 0x0ffff };

        const int block_size = local_block_size( m_functor );

        const int grid_max =
          ( block_size * block_size ) < GridMaxComputeCapability_2x ?
          ( block_size * block_size ) : GridMaxComputeCapability_2x ;

        // At most 'max_grid' blocks:
        const int max_grid = std::min( int(grid_max) , int(( nwork + block_size - 1 ) / block_size ));

        // How much work per block:
        const int work_per_block = ( nwork + max_grid - 1 ) / max_grid ;

        // How many block are really needed for this much work:
        const int grid_x = ( nwork + work_per_block - 1 ) / work_per_block ;

        m_scratch_space = cuda_internal_scratch_space( ValueTraits::value_size( m_functor ) * grid_x );
        m_scratch_flags = cuda_internal_scratch_flags( sizeof(size_type) * 1 );

        const dim3 grid( grid_x , 1 , 1 );
        const dim3 block( 1 , block_size , 1 ); // REQUIRED DIMENSIONS ( 1 , N , 1 )
        const int shmem = ValueTraits::value_size( m_functor ) * ( block_size + 2 );

        m_final = false ;
        CudaParallelLaunch< ParallelScan >( *this, grid, block, shmem ); // copy to device and execute

        m_final = true ;
        CudaParallelLaunch< ParallelScan >( *this, grid, block, shmem ); // copy to device and execute
      }
    }

  ParallelScan( const FunctorType  & arg_functor ,
                const Policy       & arg_policy )
  : m_functor( arg_functor )
  , m_policy( arg_policy )
  , m_scratch_space( 0 )
  , m_scratch_flags( 0 )
  , m_final( false )
  { }
};

} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {
  template<typename iType>
  struct TeamThreadRangeBoundariesStruct<iType,CudaTeamMember> {
    typedef iType index_type;
    const iType start;
    const iType end;
    const iType increment;
    const CudaTeamMember& thread;

#ifdef __CUDA_ARCH__
    __device__ inline
    TeamThreadRangeBoundariesStruct (const CudaTeamMember& thread_, const iType& count):
      start( threadIdx.y ),
      end( count ),
      increment( blockDim.y ),
      thread(thread_)
    {}
    __device__ inline
    TeamThreadRangeBoundariesStruct (const CudaTeamMember& thread_, const iType& begin_, const iType& end_):
      start( begin_+threadIdx.y ),
      end( end_ ),
      increment( blockDim.y ),
      thread(thread_)
    {}
#else
    KOKKOS_INLINE_FUNCTION
    TeamThreadRangeBoundariesStruct (const CudaTeamMember& thread_, const iType& count):
      start( 0 ),
      end( count ),
      increment( 1 ),
      thread(thread_)
    {}
    KOKKOS_INLINE_FUNCTION
    TeamThreadRangeBoundariesStruct (const CudaTeamMember& thread_,  const iType& begin_, const iType& end_):
      start( begin_ ),
      end( end_ ),
      increment( 1 ),
      thread(thread_)
    {}
#endif
  };

  template<typename iType>
  struct ThreadVectorRangeBoundariesStruct<iType,CudaTeamMember> {
    typedef iType index_type;
    const iType start;
    const iType end;
    const iType increment;

#ifdef __CUDA_ARCH__
    __device__ inline
    ThreadVectorRangeBoundariesStruct (const CudaTeamMember, const iType& count):
    start( threadIdx.x ),
    end( count ),
    increment( blockDim.x )
    {}
    __device__ inline
    ThreadVectorRangeBoundariesStruct (const iType& count):
        start( threadIdx.x ),
        end( count ),
        increment( blockDim.x )
     {}
#else
    KOKKOS_INLINE_FUNCTION
    ThreadVectorRangeBoundariesStruct (const CudaTeamMember, const iType& count):
      start( 0 ),
      end( count ),
      increment( 1 )
    {}
    KOKKOS_INLINE_FUNCTION
        ThreadVectorRangeBoundariesStruct (const iType& count):
          start( 0 ),
          end( count ),
          increment( 1 )
        {}
#endif
    };

} // namespace Impl

template<typename iType>
KOKKOS_INLINE_FUNCTION
Impl::TeamThreadRangeBoundariesStruct< iType, Impl::CudaTeamMember >
TeamThreadRange( const Impl::CudaTeamMember & thread, const iType & count ) {
  return Impl::TeamThreadRangeBoundariesStruct< iType, Impl::CudaTeamMember >( thread, count );
}

template< typename iType1, typename iType2 >
KOKKOS_INLINE_FUNCTION
Impl::TeamThreadRangeBoundariesStruct< typename std::common_type< iType1, iType2 >::type,
                                       Impl::CudaTeamMember >
TeamThreadRange( const Impl::CudaTeamMember & thread, const iType1 & begin, const iType2 & end ) {
  typedef typename std::common_type< iType1, iType2 >::type iType;
  return Impl::TeamThreadRangeBoundariesStruct< iType, Impl::CudaTeamMember >( thread, iType(begin), iType(end) );
}

template<typename iType>
KOKKOS_INLINE_FUNCTION
Impl::ThreadVectorRangeBoundariesStruct<iType,Impl::CudaTeamMember >
ThreadVectorRange(const Impl::CudaTeamMember& thread, const iType& count) {
  return Impl::ThreadVectorRangeBoundariesStruct<iType,Impl::CudaTeamMember >(thread,count);
}

KOKKOS_INLINE_FUNCTION
Impl::ThreadSingleStruct<Impl::CudaTeamMember> PerTeam(const Impl::CudaTeamMember& thread) {
  return Impl::ThreadSingleStruct<Impl::CudaTeamMember>(thread);
}

KOKKOS_INLINE_FUNCTION
Impl::VectorSingleStruct<Impl::CudaTeamMember> PerThread(const Impl::CudaTeamMember& thread) {
  return Impl::VectorSingleStruct<Impl::CudaTeamMember>(thread);
}

} // namespace Kokkos

namespace Kokkos {

  /** \brief  Inter-thread parallel_for. Executes lambda(iType i) for each i=0..N-1.
   *
   * The range i=0..N-1 is mapped to all threads of the the calling thread team.
   * This functionality requires C++11 support.*/
template<typename iType, class Lambda>
KOKKOS_INLINE_FUNCTION
void parallel_for(const Impl::TeamThreadRangeBoundariesStruct<iType,Impl::CudaTeamMember>& loop_boundaries, const Lambda& lambda) {
  #ifdef __CUDA_ARCH__
  for( iType i = loop_boundaries.start; i < loop_boundaries.end; i+=loop_boundaries.increment)
    lambda(i);
  #endif
}

/** \brief  Inter-thread vector parallel_reduce. Executes lambda(iType i, ValueType & val) for each i=0..N-1.
 *
 * The range i=0..N-1 is mapped to all threads of the the calling thread team and a summation of
 * val is performed and put into result. This functionality requires C++11 support.*/
template< typename iType, class Lambda, typename ValueType >
KOKKOS_INLINE_FUNCTION
void parallel_reduce(const Impl::TeamThreadRangeBoundariesStruct<iType,Impl::CudaTeamMember>& loop_boundaries,
                     const Lambda & lambda, ValueType& result) {

#ifdef __CUDA_ARCH__
  result = ValueType();

  for( iType i = loop_boundaries.start; i < loop_boundaries.end; i+=loop_boundaries.increment) {
    lambda(i,result);
  }

  Impl::cuda_intra_warp_reduction(result,[&] (ValueType& dst, const ValueType& src)
      { dst+=src; });
  Impl::cuda_inter_warp_reduction(result,[&] (ValueType& dst, const ValueType& src)
      { dst+=src; });
#endif
}

/** \brief  Intra-thread vector parallel_reduce. Executes lambda(iType i, ValueType & val) for each i=0..N-1.
 *
 * The range i=0..N-1 is mapped to all vector lanes of the the calling thread and a reduction of
 * val is performed using JoinType(ValueType& val, const ValueType& update) and put into init_result.
 * The input value of init_result is used as initializer for temporary variables of ValueType. Therefore
 * the input value should be the neutral element with respect to the join operation (e.g. '0 for +-' or
 * '1 for *'). This functionality requires C++11 support.*/
template< typename iType, class Lambda, typename ValueType, class JoinType >
KOKKOS_INLINE_FUNCTION
void parallel_reduce(const Impl::TeamThreadRangeBoundariesStruct<iType,Impl::CudaTeamMember>& loop_boundaries,
                     const Lambda & lambda, const JoinType& join, ValueType& init_result) {

#ifdef __CUDA_ARCH__
  ValueType result = init_result;

  for( iType i = loop_boundaries.start; i < loop_boundaries.end; i+=loop_boundaries.increment) {
    lambda(i,result);
  }

  Impl::cuda_intra_warp_reduction(result, join );
  Impl::cuda_inter_warp_reduction(result, join );

  init_result = result;
#endif
}

} //namespace Kokkos

namespace Kokkos {
/** \brief  Intra-thread vector parallel_for. Executes lambda(iType i) for each i=0..N-1.
 *
 * The range i=0..N-1 is mapped to all vector lanes of the the calling thread.
 * This functionality requires C++11 support.*/
template<typename iType, class Lambda>
KOKKOS_INLINE_FUNCTION
void parallel_for(const Impl::ThreadVectorRangeBoundariesStruct<iType,Impl::CudaTeamMember >&
    loop_boundaries, const Lambda& lambda) {
#ifdef __CUDA_ARCH__
  for( iType i = loop_boundaries.start; i < loop_boundaries.end; i+=loop_boundaries.increment)
    lambda(i);
#endif
}

/** \brief  Intra-thread vector parallel_reduce.
 *
 *  Calls lambda(iType i, ValueType & val) for each i=[0..N).
 *
 *  The range [0..N) is mapped to all vector lanes of
 *  the calling thread and a reduction of val is performed using +=
 *  and output into result.
 *
 *  The identity value for the += operator is assumed to be the default
 *  constructed value.
 */
template< typename iType, class Lambda, typename ValueType >
KOKKOS_INLINE_FUNCTION
void parallel_reduce
  ( Impl::ThreadVectorRangeBoundariesStruct<iType,Impl::CudaTeamMember >
      const & loop_boundaries
  , Lambda const & lambda
  , ValueType & result )
{
#ifdef __CUDA_ARCH__
  result = ValueType();

  for( iType i = loop_boundaries.start; i < loop_boundaries.end; i+=loop_boundaries.increment) {
    lambda(i,result);
  }

  Impl::cuda_intra_warp_vector_reduce(
    Impl::Reducer< ValueType , Impl::ReduceSum< ValueType > >( & result ) );

#endif
}

/** \brief  Intra-thread vector parallel_reduce.
 *
 *  Calls lambda(iType i, ValueType & val) for each i=[0..N).
 *
 *  The range [0..N) is mapped to all vector lanes of
 *  the calling thread and a reduction of val is performed
 *  using JoinType::operator()(ValueType& val, const ValueType& update)
 *  and output into result.
 *
 *  The input value of result must be the identity value for the
 *  reduction operation; e.g., ( 0 , += ) or ( 1 , *= ).
 */
template< typename iType, class Lambda, typename ValueType, class JoinType >
KOKKOS_INLINE_FUNCTION
void parallel_reduce
  ( Impl::ThreadVectorRangeBoundariesStruct<iType,Impl::CudaTeamMember >
      const & loop_boundaries
  , Lambda const & lambda
  , JoinType const & join
  , ValueType & result )
{
#ifdef __CUDA_ARCH__

  for( iType i = loop_boundaries.start; i < loop_boundaries.end; i+=loop_boundaries.increment) {
    lambda(i,result);
  }

  Impl::cuda_intra_warp_vector_reduce(
    Impl::Reducer< ValueType , JoinType >( join , & result ) );

#endif
}

/** \brief  Intra-thread vector parallel exclusive prefix sum. Executes lambda(iType i, ValueType & val, bool final)
 *          for each i=0..N-1.
 *
 * The range i=0..N-1 is mapped to all vector lanes in the thread and a scan operation is performed.
 * Depending on the target execution space the operator might be called twice: once with final=false
 * and once with final=true. When final==true val contains the prefix sum value. The contribution of this
 * "i" needs to be added to val no matter whether final==true or not. In a serial execution
 * (i.e. team_size==1) the operator is only called once with final==true. Scan_val will be set
 * to the final sum value over all vector lanes.
 * This functionality requires C++11 support.*/
template< typename iType, class FunctorType >
KOKKOS_INLINE_FUNCTION
void parallel_scan(const Impl::ThreadVectorRangeBoundariesStruct<iType,Impl::CudaTeamMember >&
      loop_boundaries, const FunctorType & lambda) {

#ifdef __CUDA_ARCH__
  typedef Kokkos::Impl::FunctorValueTraits< FunctorType , void > ValueTraits ;
  typedef typename ValueTraits::value_type value_type ;

  value_type scan_val = value_type();
  const int VectorLength = blockDim.x;

  iType loop_bound = ((loop_boundaries.end+VectorLength-1)/VectorLength) * VectorLength;
  for(int _i = threadIdx.x; _i < loop_bound; _i += VectorLength) {
    value_type val = value_type();
    if(_i<loop_boundaries.end)
      lambda(_i , val , false);

    value_type tmp = val;
    value_type result_i;

    if(threadIdx.x%VectorLength == 0)
      result_i = tmp;
    if (VectorLength > 1) {
      const value_type tmp2 = shfl_up(tmp, 1,VectorLength);
      if(threadIdx.x > 0)
        tmp+=tmp2;
    }
    if(threadIdx.x%VectorLength == 1)
      result_i = tmp;
    if (VectorLength > 3) {
      const value_type tmp2 = shfl_up(tmp, 2,VectorLength);
      if(threadIdx.x > 1)
        tmp+=tmp2;
    }
    if ((threadIdx.x%VectorLength >= 2) &&
        (threadIdx.x%VectorLength < 4))
      result_i = tmp;
    if (VectorLength > 7) {
      const value_type tmp2 = shfl_up(tmp, 4,VectorLength);
      if(threadIdx.x > 3)
        tmp+=tmp2;
    }
    if ((threadIdx.x%VectorLength >= 4) &&
        (threadIdx.x%VectorLength < 8))
      result_i = tmp;
    if (VectorLength > 15) {
      const value_type tmp2 = shfl_up(tmp, 8,VectorLength);
      if(threadIdx.x > 7)
        tmp+=tmp2;
    }
    if ((threadIdx.x%VectorLength >= 8) &&
        (threadIdx.x%VectorLength < 16))
      result_i = tmp;
    if (VectorLength > 31) {
      const value_type tmp2 = shfl_up(tmp, 16,VectorLength);
      if(threadIdx.x > 15)
        tmp+=tmp2;
    }
    if (threadIdx.x%VectorLength >= 16)
      result_i = tmp;

    val = scan_val + result_i - val;
    scan_val += shfl(tmp,VectorLength-1,VectorLength);
    if(_i<loop_boundaries.end)
      lambda(_i , val , true);
  }
#endif
}

}

namespace Kokkos {

template<class FunctorType>
KOKKOS_INLINE_FUNCTION
void single(const Impl::VectorSingleStruct<Impl::CudaTeamMember>& , const FunctorType& lambda) {
#ifdef __CUDA_ARCH__
  if(threadIdx.x == 0) lambda();
#endif
}

template<class FunctorType>
KOKKOS_INLINE_FUNCTION
void single(const Impl::ThreadSingleStruct<Impl::CudaTeamMember>& , const FunctorType& lambda) {
#ifdef __CUDA_ARCH__
  if(threadIdx.x == 0 && threadIdx.y == 0) lambda();
#endif
}

template<class FunctorType, class ValueType>
KOKKOS_INLINE_FUNCTION
void single(const Impl::VectorSingleStruct<Impl::CudaTeamMember>& , const FunctorType& lambda, ValueType& val) {
#ifdef __CUDA_ARCH__
  if(threadIdx.x == 0) lambda(val);
  val = shfl(val,0,blockDim.x);
#endif
}

template<class FunctorType, class ValueType>
KOKKOS_INLINE_FUNCTION
void single(const Impl::ThreadSingleStruct<Impl::CudaTeamMember>& single_struct, const FunctorType& lambda, ValueType& val) {
#ifdef __CUDA_ARCH__
  if(threadIdx.x == 0 && threadIdx.y == 0) {
    lambda(val);
  }
  single_struct.team_member.team_broadcast(val,0);
#endif
}

}

namespace Kokkos {

namespace Impl {
  template< class FunctorType, class ExecPolicy, class ValueType , class Tag = typename ExecPolicy::work_tag>
  struct CudaFunctorAdapter {
    const FunctorType f;
    typedef ValueType value_type;
    CudaFunctorAdapter(const FunctorType& f_):f(f_) {}

    __device__ inline
    void operator() (typename ExecPolicy::work_tag, const typename ExecPolicy::member_type& i, ValueType& val) const {
      //Insert Static Assert with decltype on ValueType equals third argument type of FunctorType::operator()
      f(typename ExecPolicy::work_tag(), i,val);
    }
  };

  template< class FunctorType, class ExecPolicy, class ValueType >
  struct CudaFunctorAdapter<FunctorType,ExecPolicy,ValueType,void> {
    const FunctorType f;
    typedef ValueType value_type;
    CudaFunctorAdapter(const FunctorType& f_):f(f_) {}

    __device__ inline
    void operator() (const typename ExecPolicy::member_type& i, ValueType& val) const {
      //Insert Static Assert with decltype on ValueType equals second argument type of FunctorType::operator()
      f(i,val);
    }
    __device__ inline
    void operator() (typename ExecPolicy::member_type& i, ValueType& val) const {
      //Insert Static Assert with decltype on ValueType equals second argument type of FunctorType::operator()
      f(i,val);
    }

  };

  template< class FunctorType, class Enable = void>
  struct ReduceFunctorHasInit {
    enum {value = false};
  };

  template< class FunctorType>
  struct ReduceFunctorHasInit<FunctorType, typename Impl::enable_if< 0 < sizeof( & FunctorType::init ) >::type > {
    enum {value = true};
  };

  template< class FunctorType, class Enable = void>
  struct ReduceFunctorHasJoin {
    enum {value = false};
  };

  template< class FunctorType>
  struct ReduceFunctorHasJoin<FunctorType, typename Impl::enable_if< 0 < sizeof( & FunctorType::join ) >::type > {
    enum {value = true};
  };

  template< class FunctorType, class Enable = void>
  struct ReduceFunctorHasFinal {
    enum {value = false};
  };

  template< class FunctorType>
  struct ReduceFunctorHasFinal<FunctorType, typename Impl::enable_if< 0 < sizeof( & FunctorType::final ) >::type > {
    enum {value = true};
  };

  template< class FunctorType, class Enable = void>
    struct ReduceFunctorHasShmemSize {
      enum {value = false};
    };

    template< class FunctorType>
    struct ReduceFunctorHasShmemSize<FunctorType, typename Impl::enable_if< 0 < sizeof( & FunctorType::team_shmem_size ) >::type > {
      enum {value = true};
    };

  template< class FunctorType, bool Enable =
      ( FunctorDeclaresValueType<FunctorType,void>::value) ||
      ( ReduceFunctorHasInit<FunctorType>::value  ) ||
      ( ReduceFunctorHasJoin<FunctorType>::value  ) ||
      ( ReduceFunctorHasFinal<FunctorType>::value ) ||
      ( ReduceFunctorHasShmemSize<FunctorType>::value )
      >
  struct IsNonTrivialReduceFunctor {
    enum {value = false};
  };

  template< class FunctorType>
  struct IsNonTrivialReduceFunctor<FunctorType, true> {
    enum {value = true};
  };

  template<class FunctorType, class ResultType, class Tag, bool Enable = IsNonTrivialReduceFunctor<FunctorType>::value >
  struct FunctorReferenceType {
    typedef ResultType& reference_type;
  };

  template<class FunctorType, class ResultType, class Tag>
  struct FunctorReferenceType<FunctorType, ResultType, Tag, true> {
    typedef typename Kokkos::Impl::FunctorValueTraits< FunctorType ,Tag >::reference_type reference_type;
  };

  template< class FunctorTypeIn, class ExecPolicy, class ValueType>
  struct ParallelReduceFunctorType<FunctorTypeIn,ExecPolicy,ValueType,Cuda> {

    enum {FunctorHasValueType = IsNonTrivialReduceFunctor<FunctorTypeIn>::value };
    typedef typename Kokkos::Impl::if_c<FunctorHasValueType, FunctorTypeIn, Impl::CudaFunctorAdapter<FunctorTypeIn,ExecPolicy,ValueType> >::type functor_type;
    static functor_type functor(const FunctorTypeIn& functor_in) {
      return Impl::if_c<FunctorHasValueType,FunctorTypeIn,functor_type>::select(functor_in,functor_type(functor_in));
    }
  };

}

} // namespace Kokkos
#endif /* defined( __CUDACC__ ) */

#endif /* #ifndef KOKKOS_CUDA_PARALLEL_HPP */
