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

#ifndef KOKKOS_IMPL_CUDA_TASK_HPP
#define KOKKOS_IMPL_CUDA_TASK_HPP

#if defined( KOKKOS_ENABLE_TASKPOLICY )

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {
namespace {

template< typename TaskType >
__global__
void set_cuda_task_base_apply_function_pointer
  ( TaskBase<Kokkos::Cuda,void,void>::function_type * ptr )
{ *ptr = TaskType::apply ; }

}

template<>
class TaskQueueSpecialization< Kokkos::Cuda >
{
public:

  using execution_space = Kokkos::Cuda ;
  using memory_space    = Kokkos::CudaUVMSpace ;
  using queue_type      = TaskQueue< execution_space > ;

  __device__
  static void driver( queue_type * const );

  static
  void execute( queue_type * const );

  template< typename FunctorType >
  static
  void proc_set_apply( TaskBase<execution_space,void,void>::function_type * ptr )
    {
      using TaskType = TaskBase< execution_space
                               , typename FunctorType::value_type
                               , FunctorType > ;

      CUDA_SAFE_CALL( cudaDeviceSynchronize() );

      set_cuda_task_base_apply_function_pointer<TaskType><<<1,1>>>(ptr);

      CUDA_SAFE_CALL( cudaGetLastError() );
      CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    }
};

extern template class TaskQueue< Kokkos::Cuda > ;

//----------------------------------------------------------------------------

template<>
class TaskExec< Kokkos::Cuda >
{
private:

  TaskExec( TaskExec && ) = delete ;
  TaskExec( TaskExec const & ) = delete ;
  TaskExec & operator = ( TaskExec && ) = delete ;
  TaskExec & operator = ( TaskExec const & ) = delete ;

  friend class Kokkos::Impl::TaskQueue< Kokkos::Cuda > ;
  friend class Kokkos::Impl::TaskQueueSpecialization< Kokkos::Cuda > ;

  const int m_team_size ;

  __device__
  TaskExec( int arg_team_size = blockDim.y )
    : m_team_size( arg_team_size ) {}

public:

#if defined( __CUDA_ARCH__ )
  __device__ void team_barrier() { __threadfence_block(); }
  __device__ int  team_rank() const { return threadIdx.y ; }
  __device__ int  team_size() const { return blockDim.y ; }
#else
  __host__ void team_barrier() {}
  __host__ int  team_rank() const { return 0 ; }
  __host__ int  team_size() const { return 0 ; }
#endif

};

template<typename iType>
struct TeamThreadRangeBoundariesStruct<iType, TaskExec< Kokkos::Cuda > >
{
  typedef iType index_type;
  const iType begin ;
  const iType end ;
  const iType increment ;
  const TaskExec< Kokkos::Cuda > & thread;

#if defined( __CUDA_ARCH__ )

  __device__ inline
  TeamThreadRangeBoundariesStruct
    ( const TaskExec< Kokkos::Cuda > & arg_thread, const iType& arg_count)
    : begin( threadIdx.y )
    , end(arg_count)
    , increment( blockDim.y )
    , thread(arg_thread)
    {}

  __device__ inline
  TeamThreadRangeBoundariesStruct
    ( const TaskExec< Kokkos::Cuda > & arg_thread
    , const iType & arg_begin
    , const iType & arg_end
    )
    : begin( arg_begin + threadIdx.y )
    , end(   arg_end)
    , increment( blockDim.y )
    , thread( arg_thread )
    {}

#else

  TeamThreadRangeBoundariesStruct
    ( const TaskExec< Kokkos::Cuda > & arg_thread, const iType& arg_count);

  TeamThreadRangeBoundariesStruct
    ( const TaskExec< Kokkos::Cuda > & arg_thread
    , const iType & arg_begin
    , const iType & arg_end
    );

#endif

};

}} /* namespace Kokkos::Impl */

//----------------------------------------------------------------------------

namespace Kokkos {

template<typename iType>
KOKKOS_INLINE_FUNCTION
Impl::TeamThreadRangeBoundariesStruct<iType,Impl::TaskExec< Kokkos::Cuda > >
TeamThreadRange( const Impl::TaskExec< Kokkos::Cuda > & thread
               , const iType & count )
{
  return Impl::TeamThreadRangeBoundariesStruct<iType,Impl::TaskExec< Kokkos::Cuda > >(thread,count);
}

template<typename iType>
KOKKOS_INLINE_FUNCTION
Impl::TeamThreadRangeBoundariesStruct<iType,Impl::TaskExec< Kokkos::Cuda > >
TeamThreadRange( const Impl::TaskExec< Kokkos::Cuda > & thread, const iType & begin , const iType & end )
{
  return Impl::TeamThreadRangeBoundariesStruct<iType,Impl:: TaskExec< Kokkos::Cuda > >(thread,begin,end);
}

  /** \brief  Inter-thread parallel_for. Executes lambda(iType i) for each i=0..N-1.
   *
   * The range i=0..N-1 is mapped to all threads of the the calling thread team.
   * This functionality requires C++11 support.*/
template<typename iType, class Lambda>
KOKKOS_INLINE_FUNCTION
void parallel_for
  ( const Impl::TeamThreadRangeBoundariesStruct<iType,Impl:: TaskExec< Kokkos::Cuda > >& loop_boundaries
  , const Lambda& lambda
  )
{
  for( iType i = loop_boundaries.begin; i < loop_boundaries.end; i+=loop_boundaries.increment) {
    lambda(i);
  }
}

} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #if defined( KOKKOS_ENABLE_TASKPOLICY ) */
#endif /* #ifndef KOKKOS_IMPL_CUDA_TASK_HPP */

