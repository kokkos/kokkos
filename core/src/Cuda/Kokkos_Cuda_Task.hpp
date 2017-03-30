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

#include <Kokkos_Macros.hpp>

#if defined( KOKKOS_ENABLE_TASKDAG )

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

template< class > class TaskExec ;

template<>
class TaskQueueSpecialization< Kokkos::Cuda >
{
public:

  using execution_space = Kokkos::Cuda ;
  using memory_space    = Kokkos::CudaUVMSpace ;
  using queue_type      = TaskQueue< execution_space > ;
  using member_type     = TaskExec< Kokkos::Cuda > ;

  static
  void iff_single_thread_recursive_execute( queue_type * const ) {}

  __device__
  static void driver( queue_type * const );

  static
  void execute( queue_type * const );

  template< typename TaskType >
  static
  typename TaskType::function_type
  get_function_pointer()
    {
      using function_type = typename TaskType::function_type ;

      function_type * const ptr =
        (function_type*) cuda_internal_scratch_unified( sizeof(function_type) );

      CUDA_SAFE_CALL( cudaDeviceSynchronize() );

      set_cuda_task_base_apply_function_pointer<TaskType><<<1,1>>>(ptr);

      CUDA_SAFE_CALL( cudaGetLastError() );
      CUDA_SAFE_CALL( cudaDeviceSynchronize() );

      return *ptr ;
    }
};

extern template class TaskQueue< Kokkos::Cuda > ;

//----------------------------------------------------------------------------
/**\brief  Impl::TaskExec<Cuda> is the TaskScheduler<Cuda>::member_type
 *         passed to tasks running in a Cuda space.
 *
 *  Cuda thread blocks for tasking are dimensioned:
 *    blockDim.x == vector length
 *    blockDim.y == team size
 *    blockDim.z == number of teams
 *  where
 *    blockDim.x * blockDim.y == WarpSize
 *
 *  Both single thread and thread team tasks are run by a full Cuda warp.
 *  A single thread task is called by warp lane #0 and the remaining
 *  lanes of the warp are idle.
 */
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
  __device__ void team_barrier() { /* __threadfence_block(); */ }
  __device__ int  team_rank() const { return threadIdx.y ; }
  __device__ int  team_size() const { return m_team_size ; }
#else
  __host__ void team_barrier() {}
  __host__ int  team_rank() const { return 0 ; }
  __host__ int  team_size() const { return 0 ; }
#endif

};

//----------------------------------------------------------------------------

template<typename iType>
struct TeamThreadRangeBoundariesStruct<iType, TaskExec< Kokkos::Cuda > >
{
  typedef iType index_type;
  const iType start ;
  const iType end ;
  const iType increment ;
  const TaskExec< Kokkos::Cuda > & thread;

#if defined( __CUDA_ARCH__ )

  __device__ inline
  TeamThreadRangeBoundariesStruct
    ( const TaskExec< Kokkos::Cuda > & arg_thread, const iType& arg_count)
    : start( threadIdx.y )
    , end(arg_count)
    , increment( blockDim.y )
    , thread(arg_thread)
    {}

  __device__ inline
  TeamThreadRangeBoundariesStruct
    ( const TaskExec< Kokkos::Cuda > & arg_thread
    , const iType & arg_start
    , const iType & arg_end
    )
    : start( arg_start + threadIdx.y )
    , end(   arg_end)
    , increment( blockDim.y )
    , thread( arg_thread )
    {}

#else

  TeamThreadRangeBoundariesStruct
    ( const TaskExec< Kokkos::Cuda > & arg_thread, const iType& arg_count);

  TeamThreadRangeBoundariesStruct
    ( const TaskExec< Kokkos::Cuda > & arg_thread
    , const iType & arg_start
    , const iType & arg_end
    );

#endif

};

//----------------------------------------------------------------------------

template<typename iType>
struct ThreadVectorRangeBoundariesStruct<iType, TaskExec< Kokkos::Cuda > >
{
  typedef iType index_type;
  const iType start ;
  const iType end ;
  const iType increment ;
  const TaskExec< Kokkos::Cuda > & thread;

#if defined( __CUDA_ARCH__ )

  __device__ inline
  ThreadVectorRangeBoundariesStruct
    ( const TaskExec< Kokkos::Cuda > & arg_thread, const iType& arg_count)
    : start( threadIdx.x )
    , end(arg_count)
    , increment( blockDim.x )
    , thread(arg_thread)
    {}

#else

  ThreadVectorRangeBoundariesStruct
    ( const TaskExec< Kokkos::Cuda > & arg_thread, const iType& arg_count);

#endif

};

}} /* namespace Kokkos::Impl */

//----------------------------------------------------------------------------

namespace Kokkos {

template<typename iType>
KOKKOS_INLINE_FUNCTION
Impl::TeamThreadRangeBoundariesStruct< iType, Impl::TaskExec< Kokkos::Cuda > >
TeamThreadRange( const Impl::TaskExec< Kokkos::Cuda > & thread, const iType & count )
{
  return Impl::TeamThreadRangeBoundariesStruct< iType, Impl::TaskExec< Kokkos::Cuda > >( thread, count );
}

template<typename iType1, typename iType2>
KOKKOS_INLINE_FUNCTION
Impl::TeamThreadRangeBoundariesStruct
  < typename std::common_type<iType1,iType2>::type
  , Impl::TaskExec< Kokkos::Cuda > >
TeamThreadRange( const Impl::TaskExec< Kokkos::Cuda > & thread
               , const iType1 & begin, const iType2 & end )
{
  typedef typename std::common_type< iType1, iType2 >::type iType;
  return Impl::TeamThreadRangeBoundariesStruct< iType, Impl::TaskExec< Kokkos::Cuda > >(
           thread, iType(begin), iType(end) );
}

template<typename iType>
KOKKOS_INLINE_FUNCTION
Impl::ThreadVectorRangeBoundariesStruct<iType,Impl::TaskExec< Kokkos::Cuda > >
ThreadVectorRange( const Impl::TaskExec< Kokkos::Cuda > & thread
               , const iType & count )
{
  return Impl::ThreadVectorRangeBoundariesStruct<iType,Impl::TaskExec< Kokkos::Cuda > >(thread,count);
}

/** \brief  Inter-thread parallel_for. Executes lambda(iType i) for each i=0..N-1.
 *
 * The range i=0..N-1 is mapped to all threads of the the calling thread team.
 * This functionality requires C++11 support.
*/
template<typename iType, class Lambda>
KOKKOS_INLINE_FUNCTION
void parallel_for
  ( const Impl::TeamThreadRangeBoundariesStruct<iType,Impl:: TaskExec< Kokkos::Cuda > >& loop_boundaries
  , const Lambda& lambda
  )
{
  for( iType i = loop_boundaries.start; i < loop_boundaries.end; i+=loop_boundaries.increment) {
    lambda(i);
  }
}

// reduce across corresponding lanes between team members within warp
// assume stride*team_size == warp_size
template< typename ValueType, class JoinType >
KOKKOS_INLINE_FUNCTION
void strided_shfl_warp_reduction
  (const JoinType& join,
   ValueType& val,
   int team_size,
   int stride)
{
  for (int lane_delta=(team_size*stride)>>1; lane_delta>=stride; lane_delta>>=1) {
    join(val, Kokkos::shfl_down(val, lane_delta, team_size*stride));
  }
}

// multiple within-warp non-strided reductions
template< typename ValueType, class JoinType >
KOKKOS_INLINE_FUNCTION
void multi_shfl_warp_reduction
  (const JoinType& join,
   ValueType& val,
   int vec_length)
{
  for (int lane_delta=vec_length>>1; lane_delta; lane_delta>>=1) {
    join(val, Kokkos::shfl_down(val, lane_delta, vec_length));
  }
}

// broadcast within warp
template< class ValueType >
KOKKOS_INLINE_FUNCTION
ValueType shfl_warp_broadcast
  (ValueType& val,
   int src_lane,
   int width)
{
  return Kokkos::shfl(val, src_lane, width);
}

// all-reduce across corresponding vector lanes between team members within warp
// assume vec_length*team_size == warp_size
// blockDim.x == vec_length == stride
// blockDim.y == team_size
// threadIdx.x == position in vec
// threadIdx.y == member number
template< typename iType, class Lambda, typename ValueType, class JoinType >
KOKKOS_INLINE_FUNCTION
void parallel_reduce
  (const Impl::TeamThreadRangeBoundariesStruct<iType,Impl::TaskExec< Kokkos::Cuda > >& loop_boundaries,
   const Lambda & lambda,
   const JoinType& join,
   ValueType& initialized_result) {

  ValueType result = initialized_result;
  for( iType i = loop_boundaries.start; i < loop_boundaries.end; i+=loop_boundaries.increment) {
    lambda(i,result);
  }
  initialized_result = result;

  strided_shfl_warp_reduction<ValueType, JoinType>(
                          join,
                          initialized_result,
                          loop_boundaries.thread.team_size(),
                          blockDim.x);
  initialized_result = shfl_warp_broadcast<ValueType>( initialized_result, threadIdx.x, Impl::CudaTraits::WarpSize );
}

// all-reduce across corresponding vector lanes between team members within warp
// if no join() provided, use sum
// assume vec_length*team_size == warp_size
// blockDim.x == vec_length == stride
// blockDim.y == team_size
// threadIdx.x == position in vec
// threadIdx.y == member number
template< typename iType, class Lambda, typename ValueType >
KOKKOS_INLINE_FUNCTION
void parallel_reduce
  (const Impl::TeamThreadRangeBoundariesStruct<iType,Impl::TaskExec< Kokkos::Cuda > >& loop_boundaries,
   const Lambda & lambda,
   ValueType& initialized_result) {

  //TODO what is the point of creating this temporary?
  ValueType result = initialized_result;
  for( iType i = loop_boundaries.start; i < loop_boundaries.end; i+=loop_boundaries.increment) {
    lambda(i,result);
  }
  initialized_result = result;

  strided_shfl_warp_reduction(
                          [&] (ValueType& val1, const ValueType& val2) { val1 += val2; },
                          initialized_result,
                          loop_boundaries.thread.team_size(),
                          blockDim.x);
  initialized_result = shfl_warp_broadcast<ValueType>( initialized_result, threadIdx.x, Impl::CudaTraits::WarpSize );
}

// all-reduce within team members within warp
// assume vec_length*team_size == warp_size
// blockDim.x == vec_length == stride
// blockDim.y == team_size
// threadIdx.x == position in vec
// threadIdx.y == member number
template< typename iType, class Lambda, typename ValueType, class JoinType >
KOKKOS_INLINE_FUNCTION
void parallel_reduce
  (const Impl::ThreadVectorRangeBoundariesStruct<iType,Impl::TaskExec< Kokkos::Cuda > >& loop_boundaries,
   const Lambda & lambda,
   const JoinType& join,
   ValueType& initialized_result) {

  ValueType result = initialized_result;
  for( iType i = loop_boundaries.start; i < loop_boundaries.end; i+=loop_boundaries.increment) {
    lambda(i,result);
  }
  initialized_result = result;

  multi_shfl_warp_reduction<ValueType, JoinType>(join, initialized_result, blockDim.x);
  initialized_result = shfl_warp_broadcast<ValueType>( initialized_result, 0, blockDim.x );
}

// all-reduce within team members within warp
// if no join() provided, use sum
// assume vec_length*team_size == warp_size
// blockDim.x == vec_length == stride
// blockDim.y == team_size
// threadIdx.x == position in vec
// threadIdx.y == member number
template< typename iType, class Lambda, typename ValueType >
KOKKOS_INLINE_FUNCTION
void parallel_reduce
  (const Impl::ThreadVectorRangeBoundariesStruct<iType,Impl::TaskExec< Kokkos::Cuda > >& loop_boundaries,
   const Lambda & lambda,
   ValueType& initialized_result) {

  ValueType result = initialized_result;

  for( iType i = loop_boundaries.start; i < loop_boundaries.end; i+=loop_boundaries.increment) {
    lambda(i,result);
  }

  initialized_result = result;

  //initialized_result = multi_shfl_warp_reduction(
  multi_shfl_warp_reduction(
                          [&] (ValueType& val1, const ValueType& val2) { val1 += val2; },
                          initialized_result,
                          blockDim.x);
  initialized_result = shfl_warp_broadcast<ValueType>( initialized_result, 0, blockDim.x );
}

// scan across corresponding vector lanes between team members within warp
// assume vec_length*team_size == warp_size
// blockDim.x == vec_length == stride
// blockDim.y == team_size
// threadIdx.x == position in vec
// threadIdx.y == member number
template< typename iType, class Closure >
KOKKOS_INLINE_FUNCTION
void parallel_scan
  (const Impl::TeamThreadRangeBoundariesStruct<iType,Impl::TaskExec< Kokkos::Cuda > >& loop_boundaries,
   const Closure & closure )
{
  // Extract value_type from closure

  using value_type =
    typename Kokkos::Impl::FunctorAnalysis
      < Kokkos::Impl::FunctorPatternInterface::SCAN
      , void
      , Closure >::value_type ;

  value_type accum = 0 ;
  value_type val, y, local_total;

  for( iType i = loop_boundaries.start; i < loop_boundaries.end; i+=loop_boundaries.increment) {
    val = 0;
    closure(i,val,false);

    // intra-blockDim.y exclusive scan on 'val'
    // accum = accumulated, sum in total for this iteration

    // INCLUSIVE scan
    for( int offset = blockDim.x ; offset < Impl::CudaTraits::WarpSize ; offset <<= 1 ) {
      y = Kokkos::shfl_up(val, offset, Impl::CudaTraits::WarpSize);
      if(threadIdx.y*blockDim.x >= offset) { val += y; }
    }

    // pass accum to all threads
    local_total = shfl_warp_broadcast<value_type>(val,
                                            threadIdx.x+Impl::CudaTraits::WarpSize-blockDim.x,
                                            Impl::CudaTraits::WarpSize);

    // make EXCLUSIVE scan by shifting values over one
    val = Kokkos::shfl_up(val, blockDim.x, Impl::CudaTraits::WarpSize);
    if ( threadIdx.y == 0 ) { val = 0 ; }

    val += accum;
    closure(i,val,true);
    accum += local_total;
  }
}

// scan within team member (vector) within warp
// assume vec_length*team_size == warp_size
// blockDim.x == vec_length == stride
// blockDim.y == team_size
// threadIdx.x == position in vec
// threadIdx.y == member number
template< typename iType, class Closure >
KOKKOS_INLINE_FUNCTION
void parallel_scan
  (const Impl::ThreadVectorRangeBoundariesStruct<iType,Impl::TaskExec< Kokkos::Cuda > >& loop_boundaries,
   const Closure & closure )
{
  // Extract value_type from closure

  using value_type =
    typename Kokkos::Impl::FunctorAnalysis
      < Kokkos::Impl::FunctorPatternInterface::SCAN
      , void
      , Closure >::value_type ;

  value_type accum = 0 ;
  value_type val, y, local_total;

  for( iType i = loop_boundaries.start; i < loop_boundaries.end; i+=loop_boundaries.increment) {
    val = 0;
    closure(i,val,false);

    // intra-blockDim.x exclusive scan on 'val'
    // accum = accumulated, sum in total for this iteration

    // INCLUSIVE scan
    for( int offset = 1 ; offset < blockDim.x ; offset <<= 1 ) {
      y = Kokkos::shfl_up(val, offset, blockDim.x);
      if(threadIdx.x >= offset) { val += y; }
    }

    // pass accum to all threads
    local_total = shfl_warp_broadcast<value_type>(val, blockDim.x-1, blockDim.x);

    // make EXCLUSIVE scan by shifting values over one
    val = Kokkos::shfl_up(val, 1, blockDim.x);
    if ( threadIdx.x == 0 ) { val = 0 ; }

    val += accum;
    closure(i,val,true);
    accum += local_total;
  }
}

} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #if defined( KOKKOS_ENABLE_TASKDAG ) */
#endif /* #ifndef KOKKOS_IMPL_CUDA_TASK_HPP */

