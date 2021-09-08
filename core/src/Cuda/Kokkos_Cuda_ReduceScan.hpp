/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
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
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_CUDA_REDUCESCAN_HPP
#define KOKKOS_CUDA_REDUCESCAN_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_CUDA)

#include <utility>

#include <Kokkos_Parallel.hpp>
#include <impl/Kokkos_FunctorAdapter.hpp>
#include <impl/Kokkos_Error.hpp>
#include <Cuda/Kokkos_Cuda_Vectorization.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

//----------------------------------------------------------------------------
/*
 *  Algorithmic constraints:
 *   (a) threads with same threadIdx.y have same value
 *   (b) blockDim.x == power of two
 *   (c) blockDim.z == 1
 */

template <class ValueType, class JoinOp>
__device__ inline
    typename std::enable_if<!Kokkos::is_reducer<ValueType>::value>::type
    cuda_intra_warp_reduction(ValueType& result, const JoinOp& join,
                              const uint32_t max_active_thread = blockDim.y) {
  unsigned int shift = 1;

  // Reduce over values from threads with different threadIdx.y
  while (blockDim.x * shift < 32) {
    const ValueType tmp = shfl_down(result, blockDim.x * shift, 32u);
    // Only join if upper thread is active (this allows non power of two for
    // blockDim.y
    if (threadIdx.y + shift < max_active_thread) join(result, tmp);
    shift *= 2;
  }

  result = shfl(result, 0, 32);
}

template <class ValueType, class JoinOp>
__device__ inline
    typename std::enable_if<!Kokkos::is_reducer<ValueType>::value>::type
    cuda_inter_warp_reduction(ValueType& value, const JoinOp& join,
                              const int max_active_thread = blockDim.y) {
#define STEP_WIDTH 4
  // Depending on the ValueType _shared__ memory must be aligned up to 8byte
  // boundaries The reason not to use ValueType directly is that for types with
  // constructors it could lead to race conditions
  alignas(alignof(ValueType) > alignof(double) ? alignof(ValueType)
                                               : alignof(double))
      __shared__ double sh_result[(sizeof(ValueType) + 7) / 8 * STEP_WIDTH];
  ValueType* result = (ValueType*)&sh_result;
  const int step    = 32 / blockDim.x;
  int shift         = STEP_WIDTH;
  const int id      = threadIdx.y % step == 0 ? threadIdx.y / step : 65000;
  if (id < STEP_WIDTH) {
    result[id] = value;
  }
  __syncthreads();
  while (shift <= max_active_thread / step) {
    if (shift <= id && shift + STEP_WIDTH > id && threadIdx.x == 0) {
      join(result[id % STEP_WIDTH], value);
    }
    __syncthreads();
    shift += STEP_WIDTH;
  }

  value = result[0];
  for (int i = 1; (i * step < max_active_thread) && i < STEP_WIDTH; i++)
    join(value, result[i]);
}

template <class ValueType, class JoinOp>
__device__ inline
    typename std::enable_if<!Kokkos::is_reducer<ValueType>::value>::type
    cuda_intra_block_reduction(ValueType& value, const JoinOp& join,
                               const int max_active_thread = blockDim.y) {
  cuda_intra_warp_reduction(value, join, max_active_thread);
  cuda_inter_warp_reduction(value, join, max_active_thread);
}

template <class FunctorType, class JoinOp, class ArgTag = void>
__device__ bool cuda_inter_block_reduction(
    typename FunctorValueTraits<FunctorType, ArgTag>::reference_type value,
    typename FunctorValueTraits<FunctorType, ArgTag>::reference_type neutral,
    const JoinOp& join, Cuda::size_type* const m_scratch_space,
    typename FunctorValueTraits<FunctorType,
                                ArgTag>::pointer_type const /*result*/,
    Cuda::size_type* const m_scratch_flags,
    const int max_active_thread = blockDim.y) {
#ifdef __CUDA_ARCH__
  using pointer_type =
      typename FunctorValueTraits<FunctorType, ArgTag>::pointer_type;
  using value_type =
      typename FunctorValueTraits<FunctorType, ArgTag>::value_type;

  // Do the intra-block reduction with shfl operations and static shared memory
  cuda_intra_block_reduction(value, join, max_active_thread);

  const int id = threadIdx.y * blockDim.x + threadIdx.x;

  // One thread in the block writes block result to global scratch_memory
  if (id == 0) {
    pointer_type global = ((pointer_type)m_scratch_space) + blockIdx.x;
    *global             = value;
  }

  // One warp of last block performs inter block reduction through loading the
  // block values from global scratch_memory
  bool last_block = false;
  __threadfence();
  __syncthreads();
  if (id < 32) {
    Cuda::size_type count;

    // Figure out whether this is the last block
    if (id == 0) count = Kokkos::atomic_fetch_add(m_scratch_flags, 1);
    count = Kokkos::shfl(count, 0, 32);

    // Last block does the inter block reduction
    if (count == gridDim.x - 1) {
      // set flag back to zero
      if (id == 0) *m_scratch_flags = 0;
      last_block = true;
      value      = neutral;

      pointer_type const volatile global = (pointer_type)m_scratch_space;

      // Reduce all global values with splitting work over threads in one warp
      const int step_size =
          blockDim.x * blockDim.y < 32 ? blockDim.x * blockDim.y : 32;
      for (int i = id; i < (int)gridDim.x; i += step_size) {
        value_type tmp = global[i];
        join(value, tmp);
      }

      // Perform shfl reductions within the warp only join if contribution is
      // valid (allows gridDim.x non power of two and <32)
      if (int(blockDim.x * blockDim.y) > 1) {
        value_type tmp = Kokkos::shfl_down(value, 1, 32);
        if (id + 1 < int(gridDim.x)) join(value, tmp);
      }
      unsigned int mask = __activemask();
      int active        = __ballot_sync(mask, 1);
      if (int(blockDim.x * blockDim.y) > 2) {
        value_type tmp = Kokkos::shfl_down(value, 2, 32);
        if (id + 2 < int(gridDim.x)) join(value, tmp);
      }
      active += __ballot_sync(mask, 1);
      if (int(blockDim.x * blockDim.y) > 4) {
        value_type tmp = Kokkos::shfl_down(value, 4, 32);
        if (id + 4 < int(gridDim.x)) join(value, tmp);
      }
      active += __ballot_sync(mask, 1);
      if (int(blockDim.x * blockDim.y) > 8) {
        value_type tmp = Kokkos::shfl_down(value, 8, 32);
        if (id + 8 < int(gridDim.x)) join(value, tmp);
      }
      active += __ballot_sync(mask, 1);
      if (int(blockDim.x * blockDim.y) > 16) {
        value_type tmp = Kokkos::shfl_down(value, 16, 32);
        if (id + 16 < int(gridDim.x)) join(value, tmp);
      }
      active += __ballot_sync(mask, 1);
    }
  }
  // The last block has in its thread=0 the global reduction value through
  // "value"
  return last_block;
#else
  (void)value;
  (void)neutral;
  (void)join;
  (void)m_scratch_space;
  (void)m_scratch_flags;
  (void)max_active_thread;
  return true;
#endif
}

template <class ReducerType>
__device__ inline
    typename std::enable_if<Kokkos::is_reducer<ReducerType>::value>::type
    cuda_intra_warp_reduction(const ReducerType& reducer,
                              typename ReducerType::value_type& result,
                              const uint32_t max_active_thread = blockDim.y) {
  using ValueType = typename ReducerType::value_type;

  unsigned int shift = 1;

  // Reduce over values from threads with different threadIdx.y
  while (blockDim.x * shift < 32) {
    const ValueType tmp = shfl_down(result, blockDim.x * shift, 32u);
    // Only join if upper thread is active (this allows non power of two for
    // blockDim.y
    if (threadIdx.y + shift < max_active_thread) reducer.join(result, tmp);
    shift *= 2;
  }

  result              = shfl(result, 0, 32);
  reducer.reference() = result;
}

template <class ReducerType>
__device__ inline
    typename std::enable_if<Kokkos::is_reducer<ReducerType>::value>::type
    cuda_inter_warp_reduction(const ReducerType& reducer,
                              typename ReducerType::value_type value,
                              const int max_active_thread = blockDim.y) {
  using ValueType = typename ReducerType::value_type;

#define STEP_WIDTH 4
  // Depending on the ValueType _shared__ memory must be aligned up to 8byte
  // boundaries The reason not to use ValueType directly is that for types with
  // constructors it could lead to race conditions
  alignas(alignof(ValueType) > alignof(double) ? alignof(ValueType)
                                               : alignof(double))
      __shared__ double sh_result[(sizeof(ValueType) + 7) / 8 * STEP_WIDTH];
  ValueType* result = (ValueType*)&sh_result;
  const int step    = 32 / blockDim.x;
  int shift         = STEP_WIDTH;
  const int id      = threadIdx.y % step == 0 ? threadIdx.y / step : 65000;
  if (id < STEP_WIDTH) {
    result[id] = value;
  }
  __syncthreads();
  while (shift <= max_active_thread / step) {
    if (shift <= id && shift + STEP_WIDTH > id && threadIdx.x == 0) {
      reducer.join(result[id % STEP_WIDTH], value);
    }
    __syncthreads();
    shift += STEP_WIDTH;
  }

  value = result[0];
  for (int i = 1; (i * step < max_active_thread) && i < STEP_WIDTH; i++)
    reducer.join(value, result[i]);

  reducer.reference() = value;
}

template <class ReducerType>
__device__ inline
    typename std::enable_if<Kokkos::is_reducer<ReducerType>::value>::type
    cuda_intra_block_reduction(const ReducerType& reducer,
                               typename ReducerType::value_type value,
                               const int max_active_thread = blockDim.y) {
  cuda_intra_warp_reduction(reducer, value, max_active_thread);
  cuda_inter_warp_reduction(reducer, value, max_active_thread);
}

template <class ReducerType>
__device__ inline
    typename std::enable_if<Kokkos::is_reducer<ReducerType>::value>::type
    cuda_intra_block_reduction(const ReducerType& reducer,
                               const int max_active_thread = blockDim.y) {
  cuda_intra_block_reduction(reducer, reducer.reference(), max_active_thread);
}

template <class ReducerType>
__device__ inline
    typename std::enable_if<Kokkos::is_reducer<ReducerType>::value, bool>::type
    cuda_inter_block_reduction(const ReducerType& reducer,
                               Cuda::size_type* const m_scratch_space,
                               Cuda::size_type* const m_scratch_flags,
                               const int max_active_thread = blockDim.y) {
#ifdef __CUDA_ARCH__
  using pointer_type = typename ReducerType::value_type*;
  using value_type   = typename ReducerType::value_type;

  // Do the intra-block reduction with shfl operations and static shared memory
  cuda_intra_block_reduction(reducer, max_active_thread);

  value_type value = reducer.reference();

  const int id = threadIdx.y * blockDim.x + threadIdx.x;

  // One thread in the block writes block result to global scratch_memory
  if (id == 0) {
    pointer_type global = ((pointer_type)m_scratch_space) + blockIdx.x;
    *global             = value;
  }

  // One warp of last block performs inter block reduction through loading the
  // block values from global scratch_memory
  bool last_block = false;

  __threadfence();
  __syncthreads();
  if (id < 32) {
    Cuda::size_type count;

    // Figure out whether this is the last block
    if (id == 0) count = Kokkos::atomic_fetch_add(m_scratch_flags, 1);
    count = Kokkos::shfl(count, 0, 32);

    // Last block does the inter block reduction
    if (count == gridDim.x - 1) {
      // set flag back to zero
      if (id == 0) *m_scratch_flags = 0;
      last_block = true;
      reducer.init(value);

      pointer_type const volatile global = (pointer_type)m_scratch_space;

      // Reduce all global values with splitting work over threads in one warp
      const int step_size =
          blockDim.x * blockDim.y < 32 ? blockDim.x * blockDim.y : 32;
      for (int i = id; i < (int)gridDim.x; i += step_size) {
        value_type tmp = global[i];
        reducer.join(value, tmp);
      }

      // Perform shfl reductions within the warp only join if contribution is
      // valid (allows gridDim.x non power of two and <32)
      if (int(blockDim.x * blockDim.y) > 1) {
        value_type tmp = Kokkos::shfl_down(value, 1, 32);
        if (id + 1 < int(gridDim.x)) reducer.join(value, tmp);
      }
      unsigned int mask = __activemask();
      int active        = __ballot_sync(mask, 1);
      if (int(blockDim.x * blockDim.y) > 2) {
        value_type tmp = Kokkos::shfl_down(value, 2, 32);
        if (id + 2 < int(gridDim.x)) reducer.join(value, tmp);
      }
      active += __ballot_sync(mask, 1);
      if (int(blockDim.x * blockDim.y) > 4) {
        value_type tmp = Kokkos::shfl_down(value, 4, 32);
        if (id + 4 < int(gridDim.x)) reducer.join(value, tmp);
      }
      active += __ballot_sync(mask, 1);
      if (int(blockDim.x * blockDim.y) > 8) {
        value_type tmp = Kokkos::shfl_down(value, 8, 32);
        if (id + 8 < int(gridDim.x)) reducer.join(value, tmp);
      }
      active += __ballot_sync(mask, 1);
      if (int(blockDim.x * blockDim.y) > 16) {
        value_type tmp = Kokkos::shfl_down(value, 16, 32);
        if (id + 16 < int(gridDim.x)) reducer.join(value, tmp);
      }
      active += __ballot_sync(mask, 1);
    }
  }

  // The last block has in its thread=0 the global reduction value through
  // "value"
  return last_block;
#else
  (void)reducer;
  (void)m_scratch_space;
  (void)m_scratch_flags;
  (void)max_active_thread;
  return true;
#endif
}

template <class FunctorType, class ArgTag, bool DoScan, bool UseShfl>
struct CudaReductionsFunctor;

template <class FunctorType, class ArgTag>
struct CudaReductionsFunctor<FunctorType, ArgTag, false, true> {
  using ValueTraits  = FunctorValueTraits<FunctorType, ArgTag>;
  using ValueJoin    = FunctorValueJoin<FunctorType, ArgTag>;
  using ValueInit    = FunctorValueInit<FunctorType, ArgTag>;
  using ValueOps     = FunctorValueOps<FunctorType, ArgTag>;
  using pointer_type = typename ValueTraits::pointer_type;
  using Scalar       = typename ValueTraits::value_type;

  __device__ static inline void scalar_intra_warp_reduction(
      const FunctorType& functor,
      Scalar value,            // Contribution
      const bool skip_vector,  // Skip threads if Kokkos vector lanes are not
                               // part of the reduction
      const int width,         // How much of the warp participates
      Scalar& result) {
    unsigned mask =
        width == 32
            ? 0xffffffff
            : ((1 << width) - 1)
                  << ((threadIdx.y * blockDim.x + threadIdx.x) / width) * width;
    for (int delta = skip_vector ? blockDim.x : 1; delta < width; delta *= 2) {
      Scalar tmp = Kokkos::shfl_down(value, delta, width, mask);
      ValueJoin::join(functor, &value, &tmp);
    }

    Impl::in_place_shfl(result, value, 0, width, mask);
  }

  __device__ static inline void scalar_intra_block_reduction(
      const FunctorType& functor, Scalar value, const bool skip,
      Scalar* my_global_team_buffer_element, const int shared_elements,
      Scalar* shared_team_buffer_element) {
    const int warp_id = (threadIdx.y * blockDim.x) / 32;
    Scalar* const my_shared_team_buffer_element =
        shared_team_buffer_element + warp_id % shared_elements;

    // Warp Level Reduction, ignoring Kokkos vector entries
    scalar_intra_warp_reduction(functor, value, skip, 32, value);

    if (warp_id < shared_elements) {
      *my_shared_team_buffer_element = value;
    }
    // Wait for every warp to be done before using one warp to do final cross
    // warp reduction
    __syncthreads();

    const int num_warps = blockDim.x * blockDim.y / 32;
    for (int w = shared_elements; w < num_warps; w += shared_elements) {
      if (warp_id >= w && warp_id < w + shared_elements) {
        if ((threadIdx.y * blockDim.x + threadIdx.x) % 32 == 0)
          ValueJoin::join(functor, my_shared_team_buffer_element, &value);
      }
      __syncthreads();
    }

    if (warp_id == 0) {
      ValueInit::init(functor, &value);
      for (unsigned int i = threadIdx.y * blockDim.x + threadIdx.x;
           i < blockDim.y * blockDim.x / 32; i += 32)
        ValueJoin::join(functor, &value, &shared_team_buffer_element[i]);
      scalar_intra_warp_reduction(functor, value, false, 32,
                                  *my_global_team_buffer_element);
    }
  }

  __device__ static inline bool scalar_inter_block_reduction(
      const FunctorType& functor, const Cuda::size_type /*block_id*/,
      const Cuda::size_type block_count, Cuda::size_type* const shared_data,
      Cuda::size_type* const global_data, Cuda::size_type* const global_flags) {
    Scalar* const global_team_buffer_element = ((Scalar*)global_data);
    Scalar* const my_global_team_buffer_element =
        global_team_buffer_element + blockIdx.x;
    Scalar* shared_team_buffer_elements = ((Scalar*)shared_data);
    Scalar value        = shared_team_buffer_elements[threadIdx.y];
    int shared_elements = blockDim.x * blockDim.y / 32;
    int global_elements = block_count;
    __syncthreads();

    scalar_intra_block_reduction(functor, value, true,
                                 my_global_team_buffer_element, shared_elements,
                                 shared_team_buffer_elements);
    __threadfence();
    __syncthreads();
    unsigned int num_teams_done = 0;
    // The cast in the atomic call is necessary to find matching call with
    // MSVC/NVCC
    if (threadIdx.x + threadIdx.y == 0) {
      num_teams_done =
          Kokkos::atomic_fetch_add(global_flags, static_cast<unsigned int>(1)) +
          1;
    }
    bool is_last_block = false;
    if (__syncthreads_or(num_teams_done == gridDim.x)) {
      is_last_block = true;
      *global_flags = 0;
      ValueInit::init(functor, &value);
      for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < global_elements;
           i += blockDim.x * blockDim.y) {
        ValueJoin::join(functor, &value, &global_team_buffer_element[i]);
      }
      scalar_intra_block_reduction(
          functor, value, false, shared_team_buffer_elements + (blockDim.y - 1),
          shared_elements, shared_team_buffer_elements);
    }
    return is_last_block;
  }
};

template <class FunctorType, class ArgTag>
struct CudaReductionsFunctor<FunctorType, ArgTag, false, false> {
  using ValueTraits  = FunctorValueTraits<FunctorType, ArgTag>;
  using ValueJoin    = FunctorValueJoin<FunctorType, ArgTag>;
  using ValueInit    = FunctorValueInit<FunctorType, ArgTag>;
  using ValueOps     = FunctorValueOps<FunctorType, ArgTag>;
  using pointer_type = typename ValueTraits::pointer_type;
  using Scalar       = typename ValueTraits::value_type;

  __device__ static inline void scalar_intra_warp_reduction(
      const FunctorType& functor,
      Scalar* value,           // Contribution
      const bool skip_vector,  // Skip threads if Kokkos vector lanes are not
                               // part of the reduction
      const int width)         // How much of the warp participates
  {
    unsigned mask =
        width == 32
            ? 0xffffffff
            : ((1 << width) - 1)
                  << ((threadIdx.y * blockDim.x + threadIdx.x) / width) * width;
    const int lane_id = (threadIdx.y * blockDim.x + threadIdx.x) % 32;
    for (int delta = skip_vector ? blockDim.x : 1; delta < width; delta *= 2) {
      if (lane_id + delta < 32) {
        ValueJoin::join(functor, value, value + delta);
      }
      __syncwarp(mask);
    }
    *value = *(value - lane_id);
  }

  __device__ static inline void scalar_intra_block_reduction(
      const FunctorType& functor, Scalar value, const bool skip, Scalar* result,
      const int /*shared_elements*/, Scalar* shared_team_buffer_element) {
    const int warp_id = (threadIdx.y * blockDim.x) / 32;
    Scalar* const my_shared_team_buffer_element =
        shared_team_buffer_element + threadIdx.y * blockDim.x + threadIdx.x;
    *my_shared_team_buffer_element = value;
    // Warp Level Reduction, ignoring Kokkos vector entries
    scalar_intra_warp_reduction(functor, my_shared_team_buffer_element, skip,
                                32);
    // Wait for every warp to be done before using one warp to do final cross
    // warp reduction
    __syncthreads();

    if (warp_id == 0) {
      const unsigned int delta = (threadIdx.y * blockDim.x + threadIdx.x) * 32;
      if (delta < blockDim.x * blockDim.y)
        *my_shared_team_buffer_element = shared_team_buffer_element[delta];
      __syncwarp(0xffffffff);
      scalar_intra_warp_reduction(functor, my_shared_team_buffer_element, false,
                                  blockDim.x * blockDim.y / 32);
      if (threadIdx.x + threadIdx.y == 0) *result = *shared_team_buffer_element;
    }
  }

  template <class SizeType = Cuda::size_type>
  __device__ static inline bool scalar_inter_block_reduction(
      const FunctorType& functor, const Cuda::size_type /*block_id*/,
      const Cuda::size_type block_count, SizeType* const shared_data,
      SizeType* const global_data, Cuda::size_type* const global_flags) {
    Scalar* const global_team_buffer_element = ((Scalar*)global_data);
    Scalar* const my_global_team_buffer_element =
        global_team_buffer_element + blockIdx.x;
    Scalar* shared_team_buffer_elements = ((Scalar*)shared_data);
    Scalar value        = shared_team_buffer_elements[threadIdx.y];
    int shared_elements = blockDim.x * blockDim.y / 32;
    int global_elements = block_count;
    __syncthreads();

    scalar_intra_block_reduction(functor, value, true,
                                 my_global_team_buffer_element, shared_elements,
                                 shared_team_buffer_elements);
    __threadfence();
    __syncthreads();

    unsigned int num_teams_done = 0;
    // The cast in the atomic call is necessary to find matching call with
    // MSVC/NVCC
    if (threadIdx.x + threadIdx.y == 0) {
      num_teams_done =
          Kokkos::atomic_fetch_add(global_flags, static_cast<unsigned int>(1)) +
          1;
    }
    bool is_last_block = false;
    if (__syncthreads_or(num_teams_done == gridDim.x)) {
      is_last_block = true;
      *global_flags = 0;
      ValueInit::init(functor, &value);
      for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < global_elements;
           i += blockDim.x * blockDim.y) {
        ValueJoin::join(functor, &value, &global_team_buffer_element[i]);
      }
      scalar_intra_block_reduction(
          functor, value, false, shared_team_buffer_elements + (blockDim.y - 1),
          shared_elements, shared_team_buffer_elements);
    }
    return is_last_block;
  }
};
//----------------------------------------------------------------------------
// See section B.17 of Cuda C Programming Guide Version 3.2
// for discussion of
//   __launch_bounds__(maxThreadsPerBlock,minBlocksPerMultiprocessor)
// function qualifier which could be used to improve performance.
//----------------------------------------------------------------------------
// Maximize shared memory and minimize L1 cache:
//   cudaFuncSetCacheConfig(MyKernel, cudaFuncCachePreferShared );
// For 2.0 capability: 48 KB shared and 16 KB L1
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
/*
 *  Algorithmic constraints:
 *   (a) blockDim.y is a power of two
 *   (b) blockDim.y <= 1024
 *   (c) blockDim.x == blockDim.z == 1
 */

template <bool DoScan, class FunctorType, class ArgTag>
__device__ void cuda_intra_block_reduce_scan(
    const FunctorType& functor,
    const typename FunctorValueTraits<FunctorType, ArgTag>::pointer_type
        base_data) {
  using ValueTraits = FunctorValueTraits<FunctorType, ArgTag>;
  using ValueJoin   = FunctorValueJoin<FunctorType, ArgTag>;

  using pointer_type = typename ValueTraits::pointer_type;

  const unsigned value_count   = ValueTraits::value_count(functor);
  const unsigned BlockSizeMask = blockDim.y - 1;

  // Must have power of two thread count

  if (BlockSizeMask & blockDim.y) {
    Kokkos::abort("Cuda::cuda_intra_block_scan requires power-of-two blockDim");
  }

#define BLOCK_REDUCE_STEP(R, TD, S)                          \
  if (!(R & ((1 << (S + 1)) - 1))) {                         \
    ValueJoin::join(functor, TD, (TD - (value_count << S))); \
  }

#define BLOCK_SCAN_STEP(TD, N, S)                            \
  if (N == (1 << S)) {                                       \
    ValueJoin::join(functor, TD, (TD - (value_count << S))); \
  }

  const unsigned rtid_intra      = threadIdx.y ^ BlockSizeMask;
  const pointer_type tdata_intra = base_data + value_count * threadIdx.y;

  {  // Intra-warp reduction:
    __syncwarp(0xffffffff);
    BLOCK_REDUCE_STEP(rtid_intra, tdata_intra, 0)
    __syncwarp(0xffffffff);
    BLOCK_REDUCE_STEP(rtid_intra, tdata_intra, 1)
    __syncwarp(0xffffffff);
    BLOCK_REDUCE_STEP(rtid_intra, tdata_intra, 2)
    __syncwarp(0xffffffff);
    BLOCK_REDUCE_STEP(rtid_intra, tdata_intra, 3)
    __syncwarp(0xffffffff);
    BLOCK_REDUCE_STEP(rtid_intra, tdata_intra, 4)
    __syncwarp(0xffffffff);
  }

  __syncthreads();  // Wait for all warps to reduce

  {  // Inter-warp reduce-scan by a single warp to avoid extra synchronizations
    const unsigned rtid_inter = (threadIdx.y ^ BlockSizeMask)
                                << CudaTraits::WarpIndexShift;

    unsigned inner_mask = __ballot_sync(0xffffffff, (rtid_inter < blockDim.y));
    if (rtid_inter < blockDim.y) {
      const pointer_type tdata_inter =
          base_data + value_count * (rtid_inter ^ BlockSizeMask);

      if ((1 << 5) < BlockSizeMask) {
        __syncwarp(inner_mask);
        BLOCK_REDUCE_STEP(rtid_inter, tdata_inter, 5)
      }
      if ((1 << 6) < BlockSizeMask) {
        __syncwarp(inner_mask);
        BLOCK_REDUCE_STEP(rtid_inter, tdata_inter, 6)
      }
      if ((1 << 7) < BlockSizeMask) {
        __syncwarp(inner_mask);
        BLOCK_REDUCE_STEP(rtid_inter, tdata_inter, 7)
      }
      if ((1 << 8) < BlockSizeMask) {
        __syncwarp(inner_mask);
        BLOCK_REDUCE_STEP(rtid_inter, tdata_inter, 8)
      }
      if ((1 << 9) < BlockSizeMask) {
        __syncwarp(inner_mask);
        BLOCK_REDUCE_STEP(rtid_inter, tdata_inter, 9)
      }

      if (DoScan) {
        int n =
            (rtid_inter & 32)
                ? 32
                : ((rtid_inter & 64)
                       ? 64
                       : ((rtid_inter & 128) ? 128
                                             : ((rtid_inter & 256) ? 256 : 0)));

        if (!(rtid_inter + n < blockDim.y)) n = 0;

        __syncwarp(inner_mask);
        BLOCK_SCAN_STEP(tdata_inter, n, 8)
        __syncwarp(inner_mask);
        BLOCK_SCAN_STEP(tdata_inter, n, 7)
        __syncwarp(inner_mask);
        BLOCK_SCAN_STEP(tdata_inter, n, 6)
        __syncwarp(inner_mask);
        BLOCK_SCAN_STEP(tdata_inter, n, 5)
      }
    }
  }

  __syncthreads();  // Wait for inter-warp reduce-scan to complete

  if (DoScan) {
    int n =
        (rtid_intra & 1)
            ? 1
            : ((rtid_intra & 2)
                   ? 2
                   : ((rtid_intra & 4)
                          ? 4
                          : ((rtid_intra & 8) ? 8
                                              : ((rtid_intra & 16) ? 16 : 0))));

    if (!(rtid_intra + n < blockDim.y)) n = 0;
    __syncwarp(0xffffffff);
    BLOCK_SCAN_STEP(tdata_intra, n, 4) __threadfence_block();
    __syncwarp(0xffffffff);
    BLOCK_SCAN_STEP(tdata_intra, n, 3) __threadfence_block();
    __syncwarp(0xffffffff);
    BLOCK_SCAN_STEP(tdata_intra, n, 2) __threadfence_block();
    __syncwarp(0xffffffff);
    BLOCK_SCAN_STEP(tdata_intra, n, 1) __threadfence_block();
    __syncwarp(0xffffffff);
    BLOCK_SCAN_STEP(tdata_intra, n, 0) __threadfence_block();
    __syncwarp(0xffffffff);
  }

#undef BLOCK_SCAN_STEP
#undef BLOCK_REDUCE_STEP
}

//----------------------------------------------------------------------------
/**\brief  Input value-per-thread starting at 'shared_data'.
 *         Reduction value at last thread's location.
 *
 *  If 'DoScan' then write blocks' scan values and block-groups' scan values.
 *
 *  Global reduce result is in the last threads' 'shared_data' location.
 */

template <bool DoScan, class FunctorType, class ArgTag,
          class SizeType = Cuda::size_type>
__device__ bool cuda_single_inter_block_reduce_scan2(
    const FunctorType& functor, const Cuda::size_type block_id,
    const Cuda::size_type block_count, SizeType* const shared_data,
    SizeType* const global_data, Cuda::size_type* const global_flags) {
  using size_type   = SizeType;
  using ValueTraits = FunctorValueTraits<FunctorType, ArgTag>;
  using ValueJoin   = FunctorValueJoin<FunctorType, ArgTag>;
  using ValueInit   = FunctorValueInit<FunctorType, ArgTag>;
  using ValueOps    = FunctorValueOps<FunctorType, ArgTag>;

  using pointer_type = typename ValueTraits::pointer_type;

  // '__ffs' = position of the least significant bit set to 1.
  // 'blockDim.y' is guaranteed to be a power of two so this
  // is the integral shift value that can replace an integral divide.
  const unsigned BlockSizeShift = __ffs(blockDim.y) - 1;
  const unsigned BlockSizeMask  = blockDim.y - 1;

  // Must have power of two thread count
  if (BlockSizeMask & blockDim.y) {
    Kokkos::abort(
        "Cuda::cuda_single_inter_block_reduce_scan requires power-of-two "
        "blockDim");
  }

  const integral_nonzero_constant<size_type, ValueTraits::StaticValueSize /
                                                 sizeof(size_type)>
      word_count(ValueTraits::value_size(functor) / sizeof(size_type));

  // Reduce the accumulation for the entire block.
  cuda_intra_block_reduce_scan<false, FunctorType, ArgTag>(
      functor, pointer_type(shared_data));

  {
    // Write accumulation total to global scratch space.
    // Accumulation total is the last thread's data.
    size_type* const shared = shared_data + word_count.value * BlockSizeMask;
    size_type* const global = global_data + word_count.value * block_id;

    for (int i = int(threadIdx.y); i < int(word_count.value);
         i += int(blockDim.y)) {
      global[i] = shared[i];
    }
  }
  __threadfence();

  // Contributing blocks note that their contribution has been completed via an
  // atomic-increment flag If this block is not the last block to contribute to
  // this group then the block is done.
  const bool is_last_block = !__syncthreads_or(
      threadIdx.y
          ? 0
          : (1 + atomicInc(global_flags, block_count - 1) < block_count));

  if (is_last_block) {
    const size_type b =
        (long(block_count) * long(threadIdx.y)) >> BlockSizeShift;
    const size_type e =
        (long(block_count) * long(threadIdx.y + 1)) >> BlockSizeShift;

    {
      void* const shared_ptr = shared_data + word_count.value * threadIdx.y;
      /* reference_type shared_value = */ ValueInit::init(functor, shared_ptr);

      for (size_type i = b; i < e; ++i) {
        ValueJoin::join(functor, shared_ptr,
                        global_data + word_count.value * i);
      }
    }

    cuda_intra_block_reduce_scan<DoScan, FunctorType, ArgTag>(
        functor, pointer_type(shared_data));

    if (DoScan) {
      size_type* const shared_value =
          shared_data +
          word_count.value * (threadIdx.y ? threadIdx.y - 1 : blockDim.y);

      if (!threadIdx.y) {
        ValueInit::init(functor, shared_value);
      }

      // Join previous inclusive scan value to each member
      for (size_type i = b; i < e; ++i) {
        size_type* const global_value = global_data + word_count.value * i;
        ValueJoin::join(functor, shared_value, global_value);
        ValueOps ::copy(functor, global_value, shared_value);
      }
    }
  }

  return is_last_block;
}

template <bool DoScan, class FunctorType, class ArgTag,
          class SizeType = Cuda::size_type>
__device__ bool cuda_single_inter_block_reduce_scan(
    const FunctorType& functor, const Cuda::size_type block_id,
    const Cuda::size_type block_count, SizeType* const shared_data,
    SizeType* const global_data, Cuda::size_type* const global_flags) {
  using ValueTraits = FunctorValueTraits<FunctorType, ArgTag>;
  if (!DoScan && ValueTraits::StaticValueSize > 0)
    return Kokkos::Impl::CudaReductionsFunctor<
        FunctorType, ArgTag, false, (ValueTraits::StaticValueSize > 16)>::
        scalar_inter_block_reduction(functor, block_id, block_count,
                                     shared_data, global_data, global_flags);
  else
    return cuda_single_inter_block_reduce_scan2<DoScan, FunctorType, ArgTag>(
        functor, block_id, block_count, shared_data, global_data, global_flags);
}

// Size in bytes required for inter block reduce or scan
template <bool DoScan, class FunctorType, class ArgTag>
inline unsigned cuda_single_inter_block_reduce_scan_shmem(
    const FunctorType& functor, const unsigned BlockSize) {
  return (BlockSize + 2) *
         Impl::FunctorValueTraits<FunctorType, ArgTag>::value_size(functor);
}

}  // namespace Impl
}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #if defined(KOKKOS_ENABLE_CUDA) */
#endif /* KOKKOS_CUDA_REDUCESCAN_HPP */
