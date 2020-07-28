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

#ifndef KOKKOS_HIP_SHUFFLE_REDUCE_HPP
#define KOKKOS_HIP_SHUFFLE_REDUCE_HPP

#include <Kokkos_Macros.hpp>

#if defined(__HIPCC__)

#include <HIP/Kokkos_HIP_Vectorization.hpp>

#include <climits>

namespace Kokkos {
namespace Impl {

/* Algorithmic constraints:
 *   (a) threads with the same threadIdx.x have same value
 *   (b) blockDim.x == power of two
 *   (x) blockDim.z == 1
 */
template <typename ValueType, typename JoinOp,
          typename std::enable_if<!Kokkos::is_reducer<ValueType>::value,
                                  int>::type = 0>
__device__ inline void hip_intra_warp_shuffle_reduction(
    ValueType& result, JoinOp const& join,
    uint32_t const max_active_thread = blockDim.y) {
  unsigned int shift = 1;

  // Reduce over values from threads with different threadIdx.y
  unsigned int constexpr warp_size =
      Kokkos::Experimental::Impl::HIPTraits::WarpSize;
  while (blockDim.x * shift < warp_size) {
    ValueType const tmp =
        Kokkos::Experimental::shfl_down(result, blockDim.x * shift, warp_size);
    // Only join if upper thread is active (this allows non power of two for
    // blockDim.y)
    if (threadIdx.y + shift < max_active_thread) {
      join(result, tmp);
    }
    shift *= 2;
    // Not sure why there is a race condition here but we need to wait for the
    // join operation to be finished to perform the next shuffle. Note that the
    // problem was also found in the CUDA backend with CUDA clang
    // (https://github.com/kokkos/kokkos/issues/941)
    __syncthreads();
  }

  // Broadcast the result to all the threads in the warp
  result = Kokkos::Experimental::shfl(result, 0, warp_size);
}

template <typename ValueType, typename JoinOp,
          typename std::enable_if<!Kokkos::is_reducer<ValueType>::value,
                                  int>::type = 0>
__device__ inline void hip_inter_warp_shuffle_reduction(
    ValueType& value, const JoinOp& join,
    const int max_active_thread = blockDim.y) {
  unsigned int constexpr warp_size =
      Kokkos::Experimental::Impl::HIPTraits::WarpSize;
  int constexpr step_width = 8;
  // Depending on the ValueType __shared__ memory must be aligned up to 8 byte
  // boundaries. The reason not to use ValueType directly is that for types with
  // constructors it could lead to race conditions.
  __shared__ double sh_result[(sizeof(ValueType) + 7) / 8 * step_width];
  ValueType* result = reinterpret_cast<ValueType*>(&sh_result);
  int const step    = warp_size / blockDim.x;
  int shift         = step_width;
  // Skip the code below if  threadIdx.y % step != 0
  int const id = threadIdx.y % step == 0 ? threadIdx.y / step : INT_MAX;
  if (id < step_width) {
    result[id] = value;
  }
  __syncthreads();
  while (shift <= max_active_thread / step) {
    if (shift <= id && shift + step_width > id && threadIdx.x == 0) {
      join(result[id % step_width], value);
    }
    __syncthreads();
    shift += step_width;
  }

  value = result[0];
  for (int i = 1; (i * step < max_active_thread) && (i < step_width); ++i)
    join(value, result[i]);
}

template <typename ValueType, typename JoinOp,
          typename std::enable_if<!Kokkos::is_reducer<ValueType>::value,
                                  int>::type = 0>
__device__ inline void hip_intra_block_shuffle_reduction(
    ValueType& value, JoinOp const& join,
    int const max_active_thread = blockDim.y) {
  hip_intra_warp_shuffle_reduction(value, join, max_active_thread);
  hip_inter_warp_shuffle_reduction(value, join, max_active_thread);
}

template <class FunctorType, class JoinOp, class ArgTag = void>
__device__ inline bool hip_inter_block_shuffle_reduction(
    typename FunctorValueTraits<FunctorType, ArgTag>::reference_type value,
    typename FunctorValueTraits<FunctorType, ArgTag>::reference_type neutral,
    JoinOp const& join,
    Kokkos::Experimental::HIP::size_type* const m_scratch_space,
    typename FunctorValueTraits<FunctorType,
                                ArgTag>::pointer_type const /*result*/,
    Kokkos::Experimental::HIP::size_type* const m_scratch_flags,
    int const max_active_thread = blockDim.y) {
  using pointer_type =
      typename FunctorValueTraits<FunctorType, ArgTag>::pointer_type;
  using value_type =
      typename FunctorValueTraits<FunctorType, ArgTag>::value_type;

  // Do the intra-block reduction with shfl operations for the intra warp
  // reduction and static shared memory for the inter warp reduction
  hip_intra_block_shuffle_reduction(value, join, max_active_thread);

  int const id = threadIdx.y * blockDim.x + threadIdx.x;

  // One thread in the block writes block result to global scratch_memory
  if (id == 0) {
    pointer_type global =
        reinterpret_cast<pointer_type>(m_scratch_space) + blockIdx.x;
    *global = value;
  }

  // One warp of last block performs inter block reduction through loading the
  // block values from global scratch_memory
  bool last_block = false;
  __threadfence();
  __syncthreads();
  int constexpr warp_size = Kokkos::Experimental::Impl::HIPTraits::WarpSize;
  if (id < warp_size) {
    Kokkos::Experimental::HIP::size_type count;

    // Figure out whether this is the last block
    if (id == 0) count = Kokkos::atomic_fetch_add(m_scratch_flags, 1);
    count = Kokkos::Experimental::shfl(count, 0, warp_size);

    // Last block does the inter block reduction
    if (count == gridDim.x - 1) {
      // set flag back to zero
      if (id == 0) *m_scratch_flags = 0;
      last_block = true;
      value      = neutral;

      pointer_type const volatile global =
          reinterpret_cast<pointer_type>(m_scratch_space);

      // Reduce all global values with splitting work over threads in one warp
      const int step_size = blockDim.x * blockDim.y < warp_size
                                ? blockDim.x * blockDim.y
                                : warp_size;
      for (int i = id; i < static_cast<int>(gridDim.x); i += step_size) {
        value_type tmp = global[i];
        join(value, tmp);
      }

      // Perform shfl reductions within the warp only join if contribution is
      // valid (allows gridDim.x non power of two and <warp_size)
      for (unsigned int i = 1; i < warp_size; i *= 2) {
        if ((blockDim.x * blockDim.y) > i) {
          value_type tmp = Kokkos::Experimental::shfl_down(value, i, warp_size);
          if (id + i < gridDim.x) join(value, tmp);
        }
        __syncthreads();
      }
    }
  }
  // The last block has in its thread=0 the global reduction value through
  // "value"
  return last_block;
}

// We implemente the same functions as above but the user provide a Reducer
// instead of JoinOP
template <typename ReducerType,
          typename std::enable_if<Kokkos::is_reducer<ReducerType>::value,
                                  int>::type = 0>
__device__ inline void hip_intra_warp_shuffle_reduction(
    const ReducerType& reducer, typename ReducerType::value_type& result,
    const uint32_t max_active_thread = blockDim.y) {
  using ValueType = typename ReducerType::value_type;
  auto join_op    = [&](ValueType& result, ValueType const& tmp) {
    reducer.join(result, tmp);
  };
  hip_intra_warp_shuffle_reduction(result, join_op, max_active_thread);

  reducer.reference() = result;
}

template <typename ReducerType,
          typename std::enable_if<Kokkos::is_reducer<ReducerType>::value,
                                  int>::type = 0>
__device__ inline void hip_inter_warp_shuffle_reduction(
    ReducerType const& reducer, typename ReducerType::value_type value,
    int const max_active_thread = blockDim.y) {
  using ValueType = typename ReducerType::value_type;
  auto join_op    = [&](ValueType& a, ValueType& b) { reducer.join(a, b); };
  hip_inter_warp_shuffle_reduction(value, join_op, max_active_thread);

  reducer.reference() = value;
}

template <typename ReducerType,
          typename std::enable_if<Kokkos::is_reducer<ReducerType>::value,
                                  int>::type = 0>
__device__ inline void hip_intra_block_shuffle_reduction(
    ReducerType const& reducer, typename ReducerType::value_type value,
    int const max_active_thread = blockDim.y) {
  hip_intra_warp_shuffle_reduction(reducer, value, max_active_thread);
  hip_inter_warp_shuffle_reduction(reducer, value, max_active_thread);
}

template <typename ReducerType,
          typename std::enable_if<Kokkos::is_reducer<ReducerType>::value,
                                  int>::type = 0>
__device__ inline void hip_intra_block_shuffle_reduction(
    ReducerType const& reducer, int const max_active_thread = blockDim.y) {
  hip_intra_block_shuffle_reduction(reducer, reducer.reference(),
                                    max_active_thread);
}

template <typename ReducerType,
          typename std::enable_if<Kokkos::is_reducer<ReducerType>::value,
                                  int>::type = 0>
__device__ inline bool hip_inter_block_shuffle_reduction(
    ReducerType const& reducer,
    Kokkos::Experimental::HIP::size_type* const m_scratch_space,
    Kokkos::Experimental::HIP::size_type* const m_scratch_flags,
    int const max_active_thread = blockDim.y) {
  using pointer_type = typename ReducerType::value_type*;
  using value_type   = typename ReducerType::value_type;

  // Do the intra-block reduction with shfl operations for the intra warp
  // reduction and static shared memory for the inter warp reduction
  hip_intra_block_shuffle_reduction(reducer, max_active_thread);

  value_type value = reducer.reference();

  int const id = threadIdx.y * blockDim.x + threadIdx.x;

  // One thread in the block writes block result to global scratch_memory
  if (id == 0) {
    pointer_type global =
        reinterpret_cast<pointer_type>(m_scratch_space) + blockIdx.x;
    *global = value;
  }

  // One warp of last block performs inter block reduction through loading the
  // block values from global scratch_memory
  bool last_block = false;

  __threadfence();
  __syncthreads();
  int constexpr warp_size = Kokkos::Experimental::Impl::HIPTraits::WarpSize;
  if (id < warp_size) {
    Kokkos::Experimental::HIP::size_type count;

    // Figure out whether this is the last block
    if (id == 0) count = Kokkos::atomic_fetch_add(m_scratch_flags, 1);
    count = Kokkos::Experimental::shfl(count, 0, warp_size);

    // Last block does the inter block reduction
    if (count == gridDim.x - 1) {
      // Set flag back to zero
      if (id == 0) *m_scratch_flags = 0;
      last_block = true;
      reducer.init(value);

      pointer_type const volatile global =
          reinterpret_cast<pointer_type>(m_scratch_space);

      // Reduce all global values with splitting work over threads in one warp
      int const step_size = blockDim.x * blockDim.y < warp_size
                                ? blockDim.x * blockDim.y
                                : warp_size;
      for (int i = id; i < static_cast<int>(gridDim.x); i += step_size) {
        value_type tmp = global[i];
        reducer.join(value, tmp);
      }

      // Perform shfl reductions within the warp only join if contribution is
      // valid (allows gridDim.x non power of two and <warp_size)
      for (unsigned int i = 1; i < warp_size; i *= 2) {
        if ((blockDim.x * blockDim.y) > i) {
          value_type tmp = Kokkos::Experimental::shfl_down(value, i, warp_size);
          if (id + i < gridDim.x) reducer.join(value, tmp);
        }
        __syncthreads();
      }
    }
  }

  // The last block has in its thread = 0 the global reduction value through
  // "value"
  return last_block;
}
}  // namespace Impl
}  // namespace Kokkos

#endif

#endif
