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
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_HIP_REDUCESCAN_HPP
#define KOKKOS_HIP_REDUCESCAN_HPP

#include <Kokkos_Macros.hpp>

#if defined(KOKKOS_ENABLE_HIP) && defined(__HIPCC__)

namespace Kokkos {
namespace Impl {
template <class FunctorType, class ArgTag, bool DoScan, bool UseShfl>
struct HIPReductionsFunctor;

template <typename FunctorType, typename ArgTag>
struct HIPReductionsFunctor<FunctorType, ArgTag, false, false> {
  using ValueTraits  = FunctorValueTraits<FunctorType, ArgTag>;
  using ValueJoin    = FunctorValueJoin<FunctorType, ArgTag>;
  using ValueInit    = FunctorValueInit<FunctorType, ArgTag>;
  using ValueOps     = FunctorValueOps<FunctorType, ArgTag>;
  using pointer_type = typename ValueTraits::pointer_type;
  using Scalar       = typename ValueTraits::value_type;

  __device__ static inline void scalar_intra_warp_reduction(
      FunctorType const& functor,
      Scalar* value,           // Contribution
      bool const skip_vector,  // Skip threads if Kokkos vector lanes are not
                               // part of the reduction
      int const width)         // How much of the warp participates
  {
    int const lane_id = (hipThreadIdx_y * hipBlockDim_x + hipThreadIdx_x) %
                        ::Kokkos::Experimental::Impl::HIPTraits::WarpSize;
    for (int delta = skip_vector ? hipBlockDim_x : 1; delta < width;
         delta *= 2) {
      if (lane_id + delta < ::Kokkos::Experimental::Impl::HIPTraits::WarpSize) {
        ValueJoin::join(functor, value, value + delta);
      }
    }
    *value = *(value - lane_id);
  }

  __device__ static inline void scalar_intra_block_reduction(
      FunctorType const& functor, Scalar value, bool const skip, Scalar* result,
      int const shared_elements, Scalar* shared_team_buffer_element) {
    int const warp_id = (hipThreadIdx_y * hipBlockDim_x) /
                        ::Kokkos::Experimental::Impl::HIPTraits::WarpSize;
    Scalar* const my_shared_team_buffer_element =
        shared_team_buffer_element + hipThreadIdx_y * hipBlockDim_x +
        hipThreadIdx_x;
    *my_shared_team_buffer_element = value;
    // Warp Level Reduction, ignoring Kokkos vector entries
    scalar_intra_warp_reduction(
        functor, my_shared_team_buffer_element, skip,
        ::Kokkos::Experimental::Impl::HIPTraits::WarpSize);
    // Wait for every warp to be done before using one warp to do final cross
    // warp reduction
    __syncthreads();

    if (warp_id == 0) {
      const unsigned int delta =
          (hipThreadIdx_y * hipBlockDim_x + hipThreadIdx_x) *
          ::Kokkos::Experimental::Impl::HIPTraits::WarpSize;
      if (delta < hipBlockDim_x * hipBlockDim_y)
        *my_shared_team_buffer_element = shared_team_buffer_element[delta];
      scalar_intra_warp_reduction(
          functor, my_shared_team_buffer_element, false,
          hipBlockDim_x * hipBlockDim_y /
              ::Kokkos::Experimental::Impl::HIPTraits::WarpSize);
      if (hipThreadIdx_x + hipThreadIdx_y == 0)
        *result = *shared_team_buffer_element;
    }
  }

  __device__ static inline bool scalar_inter_block_reduction(
      FunctorType const& functor,
      ::Kokkos::Experimental::HIP::size_type const block_id,
      ::Kokkos::Experimental::HIP::size_type const block_count,
      ::Kokkos::Experimental::HIP::size_type* const shared_data,
      ::Kokkos::Experimental::HIP::size_type* const global_data,
      ::Kokkos::Experimental::HIP::size_type* const global_flags) {
    Scalar* const global_team_buffer_element =
        reinterpret_cast<Scalar*>(global_data);
    Scalar* const my_global_team_buffer_element =
        global_team_buffer_element + hipBlockIdx_x;
    Scalar* shared_team_buffer_elements =
        reinterpret_cast<Scalar*>(shared_data);
    Scalar value        = shared_team_buffer_elements[hipThreadIdx_y];
    int shared_elements = (hipBlockDim_x * hipBlockDim_y) /
                          ::Kokkos::Experimental::Impl::HIPTraits::WarpSize;
    int global_elements = block_count;
    __syncthreads();

    // Do the scalar reduction inside each block
    scalar_intra_block_reduction(functor, value, true,
                                 my_global_team_buffer_element, shared_elements,
                                 shared_team_buffer_elements);
    __syncthreads();

    // Use the last block that is done to do the do the reduction across the
    // block
    __shared__ unsigned int num_teams_done;
    if (hipThreadIdx_x + hipThreadIdx_y == 0) {
      __threadfence();
      num_teams_done = Kokkos::atomic_fetch_add(global_flags, 1) + 1;
    }
    bool is_last_block = false;
    // TODO HIP does not support syncthreads_or. That's why we need to make
    // num_teams_done __shared__
    // if (__syncthreads_or(num_teams_done == hipGridDim_x)) {*/
    __syncthreads();
    if (num_teams_done == hipGridDim_x) {
      is_last_block = true;
      *global_flags = 0;
      ValueInit::init(functor, &value);
      for (int i = hipThreadIdx_y * hipBlockDim_x + hipThreadIdx_x;
           i < global_elements; i += hipBlockDim_x * hipBlockDim_y) {
        ValueJoin::join(functor, &value, &global_team_buffer_element[i]);
      }
      scalar_intra_block_reduction(
          functor, value, false,
          shared_team_buffer_elements + (hipBlockDim_y - 1), shared_elements,
          shared_team_buffer_elements);
    }

    return is_last_block;
  }
};

//----------------------------------------------------------------------------
/*
 *  Algorithmic constraints:
 *   (a) hipBlockDim_y is a power of two
 *   (b) hipBlockDim_y <= 1024
 *   (c) hipBlockDim_x == hipBlockDim_z == 1
 */

template <bool DoScan, class FunctorType, class ArgTag>
__device__ void hip_intra_block_reduce_scan(
    FunctorType const& functor,
    typename FunctorValueTraits<FunctorType, ArgTag>::pointer_type const
        base_data) {
  using ValueTraits = FunctorValueTraits<FunctorType, ArgTag>;
  using ValueJoin   = FunctorValueJoin<FunctorType, ArgTag>;

  using pointer_type = typename ValueTraits::pointer_type;

  unsigned int const value_count   = ValueTraits::value_count(functor);
  unsigned int const BlockSizeMask = hipBlockDim_y - 1;

  // Must have power of two thread count

  if (BlockSizeMask & hipBlockDim_y) {
    Kokkos::abort("HIP::hip_intra_block_scan requires power-of-two blockDim");
  }

  auto block_reduce_step =
      [&functor, value_count](int const R, pointer_type const TD, int const S) {
        if (!(R & ((1 << (S + 1)) - 1))) {
          ValueJoin::join(functor, TD, (TD - (value_count << S)));
        }
      };

  auto block_scan_step = [&functor, value_count](pointer_type const TD,
                                                 int const N, int const S) {
    if (N == (1 << S)) {
      ValueJoin::join(functor, TD, (TD - (value_count << S)));
    }
  };

  const unsigned rtid_intra      = hipThreadIdx_y ^ BlockSizeMask;
  const pointer_type tdata_intra = base_data + value_count * hipThreadIdx_y;

  {  // Intra-warp reduction:
    block_reduce_step(rtid_intra, tdata_intra, 0);
    block_reduce_step(rtid_intra, tdata_intra, 1);
    block_reduce_step(rtid_intra, tdata_intra, 2);
    block_reduce_step(rtid_intra, tdata_intra, 3);
    block_reduce_step(rtid_intra, tdata_intra, 4);
    // TODO check: one more than CUDA because bigger warpsize
    block_reduce_step(rtid_intra, tdata_intra, 5);
  }

  __syncthreads();  // Wait for all warps to reduce

  {  // Inter-warp reduce-scan by a single warp to avoid extra synchronizations
    unsigned int const rtid_inter =
        (hipThreadIdx_y ^ BlockSizeMask)
        << ::Kokkos::Experimental::Impl::HIPTraits::WarpIndexShift;

    if (rtid_inter < hipBlockDim_y) {
      pointer_type const tdata_inter =
          base_data + value_count * (rtid_inter ^ BlockSizeMask);

      if ((1 << 6) < BlockSizeMask) {
        block_reduce_step(rtid_inter, tdata_inter, 6);
      }
      if ((1 << 7) < BlockSizeMask) {
        block_reduce_step(rtid_inter, tdata_inter, 7);
      }
      if ((1 << 8) < BlockSizeMask) {
        block_reduce_step(rtid_inter, tdata_inter, 8);
      }
      if ((1 << 9) < BlockSizeMask) {
        block_reduce_step(rtid_inter, tdata_inter, 9);
      }
      if ((1 << 10) < BlockSizeMask) {
        block_reduce_step(rtid_inter, tdata_inter, 10);
      }

      int constexpr warp_size =
          ::Kokkos::Experimental::Impl::HIPTraits::WarpSize;
      if (DoScan) {
        int n =
            (rtid_inter & warp_size)
                ? warp_size
                : ((rtid_inter & 2 * warp_size)
                       ? 2 * warp_size
                       : ((rtid_inter & 4 * warp_size)
                              ? 4 * warp_size
                              : ((rtid_inter & 8 * warp_size) ? 8 * warp_size
                                                              : 0)));

        if (!(rtid_inter + n < hipBlockDim_y)) n = 0;

        block_scan_step(tdata_inter, n, 9);
        block_scan_step(tdata_inter, n, 8);
        block_scan_step(tdata_inter, n, 7);
        block_scan_step(tdata_inter, n, 6);
      }
    }
  }

  __syncthreads();  // Wait for inter-warp reduce-scan to complete

  // TODO check: increased size because of warpsize difference
  // TODO the runtime check could be avoided
  if (DoScan) {
    int n = (rtid_intra & 1)
                ? 1
                : ((rtid_intra & 2)
                       ? 2
                       : ((rtid_intra & 4)
                              ? 4
                              : ((rtid_intra & 8)
                                     ? 8
                                     : ((rtid_intra & 16)
                                            ? 16
                                            : ((rtid_intra & 32) ? 32 : 0)))));

    if (!(rtid_intra + n < hipBlockDim_y)) n = 0;
    block_scan_step(tdata_intra, n, 5);
    __threadfence_block();
    block_scan_step(tdata_intra, n, 4);
    __threadfence_block();
    block_scan_step(tdata_intra, n, 3);
    __threadfence_block();
    block_scan_step(tdata_intra, n, 2);
    __threadfence_block();
    block_scan_step(tdata_intra, n, 1);
    __threadfence_block();
    block_scan_step(tdata_intra, n, 0);
    __threadfence_block();
  }
}

//----------------------------------------------------------------------------
/**\brief  Input value-per-thread starting at 'shared_data'.
 *         Reduction value at last thread's location.
 *
 *  If 'DoScan' then write blocks' scan values and block-groups' scan values.
 *
 *  Global reduce result is in the last threads' 'shared_data' location.
 */

template <bool DoScan, class FunctorType, class ArgTag>
__device__ bool hip_single_inter_block_reduce_scan2(
    FunctorType const& functor,
    ::Kokkos::Experimental::HIP::size_type const block_id,
    ::Kokkos::Experimental::HIP::size_type const block_count,
    ::Kokkos::Experimental::HIP::size_type* const shared_data,
    ::Kokkos::Experimental::HIP::size_type* const global_data,
    ::Kokkos::Experimental::HIP::size_type* const global_flags) {
  using size_type   = ::Kokkos::Experimental::HIP::size_type;
  using ValueTraits = FunctorValueTraits<FunctorType, ArgTag>;
  using ValueJoin   = FunctorValueJoin<FunctorType, ArgTag>;
  using ValueInit   = FunctorValueInit<FunctorType, ArgTag>;
  using ValueOps    = FunctorValueOps<FunctorType, ArgTag>;

  using pointer_type = typename ValueTraits::pointer_type;

  // '__ffs' = position of the least significant bit set to 1.
  // 'hipBlockDim_y' is guaranteed to be a power of two so this
  // is the integral shift value that can replace an integral divide.
  unsigned int const BlockSizeShift = __ffs(hipBlockDim_y) - 1;
  unsigned int const BlockSizeMask  = hipBlockDim_y - 1;

  // Must have power of two thread count
  if (BlockSizeMask & hipBlockDim_y) {
    Kokkos::abort(
        "HIP::hip_single_inter_block_reduce_scan requires power-of-two "
        "blockDim");
  }

  integral_nonzero_constant<size_type, ValueTraits::StaticValueSize /
                                           sizeof(size_type)> const
      word_count(ValueTraits::value_size(functor) / sizeof(size_type));

  // Reduce the accumulation for the entire block.
  hip_intra_block_reduce_scan<false, FunctorType, ArgTag>(
      functor, pointer_type(shared_data));

  {
    // Write accumulation total to global scratch space.
    // Accumulation total is the last thread's data.
    size_type* const shared = shared_data + word_count.value * BlockSizeMask;
    size_type* const global = global_data + word_count.value * block_id;

    for (size_t i = hipThreadIdx_y; i < word_count.value; i += hipBlockDim_y) {
      global[i] = shared[i];
    }
  }

  // Contributing blocks note that their contribution has been completed via an
  // atomic-increment flag If this block is not the last block to contribute to
  // this group then the block is done.
  // TODO __syncthreads_or is not supported by HIP yet.
  // const bool is_last_block = !__syncthreads_or(
  //    threadIdx.y
  //        ? 0
  //        : (1 + atomicInc(global_flags, block_count - 1) < block_count));
  __shared__ int n_done;
  n_done = 0;
  __syncthreads();
  if (hipThreadIdx_y == 0) {
    __threadfence();
    n_done = 1 + atomicInc(global_flags, block_count - 1);
  }
  __syncthreads();
  bool const is_last_block = (n_done == static_cast<int>(block_count));

  if (is_last_block) {
    size_type const b = (static_cast<long long int>(block_count) *
                         static_cast<long long int>(hipThreadIdx_y)) >>
                        BlockSizeShift;
    size_type const e = (static_cast<long long int>(block_count) *
                         static_cast<long long int>(hipThreadIdx_y + 1)) >>
                        BlockSizeShift;

    {
      void* const shared_ptr = shared_data + word_count.value * hipThreadIdx_y;
      /* reference_type shared_value = */ ValueInit::init(functor, shared_ptr);

      for (size_type i = b; i < e; ++i) {
        ValueJoin::join(functor, shared_ptr,
                        global_data + word_count.value * i);
      }
    }

    hip_intra_block_reduce_scan<DoScan, FunctorType, ArgTag>(
        functor, pointer_type(shared_data));

    if (DoScan) {
      size_type* const shared_value =
          shared_data + word_count.value * (hipThreadIdx_y ? hipThreadIdx_y - 1
                                                           : hipBlockDim_y);

      if (!hipThreadIdx_y) {
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

template <bool DoScan, typename FunctorType, typename ArgTag>
__device__ bool hip_single_inter_block_reduce_scan(
    FunctorType const& functor,
    ::Kokkos::Experimental::HIP::size_type const block_id,
    ::Kokkos::Experimental::HIP::size_type const block_count,
    ::Kokkos::Experimental::HIP::size_type* const shared_data,
    ::Kokkos::Experimental::HIP::size_type* const global_data,
    ::Kokkos::Experimental::HIP::size_type* const global_flags) {
  using ValueTraits = FunctorValueTraits<FunctorType, ArgTag>;
  if (!DoScan && /*FIXME*/ (bool)ValueTraits::StaticValueSize)
    // TODO For now we don't use shuffle
    // return Kokkos::Impl::HIPReductionsFunctor<
    //    FunctorType, ArgTag, false, (ValueTraits::StaticValueSize > 16)>::
    //    scalar_inter_block_reduction(functor, block_id, block_count,
    //                                 shared_data, global_data, global_flags);
    return Kokkos::Impl::HIPReductionsFunctor<
        FunctorType, ArgTag, false,
        false>::scalar_inter_block_reduction(functor, block_id, block_count,
                                             shared_data, global_data,
                                             global_flags);
  else {
    return hip_single_inter_block_reduce_scan2<DoScan, FunctorType, ArgTag>(
        functor, block_id, block_count, shared_data, global_data, global_flags);
  }
}

// Size in bytes required for inter block reduce or scan
template <bool DoScan, class FunctorType, class ArgTag>
inline unsigned hip_single_inter_block_reduce_scan_shmem(
    const FunctorType& functor, const unsigned BlockSize) {
  return (BlockSize + 2) *
         Impl::FunctorValueTraits<FunctorType, ArgTag>::value_size(functor);
}

}  // namespace Impl
}  // namespace Kokkos

#endif

#endif
