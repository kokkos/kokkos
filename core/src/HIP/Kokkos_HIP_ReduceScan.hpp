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

template <bool DoScan, typename FunctorType, typename ArgTag>
__device__ bool hip_single_inter_block_reduce_scan(
    FunctorType const& functor,
    ::Kokkos::Experimental::HIP::size_type const block_id,
    ::Kokkos::Experimental::HIP::size_type const block_count,
    ::Kokkos::Experimental::HIP::size_type* const shared_data,
    ::Kokkos::Experimental::HIP::size_type* const global_data,
    ::Kokkos::Experimental::HIP::size_type* const global_flags) {
  using ValueTraits = FunctorValueTraits<FunctorType, ArgTag>;
  if (!DoScan && ValueTraits::StaticValueSize)
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
    // TODO implement
    return false;
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
