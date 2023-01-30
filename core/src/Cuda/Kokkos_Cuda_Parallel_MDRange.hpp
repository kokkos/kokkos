//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef KOKKOS_CUDA_PARALLEL_MD_RANGE_HPP
#define KOKKOS_CUDA_PARALLEL_MD_RANGE_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_CUDA)

#include <algorithm>
#include <string>

#include <Kokkos_Parallel.hpp>

#include <Cuda/Kokkos_Cuda_KernelLaunch.hpp>
#include <Cuda/Kokkos_Cuda_ReduceScan.hpp>
#include <Cuda/Kokkos_Cuda_BlockSize_Deduction.hpp>
#include <Kokkos_MinMaxClamp.hpp>

#include <impl/Kokkos_Tools.hpp>
#include <typeinfo>

#include <KokkosExp_MDRangePolicy.hpp>
#include <impl/KokkosExp_IterateTileGPU.hpp>

namespace Kokkos {
namespace Impl {

template <typename ParallelType, typename Policy, typename LaunchBounds>
int max_tile_size_product_helper(const Policy& pol, const LaunchBounds&) {
  cudaFuncAttributes attr =
      CudaParallelLaunch<ParallelType,
                         LaunchBounds>::get_cuda_func_attributes();
  auto const& prop = pol.space().cuda_device_prop();

  // Limits due to registers/SM, MDRange doesn't have
  // shared memory constraints
  int const optimal_block_size =
      cuda_get_opt_block_size_no_shmem(prop, attr, LaunchBounds{});

  // Compute how many blocks of this size we can launch, based on warp
  // constraints
  int const max_warps_per_sm_registers =
      Kokkos::Impl::cuda_max_warps_per_sm_registers(prop, attr);
  int const max_num_threads_from_warps =
      max_warps_per_sm_registers * prop.warpSize;
  int const max_num_blocks = max_num_threads_from_warps / optimal_block_size;

  // Compute the total number of threads
  int const max_threads_per_sm = optimal_block_size * max_num_blocks;

  return std::min(
      max_threads_per_sm,
      static_cast<int>(Kokkos::Impl::CudaTraits::MaxHierarchicalParallelism));
}

template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::MDRangePolicy<Traits...>, Kokkos::Cuda> {
 public:
  using Policy       = Kokkos::MDRangePolicy<Traits...>;
  using functor_type = FunctorType;

 private:
  using RP               = Policy;
  using array_index_type = typename Policy::array_index_type;
  using index_type       = typename Policy::index_type;
  using LaunchBounds     = typename Policy::launch_bounds;

  const FunctorType m_functor;
  const Policy m_rp;

 public:
  template <typename Policy, typename Functor>
  static int max_tile_size_product(const Policy& pol, const Functor&) {
    return max_tile_size_product_helper<ParallelFor>(pol, LaunchBounds{});
  }
  Policy const& get_policy() const { return m_rp; }
  inline __device__ void operator()() const {
    Kokkos::Impl::DeviceIterateTile<Policy::rank, Policy, FunctorType,
                                    typename Policy::work_tag>(m_rp, m_functor)
        .exec_range();
  }

  inline void execute() const {
    if (m_rp.m_num_tiles == 0) return;
    const auto maxblocks = cuda_internal_maximum_grid_count();
    if (RP::rank == 2) {
      const dim3 block(m_rp.m_tile[0], m_rp.m_tile[1], 1);
      KOKKOS_ASSERT(block.x > 0);
      KOKKOS_ASSERT(block.y > 0);
      const dim3 grid(
          std::min<array_index_type>(
              (m_rp.m_upper[0] - m_rp.m_lower[0] + block.x - 1) / block.x,
              maxblocks[0]),
          std::min<array_index_type>(
              (m_rp.m_upper[1] - m_rp.m_lower[1] + block.y - 1) / block.y,
              maxblocks[1]),
          1);
      CudaParallelLaunch<ParallelFor, LaunchBounds>(
          *this, grid, block, 0, m_rp.space().impl_internal_space_instance());
    } else if (RP::rank == 3) {
      const dim3 block(m_rp.m_tile[0], m_rp.m_tile[1], m_rp.m_tile[2]);
      KOKKOS_ASSERT(block.x > 0);
      KOKKOS_ASSERT(block.y > 0);
      KOKKOS_ASSERT(block.z > 0);
      const dim3 grid(
          std::min<array_index_type>(
              (m_rp.m_upper[0] - m_rp.m_lower[0] + block.x - 1) / block.x,
              maxblocks[0]),
          std::min<array_index_type>(
              (m_rp.m_upper[1] - m_rp.m_lower[1] + block.y - 1) / block.y,
              maxblocks[1]),
          std::min<array_index_type>(
              (m_rp.m_upper[2] - m_rp.m_lower[2] + block.z - 1) / block.z,
              maxblocks[2]));
      CudaParallelLaunch<ParallelFor, LaunchBounds>(
          *this, grid, block, 0, m_rp.space().impl_internal_space_instance());
    } else if (RP::rank == 4) {
      // id0,id1 encoded within threadIdx.x; id2 to threadIdx.y; id3 to
      // threadIdx.z
      const dim3 block(m_rp.m_tile[0] * m_rp.m_tile[1], m_rp.m_tile[2],
                       m_rp.m_tile[3]);
      KOKKOS_ASSERT(block.y > 0);
      KOKKOS_ASSERT(block.z > 0);
      const dim3 grid(
          std::min<array_index_type>(m_rp.m_tile_end[0] * m_rp.m_tile_end[1],
                                     maxblocks[0]),
          std::min<array_index_type>(
              (m_rp.m_upper[2] - m_rp.m_lower[2] + block.y - 1) / block.y,
              maxblocks[1]),
          std::min<array_index_type>(
              (m_rp.m_upper[3] - m_rp.m_lower[3] + block.z - 1) / block.z,
              maxblocks[2]));
      CudaParallelLaunch<ParallelFor, LaunchBounds>(
          *this, grid, block, 0, m_rp.space().impl_internal_space_instance());
    } else if (RP::rank == 5) {
      // id0,id1 encoded within threadIdx.x; id2,id3 to threadIdx.y; id4 to
      // threadIdx.z
      const dim3 block(m_rp.m_tile[0] * m_rp.m_tile[1],
                       m_rp.m_tile[2] * m_rp.m_tile[3], m_rp.m_tile[4]);
      KOKKOS_ASSERT(block.z > 0);
      const dim3 grid(
          std::min<array_index_type>(m_rp.m_tile_end[0] * m_rp.m_tile_end[1],
                                     maxblocks[0]),
          std::min<array_index_type>(m_rp.m_tile_end[2] * m_rp.m_tile_end[3],
                                     maxblocks[1]),
          std::min<array_index_type>(
              (m_rp.m_upper[4] - m_rp.m_lower[4] + block.z - 1) / block.z,
              maxblocks[2]));
      CudaParallelLaunch<ParallelFor, LaunchBounds>(
          *this, grid, block, 0, m_rp.space().impl_internal_space_instance());
    } else if (RP::rank == 6) {
      // id0,id1 encoded within threadIdx.x; id2,id3 to threadIdx.y; id4,id5 to
      // threadIdx.z
      const dim3 block(m_rp.m_tile[0] * m_rp.m_tile[1],
                       m_rp.m_tile[2] * m_rp.m_tile[3],
                       m_rp.m_tile[4] * m_rp.m_tile[5]);
      const dim3 grid(
          std::min<array_index_type>(m_rp.m_tile_end[0] * m_rp.m_tile_end[1],
                                     maxblocks[0]),
          std::min<array_index_type>(m_rp.m_tile_end[2] * m_rp.m_tile_end[3],
                                     maxblocks[1]),
          std::min<array_index_type>(m_rp.m_tile_end[4] * m_rp.m_tile_end[5],
                                     maxblocks[2]));
      CudaParallelLaunch<ParallelFor, LaunchBounds>(
          *this, grid, block, 0, m_rp.space().impl_internal_space_instance());
    } else {
      Kokkos::abort("Kokkos::MDRange Error: Exceeded rank bounds with Cuda\n");
    }

  }  // end execute

  //  inline
  ParallelFor(const FunctorType& arg_functor, Policy arg_policy)
      : m_functor(arg_functor), m_rp(arg_policy) {}
};

template <class FunctorType, class ReducerType, class... Traits>
class ParallelReduce<FunctorType, Kokkos::MDRangePolicy<Traits...>, ReducerType,
                     Kokkos::Cuda> {
 public:
  using Policy = Kokkos::MDRangePolicy<Traits...>;

 private:
  using array_index_type = typename Policy::array_index_type;
  using index_type       = typename Policy::index_type;

  using WorkTag      = typename Policy::work_tag;
  using Member       = typename Policy::member_type;
  using LaunchBounds = typename Policy::launch_bounds;

  using ReducerConditional =
      Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                         FunctorType, ReducerType>;
  using ReducerTypeFwd = typename ReducerConditional::type;
  using WorkTagFwd =
      typename Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                                  WorkTag, void>::type;

  using Analysis =
      Kokkos::Impl::FunctorAnalysis<FunctorPatternInterface::REDUCE, Policy,
                                    ReducerTypeFwd>;

 public:
  using pointer_type   = typename Analysis::pointer_type;
  using value_type     = typename Analysis::value_type;
  using reference_type = typename Analysis::reference_type;
  using functor_type   = FunctorType;
  using size_type      = Cuda::size_type;
  using reducer_type   = ReducerType;

  // Algorithmic constraints: blockSize is a power of two AND blockDim.y ==
  // blockDim.z == 1

  const FunctorType m_functor;
  const Policy m_policy;  // used for workrange and nwork
  const ReducerType m_reducer;
  const pointer_type m_result_ptr;
  const bool m_result_ptr_device_accessible;
  size_type* m_scratch_space;
  size_type* m_scratch_flags;
  size_type* m_unified_space;

  using DeviceIteratePattern = typename Kokkos::Impl::Reduce::DeviceIterateTile<
      Policy::rank, Policy, FunctorType, typename Policy::work_tag,
      reference_type>;

  // Shall we use the shfl based reduction or not (only use it for static sized
  // types of more than 128bit
  static constexpr bool UseShflReduction = false;
  //((sizeof(value_type)>2*sizeof(double)) && Analysis::StaticValueSize)
  // Some crutch to do function overloading

 public:
  template <typename Policy, typename Functor>
  static int max_tile_size_product(const Policy& pol, const Functor&) {
    return max_tile_size_product_helper<ParallelReduce>(pol, LaunchBounds{});
  }
  Policy const& get_policy() const { return m_policy; }
  inline __device__ void exec_range(reference_type update) const {
    Kokkos::Impl::Reduce::DeviceIterateTile<Policy::rank, Policy, FunctorType,
                                            typename Policy::work_tag,
                                            reference_type>(m_policy, m_functor,
                                                            update)
        .exec_range();
  }

  inline __device__ void operator()() const {
    typename Analysis::Reducer final_reducer(
        &ReducerConditional::select(m_functor, m_reducer));
    const integral_nonzero_constant<size_type, Analysis::StaticValueSize /
                                                   sizeof(size_type)>
        word_count(Analysis::value_size(
                       ReducerConditional::select(m_functor, m_reducer)) /
                   sizeof(size_type));

    {
      reference_type value = final_reducer.init(reinterpret_cast<pointer_type>(
          kokkos_impl_cuda_shared_memory<size_type>() +
          threadIdx.y * word_count.value));

      // Number of blocks is bounded so that the reduction can be limited to two
      // passes. Each thread block is given an approximately equal amount of
      // work to perform. Accumulate the values for this block. The accumulation
      // ordering does not match the final pass, but is arithmatically
      // equivalent.

      this->exec_range(value);
    }

    // Reduce with final value at blockDim.y - 1 location.
    // Problem: non power-of-two blockDim
    if (cuda_single_inter_block_reduce_scan<false>(
            final_reducer, blockIdx.x, gridDim.x,
            kokkos_impl_cuda_shared_memory<size_type>(), m_scratch_space,
            m_scratch_flags)) {
      // This is the final block with the final result at the final threads'
      // location
      size_type* const shared = kokkos_impl_cuda_shared_memory<size_type>() +
                                (blockDim.y - 1) * word_count.value;
      size_type* const global =
          m_result_ptr_device_accessible
              ? reinterpret_cast<size_type*>(m_result_ptr)
              : (m_unified_space ? m_unified_space : m_scratch_space);

      if (threadIdx.y == 0) {
        final_reducer.final(reinterpret_cast<value_type*>(shared));
      }

      if (CudaTraits::WarpSize < word_count.value) {
        __syncthreads();
      }

      for (unsigned i = threadIdx.y; i < word_count.value; i += blockDim.y) {
        global[i] = shared[i];
      }
    }
  }

  // Determine block size constrained by shared memory:
  inline unsigned local_block_size(const FunctorType& f) {
    unsigned n = CudaTraits::WarpSize * 8;
    int shmem_size =
        cuda_single_inter_block_reduce_scan_shmem<false, FunctorType, WorkTag>(
            f, n);
    using closure_type = Impl::ParallelReduce<FunctorType, Policy, ReducerType>;
    cudaFuncAttributes attr =
        CudaParallelLaunch<closure_type,
                           LaunchBounds>::get_cuda_func_attributes();
    while (
        (n &&
         (m_policy.space().impl_internal_space_instance()->m_maxShmemPerBlock <
          shmem_size)) ||
        (n >
         static_cast<unsigned>(
             Kokkos::Impl::cuda_get_max_block_size<FunctorType, LaunchBounds>(
                 m_policy.space().impl_internal_space_instance(), attr, f, 1,
                 shmem_size, 0)))) {
      n >>= 1;
      shmem_size = cuda_single_inter_block_reduce_scan_shmem<false, FunctorType,
                                                             WorkTag>(f, n);
    }
    return n;
  }

  inline void execute() {
    typename Analysis::Reducer final_reducer(
        &ReducerConditional::select(m_functor, m_reducer));

    const auto nwork = m_policy.m_num_tiles;
    if (nwork) {
      int block_size = m_policy.m_prod_tile_dims;
      // CONSTRAINT: Algorithm requires block_size >= product of tile dimensions
      // Nearest power of two
      int exponent_pow_two    = std::ceil(std::log2(block_size));
      block_size              = std::pow(2, exponent_pow_two);
      int suggested_blocksize = local_block_size(m_functor);

      block_size = (block_size > suggested_blocksize)
                       ? block_size
                       : suggested_blocksize;  // Note: block_size must be less
                                               // than or equal to 512

      m_scratch_space = cuda_internal_scratch_space(
          m_policy.space(), Analysis::value_size(ReducerConditional::select(
                                m_functor, m_reducer)) *
                                block_size /* block_size == max block_count */);
      m_scratch_flags =
          cuda_internal_scratch_flags(m_policy.space(), sizeof(size_type));
      m_unified_space = cuda_internal_scratch_unified(
          m_policy.space(), Analysis::value_size(ReducerConditional::select(
                                m_functor, m_reducer)));

      // REQUIRED ( 1 , N , 1 )
      const dim3 block(1, block_size, 1);
      // Required grid.x <= block.y
      const dim3 grid(std::min(int(block.y), int(nwork)), 1, 1);

      // TODO @graph We need to effectively insert this in to the graph
      const int shmem =
          UseShflReduction
              ? 0
              : cuda_single_inter_block_reduce_scan_shmem<false, FunctorType,
                                                          WorkTag>(m_functor,
                                                                   block.y);

      CudaParallelLaunch<ParallelReduce, LaunchBounds>(
          *this, grid, block, shmem,
          m_policy.space()
              .impl_internal_space_instance());  // copy to device and execute

      if (!m_result_ptr_device_accessible) {
        if (m_result_ptr) {
          if (m_unified_space) {
            m_policy.space().fence(
                "Kokkos::Impl::ParallelReduce<Cuda, MDRangePolicy>::execute: "
                "Result Not Device Accessible");

            const int count = Analysis::value_count(
                ReducerConditional::select(m_functor, m_reducer));
            for (int i = 0; i < count; ++i) {
              m_result_ptr[i] = pointer_type(m_unified_space)[i];
            }
          } else {
            const int size = Analysis::value_size(
                ReducerConditional::select(m_functor, m_reducer));
            DeepCopy<HostSpace, CudaSpace, Cuda>(m_policy.space(), m_result_ptr,
                                                 m_scratch_space, size);
          }
        }
      }
    } else {
      if (m_result_ptr) {
        // TODO @graph We need to effectively insert this in to the graph
        final_reducer.init(m_result_ptr);
      }
    }
  }

  template <class ViewType>
  ParallelReduce(
      const FunctorType& arg_functor, const Policy& arg_policy,
      const ViewType& arg_result,
      std::enable_if_t<Kokkos::is_view<ViewType>::value, void*> = nullptr)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(InvalidType()),
        m_result_ptr(arg_result.data()),
        m_result_ptr_device_accessible(
            MemorySpaceAccess<Kokkos::CudaSpace,
                              typename ViewType::memory_space>::accessible),
        m_scratch_space(nullptr),
        m_scratch_flags(nullptr),
        m_unified_space(nullptr) {
    check_reduced_view_shmem_size<WorkTag>(m_policy, m_functor);
  }

  ParallelReduce(const FunctorType& arg_functor, const Policy& arg_policy,
                 const ReducerType& reducer)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(reducer),
        m_result_ptr(reducer.view().data()),
        m_result_ptr_device_accessible(
            MemorySpaceAccess<Kokkos::CudaSpace,
                              typename ReducerType::result_view_type::
                                  memory_space>::accessible),
        m_scratch_space(nullptr),
        m_scratch_flags(nullptr),
        m_unified_space(nullptr) {
    check_reduced_view_shmem_size<WorkTag>(m_policy, m_functor);
  }
};
}  // namespace Impl
}  // namespace Kokkos
#endif

#endif
