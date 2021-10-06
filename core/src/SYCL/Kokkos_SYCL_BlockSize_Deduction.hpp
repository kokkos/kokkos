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

#ifndef KOKKOS_SYCL_INTERNAL_HPP
#define KOKKOS_SYCL_INTERNAL_HPP

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_SYCL

#include <Cuda/Kokkos_Cuda_Error.hpp>

namespace Kokkos {
namespace Impl {

inline int sycl_max_active_blocks_per_sm(int block_size, size_t dynamic_shmem, const int regs_per_thread) {

// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
#if defined(KOKKOS_ARCH_AMPERE80)
  int const regs_per_sm     = 65536;
  size_t const shmem_per_sm = 167936;
  size_t const shmem_per_block = 49152;
  int const max_blocks_per_sm = 32;
#else
  // TODO other arches
#endif

  // Limits due to registers/SM
  int const max_blocks_regs = regs_per_sm / (regs_per_thread * block_size);

  int const max_blocks_shmem =
      dynamic_shmem > shmem_per_block
          ? 0
          : (dynamic_shmem > 0 ? (int)shmem_per_sm / dynamic_shmem
                             : max_blocks_regs);

  // Overall occupancy in blocks
  return std::min({max_blocks_regs, max_blocks_shmem, max_blocks_per_sm});
}

// Implemented for sycl cuda backend only
template <typename UnaryFunction, typename FunctorType, typename LaunchBounds, template <typename> class Wrapper>
inline int sycl_deduce_block_size(bool early_termination,
				  const sycl::queue& q,
				  const FunctorType& f,
                                  UnaryFunction block_size_to_dynamic_shmem,
                                  LaunchBounds) {

  // Get the device & check backend
  const sycl::device sycl_device = q.get_device();
  KOKKOS_ASSERT(sycl_device.get_backend() == sycl::backend::cuda);

  // Get the compiled kernel to query register & memory usage
  sycl::program p{q.get_context()};
  p.build_with_kernel_type<Wrapper<FunctorType>>();
  auto k = p.get_kernel<Wrapper<FunctorType>>();

  auto num_regs = k.template get_info<
      sycl::info::kernel_device_specific::ext_codeplay_num_regs>(sycl_device);

  size_t kernelMaxThreadsPerBlock = k.template get_info<
      sycl::info::kernel_device_specific::work_group_size>(sycl_device);

#if defined(KOKKOS_ARCH_AMPERE80)
  int const max_threads_per_sm = 2048;
#else
  // TODO other arches
#endif

  int const device_max_threads_per_block =
    sycl_device.template get_info<sycl::info::device::max_work_group_size>();

  int const max_threads_per_block =
      std::min(LaunchBounds::maxTperB == 0 ? (int)device_max_threads_per_block
                                           : (int)LaunchBounds::maxTperB,
               (int)kernelMaxThreadsPerBlock);

  int const min_blocks_per_sm =
      LaunchBounds::minBperSM == 0 ? 1 : LaunchBounds::minBperSM;

  // Recorded maximum
  int opt_block_size     = 0;
  int opt_threads_per_sm = 0;

  size_t wg_size_multiple = k.template get_info<
      sycl::info::kernel_device_specific::preferred_work_group_size_multiple>(sycl_device);

  assert(max_threads_per_block % wg_size_multiple == 0);

  for (int block_size = max_threads_per_block; block_size > 0;
       block_size -= wg_size_multiple) {

    // 'dynamic_shmem' is a misnomer here. It's allocated before launch by
    // the host & it's sycl 'local' memory.
    size_t const dynamic_shmem = block_size_to_dynamic_shmem(block_size);

    int blocks_per_sm = sycl_max_active_blocks_per_sm(
        block_size, dynamic_shmem, num_regs);

    int threads_per_sm = blocks_per_sm * block_size;
    if (threads_per_sm > max_threads_per_sm) {
      blocks_per_sm  = max_threads_per_sm / block_size;
      threads_per_sm = blocks_per_sm * block_size;
    }

    if (blocks_per_sm >= min_blocks_per_sm) {
      if (threads_per_sm >= opt_threads_per_sm) {
        opt_block_size     = block_size;
        opt_threads_per_sm = threads_per_sm;
      }
    }

    if (early_termination && opt_block_size != 0) break;
  }

  return opt_block_size;
}

template <class FunctorType, class LaunchBounds,
          template <typename> class Wrapper>
int sycl_get_opt_block_size(const sycl::queue& q, const FunctorType& f,
                            const size_t vector_length,
                            const size_t shmem_block,
                            const size_t shmem_thread) {

  // TODO - cuda equiv here calls:
  // auto const& prop = Kokkos::Cuda().cuda_device_prop();
  // i.e. device info caching.

  // The shared memory here is that which is explicitly allocated by Kokkos
  // for either team operations or because of an explicit user request (functor_shmem).
  // SYCL kernels cannot directly request shared (AKA local) memory. This can be achieved
  // via local_accessors but a Kokkos kernel can't create those.
  auto const block_size_to_dynamic_shmem = [&f, vector_length, shmem_block,
                                            shmem_thread](int block_size) {
    size_t const functor_shmem =
        Kokkos::Impl::FunctorTeamShmemSize<FunctorType>::value(
            f, block_size / vector_length);

    size_t const dynamic_shmem = shmem_block +
                                 shmem_thread * (block_size / vector_length) +
                                 functor_shmem;
    return dynamic_shmem;
  };


  return sycl_deduce_block_size<decltype(block_size_to_dynamic_shmem), FunctorType, LaunchBounds, Wrapper>(false, q, f, block_size_to_dynamic_shmem, LaunchBounds{});
}

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_ENABLE_SYCL
#endif  /* #ifndef KOKKOS_SYCL_INTERNAL_HPP */
