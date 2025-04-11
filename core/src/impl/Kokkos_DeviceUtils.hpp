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

#ifndef KOKKOS_DEVICE_UTILS_HPP
#define KOKKOS_DEVICE_UTILS_HPP

#include <cstddef>
#include <vector>
#include <utility>

namespace Kokkos {
namespace Impl {

// Host function to determine free and total device memory.
template <typename MemorySpace>
inline void get_free_total_memory(size_t& /* free_mem */,
                                  size_t& /* total_mem */) {}

#ifdef KOKKOS_ENABLE_CUDA
template <>
inline void get_free_total_memory<Kokkos::CudaSpace>(size_t& free_mem,
                                                     size_t& total_mem) {
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMemGetInfo(&free_mem, &total_mem));
}
#endif

#ifdef KOKKOS_ENABLE_HIP
template <>
inline void get_free_total_memory<Kokkos::HIPSpace>(size_t& free_mem,
                                                    size_t& total_mem) {
  KOKKOS_IMPL_HIP_SAFE_CALL(hipMemGetInfo(&free_mem, &total_mem));
}
#endif

#if defined(KOKKOS_ENABLE_SYCL)
template <>
inline void get_free_total_memory<Kokkos::SYCLDeviceUSMSpace>(
    size_t& free_mem, size_t& total_mem) {
  std::vector<sycl::device> devices = Kokkos::Impl::get_sycl_devices();
  if (devices.empty()) {
    return;
  }
  int device_id = Kokkos::Impl::SYCLInternal::m_syclDev;
  if (device_id < 0 || device_id >= static_cast<int>(devices.size())) {
    return;
  }
  auto device = devices[Impl::SYCLInternal::m_syclDev];

  total_mem = 0;
  free_mem  = 0;
  if (device.is_gpu()) {
    if (device.has(sycl::aspect::ext_intel_free_memory)) {
      free_mem = device.get_info<sycl::ext::intel::info::device::free_memory>();
      total_mem = device.get_info<sycl::info::device::global_mem_size>();
    }
  }
}
#endif

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_DEVICE_UTILS_HPP
