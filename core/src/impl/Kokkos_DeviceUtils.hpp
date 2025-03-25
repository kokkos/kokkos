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
#include <utility>

namespace Kokkos {
namespace Impl {

// Host function to determine free and total device memory.
// Will throw if execution space doesn't support this.
template <typename MemorySpace>
inline void get_free_total_memory(size_t& /* free_mem */,
                                  size_t& /* total_mem */) {}

// Host function to determine free and total device memory.
// Will throw if execution space doesn't support this.
template <typename MemorySpace>
inline void get_free_total_memory(size_t& /* free_mem */,
                                  size_t& /* total_mem */,
                                  int /* n_streams */) {}

#ifdef KOKKOS_ENABLE_CUDA
template <>
inline void get_free_total_memory<Kokkos::CudaSpace>(size_t& free_mem,
                                                     size_t& total_mem,
                                                     int n_streams) {
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMemGetInfo(&free_mem, &total_mem));
  free_mem /= n_streams;
  total_mem /= n_streams;
}
template <>
inline void get_free_total_memory<Kokkos::CudaSpace>(size_t& free_mem,
                                                     size_t& total_mem) {
  get_free_total_memory<Kokkos::CudaSpace>(free_mem, total_mem, 1);
}
template <>
inline void get_free_total_memory<Kokkos::CudaUVMSpace>(size_t& free_mem,
                                                        size_t& total_mem,
                                                        int n_streams) {
  get_free_total_memory<Kokkos::CudaSpace>(free_mem, total_mem, n_streams);
}
template <>
inline void get_free_total_memory<Kokkos::CudaUVMSpace>(size_t& free_mem,
                                                        size_t& total_mem) {
  get_free_total_memory<Kokkos::CudaUVMSpace>(free_mem, total_mem, 1);
}
template <>
inline void get_free_total_memory<Kokkos::CudaHostPinnedSpace>(
    size_t& free_mem, size_t& total_mem, int n_streams) {
  get_free_total_memory<Kokkos::CudaSpace>(free_mem, total_mem, n_streams);
}
template <>
inline void get_free_total_memory<Kokkos::CudaHostPinnedSpace>(
    size_t& free_mem, size_t& total_mem) {
  get_free_total_memory<Kokkos::CudaHostPinnedSpace>(free_mem, total_mem, 1);
}
#endif

#ifdef KOKKOS_ENABLE_HIP
template <>
inline void get_free_total_memory<Kokkos::HIPSpace>(size_t& free_mem,
                                                    size_t& total_mem,
                                                    int n_streams) {
  KOKKOS_IMPL_HIP_SAFE_CALL(hipMemGetInfo(&free_mem, &total_mem));
  free_mem /= n_streams;
  total_mem /= n_streams;
}
template <>
inline void get_free_total_memory<Kokkos::HIPManagedSpace>(size_t& free_mem,
                                                           size_t& total_mem,
                                                           int n_streams) {
  get_free_total_memory<Kokkos::HIPSpace>(free_mem, total_mem, n_streams);
}
template <>
inline void get_free_total_memory<Kokkos::HIPSpace>(size_t& free_mem,
                                                    size_t& total_mem) {
  get_free_total_memory<Kokkos::HIPSpace>(free_mem, total_mem, 1);
}
template <>
inline void get_free_total_memory<Kokkos::HIPManagedSpace>(size_t& free_mem,
                                                           size_t& total_mem) {
  get_free_total_memory<Kokkos::HIPSpace>(free_mem, total_mem, 1);
}
#endif

// FIXME_SYCL Use compiler extension instead of low level interface when
// available. Also, we assume to query memory associated with the default queue.
#if defined(KOKKOS_ENABLE_SYCL) && defined(KOKKOS_ARCH_INTEL_GPU)
template <>
inline void get_free_total_memory<Kokkos::SYCLDeviceUSMSpace>(size_t& free_mem,
                                                              size_t& total_mem,
                                                              int n_streams) {
  sycl::queue queue;
  sycl::device device = queue.get_device();
  auto level_zero_handle =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device);

  uint32_t n_memory_modules = 0;
  zesDeviceEnumMemoryModules(level_zero_handle, &n_memory_modules, nullptr);

  if (n_memory_modules == 0) {
    throw std::runtime_error(
        "Error: No memory modules for the SYCL backend found. Make sure that "
        "ZES_ENABLE_SYSMAN=1 is set at run time!");
  }

  total_mem = 0;
  free_mem  = 0;
  std::vector<zes_mem_handle_t> mem_handles(n_memory_modules);
  zesDeviceEnumMemoryModules(level_zero_handle, &n_memory_modules,
                             mem_handles.data());

  for (auto& mem_handle : mem_handles) {
    zes_mem_properties_t memory_properties{ZES_STRUCTURE_TYPE_MEM_PROPERTIES};
    zesMemoryGetProperties(mem_handle, &memory_properties);
    // Only report HBM which zeMemAllocDevice allocates from.
    if (memory_properties.type != ZES_MEM_TYPE_HBM) continue;

    zes_mem_state_t memory_states{ZES_STRUCTURE_TYPE_MEM_STATE};
    zesMemoryGetState(mem_handle, &memory_states);
    total_mem += memory_states.size;
    free_mem += memory_states.free;
  }
  free_mem /= n_streams;
  total_mem /= n_streams;
}

template <>
inline void get_free_total_memory<Kokkos::SYCLDeviceUSMSpace>(
    size_t& free_mem, size_t& total_mem) {
  get_free_total_memory<Kokkos::SYCLDeviceUSMSpace>(free_mem, total_mem, 1);
}

template <>
inline void get_free_total_memory<Kokkos::SYCLHostUSMSpace>(size_t& free_mem,
                                                            size_t& total_mem,
                                                            int n_streams) {
  get_free_total_memory<Kokkos::SYCLDeviceUSMSpace>(free_mem, total_mem,
                                                    n_streams);
}
template <>
inline void get_free_total_memory<Kokkos::SYCLHostUSMSpace>(size_t& free_mem,
                                                            size_t& total_mem) {
  get_free_total_memory<Kokkos::SYCLHostUSMSpace>(free_mem, total_mem, 1);
}
template <>
inline void get_free_total_memory<Kokkos::SYCLSharedUSMSpace>(size_t& free_mem,
                                                              size_t& total_mem,
                                                              int n_streams) {
  get_free_total_memory<Kokkos::SYCLDeviceUSMSpace>(free_mem, total_mem,
                                                    n_streams);
}
template <>
inline void get_free_total_memory<Kokkos::SYCLSharedUSMSpace>(
    size_t& free_mem, size_t& total_mem) {
  get_free_total_memory<Kokkos::SYCLSharedUSMSpace>(free_mem, total_mem, 1);
}
#endif

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_DEVICE_UTILS_HPP
