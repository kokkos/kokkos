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

#define KOKKOS_IMPL_PUBLIC_INCLUDE

#include <OpenACC/Kokkos_OpenACC.hpp>
#include <OpenACC/Kokkos_OpenACC_Instance.hpp>
#include <OpenACC/Kokkos_OpenACC_Traits.hpp>
#include <impl/Kokkos_Profiling.hpp>
#include <impl/Kokkos_ExecSpaceManager.hpp>
#include <impl/Kokkos_DeviceManagement.hpp>

#if defined(KOKKOS_IMPL_ARCH_NVIDIA_GPU)
#include <cuda_runtime.h>
#elif defined(KOKKOS_ARCH_AMD_GPU)
// FIXME_OPENACC - hip_runtime_api.h contains two implementations: one for AMD
// GPUs and the other for NVIDIA GPUs; below macro is needed to choose AMD GPUs.
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime_api.h>
#elif defined(KOKKOS_ENABLE_OPENACC_FORCE_HOST_AS_DEVICE)
#include <thread>
#endif

#include <iostream>
#include <sstream>

Kokkos::Experimental::OpenACC::OpenACC()
    : m_space_instance(
          &Kokkos::Experimental::Impl::OpenACCInternal::singleton(),
          [](Impl::OpenACCInternal*) {}) {
  Impl::OpenACCInternal::singleton().verify_is_initialized(
      "OpenACC instance constructor");
}

Kokkos::Experimental::OpenACC::OpenACC(int async_arg)
    : m_space_instance(new Kokkos::Experimental::Impl::OpenACCInternal,
                       [](Impl::OpenACCInternal* ptr) {
                         ptr->finalize();
                         delete ptr;
                       }) {
  Impl::OpenACCInternal::singleton().verify_is_initialized(
      "OpenACC instance constructor");
  m_space_instance->initialize(async_arg);
}

void Kokkos::Experimental::OpenACC::impl_initialize(
    InitializationSettings const& settings) {
  Impl::OpenACCInternal::m_concurrency =
      256000;  // FIXME_OPENACC - random guess when cannot compute
  if (Impl::OpenACC_Traits::may_fallback_to_host &&
      acc_get_num_devices(Impl::OpenACC_Traits::dev_type) == 0 &&
      !settings.has_device_id()) {
    if (show_warnings()) {
      std::cerr << "Warning: No GPU available for execution, falling back to"
                   " using the host!"
                << std::endl;
    }
    acc_set_device_type(acc_device_host);
    Impl::OpenACCInternal::m_acc_device_num =
        acc_get_device_num(acc_device_host);
  } else {
    using Kokkos::Impl::get_visible_devices;
    acc_set_device_type(Impl::OpenACC_Traits::dev_type);
    std::vector<int> const& visible_devices = get_visible_devices();
    using Kokkos::Impl::get_gpu;
    int const dev_num = get_gpu(settings).value_or(visible_devices[0]);
    acc_set_device_num(dev_num, Impl::OpenACC_Traits::dev_type);
    Impl::OpenACCInternal::m_acc_device_num = dev_num;
#if defined(KOKKOS_IMPL_ARCH_NVIDIA_GPU)
    cudaDeviceProp deviceProp;
    cudaError error = cudaGetDeviceProperties(&deviceProp, dev_num);
    if (error != cudaSuccess) {
      std::ostringstream msg;
      msg << "Error: During OpenACC backend initialization, failed to retrieve "
          << "CUDA device properties: (" << cudaGetErrorName(error)
          << "): " << cudaGetErrorString(error);
      Kokkos::Impl::host_abort(msg.str().c_str());
    }
    Impl::OpenACCInternal::m_concurrency =
        deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount;
#elif defined(KOKKOS_ARCH_AMD_GPU)
    hipDeviceProp_t deviceProp;
    hipError_t error = hipGetDeviceProperties(&deviceProp, dev_num);
    if (error != hipSuccess) {
      std::ostringstream msg;
      msg << "Error: During OpenACC backend initialization, failed to retrieve "
          << "HIP device properties: (" << hipGetErrorName(error)
          << "): " << hipGetErrorString(error);
      Kokkos::Impl::host_abort(msg.str().c_str());
    }
    Impl::OpenACCInternal::m_concurrency =
        deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount;
#elif defined(KOKKOS_ENABLE_OPENACC_FORCE_HOST_AS_DEVICE)
    Impl::OpenACCInternal::m_concurrency = std::thread::hardware_concurrency();
    if (Impl::OpenACCInternal::m_concurrency == 0) {
      Kokkos::Impl::host_abort(
          "Error: During OpenACC backend initialization, failed to retrieve "
          "CPU hardware concurrency");
    }
#else
    // FIXME_OPENACC: Compute Impl::OpenACCInternal::m_concurrency correctly.
#endif
  }
  Impl::OpenACCInternal::singleton().initialize();
}

void Kokkos::Experimental::OpenACC::impl_finalize() {
  Impl::OpenACCInternal::singleton().finalize();
}

bool Kokkos::Experimental::OpenACC::impl_is_initialized() {
  return Impl::OpenACCInternal::singleton().is_initialized();
}

void Kokkos::Experimental::OpenACC::print_configuration(std::ostream& os,
                                                        bool verbose) const {
  os << "Device Execution Space:\n";
  os << "  KOKKOS_ENABLE_OPENACC: yes\n";
  os << "OpenACC Options:\n";
  os << "  KOKKOS_ENABLE_OPENACC_COLLAPSE_HIERARCHICAL_CONSTRUCTS: ";
#ifdef KOKKOS_ENABLE_OPENACC_COLLAPSE_HIERARCHICAL_CONSTRUCTS
  os << "yes\n";
#else
  os << "no\n";
#endif
  os << "  KOKKOS_ENABLE_OPENACC_FORCE_HOST_AS_DEVICE: ";
#if defined(KOKKOS_ENABLE_OPENACC_FORCE_HOST_AS_DEVICE)
  os << "yes\n";
#else
  os << "no\n";
#endif
  m_space_instance->print_configuration(os, verbose);
}

void Kokkos::Experimental::OpenACC::fence(std::string const& name) const {
  m_space_instance->fence(name);
}

void Kokkos::Experimental::OpenACC::impl_static_fence(std::string const& name) {
  Kokkos::Tools::Experimental::Impl::profile_fence_event<
      Kokkos::Experimental::OpenACC>(
      name,
      Kokkos::Tools::Experimental::SpecialSynchronizationCases::
          GlobalDeviceSynchronization,
      [&]() { acc_wait_all(); });
}

uint32_t Kokkos::Experimental::OpenACC::impl_instance_id() const noexcept {
  return m_space_instance->instance_id();
}

int Kokkos::Experimental::OpenACC::acc_async_queue() const {
  return m_space_instance->m_async_arg;
}

int Kokkos::Experimental::OpenACC::acc_device_number() const {
  return Impl::OpenACCInternal::m_acc_device_num;
}

namespace Kokkos {
namespace Impl {
int g_openacc_space_factory_initialized =
    initialize_space_factory<Experimental::OpenACC>("170_OpenACC");
}  // namespace Impl
}  // Namespace Kokkos
