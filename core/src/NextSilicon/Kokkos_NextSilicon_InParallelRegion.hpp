// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_NEXTSILICON_IN_PARALLEL_REGION_HPP
#define KOKKOS_NEXTSILICON_IN_PARALLEL_REGION_HPP

#include <cstddef>

// used to implement KOKKOS_IF_ON_HOST / KOKKOS_IF_ON_DEVICE
namespace Kokkos::Impl {

// NextSiliconParallelRegionScopeGuard maintains a flag that
// tracks whether the current thread is executing under the NextSilicon
// device execution space. This is needed to correctly implement
// KOKKOS_IF_ON_DEVICE/KOKKOS_IF_ON_HOST semantics: unlike CUDA or SYCL,
// NextSilicon kernels may run on the host during training/telemetry
// collection while still being under the device execution space, so we
// cannot simply equate "on device" with "offloaded to device".

class NextSiliconParallelRegionScopeGuard {
  // FIXME_NEXTSILICON: It quacks like a bool, but sizeof and alignas -> SIZE.
  // It's pinned to the host to prevent migration to device memory, which would
  // cause problems when we try to access it from the host. This should be a
  // facility provided by the toolchain.
  struct HostPinnedBool {
    static constexpr int SIZE = 4096;  // Page-aligned for nextapi_mem_migrate.

    union {
      bool b_;
      std::byte pad[SIZE];
    };

   public:
    constexpr operator bool&() { return b_; }

    HostPinnedBool(const bool& b);
  };

  static_assert(sizeof(HostPinnedBool) == 4096);
  static HostPinnedBool s_is_in_parallel_region;

 public:
  static bool in() { return s_is_in_parallel_region; }

  // This constructor is in RAII context to set the flag to true when the
  // object is created and to false when the object is destroyed.
  NextSiliconParallelRegionScopeGuard();
  ~NextSiliconParallelRegionScopeGuard();
  NextSiliconParallelRegionScopeGuard(
      NextSiliconParallelRegionScopeGuard const&) = delete;
  NextSiliconParallelRegionScopeGuard& operator=(
      NextSiliconParallelRegionScopeGuard const&) = delete;
  NextSiliconParallelRegionScopeGuard(NextSiliconParallelRegionScopeGuard&&) =
      default;
  NextSiliconParallelRegionScopeGuard& operator=(
      NextSiliconParallelRegionScopeGuard&&) = default;
};

}  // namespace Kokkos::Impl

#endif
