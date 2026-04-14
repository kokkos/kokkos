// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <NextSilicon/Kokkos_NextSilicon_InParallelRegion.hpp>
#include <Kokkos_Abort.hpp>
#include <Kokkos_Atomic.hpp>
#include <nextapi/memory.h>

namespace Kokkos::Impl {
/*static*/ NextSiliconParallelRegionScopeGuard::HostPinnedBool
    NextSiliconParallelRegionScopeGuard::s_is_in_parallel_region = false;

NextSiliconParallelRegionScopeGuard::HostPinnedBool::HostPinnedBool(
    const bool &b)
    : b_(b) {
  // Upon construction, pin this variable to host memory to prevent it from
  // being migrated to device.
  nextapi_mem_migrate(this, sizeof(*this), NEXTAPI_PAGE_LOC_HOST, true);
}

NextSiliconParallelRegionScopeGuard::NextSiliconParallelRegionScopeGuard() {
  // FIXME_NEXTSILICON: Currently parallel dispatch from multiple host threads
  // is not supported. Remove this atomic and assert when when support is added.
  bool old = Kokkos::atomic_exchange(
      &static_cast<bool &>(s_is_in_parallel_region), true);
  if (old != false) {
    Kokkos::abort("Already in a parallel region");
  }
}
NextSiliconParallelRegionScopeGuard::~NextSiliconParallelRegionScopeGuard() {
  // FIXME_NEXTSILICON: Currently parallel dispatch from multiple host threads
  // is not supported. Remove this atomic and assert when when support is added.
  bool old = Kokkos::atomic_exchange(
      &static_cast<bool &>(s_is_in_parallel_region), false);
  if (old != true) {
    Kokkos::abort("Not in a parallel region");
  }
}

}  // namespace Kokkos::Impl
