// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#endif

#include <Kokkos_Core.hpp>

#include "impl/Kokkos_Profiling.hpp"
#include <ostream>
#include <cstdint>

#if defined(KOKKOS_ENABLE_IMPL_NEXTSILICON_ADD_TELEM_REGIONS)
#include <nsapi/telem.h>
#endif

namespace Kokkos::Experimental::Impl {

NextSiliconInternal *NextSiliconInternal::singleton() {
  static NextSiliconInternal self;
  return &self;
}

void NextSiliconInternal::initialize() {
#if defined(KOKKOS_ENABLE_IMPL_NEXTSILICON_ADD_TELEM_REGIONS)
  /* Wrap the entire program with a nsapi telem region.
   * Required for performance estimation to take into account all kernel
   * invocations.
   */
  nsapi_telem_region_enter();
#endif
  m_is_initialized = true;
}

void NextSiliconInternal::finalize() {
#if defined(KOKKOS_ENABLE_IMPL_NEXTSILICON_ADD_TELEM_REGIONS)
  nsapi_telem_region_exit();
#endif
  m_is_initialized = false;
}

bool NextSiliconInternal::is_initialized() const { return m_is_initialized; }

void NextSiliconInternal::print_configuration(std::ostream &os) const {
#if defined(KOKKOS_ENABLE_NEXTSILICON)
  os << "macro  KOKKOS_ENABLE_NEXTSILICON      : defined\n";
#endif
  // FIXME_NEXTSILICON print_configuration doesn't do anything useful, fix
  // once device properties nsapi is available
}

void NextSiliconInternal::fence(std::string const &name) const {
  // FIXME_NEXTSILICON: all APIs are synchronous, fence is a no-op
  Kokkos::Tools::Experimental::Impl::profile_fence_event<NextSilicon>(
      name,
      Kokkos::Tools::Experimental::Impl::DirectFenceIDHandle{instance_id()},
      [&]() {});
}

uint32_t NextSiliconInternal::instance_id() const noexcept {
  return Kokkos::Tools::Experimental::Impl::idForInstance<
      Kokkos::Experimental::NextSilicon>(reinterpret_cast<uintptr_t>(this));
}

}  // namespace Kokkos::Experimental::Impl
