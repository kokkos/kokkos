// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#endif

#include <Kokkos_Core.hpp>

#include <impl/Kokkos_Profiling.hpp>
#include <ostream>
#include <cstdint>

namespace Kokkos::Experimental::Impl {

Kokkos::Impl::HostSharedPtr<Kokkos::Experimental::Impl::NextSiliconInternal>
    Kokkos::Experimental::Impl::NextSiliconInternal::default_instance;

NextSiliconInternal::NextSiliconInternal() {}

NextSiliconInternal::~NextSiliconInternal() {
  fence(
      "Kokkos::Experimental::Impl::NextSiliconInternal: fence on destruction");
}

void NextSiliconInternal::print_configuration(std::ostream &os) const {
  os << "macro  KOKKOS_ENABLE_NEXTSILICON      : defined\n";
  // FIXME_NEXTSILICON print_configuration doesn't do anything useful, fix
  // once device properties nextapi is available
}

void NextSiliconInternal::fence(std::string const &name) const {
  // FIXME_NEXTSILICON fence doesn't do anything in the OpenMP-style interface

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
