// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <NextSilicon/Kokkos_NextSilicon_PageAlignedData.hpp>
#include <NextSilicon/Kokkos_NextSilicon_InitializationCallbacks.hpp>

#include <Kokkos_InitializeFinalize.hpp>

#include <nextapi/memory.h>

namespace {
void impl_migrate_after_initialize(void *obj, size_t size, auto loc) {
  auto pin = [=] { nextapi_mem_migrate(obj, size, loc, /*pin=*/true); };
  if (Kokkos::is_initialized()) {
    pin();
  } else {
    Kokkos::Impl::register_nextsilicon_initialization_callback(std::move(pin));
  }
}
}  // namespace

namespace Kokkos::Impl {

template <>
void migrate_after_initialize<PageLocation::Any>(void *, size_t) {
  // no specific migration requested
}

template <>
void migrate_after_initialize<PageLocation::Host>(void *obj, size_t size) {
  impl_migrate_after_initialize(obj, size, NEXTAPI_PAGE_LOC_HOST);
}

template <>
void migrate_after_initialize<PageLocation::Device>(void *obj, size_t size) {
  impl_migrate_after_initialize(obj, size, NEXTAPI_PAGE_LOC_DEVICE);
}

}  //  namespace Kokkos::Impl
