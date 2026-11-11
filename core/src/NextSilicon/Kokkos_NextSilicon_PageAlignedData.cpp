// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <NextSilicon/Kokkos_NextSilicon_PageAlignedData.hpp>
#include <NextSilicon/Kokkos_NextSilicon_InitializationCallbacks.hpp>

#include <Kokkos_InitializeFinalize.hpp>

#include <nextapi/memory.h>

#include <utility>

namespace {
Kokkos::Impl::NextSiliconInitializationCallbackHandle
impl_migrate_after_initialize(void* obj, std::size_t size, auto loc) {
  auto pin = [obj, size, loc] {
    nextapi_mem_migrate(obj, size, loc, /*pin=*/true);
  };
  if (Kokkos::is_initialized()) {
    pin();
    return Kokkos::Impl::invalid_nextsilicon_initialization_callback_handle;
  } else {
    return Kokkos::Impl::register_nextsilicon_initialization_callback(
        std::move(pin));
  }
}

void impl_release_page_migration(
    Kokkos::Impl::NextSiliconInitializationCallbackHandle handle, void* obj,
    std::size_t size, auto loc) {
  if (handle ==
      Kokkos::Impl::invalid_nextsilicon_initialization_callback_handle) {
    // ctor after Kokkos initialization: no callback; pinned immediately; unpin
    // now.
    nextapi_mem_migrate(obj, size, loc, /*pin=*/false);
  } else if (!Kokkos::Impl::retrieve_nextsilicon_initialization_callback(
                 handle)) {
    // ctor before Kokkos initialization, dtor after callback ran: unpin now.
    nextapi_mem_migrate(obj, size, loc, /*pin=*/false);
  }
  // ctor before Kokkos initialization, dtor before initialization: retrieving
  // callback above destroys it, and we are not pinned. Nothing to do.
}
}  // namespace

namespace Kokkos::Impl {

template <>
NextSiliconInitializationCallbackHandle
migrate_after_initialize<PageLocation::Any>(void*, std::size_t) {
  // no specific migration requested
  return invalid_nextsilicon_initialization_callback_handle;
}

template <>
NextSiliconInitializationCallbackHandle
migrate_after_initialize<PageLocation::Host>(void* obj, std::size_t size) {
  return impl_migrate_after_initialize(obj, size, NEXTAPI_PAGE_LOC_HOST);
}

template <>
NextSiliconInitializationCallbackHandle
migrate_after_initialize<PageLocation::Device>(void* obj, std::size_t size) {
  return impl_migrate_after_initialize(obj, size, NEXTAPI_PAGE_LOC_DEVICE);
}

template <>
void release_page_migration<PageLocation::Any>(
    NextSiliconInitializationCallbackHandle, void*, std::size_t) {
  // no specific migration requested
}

template <>
void release_page_migration<PageLocation::Host>(
    NextSiliconInitializationCallbackHandle handle, void* obj,
    std::size_t size) {
  impl_release_page_migration(handle, obj, size, NEXTAPI_PAGE_LOC_HOST);
}

template <>
void release_page_migration<PageLocation::Device>(
    NextSiliconInitializationCallbackHandle handle, void* obj,
    std::size_t size) {
  impl_release_page_migration(handle, obj, size, NEXTAPI_PAGE_LOC_DEVICE);
}

}  //  namespace Kokkos::Impl
