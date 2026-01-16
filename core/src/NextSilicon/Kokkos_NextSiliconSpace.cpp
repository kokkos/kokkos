// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#define KOKKOS_IMPL_PUBLIC_INCLUDE

#include <nsapi/memory.h>

#include <NextSilicon/Kokkos_NextSilicon.hpp>
#include <NextSilicon/Kokkos_NextSiliconSpace.hpp>
#include <NextSilicon/Kokkos_NextSilicon_DeepCopy.hpp>
#include <impl/Kokkos_Profiling_Interface.hpp>

namespace Kokkos {
namespace Experimental {

void *NextSiliconSpace::allocate(const size_t arg_alloc_size) const {
  return allocate("[unlabeled]", arg_alloc_size);
}

void *NextSiliconSpace::allocate(const char *arg_label,
                                 const size_t arg_alloc_size,
                                 const size_t arg_logical_size) const {
  return impl_allocate(arg_label, arg_alloc_size, arg_logical_size);
}

void *NextSiliconSpace::impl_allocate(
    const char *arg_label, const size_t arg_alloc_size,
    const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  static_assert(sizeof(void *) == sizeof(uintptr_t),
                "Error sizeof(void*) != sizeof(uintptr_t)");

  void *ptr = nullptr;
#if defined(KOKKOS_ENABLE_IMPL_NEXTSILICON_DISTRIBUTE_MEMORY)
  ptr = std::aligned_alloc(arg_alloc_size, arg_alloc_size);
  nsapi_mem_migrate_distributed(ptr, arg_alloc_size, NSAPI_PAGE_LOC_DEVICE,
                                false);
#else
  ptr = llns_memory_device_allocate(
      arg_alloc_size, /* loopref */ {}, /* cache_capacity */ {},
      /* hit_throughput */ {}, /* miss_throughput */ {});
#endif
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    const size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    Kokkos::Profiling::allocateData(arg_handle, arg_label, ptr, reported_size);
  }
  return ptr;
}

void NextSiliconSpace::deallocate(void *const arg_alloc_ptr,
                                  const size_t arg_alloc_size) const {
  deallocate("[unlabeled]", arg_alloc_ptr, arg_alloc_size);
}

void NextSiliconSpace::deallocate(const char *arg_label,
                                  void *const arg_alloc_ptr,
                                  const size_t arg_alloc_size,
                                  const size_t arg_logical_size) const {
  impl_deallocate(arg_label, arg_alloc_ptr, arg_alloc_size, arg_logical_size);
}

void NextSiliconSpace::impl_deallocate(
    const char *arg_label, void *const arg_alloc_ptr,
    const size_t arg_alloc_size, const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    const size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    Kokkos::Profiling::deallocateData(arg_handle, arg_label, arg_alloc_ptr,
                                      reported_size);
  }

  if (arg_alloc_ptr) {
#if defined(KOKKOS_ENABLE_IMPL_NEXTSILICON_DISTRIBUTE_MEMORY)
    free(arg_alloc_ptr);
#else
    llns_memory_free(arg_alloc_ptr);
#endif
  }
}

void *NextSiliconSharedSpace::allocate(const size_t arg_alloc_size) const {
  return allocate("[unlabeled]", arg_alloc_size);
}

void *NextSiliconSharedSpace::allocate(const char *arg_label,
                                       const size_t arg_alloc_size,
                                       const size_t arg_logical_size) const {
  return impl_allocate(arg_label, arg_alloc_size, arg_logical_size);
}

void *NextSiliconSharedSpace::impl_allocate(
    const char *arg_label, const size_t arg_alloc_size,
    const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  static_assert(sizeof(void *) == sizeof(uintptr_t),
                "Error sizeof(void*) != sizeof(uintptr_t)");

  void *ptr = nullptr;
  // NextSilicon implements shared UVM over standard memory operations.
  ptr = malloc(arg_alloc_size);

  if (Kokkos::Profiling::profileLibraryLoaded()) {
    const size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    Kokkos::Profiling::allocateData(arg_handle, arg_label, ptr, reported_size);
  }

  return ptr;
}

void NextSiliconSharedSpace::deallocate(void *const arg_alloc_ptr,
                                        const size_t arg_alloc_size) const {
  deallocate("[unlabeled]", arg_alloc_ptr, arg_alloc_size);
}

void NextSiliconSharedSpace::deallocate(const char *arg_label,
                                        void *const arg_alloc_ptr,
                                        const size_t arg_alloc_size,
                                        const size_t arg_logical_size) const {
  impl_deallocate(arg_label, arg_alloc_ptr, arg_alloc_size, arg_logical_size);
}

void NextSiliconSharedSpace::impl_deallocate(
    const char *arg_label, void *const arg_alloc_ptr,
    const size_t arg_alloc_size, const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    const size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    Kokkos::Profiling::deallocateData(arg_handle, arg_label, arg_alloc_ptr,
                                      reported_size);
  }

  if (arg_alloc_ptr) {
    // NextSilicon implements shared UVM over standard memory operations.
    free(arg_alloc_ptr);
  }
}

}  // namespace Experimental
}  // namespace Kokkos
