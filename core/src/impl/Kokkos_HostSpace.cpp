// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#endif

#include <Kokkos_Macros.hpp>

#include <Kokkos_Atomic.hpp>
#include <Kokkos_BitManipulation.hpp>
#include <Kokkos_HostSpace.hpp>
#include <impl/Kokkos_Error.hpp>
#include <impl/Kokkos_Tools.hpp>

#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <new>

#include <iostream>
#include <sstream>
#include <cstring>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

void *HostSpace::allocate(const size_t arg_alloc_size) const {
  return allocate("[unlabeled]", arg_alloc_size);
}
void *HostSpace::allocate(const char *arg_label, const size_t arg_alloc_size,
                          const size_t arg_logical_size) const {
  return impl_allocate(arg_label, arg_alloc_size, arg_logical_size);
}
void *HostSpace::impl_allocate(
    const char *arg_label, const size_t arg_alloc_size,
    const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  void *ptr = nullptr;
  const size_t reported_size =
      (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
  static_assert(sizeof(void *) == sizeof(uintptr_t),
                "Error sizeof(void*) != sizeof(uintptr_t)");

  constexpr uintptr_t alignment      = Kokkos::Impl::MEMORY_ALIGNMENT;
  constexpr uintptr_t alignment_mask = alignment - 1;

#ifdef KOKKOS_ENABLE_HWLOC
  // only use hwloc_alloc() when membind_set is initialized (i.e. HostSpace
  // was constructed with hwloc nodes in mind), otherwise will fallback to
  // default 'operator new'
  if ((nullptr != membind_set) && arg_alloc_size) {
    const hwloc_topology_t topology = Kokkos::hwloc::get_topology();

    // We want to apply hwloc_membind_policy_t behavior uniquely to the
    // current Kokkos memory space, and not the whole process. Moreover,
    // the process memory binding can be larger than the current nodeset
    // of this space, and should not be reduced (with hwloc_set_membind())
    // for any reason.
    // - Among all membind policies, Firsttouch and Nexttouch
    //   are a bit tricky to treat: pages will be allocated (or moved) to
    //   the **local** NUMA node of the toucher-thread. This local node
    //   might not belong to the nodeset of this space and data will be
    //   spread over unwanted nodes. After multiple tests, it turns out
    //   that using:
    //   - hwloc_alloc() followed by
    //   - hwloc_set_area_membind() with
    //       policy = HWLOC_MEMBIND_BIND and
    //       flags |= (HWLOC_MEMBIND_MIGRATE | HWLOC_MEMBIND_STRICT)
    //   responds to our expected behavior.
    // - In the current implementation, HWLOC_MEMBIND_DEFAULT will be treated
    //   as HWLOC_MEMBIND_FIRSTTOUCH
    // - Other policies work straightforwardly.
    if (membind_policy == HWLOC_MEMBIND_DEFAULT ||
        membind_policy == HWLOC_MEMBIND_FIRSTTOUCH ||
        membind_policy == HWLOC_MEMBIND_NEXTTOUCH)
    {
      // allocate
      ptr = hwloc_alloc(topology, arg_alloc_size);

      // set_area_membind
      if (ptr) {
        int err_set_area =
          hwloc_set_area_membind(topology, ptr, arg_alloc_size,
            membind_set, HWLOC_MEMBIND_BIND,
            membind_flags | HWLOC_MEMBIND_MIGRATE | HWLOC_MEMBIND_STRICT);
        if (err_set_area) {
          std::cout << "ERROR: " << __PRETTY_FUNCTION__
                  << " failed to hwloc_set_area_membind() a"
                  << " [default|firsttouch|nexttouch] policy"
                  << std::endl;
          hwloc_free(topology, ptr, arg_alloc_size);
          ptr = nullptr;
        }
      }
    } else {
      // allocate with membind
      ptr = hwloc_alloc_membind(topology, arg_alloc_size,
            membind_set, membind_policy, membind_flags);
    }

#ifdef KOKKOS_ENABLE_DEBUG_HWLOC
    char *set_str;
    hwloc_bitmap_list_asprintf(&set_str, membind_set);
    std::cout << "INFO: " << __PRETTY_FUNCTION__
                << " hwloc_alloc_membind() = " << ptr
                << " arg_alloc_size: " << arg_alloc_size
                << " membind_set: " << set_str
                << " membind_policy: " << membind_policy
                << " membind_flags: " << membind_flags
                << std::endl;
    free(set_str);
#endif // KOKKOS_ENABLE_DEBUG_HWLOC
    goto check;
  }
#endif  // KOKKOS_ENABLE_HWLOC

  static_assert(Kokkos::has_single_bit(Kokkos::Impl::MEMORY_ALIGNMENT),
                "Memory alignment must be power of two");

  if (arg_alloc_size)
    ptr = operator new(arg_alloc_size, std::align_val_t(alignment),
                       std::nothrow_t{});

check:
  if (!ptr || (reinterpret_cast<uintptr_t>(ptr) == ~uintptr_t(0)) ||
      (reinterpret_cast<uintptr_t>(ptr) & alignment_mask)) {
    Impl::throw_bad_alloc(name(), arg_alloc_size, arg_label);
  }
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::allocateData(arg_handle, arg_label, ptr, reported_size);
  }

  return ptr;
}

void HostSpace::deallocate(void *const arg_alloc_ptr,
                           const size_t arg_alloc_size) const {
  deallocate("[unlabeled]", arg_alloc_ptr, arg_alloc_size);
}

void HostSpace::deallocate(const char *arg_label, void *const arg_alloc_ptr,
                           const size_t arg_alloc_size,
                           const size_t arg_logical_size) const {
  if (arg_alloc_ptr) Kokkos::fence("HostSpace::impl_deallocate before free");
  impl_deallocate(arg_label, arg_alloc_ptr, arg_alloc_size, arg_logical_size);
}
void HostSpace::impl_deallocate(
    const char *arg_label, void *const arg_alloc_ptr,
    const size_t arg_alloc_size, const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  if (arg_alloc_ptr) {
    size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    if (Kokkos::Profiling::profileLibraryLoaded()) {
      Kokkos::Profiling::deallocateData(arg_handle, arg_label, arg_alloc_ptr,
                                        reported_size);
    }

#ifdef KOKKOS_ENABLE_HWLOC
    if ((nullptr != membind_set) && arg_alloc_size) {
#ifdef KOKKOS_ENABLE_DEBUG_HWLOC
      std::cout << "INFO: " << __PRETTY_FUNCTION__
                << " hwloc_free(" << arg_alloc_ptr << ")"
                << " alloc_size: " << arg_alloc_size
                << std::endl;
#endif // KOKKOS_ENABLE_DEBUG_HWLOC
      hwloc_free(Kokkos::hwloc::get_topology(), arg_alloc_ptr, arg_alloc_size);
      return;
    }
#endif  // KOKKOS_ENABLE_HWLOC

    constexpr uintptr_t alignment = Kokkos::Impl::MEMORY_ALIGNMENT;
    operator delete(arg_alloc_ptr, std::align_val_t(alignment),
                    std::nothrow_t{});
  }
}

}  // namespace Kokkos

#include <impl/Kokkos_SharedAlloc_timpl.hpp>

KOKKOS_IMPL_SHARED_ALLOCATION_RECORD_EXPLICIT_INSTANTIATION(Kokkos::HostSpace);
