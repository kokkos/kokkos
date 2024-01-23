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

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#endif

#include <Kokkos_Macros.hpp>

#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <cstring>

#include <iostream>
#include <sstream>
#include <cstring>
#include <algorithm>

#include <Kokkos_HBWSpace.hpp>
#include <impl/Kokkos_Error.hpp>
#include <Kokkos_Atomic.hpp>
#ifdef KOKKOS_ENABLE_HBWSPACE
#include <memkind.h>
#endif

#include <impl/Kokkos_Tools.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
#ifdef KOKKOS_ENABLE_HBWSPACE
#define MEMKIND_TYPE MEMKIND_HBW  // hbw_get_kind(HBW_PAGESIZE_4KB)

/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Experimental {

/* Default allocation mechanism */
HBWSpace::HBWSpace() : m_alloc_mech(HBWSpace::STD_MALLOC) {
  printf("Init\n");
  setenv("MEMKIND_HBW_NODES", "1", 0);
}

/* Default allocation mechanism */
HBWSpace::HBWSpace(const HBWSpace::AllocationMechanism &arg_alloc_mech)
    : m_alloc_mech(HBWSpace::STD_MALLOC) {
  printf("Init2\n");
  setenv("MEMKIND_HBW_NODES", "1", 0);
  if (arg_alloc_mech == STD_MALLOC) {
    m_alloc_mech = HBWSpace::STD_MALLOC;
  }
}

void *HBWSpace::allocate(const size_t arg_alloc_size) const {
  return allocate("[unlabeled]", arg_alloc_size);
}
void *HBWSpace::allocate(const char *arg_label, const size_t arg_alloc_size,
                         const size_t arg_logical_size) const {
  return impl_allocate(arg_label, arg_alloc_size, arg_logical_size);
}
void *HBWSpace::impl_allocate(
    const char *arg_label, const size_t arg_alloc_size,
    const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  static_assert(sizeof(void *) == sizeof(uintptr_t),
                "Error sizeof(void*) != sizeof(uintptr_t)");

  static_assert(
      Kokkos::Impl::power_of_two<Kokkos::Impl::MEMORY_ALIGNMENT>::value,
      "Memory alignment must be power of two");

  constexpr uintptr_t alignment      = Kokkos::Impl::MEMORY_ALIGNMENT;
  constexpr uintptr_t alignment_mask = alignment - 1;

  void *ptr = nullptr;

  if (arg_alloc_size) {
    if (m_alloc_mech == STD_MALLOC) {
      // Over-allocate to and round up to guarantee proper alignment.
      size_t size_padded = arg_alloc_size + sizeof(void *) + alignment;

      void *alloc_ptr = memkind_malloc(MEMKIND_TYPE, size_padded);

      if (alloc_ptr) {
        uintptr_t address = reinterpret_cast<uintptr_t>(alloc_ptr);

        // offset enough to record the alloc_ptr
        address += sizeof(void *);
        uintptr_t rem    = address % alignment;
        uintptr_t offset = rem ? (alignment - rem) : 0u;
        address += offset;
        ptr = reinterpret_cast<void *>(address);
        // record the alloc'd pointer
        address -= sizeof(void *);
        *reinterpret_cast<void **>(address) = alloc_ptr;
      }
    }
  }

  if ((ptr == nullptr) || (reinterpret_cast<uintptr_t>(ptr) == ~uintptr_t(0)) ||
      (reinterpret_cast<uintptr_t>(ptr) & alignment_mask)) {
    std::ostringstream msg;
    msg << "Kokkos::Experimental::HBWSpace::allocate[ ";
    switch (m_alloc_mech) {
      case STD_MALLOC: msg << "STD_MALLOC"; break;
      case POSIX_MEMALIGN: msg << "POSIX_MEMALIGN"; break;
      case POSIX_MMAP: msg << "POSIX_MMAP"; break;
      case INTEL_MM_ALLOC: msg << "INTEL_MM_ALLOC"; break;
    }
    msg << " ]( " << arg_alloc_size << " ) FAILED";
    if (ptr == nullptr) {
      msg << " nullptr";
    } else {
      msg << " NOT ALIGNED " << ptr;
    }

    std::cerr << msg.str() << std::endl;
    std::cerr.flush();

    Kokkos::Impl::throw_runtime_exception(msg.str());
  }
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    const size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    Kokkos::Profiling::allocateData(arg_handle, arg_label, ptr, reported_size);
  }

  return ptr;
}

void HBWSpace::deallocate(void *const arg_alloc_ptr,
                          const size_t arg_alloc_size) const {
  deallocate("[unlabeled]", arg_alloc_ptr, arg_alloc_size);
}
void HBWSpace::deallocate(const char *arg_label, void *const arg_alloc_ptr,
                          const size_t arg_alloc_size,
                          const size_t arg_logical_size) const {
  impl_deallocate(arg_label, arg_alloc_ptr, arg_alloc_size, arg_logical_size);
}
void HBWSpace::impl_deallocate(
    const char *arg_label, void *const arg_alloc_ptr,
    const size_t arg_alloc_size, const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  if (arg_alloc_ptr) {
    if (Kokkos::Profiling::profileLibraryLoaded()) {
      const size_t reported_size =
          (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
      Kokkos::Profiling::deallocateData(arg_handle, arg_label, arg_alloc_ptr,
                                        reported_size);
    }

    if (m_alloc_mech == STD_MALLOC) {
      void *alloc_ptr = *(reinterpret_cast<void **>(arg_alloc_ptr) - 1);
      memkind_free(MEMKIND_TYPE, alloc_ptr);
    }
  }
}

}  // namespace Experimental
}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#include <impl/Kokkos_SharedAlloc_timpl.hpp>

KOKKOS_IMPL_SHARED_ALLOCATION_RECORD_EXPLICIT_INSTANTIATION(
    Kokkos::Experimental::HBWSpace);

#endif
