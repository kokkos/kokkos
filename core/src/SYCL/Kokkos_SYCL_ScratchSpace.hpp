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
#include <Kokkos_Macros.hpp>
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif
#ifndef KOKKOS_SYCL_SCRATCHSPACE_HPP
#define KOKKOS_SYCL_SCRATCHSPACE_HPP

#include <cstdio>
#include <cstddef>
#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Concepts.hpp>

/*--------------------------------------------------------------------------*/

namespace Kokkos {

template <int Level>
struct ScratchMemorySpaceWrapper;

/** \brief  Scratch memory space associated with an execution space.
 *
 */
template <>
class ScratchMemorySpace<Kokkos::Experimental::SYCL> {
  using ExecSpace = Kokkos::Experimental::SYCL;

  static_assert(
      is_execution_space<ExecSpace>::value,
      "Instantiating ScratchMemorySpace on non-execution-space type.");

 public:
  // Minimal overalignment used by view scratch allocations
  constexpr static int ALIGN = 8;

  template <int Level>
  using wrapper_type = ScratchMemorySpaceWrapper<Level>;

 private:
  mutable sycl::local_ptr<char> m_iter_L0  = nullptr;
  mutable sycl::device_ptr<char> m_iter_L1 = nullptr;
  sycl::local_ptr<char> m_end_L0           = nullptr;
  sycl::device_ptr<char> m_end_L1          = nullptr;

  mutable int m_multiplier    = 0;
  mutable int m_offset        = 0;
  mutable int m_default_level = 0;

 public:
  //! Tag this class as a memory space
  using memory_space    = ScratchMemorySpace<ExecSpace>;
  using execution_space = ExecSpace;
  //! This execution space preferred device_type
  using device_type = Kokkos::Device<execution_space, memory_space>;

  using array_layout = typename ExecSpace::array_layout;
  using size_type    = typename ExecSpace::size_type;

  static constexpr const char* name() { return "ScratchMemorySpace"; }

  template <typename IntType>
  KOKKOS_INLINE_FUNCTION void* get_shmem(const IntType& size,
                                         int level = -1) const {
    if (level == -1) level = m_default_level;
    constexpr bool align = false;
    if (level == 1)
      return sycl::device_ptr<void>(
          get_shmem_common<align>(size, 1, m_iter_L1, m_end_L1));
    else
      return sycl::local_ptr<void>(
          get_shmem_common<align>(size, 1, m_iter_L0, m_end_L0));
  }

  template <typename IntType>
  KOKKOS_INLINE_FUNCTION void* get_shmem_aligned(const IntType& size,
                                                 const ptrdiff_t alignment,
                                                 int level = -1) const {
    if (level == -1) level = m_default_level;
    constexpr bool align = true;
    if (level == 1) {
      return sycl::device_ptr<void>(
          get_shmem_common<align>(size, alignment, m_iter_L1, m_end_L1));
    } else {
      return sycl::local_ptr<void>(
          get_shmem_common<align>(size, alignment, m_iter_L0, m_end_L0));
    }
  }

  template <int Level, typename IntType>
  KOKKOS_INLINE_FUNCTION void* get_shmem_aligned_on_level(
      const IntType& size, const ptrdiff_t alignment) const {
    static_assert(Level == 0 || Level == 1);
    constexpr bool align = true;
    if constexpr (Level == 1) {
      return sycl::device_ptr<void>(
          get_shmem_common<align>(size, alignment, m_iter_L1, m_end_L1));
    } else {
      return sycl::local_ptr<void>(
          get_shmem_common<align>(size, alignment, m_iter_L0, m_end_L0));
    }
  }

 private:
  template <bool align, typename IntType, typename PointerType>
  KOKKOS_INLINE_FUNCTION PointerType get_shmem_common(
      const IntType& size, [[maybe_unused]] const ptrdiff_t alignment,
      PointerType& begin, const PointerType end) const {
    auto m_iter_old = begin;
    if constexpr (align) {
      const ptrdiff_t missalign = size_t(begin.get()) % alignment;
      if (missalign) begin += alignment - missalign;
    }

    // This is each thread's start pointer for its allocation
    // Note: for team scratch m_offset is 0, since every
    // thread will get back the same shared pointer
    PointerType tmp     = begin + m_offset * size;
    uintptr_t increment = size * m_multiplier;
    uintptr_t capacity  = end.get() - begin.get();

    if (increment > capacity) {
      // Request did overflow: return nullptr and reset m_iter
      begin = m_iter_old;
      tmp   = nullptr;
#ifdef KOKKOS_ENABLE_DEBUG
      // mfh 23 Jun 2015: printf call consumes 25 registers
      // in a CUDA build, so only print in debug mode.  The
      // function still returns nullptr if not enough memory.
      KOKKOS_IMPL_DO_NOT_USE_PRINTF(
          "ScratchMemorySpace<...>::get_shmem: Failed to allocate "
          "%lu byte(s); remaining capacity is %lu byte(s)\n",
          long(size), long(capacity));
#endif  // KOKKOS_ENABLE_DEBUG
    } else {
      begin += increment;
    }
    return tmp;
  }

 public:
  KOKKOS_DEFAULTED_FUNCTION
  ScratchMemorySpace() = default;

  template <typename IntType>
  KOKKOS_INLINE_FUNCTION ScratchMemorySpace(sycl::local_ptr<void> ptr_L0,
                                            const IntType& size_L0,
                                            sycl::device_ptr<void> ptr_L1,
                                            const IntType& size_L1)
      : m_iter_L0(ptr_L0),
        m_iter_L1(ptr_L1),
        m_end_L0(sycl::local_ptr<char>(ptr_L0) + size_L0),
        m_end_L1(sycl::device_ptr<char>(ptr_L1) + size_L1),
        m_multiplier(1),
        m_offset(0),
        m_default_level(0) {}

  KOKKOS_INLINE_FUNCTION
  const ScratchMemorySpace& set_team_thread_mode(const int& level,
                                                 const int& multiplier,
                                                 const int& offset) const {
    m_default_level = level;
    m_multiplier    = multiplier;
    m_offset        = offset;
    return *this;
  }
};

template <int Level>
struct ScratchMemorySpaceWrapper {
  template <typename IntType>
  KOKKOS_INLINE_FUNCTION void* get_shmem_aligned(
      const IntType& size, const ptrdiff_t alignment) const {
    return m_scratch_space.get_shmem_aligned_on_level<Level>(size, alignment);
  }

  const ScratchMemorySpace<Kokkos::Experimental::SYCL>& m_scratch_space;
};

}  // namespace Kokkos

#endif /* #ifndef KOKKOS_SYCL_SCRATCHSPACE_HPP */
