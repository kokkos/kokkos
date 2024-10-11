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
#ifndef KOKKOS_SCRATCHSPACE_HPP
#define KOKKOS_SCRATCHSPACE_HPP

#include <cstdio>
#include <cstddef>
#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Concepts.hpp>

/*--------------------------------------------------------------------------*/

namespace Kokkos {

struct L0Tag {};

struct L1Tag {};

template <typename ExecSpace, typename Tag>
class ScratchMemorySpaceBase {
  static_assert(
      is_execution_space<ExecSpace>::value,
      "Instantiating ScratchMemorySpaceBase on non-execution-space type.");

  using PointerType = char*;

 public:
  // Minimal overalignment used by view scratch allocations
  constexpr static int ALIGN = 8;

 private:
  mutable PointerType m_iter = nullptr;
  PointerType m_end          = nullptr;

  mutable int m_multiplier = 0;
  mutable int m_offset     = 0;

 public:
  //! Tag this class as a memory space
  using memory_space    = ScratchMemorySpaceBase<ExecSpace, Tag>;
  using execution_space = ExecSpace;
  //! This execution space preferred device_type
  using device_type = Kokkos::Device<execution_space, memory_space>;

  using array_layout = typename ExecSpace::array_layout;
  using size_type    = typename ExecSpace::size_type;

  static constexpr const char* name() { return "ScratchMemorySpaceBase"; }

  template <typename IntType>
  KOKKOS_INLINE_FUNCTION void* get_shmem(const IntType& size) const {
    return get_shmem_common<false>(size, 1);
  }

  template <typename IntType>
  KOKKOS_INLINE_FUNCTION void* get_shmem_aligned(
      const IntType& size, const ptrdiff_t alignment) const {
    return get_shmem_common<true>(size, alignment);
  }

  template <bool alignment_requested, typename IntType>
  KOKKOS_INLINE_FUNCTION void* get_shmem_common(
      const IntType& size, [[maybe_unused]] const ptrdiff_t alignment) const {
    if constexpr (alignment_requested) {
      const ptrdiff_t missalign =
          size_t(static_cast<char*>(m_iter)) % alignment;
      if (missalign) m_iter += alignment - missalign;
    }

    // This is each thread's start pointer for its allocation
    // Note: for team scratch m_offset is 0, since every
    // thread will get back the same shared pointer
    PointerType tmp     = m_iter + m_offset * size;
    uintptr_t increment = size * m_multiplier;
    uintptr_t capacity  = m_end - m_iter;

    if (increment > capacity) {
      // Request did overflow: return nullptr and reset m_iter
      tmp = nullptr;
#ifdef KOKKOS_ENABLE_DEBUG
      // mfh 23 Jun 2015: printf call consumes 25 registers
      // in a CUDA build, so only print in debug mode.  The
      // function still returns nullptr if not enough memory.
      Kokkos::printf(
          "ScratchMemorySpaceBase<...>::get_shmem: Failed to allocate "
          "%ld byte(s); remaining capacity is %ld byte(s)\n",
          long(size), long(capacity));
#endif  // KOKKOS_ENABLE_DEBUG
    } else {
      m_iter += increment;
    }
    return tmp;
  }

 public:
  KOKKOS_DEFAULTED_FUNCTION
  ScratchMemorySpaceBase() = default;

  template <typename IntType>
  KOKKOS_INLINE_FUNCTION ScratchMemorySpaceBase(PointerType ptr,
                                                const IntType& size)
      : m_iter(static_cast<char*>(ptr)),
        m_end(static_cast<char*>(ptr) + size),
        m_multiplier(1),
        m_offset(0) {}

  KOKKOS_INLINE_FUNCTION
  const ScratchMemorySpaceBase& set_team_thread_mode(const int& multiplier,
                                                     const int& offset) const {
    m_multiplier = multiplier;
    m_offset     = offset;
    return *this;
  }
};

template <typename ExecSpace>
class ScratchMemorySpace {
  ScratchMemorySpaceBase<ExecSpace, L0Tag> m_scratch_L0;
  ScratchMemorySpaceBase<ExecSpace, L1Tag> m_scratch_L1;
  mutable int m_default_level = 0;

  using PointerTypeL0 = char*;
  using PointerTypeL1 = char*;

 public:
  // Minimal overalignment used by view scratch allocations
  constexpr static int ALIGN = 8;

  //! Tag this class as a memory space
  using memory_space    = ScratchMemorySpace;
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
    if (level == 0)
      return m_scratch_L0.get_shmem(size);
    else
      return m_scratch_L1.get_shmem(size);
  }

  template <typename IntType>
  KOKKOS_INLINE_FUNCTION void* get_shmem_aligned(const IntType& size,
                                                 const ptrdiff_t alignment,
                                                 int level = -1) const {
    if (level == -1) level = m_default_level;
    if (level == 0)
      return m_scratch_L0.get_shmem_aligned(size, alignment);
    else
      return m_scratch_L1.get_shmem_aligned(size, alignment);
  }

  KOKKOS_INLINE_FUNCTION
  const ScratchMemorySpace& set_team_thread_mode(const int& level,
                                                 const int& multiplier,
                                                 const int& offset) const {
    m_default_level = level;
    if (level == 0)
      m_scratch_L0.set_team_thread_mode(multiplier, offset);
    else
      m_scratch_L1.set_team_thread_mode(multiplier, offset);
    return *this;
  }

  template <int Level>
  KOKKOS_INLINE_FUNCTION const auto& set_team_thread_mode(
      const int& multiplier, const int& offset) const {
    if constexpr (Level == 0) {
      return m_scratch_L0.set_team_thread_mode(multiplier, offset);
    } else {
      static_assert(Level == 1);
      return m_scratch_L1.set_team_thread_mode(multiplier, offset);
    }
  }

  KOKKOS_DEFAULTED_FUNCTION
  ScratchMemorySpace() = default;

  template <typename IntType>
  KOKKOS_INLINE_FUNCTION ScratchMemorySpace(void* ptr_L0,
                                            const IntType& size_L0,
                                            void* ptr_L1           = nullptr,
                                            const IntType& size_L1 = 0)
      : m_scratch_L0(PointerTypeL0(static_cast<char*>(ptr_L0)), size_L0),
        m_scratch_L1(PointerTypeL1(static_cast<char*>(ptr_L1)), size_L1) {}
};

}  // namespace Kokkos

#endif /* #ifndef KOKKOS_SCRATCHSPACE_HPP */
