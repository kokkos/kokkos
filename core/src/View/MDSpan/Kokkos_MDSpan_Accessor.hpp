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
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif

#ifndef KOKKOS_MDSPAN_ACCESSOR_HPP
#define KOKKOS_MDSPAN_ACCESSOR_HPP

#include <Kokkos_Macros.hpp>
#include <impl/Kokkos_Utilities.hpp>
#include <impl/Kokkos_SharedAlloc.hpp>
#include <Kokkos_Atomics_Desul_Wrapper.hpp>
#include "Kokkos_MDSpan_Header.hpp"

#ifdef KOKKOS_INTERNAL_NOT_PARALLEL
#define KOKKOS_DESUL_MEM_SCOPE desul::MemoryScopeCaller()
#else
#define KOKKOS_DESUL_MEM_SCOPE desul::MemoryScopeDevice()
#endif

namespace Kokkos {
namespace Impl {
template <typename T>
using AtomicRef = desul::scoped_atomic_ref<T, desul::MemoryOrderRelaxed,
                                           KOKKOS_DESUL_MEM_SCOPE>;
}  // namespace Impl

/**
 * @brief mdspan accessor compatible with Kokkos memory/execution spaces
 *
 * @tparam ElementType  the element type
 * @tparam alignment    the alignment of the elements, 0 means unaligned
 * @tparam is_restrict  whether the data is restrict
 * @tparam is_atomic    whether the data is atomic
 * @tparam ExecutionSpace the execution space
 * @tparam MemorySpace  the memory space
 */
template <class ElementType, size_t alignment = 0, bool is_restrict = false,
          bool is_atomic       = false,
          class ExecutionSpace = Kokkos::DefaultExecutionSpace,
          class MemorySpace    = typename ExecutionSpace::memory_space,
          class Enabled        = void>
class SpaceAwareAccessor {
  using offset_policy = SpaceAwareAccessor;
  using element_type  = ElementType;
  using reference =
      std::conditional_t<is_restrict, ElementType & KOKKOS_RESTRICT,
                         ElementType &>;
  using data_handle_type =
      std::conditional_t<is_restrict, ElementType * KOKKOS_RESTRICT,
                         ElementType *>;
  using execution_space = ExecutionSpace;
  using memory_space    = MemorySpace;

  constexpr SpaceAwareAccessor() = default;

  template <class OtherElementType,
            std::enable_if_t<std::is_convertible_v<OtherElementType (*)[],
                                                   element_type (*)[]>,
                             int> = 0>
  /* implicit */ constexpr SpaceAwareAccessor(
      SpaceAwareAccessor<OtherElementType, alignment, is_restrict, is_atomic,
                         ExecutionSpace, MemorySpace>) noexcept {}

  constexpr reference access(data_handle_type p, size_t i) const noexcept {
    // Same error as Kokkos view in case anyone greps for this in their tools
    Kokkos::Impl::runtime_check_memory_access_violation<memory_space>(
        "Kokkos::View ERROR: attempt to access inaccessible memory space");
    return p[i];
  }

  constexpr data_handle_type offset(data_handle_type p,
                                    size_t i) const noexcept {
    return p + i;
  }
};

/**
 * @brief Specialization for non-const atomics
 *
 * See wg21.link/p2689 for rough equivalent
 */
template <class ElementType, size_t alignment, bool is_restrict,
          class ExecutionSpace, class MemorySpace>
class SpaceAwareAccessor<ElementType, alignment, is_restrict, true,
                         ExecutionSpace, MemorySpace,
                         std::enable_if_t<!std::is_const_v<ElementType>>> {
  using offset_policy    = SpaceAwareAccessor;
  using element_type     = ElementType;
  using reference        = Impl::AtomicRef<ElementType>;
  using data_handle_type = ElementType *;
  using execution_space  = ExecutionSpace;
  using memory_space     = MemorySpace;

  constexpr SpaceAwareAccessor() = default;

  template <class OtherElementType,
            std::enable_if_t<std::is_convertible_v<OtherElementType (*)[],
                                                   element_type (*)[]>,
                             int> = 0>
  /* implicit */ constexpr SpaceAwareAccessor(
      SpaceAwareAccessor<OtherElementType, alignment, is_restrict, true,
                         ExecutionSpace, MemorySpace>) noexcept {}

  constexpr reference access(data_handle_type p, size_t i) const noexcept {
    // Same error as Kokkos view in case anyone greps for this in their tools
    Kokkos::Impl::runtime_check_memory_access_violation<memory_space>(
        "Kokkos::View ERROR: attempt to access inaccessible memory space");
    return reference(p[i]);
  }

  constexpr data_handle_type offset(data_handle_type p,
                                    size_t i) const noexcept {
    return p + i;
  }
};

/**
 * @brief Specialization for aligned (either with restrict or without)
 *
 * See wg21.link/p2897
 */
template <class ElementType, size_t alignment, bool is_restrict,
          class ExecutionSpace, class MemorySpace>
class SpaceAwareAccessor<ElementType, alignment, is_restrict, false,
                         ExecutionSpace, MemorySpace,
                         std::enable_if_t<(alignment != 0)>> {
  using offset_policy = SpaceAwareAccessor<ElementType, alignment, is_restrict,
                                           false, ExecutionSpace, MemorySpace>;
  using element_type  = ElementType;
  using reference =
      std::conditional_t<is_restrict, ElementType & KOKKOS_RESTRICT,
                         ElementType &>;
  using data_handle_type =
      std::conditional_t<is_restrict, ElementType * KOKKOS_RESTRICT,
                         ElementType *>;
  using execution_space = ExecutionSpace;
  using memory_space    = MemorySpace;

  static constexpr size_t byte_alignment = alignment;

  constexpr SpaceAwareAccessor() = default;

  template <class OtherElementType, size_t other_byte_alignment,
            std::enable_if_t<std::is_convertible_v<OtherElementType (*)[],
                                                   element_type (*)[]> &&
                                 (other_byte_alignment != 0),
                             int> = 0>
  /* implicit */ constexpr SpaceAwareAccessor(
      SpaceAwareAccessor<OtherElementType, other_byte_alignment, is_restrict,
                         false, ExecutionSpace, MemorySpace>) noexcept {
    static_assert(std::gcd(other_byte_alignment, other_byte_alignment) ==
                  byte_alignment);
  }

  template <class OtherElementType,
            std::enable_if_t<std::is_convertible_v<OtherElementType (*)[],
                                                   element_type (*)[]>,
                             int> = 0>
  explicit constexpr SpaceAwareAccessor(
      SpaceAwareAccessor<OtherElementType, 0, is_restrict, false,
                         ExecutionSpace, MemorySpace>) noexcept {}

  constexpr reference access(data_handle_type p, size_t i) const noexcept {
    // Same error as Kokkos view in case anyone greps for this in their tools
    Kokkos::Impl::runtime_check_memory_access_violation<memory_space>(
        "Kokkos::View ERROR: attempt to access inaccessible memory space");
    return KOKKOS_IMPL_ASSUME_ALIGNED(element_type, p, byte_alignment)[i];
  }

  constexpr data_handle_type offset(data_handle_type p,
                                    size_t i) const noexcept {
    return p + i;
  }

  static constexpr bool is_sufficiently_aligned(data_handle_type p) {
    return !(reinterpret_cast<uintptr_t>(p) % byte_alignment);
  }
};

}  // namespace Kokkos

#undef KOKKOS_DESUL_MEM_SCOPE

#endif
