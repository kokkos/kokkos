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
#include <Kokkos_Concepts.hpp>
#include <Kokkos_Core_fwd.hpp>

namespace Kokkos {

// For now use the accessors in Impl namespace, as an
// implementation detail for rebasing View on mdspan
namespace Impl {

template <class MemorySpace, class NestedAccessor>
struct SpaceAwareAccessor {
  // Part of Accessor Requirements
  using element_type     = typename NestedAccessor::element_type;
  using reference        = typename NestedAccessor::reference;
  using data_handle_type = typename NestedAccessor::data_handle_type;
  using offset_policy =
      SpaceAwareAccessor<MemorySpace, typename NestedAccessor::offset_policy>;

  // Specific to SpaceAwareAccessor
  using memory_space         = MemorySpace;
  using nested_accessor_type = NestedAccessor;

  static_assert(is_memory_space_v<memory_space>);

  KOKKOS_DEFAULTED_FUNCTION
  constexpr SpaceAwareAccessor() = default;

  template <
      class OtherMemorySpace, class OtherNestedAccessorType,
      std::enable_if_t<
          MemorySpaceAccess<MemorySpace, OtherMemorySpace>::assignable &&
              std::is_constructible_v<NestedAccessor, OtherNestedAccessorType>,
          int> = 0>
  KOKKOS_FUNCTION constexpr SpaceAwareAccessor(
      const SpaceAwareAccessor<OtherMemorySpace, OtherNestedAccessorType>&
          other) noexcept
      : nested_acc(other.nested_acc) {}

  KOKKOS_FUNCTION
  SpaceAwareAccessor(const NestedAccessor& acc) : nested_acc(acc) {}

  KOKKOS_FUNCTION
  explicit operator NestedAccessor() const { return nested_acc; }

  KOKKOS_FUNCTION
  constexpr reference access(data_handle_type p, size_t i) const noexcept {
    Kokkos::Impl::runtime_check_memory_access_violation<memory_space>(
        "Kokkos::SpaceAwareAccessor ERROR: attempt to access inaccessible "
        "memory space");
    return nested_acc.access(p, i);
  }

  KOKKOS_FUNCTION
  constexpr auto offset(data_handle_type p, size_t i) const noexcept {
    return nested_acc.offset(p, i);
  }

 private:
// We either compile with our custom mdspan impl
// in which case we discover inside it whether no_unique_address
// works, or we use C++23 in which case it better be available
#ifdef _MDSPAN_NO_UNIQUE_ADDRESS
  _MDSPAN_NO_UNIQUE_ADDRESS
#else
  [[no_unique_address]]
#endif
  NestedAccessor nested_acc;
  template <class, class>
  friend struct SpaceAwareAccessor;
};

template <class NestedAccessor>
struct SpaceAwareAccessor<AnonymousSpace, NestedAccessor> {
  // Part of Accessor Requirements
  using element_type     = typename NestedAccessor::element_type;
  using reference        = typename NestedAccessor::reference;
  using data_handle_type = typename NestedAccessor::data_handle_type;

  using offset_policy =
      SpaceAwareAccessor<AnonymousSpace,
                         typename NestedAccessor::offset_policy>;

  // Specific to SpaceAwareAccessor
  using memory_space         = AnonymousSpace;
  using nested_accessor_type = NestedAccessor;

  KOKKOS_DEFAULTED_FUNCTION
  constexpr SpaceAwareAccessor() = default;

  template <class OtherMemorySpace, class OtherNestedAccessorType,
            std::enable_if_t<std::is_constructible_v<NestedAccessor,
                                                     OtherNestedAccessorType>,
                             int> = 0>
  KOKKOS_FUNCTION constexpr SpaceAwareAccessor(
      const SpaceAwareAccessor<OtherMemorySpace, OtherNestedAccessorType>&
          other) noexcept
      : nested_acc(other.nested_acc) {}

  KOKKOS_FUNCTION
  SpaceAwareAccessor(const NestedAccessor& acc) : nested_acc(acc) {}

  KOKKOS_FUNCTION
  explicit operator NestedAccessor() const { return nested_acc; }

  KOKKOS_FUNCTION
  constexpr reference access(data_handle_type p, size_t i) const noexcept {
    return nested_acc.access(p, i);
  }

  KOKKOS_FUNCTION
  constexpr auto offset(data_handle_type p, size_t i) const noexcept {
    return nested_acc.offset(p, i);
  }

 private:
// We either compile with our custom mdspan impl
// in which case we discover inside it whether no_unique_address
// works, or we use C++23 in which case it better be available
#ifdef _MDSPAN_NO_UNIQUE_ADDRESS
  _MDSPAN_NO_UNIQUE_ADDRESS
#else
  [[no_unique_address]]
#endif
  NestedAccessor nested_acc;
  template <class, class>
  friend struct SpaceAwareAccessor;
};

}  // namespace Impl
}  // namespace Kokkos

#endif
