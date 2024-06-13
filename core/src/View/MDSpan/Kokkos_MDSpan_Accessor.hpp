
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
#include <impl/Kokkos_Utilities.hpp>

namespace Kokkos {

// For now use the accessors in IMPL namespace, as an
// implementation detail for rebasing View on mdspan
namespace Impl {

template <class MemorySpace, class NestedAccessor>
struct SpaceAwareAccessor {
  using element_type     = typename NestedAccessor::element_type;
  using reference        = typename NestedAccessor::reference;
  using data_handle_type = typename NestedAccessor::data_handle_type;

  using offset_policy =
      SpaceAwareAccessor<MemorySpace, typename NestedAccessor::offset_policy>;

  using memory_space = MemorySpace;

  static_assert(is_memory_space<memory_space>::value);

  KOKKOS_DEFAULTED_FUNCTION
  constexpr SpaceAwareAccessor() = default;

  template <class OtherNestedAccessorType,
            std::enable_if_t<std::is_constructible_v<NestedAccessor,
                                                     OtherNestedAccessorType>,
                             int> = 0>
  KOKKOS_FUNCTION constexpr SpaceAwareAccessor(
      const SpaceAwareAccessor<MemorySpace, OtherNestedAccessorType>&
          other) noexcept
      : nested_acc(other.nested_acc) {}

  KOKKOS_FUNCTION
  SpaceAwareAccessor(const NestedAccessor& acc) : nested_acc(acc) {}

  KOKKOS_FUNCTION
  explicit operator NestedAccessor() const { return nested_acc; }

  KOKKOS_FUNCTION
  constexpr reference access(data_handle_type p, size_t i) const noexcept {
    // Same error as Kokkos view in case anyone greps for this in their tools
    Kokkos::Impl::runtime_check_memory_access_violation<memory_space>(
        "Kokkos::View ERROR: attempt to access inaccessible memory space");
    return nested_acc.access(p, i);
  }

  KOKKOS_FUNCTION
  constexpr data_handle_type offset(data_handle_type p,
                                    size_t i) const noexcept {
    return nested_acc.offset(p, i);
  }


 private:
  [[no_unique_address]] NestedAccessor nested_acc;
  template<class, class>
  friend struct SpaceAwareAccessor;
};

}  // namespace Impl
}  // namespace Kokkos

#endif
