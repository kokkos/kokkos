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

#ifndef KOKKOS_EXPERIMENTAL_MDSPAN_LAYOUT_HPP
#define KOKKOS_EXPERIMENTAL_MDSPAN_LAYOUT_HPP

#include "Kokkos_MDSpan_Extents.hpp"
#include <impl/Kokkos_ViewDataAnalysis.hpp>

namespace Kokkos::Impl {

template <class ArrayLayout>
struct LayoutFromArrayLayout;

template <>
struct LayoutFromArrayLayout<Kokkos::LayoutLeft> {
  using type = Experimental::layout_left_padded<dynamic_extent>;
};

template <>
struct LayoutFromArrayLayout<Kokkos::LayoutRight> {
  using type = Experimental::layout_right_padded<dynamic_extent>;
};

template <>
struct LayoutFromArrayLayout<Kokkos::LayoutStride> {
  using type = layout_stride;
};

/// Convert from a mdspan extent to a Kokkos extent, inserting 0s for static
/// extents
template <class Extents>
KOKKOS_INLINE_FUNCTION auto dimension_from_extent(const Extents &e,
                                                  std::size_t r) noexcept {
  return e.extent(r);
}

template <class ArrayLayout, class MDSpanType>
KOKKOS_INLINE_FUNCTION auto array_layout_from_mapping(
    const typename MDSpanType::mapping_type &mapping) {
  using mapping_type = typename MDSpanType::mapping_type;
  using extents_type = typename mapping_type::extents_type;

  constexpr auto rank = extents_type::rank();
  const auto &ext     = mapping.extents();

  static_assert(rank <= ARRAY_LAYOUT_MAX_RANK,
                "Unsupported rank for mdspan (must be <= 8)");

  if constexpr (std::is_same_v<ArrayLayout, LayoutStride>) {
    return Kokkos::LayoutStride{
        rank > 0 ? dimension_from_extent(ext, 0) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        rank > 0 ? mapping.stride(0) : 0,
        rank > 1 ? dimension_from_extent(ext, 1) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        rank > 1 ? mapping.stride(1) : 0,
        rank > 2 ? dimension_from_extent(ext, 2) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        rank > 2 ? mapping.stride(2) : 0,
        rank > 3 ? dimension_from_extent(ext, 3) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        rank > 3 ? mapping.stride(3) : 0,
        rank > 4 ? dimension_from_extent(ext, 4) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        rank > 4 ? mapping.stride(4) : 0,
        rank > 5 ? dimension_from_extent(ext, 5) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        rank > 5 ? mapping.stride(5) : 0,
        rank > 6 ? dimension_from_extent(ext, 6) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        rank > 6 ? mapping.stride(6) : 0,
        rank > 7 ? dimension_from_extent(ext, 7) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        rank > 7 ? mapping.stride(7) : 0,
    };
  } else {
    return ArrayLayout{
        rank > 0 ? dimension_from_extent(ext, 0) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        rank > 1 ? dimension_from_extent(ext, 1) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        rank > 2 ? dimension_from_extent(ext, 2) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        rank > 3 ? dimension_from_extent(ext, 3) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        rank > 4 ? dimension_from_extent(ext, 4) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        rank > 5 ? dimension_from_extent(ext, 5) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        rank > 6 ? dimension_from_extent(ext, 6) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        rank > 7 ? dimension_from_extent(ext, 7)
                 : KOKKOS_IMPL_CTOR_DEFAULT_ARG};
  }
}

template <class MDSpanType, class VM>
KOKKOS_INLINE_FUNCTION auto mapping_from_view_mapping(const VM &view_mapping) {
  using mapping_type = typename MDSpanType::mapping_type;
  using extents_type = typename mapping_type::extents_type;

  if constexpr (std::is_same_v<typename mapping_type::layout_type,
                               Kokkos::layout_stride>) {
    // std::span is not available in C++17 (our current requirements),
    // so we need to use the std::array constructor for layout mappings.
    // When C++20 is available, we can use std::span here instead
    std::array<std::size_t, VM::Rank> strides;
    view_mapping.stride_fill(strides.data());
    return mapping_type(extents_from_view_mapping<extents_type>(view_mapping),
                        strides);
  } else {
    return mapping_type(extents_from_view_mapping<extents_type>(view_mapping));
  }
}

}  // namespace Kokkos::Impl

#endif  // KOKKOS_EXPERIMENTAL_MDSPAN_LAYOUT_HPP
