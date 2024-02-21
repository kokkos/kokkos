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
#include <Kokkos_Core_fwd.hpp>
#include <impl/Kokkos_ViewDataAnalysis.hpp>

namespace Kokkos::Experimental::Impl {
template <class Layout>
struct ArrayLayoutFromLayout;

template <std::size_t padding_value>
struct ArrayLayoutFromLayout<Experimental::layout_left_padded<padding_value>> {
  using type = Kokkos::LayoutLeft;
  static constexpr std::integral_constant<unsigned,
                                          static_cast<unsigned>(padding_value)>
      padding = {};
};

template <std::size_t padding_value>
struct ArrayLayoutFromLayout<Experimental::layout_right_padded<padding_value>> {
  using type = Kokkos::LayoutRight;
  static constexpr std::integral_constant<unsigned,
                                          static_cast<unsigned>(padding_value)>
      padding = {};
};

template <class T, class Extents, class Layout>
struct ViewOffsetFromExtents {
  using value_type   = T;
  using data_type    = typename DataTypeFromExtents<value_type, Extents>::type;
  using array_layout = typename ArrayLayoutFromLayout<Layout>::type;
  using data_analysis =
      Kokkos::Impl::ViewDataAnalysis<data_type, array_layout, value_type>;
  using type =
      Kokkos::Impl::ViewOffset<typename data_analysis::dimension, array_layout>;
};

template <class ElementType, class Extents, class LayoutPolicy,
          class AccessorPolicy>
auto array_layout_from_mdspan(
    const mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy> &mds) {
  using layout_type = typename ArrayLayoutFromLayout<LayoutPolicy>::type;
  const auto &ext   = mds.extents();

  static constexpr auto rank = Extents::rank();

  static_assert(rank <= ARRAY_LAYOUT_MAX_RANK,
                "Unsupported rank for mdspan (must be <= 8)");
  return layout_type{
      rank > 0 ? dimension_from_extent(ext, 0) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      rank > 1 ? dimension_from_extent(ext, 1) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      rank > 2 ? dimension_from_extent(ext, 2) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      rank > 3 ? dimension_from_extent(ext, 3) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      rank > 4 ? dimension_from_extent(ext, 4) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      rank > 5 ? dimension_from_extent(ext, 5) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      rank > 6 ? dimension_from_extent(ext, 6) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      rank > 7 ? dimension_from_extent(ext, 7) : KOKKOS_IMPL_CTOR_DEFAULT_ARG};
}

template <class ElementType, class Extents, class LayoutPolicy,
          class AccessorPolicy>
auto view_offset_from_mdspan(
    const mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy> &mds) {
  using offset_type =
      typename ViewOffsetFromExtents<ElementType, Extents, LayoutPolicy>::type;
  static constexpr auto padding = ArrayLayoutFromLayout<LayoutPolicy>::padding;
  return offset_type(padding, array_layout_from_mdspan(mds));
};
}  // namespace Kokkos::Experimental::Impl

#endif  // KOKKOS_EXPERIMENTAL_MDSPAN_LAYOUT_HPP
