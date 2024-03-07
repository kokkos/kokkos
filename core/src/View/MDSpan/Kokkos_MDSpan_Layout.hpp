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

template <>
struct ArrayLayoutFromLayout<layout_stride> {
  using type = Kokkos::LayoutStride;
};

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

template <class ArrayLayout>
struct ArrayLayoutFromMappingImpl;

template <class ArrayLayout, class MDSpanType>
KOKKOS_INLINE_FUNCTION auto array_layout_leftright_from_mapping_impl(
    const typename MDSpanType::mapping_type &mapping) {
  using mapping_type = typename MDSpanType::mapping_type;
  using extents_type = typename mapping_type::extents_type;

  static constexpr auto rank = extents_type::rank();
  const auto &ext            = mapping.extents();

  static_assert(rank <= ARRAY_LAYOUT_MAX_RANK,
                "Unsupported rank for mdspan (must be <= 8)");
  return ArrayLayout{
      rank > 0 ? dimension_from_extent(ext, 0) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      rank > 1 ? dimension_from_extent(ext, 1) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      rank > 2 ? dimension_from_extent(ext, 2) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      rank > 3 ? dimension_from_extent(ext, 3) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      rank > 4 ? dimension_from_extent(ext, 4) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      rank > 5 ? dimension_from_extent(ext, 5) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      rank > 6 ? dimension_from_extent(ext, 6) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      rank > 7 ? dimension_from_extent(ext, 7) : KOKKOS_IMPL_CTOR_DEFAULT_ARG};
}

template <>
struct ArrayLayoutFromMappingImpl<Kokkos::LayoutLeft> {
  template <class MDSpanType>
  KOKKOS_INLINE_FUNCTION static Kokkos::LayoutLeft construct(
      const typename MDSpanType::mapping_type &mapping) {
    return array_layout_leftright_from_mapping_impl<Kokkos::LayoutLeft,
                                                    MDSpanType>(mapping);
  }
};

template <>
struct ArrayLayoutFromMappingImpl<Kokkos::LayoutRight> {
  template <class MDSpanType>
  KOKKOS_INLINE_FUNCTION static Kokkos::LayoutRight construct(
      const typename MDSpanType::mapping_type &mapping) {
    return array_layout_leftright_from_mapping_impl<Kokkos::LayoutRight,
                                                    MDSpanType>(mapping);
  }
};

template <>
struct ArrayLayoutFromMappingImpl<Kokkos::LayoutStride> {
  template <class MDSpanType>
  KOKKOS_INLINE_FUNCTION static Kokkos::LayoutStride construct(
      const typename MDSpanType::mapping_type &mapping) {
    using mapping_type = typename MDSpanType::mapping_type;
    using extents_type = typename mapping_type::extents_type;

    static constexpr auto rank = extents_type::rank();
    const auto &ext            = mapping.extents();

    static_assert(rank <= ARRAY_LAYOUT_MAX_RANK,
                  "Unsupported rank for mdspan (must be <= 8)");
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
  }
};

template <class ArrayLayout, class MDSpanType>
KOKKOS_INLINE_FUNCTION auto array_layout_from_mapping(
    const typename MDSpanType::mapping_type &mapping) {
  return ArrayLayoutFromMappingImpl<ArrayLayout>::template construct<MDSpanType>(
      mapping);
}

template <class MDSpanType, class VM>
KOKKOS_INLINE_FUNCTION auto mapping_from_view_mapping(const VM &view_mapping) {
  using mapping_type = typename MDSpanType::mapping_type;
  using extents_type = typename mapping_type::extents_type;

  if constexpr (std::is_same_v<typename mapping_type::layout_type,
                               Kokkos::layout_stride>) {
    std::array<std::size_t, VM::Rank> strides;
    view_mapping.stride(strides.data());
    return mapping_type(extents_from_view_mapping<extents_type>(view_mapping),
                        strides);
  } else {
    return mapping_type(extents_from_view_mapping<extents_type>(view_mapping));
  }
}

template <class ElementType, class Extents, class LayoutPolicy,
          class AccessorPolicy>
KOKKOS_INLINE_FUNCTION auto view_offset_from_mdspan(
    const mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy> &mds) {
  using offset_type =
      typename ViewOffsetFromExtents<ElementType, Extents, LayoutPolicy>::type;
  static constexpr auto padding = ArrayLayoutFromLayout<LayoutPolicy>::padding;
  return offset_type(padding, array_layout_from_mdspan(mds));
};
}  // namespace Kokkos::Experimental::Impl

#endif  // KOKKOS_EXPERIMENTAL_MDSPAN_LAYOUT_HPP
