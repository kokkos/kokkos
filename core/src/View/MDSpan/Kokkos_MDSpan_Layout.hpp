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

#include <utility>
#include "Kokkos_MDSpan_Extents.hpp"
#include <View/Kokkos_ViewDataAnalysis.hpp>

namespace Kokkos::Impl {

template <class ArrayLayout>
struct LayoutFromArrayLayout;

template <>
struct LayoutFromArrayLayout<Kokkos::LayoutLeft> {
  using type = Kokkos::Experimental::layout_left_padded<dynamic_extent>;
};

template <>
struct LayoutFromArrayLayout<Kokkos::LayoutRight> {
  using type = Kokkos::Experimental::layout_right_padded<dynamic_extent>;
};

template <>
struct LayoutFromArrayLayout<Kokkos::LayoutStride> {
  using type = layout_stride;
};

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
        rank > 0 ? ext.extent(0) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        rank > 0 ? mapping.stride(0) : 0,
        rank > 1 ? ext.extent(1) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        rank > 1 ? mapping.stride(1) : 0,
        rank > 2 ? ext.extent(2) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        rank > 2 ? mapping.stride(2) : 0,
        rank > 3 ? ext.extent(3) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        rank > 3 ? mapping.stride(3) : 0,
        rank > 4 ? ext.extent(4) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        rank > 4 ? mapping.stride(4) : 0,
        rank > 5 ? ext.extent(5) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        rank > 5 ? mapping.stride(5) : 0,
        rank > 6 ? ext.extent(6) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        rank > 6 ? mapping.stride(6) : 0,
        rank > 7 ? ext.extent(7) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        rank > 7 ? mapping.stride(7) : 0,
    };
  } else {
    // FIXME: Kokkos Layouts don't store stride (it's in the mapping)
    // We could conceivably fix this by adding an extra ViewCtorProp for
    // an abritrary padding. For now we will check for this.
    if constexpr (rank > 1 &&
                  (std::is_same_v<typename mapping_type::layout_type,
                                  Kokkos::Experimental::layout_left_padded<
                                      dynamic_extent>> ||
                   std::is_same_v<typename mapping_type::layout_type,
                                  Kokkos::Experimental::layout_right_padded<
                                      dynamic_extent>>)) {
      [[maybe_unused]] constexpr size_t strided_index =
          std::is_same_v<
              typename mapping_type::layout_type,
              Kokkos::Experimental::layout_left_padded<dynamic_extent>>
              ? 1
              : rank - 2;
      [[maybe_unused]] constexpr size_t extent_index =
          std::is_same_v<
              typename mapping_type::layout_type,
              Kokkos::Experimental::layout_left_padded<dynamic_extent>>
              ? 0
              : rank - 1;
      KOKKOS_ASSERT(mapping.stride(strided_index) == ext.extent(extent_index));
    }

    return ArrayLayout{rank > 0 ? ext.extent(0) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                       rank > 1 ? ext.extent(1) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                       rank > 2 ? ext.extent(2) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                       rank > 3 ? ext.extent(3) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                       rank > 4 ? ext.extent(4) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                       rank > 5 ? ext.extent(5) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                       rank > 6 ? ext.extent(6) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                       rank > 7 ? ext.extent(7) : KOKKOS_IMPL_CTOR_DEFAULT_ARG};
  }
#ifdef KOKKOS_COMPILER_INTEL
  __builtin_unreachable();
#endif
}

template <class MappingType, class ArrayLayout, size_t... Idx>
KOKKOS_INLINE_FUNCTION auto mapping_from_array_layout_impl(
    ArrayLayout layout, std::index_sequence<Idx...>) {
  using index_type   = typename MappingType::index_type;
  using extents_type = typename MappingType::extents_type;
  return MappingType{
      extents_type{dextents<index_type, MappingType::extents_type::rank()>{
          layout.dimension[Idx]...}}};
}
template <class MappingType, size_t... Idx>
KOKKOS_INLINE_FUNCTION auto mapping_from_array_layout_impl(
    LayoutStride layout, std::index_sequence<Idx...>) {
  static_assert(
      std::is_same_v<typename MappingType::layout_type, layout_stride>);
  using index_type = typename MappingType::index_type;
  index_type strides[MappingType::extents_type::rank()] = {
      layout.stride[Idx]...};
  return MappingType{
      mdspan_non_standard_tag(),
      static_cast<typename MappingType::extents_type>(
          dextents<index_type, MappingType::extents_type::rank()>{
              layout.dimension[Idx]...}),
      strides};
}

template <class MappingType, class ArrayLayout>
KOKKOS_INLINE_FUNCTION auto mapping_from_array_layout(ArrayLayout layout) {
  return mapping_from_array_layout_impl<MappingType>(
      layout, std::make_index_sequence<MappingType::extents_type::rank()>());
}

template <class MDSpanType, class VM>
KOKKOS_INLINE_FUNCTION auto mapping_from_view_mapping(const VM &view_mapping) {
  using mapping_type = typename MDSpanType::mapping_type;
  using extents_type = typename mapping_type::extents_type;

  // std::span is not available in C++17 (our current requirements),
  // so we need to use the std::array constructor for layout mappings.
  // FIXME When C++20 is available, we can use std::span here instead
  std::size_t strides[VM::Rank];
  view_mapping.stride_fill(&strides[0]);
  if constexpr (std::is_same_v<typename mapping_type::layout_type,
                               Kokkos::layout_stride>) {
    return mapping_type(Kokkos::mdspan_non_standard,
                        extents_from_view_mapping<extents_type>(view_mapping),
                        strides);
  } else if constexpr (VM::Rank > 1 &&
                       std::is_same_v<typename mapping_type::layout_type,
                                      Kokkos::Experimental::layout_left_padded<
                                          Kokkos::dynamic_extent>>) {
    return mapping_type(extents_from_view_mapping<extents_type>(view_mapping),
                        strides[1]);
  } else if constexpr (VM::Rank > 1 &&
                       std::is_same_v<typename mapping_type::layout_type,
                                      Kokkos::Experimental::layout_right_padded<
                                          Kokkos::dynamic_extent>>) {
    return mapping_type(extents_from_view_mapping<extents_type>(view_mapping),
                        strides[VM::Rank - 2]);
  } else {
    return mapping_type(extents_from_view_mapping<extents_type>(view_mapping));
  }
#ifdef KOKKOS_COMPILER_INTEL
  __builtin_unreachable();
#endif
}

template <class LayoutType>
struct is_layout_left_padded : std::false_type {};
template <size_t PaddingValue>
struct is_layout_left_padded<
    Kokkos::Experimental::layout_left_padded<PaddingValue>> : std::true_type {};
template <class LayoutType>
struct is_layout_right_padded : std::false_type {};
template <size_t PaddingValue>
struct is_layout_right_padded<
    Kokkos::Experimental::layout_right_padded<PaddingValue>> : std::true_type {
};

// This class existed in ViewOffset
template <unsigned ScalarSize>
struct Padding {
  static constexpr unsigned div =
      ScalarSize == 0 ? 0 : MEMORY_ALIGNMENT / ScalarSize;
  static constexpr unsigned mod =
      ScalarSize == 0 ? 0 : Kokkos::Impl::MEMORY_ALIGNMENT % ScalarSize;

  // If memory alignment is a multiple of the trivial scalar size then attempt
  // to align.
  static constexpr unsigned align  = ScalarSize != 0 && mod == 0 ? div : 0;
  static constexpr unsigned div_ok = (div != 0) ? div : 1;

  KOKKOS_INLINE_FUNCTION
  static constexpr size_t stride(size_t const N) {
    return ((align != 0) &&
            ((static_cast<int>(Kokkos::Impl::MEMORY_ALIGNMENT_THRESHOLD) *
              static_cast<int>(align)) < N) &&
            ((N % div_ok) != 0))
               ? N + align - (N % div_ok)
               : N;
  }
};

template <class MappingType, size_t ScalarSize, class... P, class... Sizes>
KOKKOS_INLINE_FUNCTION auto mapping_from_ctor_and_sizes(
    const Impl::ViewCtorProp<P...> &arg_prop, const Sizes... args) {
  using layout_t = typename MappingType::layout_type;
  using ext_t    = typename MappingType::extents_type;
  ext_t ext{args...};
  constexpr bool padded = Impl::ViewCtorProp<P...>::allow_padding;
  if constexpr (is_layout_left_padded<layout_t>::value && padded &&
                ext_t::rank() > 1) {
    return MappingType(ext, Padding<ScalarSize>::stride(ext.extent(0)));
  } else if constexpr (is_layout_right_padded<layout_t>::value && padded &&
                       ext_t::rank() > 1) {
    return MappingType(
        ext, Padding<ScalarSize>::stride(ext.extent(ext_t::rank() - 1)));
  } else {
    return MappingType(ext);
  }
}

template <class MappingType, size_t ScalarSize, class... P, class... Sizes>
KOKKOS_INLINE_FUNCTION auto mapping_from_ctor_and_8sizes(
    const ViewCtorProp<P...> &arg_prop, const size_t arg_N0,
    const size_t arg_N1, const size_t arg_N2, const size_t arg_N3,
    const size_t arg_N4, const size_t arg_N5, const size_t arg_N6,
    const size_t arg_N7) {
  if constexpr (MappingType::extents_type::rank() == 0) {
    return mapping_from_ctor_and_sizes<MappingType, ScalarSize>(arg_prop);
  } else if constexpr (MappingType::extents_type::rank() == 1) {
    return mapping_from_ctor_and_sizes<MappingType, ScalarSize>(arg_prop,
                                                                arg_N0);
  } else if constexpr (MappingType::extents_type::rank() == 2) {
    return mapping_from_ctor_and_sizes<MappingType, ScalarSize>(arg_prop,
                                                                arg_N0, arg_N1);
  } else if constexpr (MappingType::extents_type::rank() == 3) {
    return mapping_from_ctor_and_sizes<MappingType, ScalarSize>(
        arg_prop, arg_N0, arg_N1, arg_N2);
  } else if constexpr (MappingType::extents_type::rank() == 4) {
    return mapping_from_ctor_and_sizes<MappingType, ScalarSize>(
        arg_prop, arg_N0, arg_N1, arg_N2, arg_N3);
  } else if constexpr (MappingType::extents_type::rank() == 5) {
    return mapping_from_ctor_and_sizes<MappingType, ScalarSize>(
        arg_prop, arg_N0, arg_N1, arg_N2, arg_N3, arg_N4);
  } else if constexpr (MappingType::extents_type::rank() == 6) {
    return mapping_from_ctor_and_sizes<MappingType, ScalarSize>(
        arg_prop, arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5);
  } else if constexpr (MappingType::extents_type::rank() == 7) {
    return mapping_from_ctor_and_sizes<MappingType, ScalarSize>(
        arg_prop, arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5, arg_N6);
  } else if constexpr (MappingType::extents_type::rank() == 8) {
    return mapping_from_ctor_and_sizes<MappingType, ScalarSize>(
        arg_prop, arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5, arg_N6,
        arg_N7);
  }
}

}  // namespace Kokkos::Impl

#endif  // KOKKOS_EXPERIMENTAL_MDSPAN_LAYOUT_HPP
