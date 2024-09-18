// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_EXPERIMENTAL_VIEWUNIFORMTYPE_HPP
#define KOKKOS_EXPERIMENTAL_VIEWUNIFORMTYPE_HPP

#include <Kokkos_Macros.hpp>

namespace Kokkos {
namespace Impl {
template <class ScalarType, int Rank>
struct ViewScalarToDataType {
  using type = typename ViewScalarToDataType<ScalarType, Rank - 1>::type *;
  using const_type =
      typename ViewScalarToDataType<ScalarType, Rank - 1>::const_type *;
};

template <class ScalarType>
struct ViewScalarToDataType<ScalarType, 0> {
  using type       = ScalarType;
  using const_type = const ScalarType;
};

template <class LayoutType, int Rank, bool is_customized>
struct ViewUniformLayout {
  using layout_type = LayoutType;
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_5
  using array_layout KOKKOS_DEPRECATED_WITH_COMMENT(
      "Use layout_type instead.") = layout_type;
#endif
};

template <class LayoutType>
struct ViewUniformLayout<LayoutType, 0, false> {
  using layout_type = Kokkos::LayoutLeft;
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_5
  using array_layout KOKKOS_DEPRECATED_WITH_COMMENT(
      "Use layout_type instead.") = layout_type;
#endif
};

template <>
struct ViewUniformLayout<Kokkos::LayoutRight, 1, false> {
  using layout_type = Kokkos::LayoutLeft;
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_5
  using array_layout KOKKOS_DEPRECATED_WITH_COMMENT(
      "Use layout_type instead.") = layout_type;
#endif
};

template <class ViewType, int Traits>
struct ViewUniformType {
// FIXME: legacy view does not support operator() with rank
#ifdef KOKKOS_ENABLE_IMPL_VIEW_LEGACY
  static constexpr size_t rank = ViewType::rank;
#else
  static constexpr size_t rank = ViewType::rank();
#endif

  using data_type       = typename ViewType::data_type;
  using const_data_type = typename ViewType::const_data_type;
  using runtime_data_type =
      typename ViewScalarToDataType<typename ViewType::value_type, rank>::type;
  using runtime_const_data_type = typename ViewScalarToDataType<
      std::add_const_t<typename ViewType::value_type>, rank>::type;

  using layout_type = typename ViewUniformLayout<
      typename ViewType::layout_type, rank,
      ViewType::traits::impl_is_customized>::layout_type;
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_5
  using array_layout KOKKOS_DEPRECATED_WITH_COMMENT(
      "Use layout_type instead.") = layout_type;
#endif
  using device_type = typename ViewType::device_type;
  using anonymous_device_type =
      typename Kokkos::Device<typename device_type::execution_space,
                              Kokkos::AnonymousSpace>;

  using memory_traits = typename Kokkos::MemoryTraits<Traits>;
  using type = Kokkos::View<data_type, layout_type, device_type, memory_traits>;
  using const_type =
      Kokkos::View<const_data_type, layout_type, device_type, memory_traits>;
  using runtime_type =
      Kokkos::View<runtime_data_type, layout_type, device_type, memory_traits>;
  using runtime_const_type = Kokkos::View<runtime_const_data_type, layout_type,
                                          device_type, memory_traits>;

  using nomemspace_type = Kokkos::View<data_type, layout_type,
                                       anonymous_device_type, memory_traits>;
  using const_nomemspace_type =
      Kokkos::View<const_data_type, layout_type, anonymous_device_type,
                   memory_traits>;
  using runtime_nomemspace_type =
      Kokkos::View<runtime_data_type, layout_type, anonymous_device_type,
                   memory_traits>;
  using runtime_const_nomemspace_type =
      Kokkos::View<runtime_const_data_type, layout_type, anonymous_device_type,
                   memory_traits>;
};
}  // namespace Impl
}  // namespace Kokkos

#endif
