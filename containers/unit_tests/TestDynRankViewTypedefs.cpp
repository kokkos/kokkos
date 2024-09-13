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

#include <Kokkos_Core.hpp>

namespace {

// clang-format off
template<class DataType>
struct data_analysis {
  using data_type = DataType;
  using const_data_type = const DataType;
  using runtime_data_type = DataType;
  using runtime_const_data_type = const DataType;
  using non_const_data_type = std::remove_const_t<DataType>;
};

template<class DataType>
struct data_analysis<DataType*> {
  using data_type = typename data_analysis<DataType>::data_type*;
  using const_data_type = typename data_analysis<DataType>::const_data_type*;
  using runtime_data_type = typename data_analysis<DataType>::runtime_data_type*;
  using runtime_const_data_type = typename data_analysis<DataType>::runtime_const_data_type*;
  using non_const_data_type = typename data_analysis<DataType>::non_const_data_type*;
};

template<class DataType, size_t N>
struct data_analysis<DataType[N]> {
  using data_type = typename data_analysis<DataType>::data_type[N];
  using const_data_type = typename data_analysis<DataType>::const_data_type[N];
  using runtime_data_type = typename data_analysis<DataType>::runtime_data_type*;
  using runtime_const_data_type = typename data_analysis<DataType>::runtime_const_data_type*;
  using non_const_data_type = typename data_analysis<DataType>::non_const_data_type[N];
};

template<class ViewType, class ViewTraitsType, class DataType, class Layout, class Space, class MemoryTraitsType,
         class HostMirrorSpace, class ValueType, class ReferenceType>
constexpr bool test_view_typedefs_impl() {
  // ========================
  // inherited from ViewTraits
  // ========================
  static_assert(std::is_same_v<typename ViewType::data_type, DataType>);
  static_assert(std::is_same_v<typename ViewType::const_data_type, typename  data_analysis<DataType>::const_data_type>);
  static_assert(std::is_same_v<typename ViewType::non_const_data_type, typename  data_analysis<DataType>::non_const_data_type>);
  
  // these should be deprecated and for proper testing (I.e. where this is different from data_type)
  // we would need ensemble types which use the hidden View dimension facility of View (i.e. which make
  // "specialize" not void
  static_assert(std::is_same_v<typename ViewType::scalar_array_type, DataType>);
  static_assert(std::is_same_v<typename ViewType::const_scalar_array_type, typename  data_analysis<DataType>::const_data_type>);
  static_assert(std::is_same_v<typename ViewType::non_const_scalar_array_type, typename  data_analysis<DataType>::non_const_data_type>);
  static_assert(std::is_same_v<typename ViewType::specialize, void>);

  // value_type definition conflicts with mdspan value_type
  static_assert(std::is_same_v<typename ViewType::value_type, ValueType>);
  static_assert(std::is_same_v<typename ViewType::const_value_type, const ValueType>);
  static_assert(std::is_same_v<typename ViewType::non_const_value_type, std::remove_const_t<ValueType>>);

  // should maybe be deprecated
  static_assert(std::is_same_v<typename ViewType::array_layout, Layout>);

  // should be deprecated and is some complicated impl type
  static_assert(!std::is_void_v<typename ViewType::dimension>);

  static_assert(std::is_same_v<typename ViewType::execution_space, typename Space::execution_space>);
  static_assert(std::is_same_v<typename ViewType::memory_space, typename Space::memory_space>);
  static_assert(std::is_same_v<typename ViewType::device_type, Kokkos::Device<typename ViewType::execution_space, typename ViewType::memory_space>>);
  static_assert(std::is_same_v<typename ViewType::memory_traits, MemoryTraitsType>);
  static_assert(std::is_same_v<typename ViewType::host_mirror_space, HostMirrorSpace>);
  static_assert(std::is_same_v<typename ViewType::size_type, typename ViewType::memory_space::size_type>);
 
  // should be deprecated in favor of reference
  static_assert(std::is_same_v<typename ViewType::reference_type, ReferenceType>);
  // should be deprecated in favor of data_handle_type
  static_assert(std::is_same_v<typename ViewType::pointer_type, ValueType*>);
 
  // =========================================
  // in Legacy View: some helper View variants
  // =========================================
  static_assert(std::is_same_v<typename ViewType::traits, ViewTraitsType>);
  static_assert(std::is_same_v<typename ViewType::array_type,
                               Kokkos::View<typename ViewType::scalar_array_type, typename ViewType::array_layout,
                                            typename ViewType::device_type, typename ViewTraitsType::hooks_policy,
                                            typename ViewType::memory_traits>>);
  static_assert(std::is_same_v<typename ViewType::const_type,
                               Kokkos::View<typename ViewType::const_data_type, typename ViewType::array_layout,
                                            typename ViewType::device_type, typename ViewTraitsType::hooks_policy,
                                            typename ViewType::memory_traits>>);
  static_assert(std::is_same_v<typename ViewType::non_const_type,
                               Kokkos::View<typename ViewType::non_const_data_type, typename ViewType::array_layout,
                                            typename ViewType::device_type, typename ViewTraitsType::hooks_policy,
                                            typename ViewType::memory_traits>>);
  static_assert(std::is_same_v<typename ViewType::host_mirror_type,
                               Kokkos::View<typename ViewType::non_const_data_type, typename ViewType::array_layout,
                                            Kokkos::Device<Kokkos::DefaultHostExecutionSpace,
                                                           typename ViewType::host_mirror_space::memory_space>,
                                            typename ViewTraitsType::hooks_policy>>);

  using uniform_layout_type = std::conditional_t<ViewType::rank()==0 || (ViewType::rank()==0 &&
                                                 std::is_same_v<Layout, Kokkos::LayoutRight>),
                                                 Kokkos::LayoutLeft, Layout>;

  // Uhm uniformtype removes all memorytraits?
  static_assert(std::is_same_v<typename ViewType::uniform_type,
                               Kokkos::View<typename ViewType::data_type, uniform_layout_type,
                                            typename ViewType::device_type, Kokkos::MemoryTraits<0>>>);
  static_assert(std::is_same_v<typename ViewType::uniform_const_type,
                               Kokkos::View<typename ViewType::const_data_type, uniform_layout_type,
                                            typename ViewType::device_type, Kokkos::MemoryTraits<0>>>);
  static_assert(std::is_same_v<typename ViewType::uniform_runtime_type,
                               Kokkos::View<typename data_analysis<DataType>::runtime_data_type, uniform_layout_type,
                                            typename ViewType::device_type, Kokkos::MemoryTraits<0>>>);
  static_assert(std::is_same_v<typename ViewType::uniform_runtime_const_type,
                               Kokkos::View<typename data_analysis<DataType>::runtime_const_data_type, uniform_layout_type,
                                            typename ViewType::device_type, Kokkos::MemoryTraits<0>>>);

  using anonymous_device_type = Kokkos::Device<typename ViewType::execution_space, Kokkos::AnonymousSpace>;
  static_assert(std::is_same_v<typename ViewType::uniform_nomemspace_type,
                               Kokkos::View<typename ViewType::data_type, uniform_layout_type,
                                            anonymous_device_type, Kokkos::MemoryTraits<0>>>);
  static_assert(std::is_same_v<typename ViewType::uniform_const_nomemspace_type,
                               Kokkos::View<typename ViewType::const_data_type, uniform_layout_type,
                                            anonymous_device_type, Kokkos::MemoryTraits<0>>>);
  static_assert(std::is_same_v<typename ViewType::uniform_runtime_nomemspace_type,
                               Kokkos::View<typename data_analysis<DataType>::runtime_data_type, uniform_layout_type,
                                            anonymous_device_type, Kokkos::MemoryTraits<0>>>);
  static_assert(std::is_same_v<typename ViewType::uniform_runtime_const_nomemspace_type,
                               Kokkos::View<typename data_analysis<DataType>::runtime_const_data_type, uniform_layout_type,
                                            anonymous_device_type, Kokkos::MemoryTraits<0>>>);


  // ==================================
  // mdspan compatibility
  // ==================================

  static_assert(std::is_same_v<typename ViewType::layout_type, Layout>);
  // Not supported yet
  // static_assert(std::is_same_v<typename ViewType::extents_type, >);
  // static_assert(std::is_same_v<typename ViewType::mapping_type, >);
  // static_assert(std::is_same_v<typename ViewType::accessor_type, >);

  static_assert(std::is_same_v<typename ViewType::element_type, ValueType>);
  // should be remove_const_t<element_type>
  static_assert(std::is_same_v<typename ViewType::value_type, ValueType>);
  // should be extents_type::index_type
  static_assert(std::is_same_v<typename ViewType::index_type, typename Space::memory_space::size_type>);
  static_assert(std::is_same_v<typename ViewType::size_type, std::make_unsigned_t<typename ViewType::index_type>>);
  static_assert(std::is_same_v<typename ViewType::rank_type, size_t>);

  // should come from accessor_type
  static_assert(std::is_same_v<typename ViewType::data_handle_type, typename ViewType::pointer_type>);
  static_assert(std::is_same_v<typename ViewType::reference, typename ViewType::reference_type>);
  return true;
};

template<class T, class ... ViewArgs>
struct ViewParams {};

template<class L, class S, class M, class HostMirrorSpace, class ValueType, class ReferenceType, class T, class ... ViewArgs>
constexpr bool test_view_typedefs(ViewParams<T, ViewArgs...>) {
  return test_view_typedefs_impl<Kokkos::View<T, ViewArgs...>, Kokkos::ViewTraits<T, ViewArgs...>,
                            T, L, S, M, HostMirrorSpace, ValueType, ReferenceType>();
}


constexpr bool is_host_exec = Kokkos::Impl::MemorySpaceAccess<
        Kokkos::DefaultExecutionSpace::memory_space,
        Kokkos::HostSpace>::accessible;

// These test take explicit template arguments for: LayoutType, Space, MemoryTraits, HostMirrorSpace, ValueType, ReferenceType
// The ViewParams is just a type pack for the View template arguments
static_assert(test_view_typedefs<
                  Kokkos::DefaultExecutionSpace::array_layout,
                  Kokkos::DefaultExecutionSpace, Kokkos::MemoryTraits<0>,
                  std::conditional_t<is_host_exec, Kokkos::DefaultExecutionSpace, Kokkos::HostSpace>,
                  int, int&>
                  (ViewParams<int>{}));
// WTF: HostMirrorSpace is different from the first one ....
static_assert(test_view_typedefs<
                  Kokkos::DefaultExecutionSpace::array_layout,
                  Kokkos::DefaultExecutionSpace, Kokkos::MemoryTraits<0>,
                  std::conditional_t<is_host_exec, Kokkos::HostSpace, Kokkos::HostSpace>,
                  int, int&>
                  (ViewParams<int, Kokkos::DefaultExecutionSpace>{}));
static_assert(test_view_typedefs<
                  Kokkos::LayoutRight,
                  Kokkos::HostSpace, Kokkos::MemoryTraits<0>,
                  Kokkos::HostSpace,
                  float, float&>
                  (ViewParams<float**, Kokkos::HostSpace>{}));
static_assert(test_view_typedefs<
                  Kokkos::LayoutLeft,
                  Kokkos::DefaultExecutionSpace, Kokkos::MemoryTraits<0>,
                  std::conditional_t<is_host_exec, Kokkos::DefaultExecutionSpace, Kokkos::HostSpace>,
                  float, float&>
                  (ViewParams<float*[3], Kokkos::LayoutLeft>{}));
static_assert(test_view_typedefs<
                  Kokkos::LayoutRight,
                  Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>, Kokkos::MemoryTraits<0>,
                  Kokkos::HostSpace,
                  float, float&>
                  (ViewParams<float[2][3], Kokkos::LayoutRight, Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>>{}));
static_assert(test_view_typedefs<
                  Kokkos::DefaultExecutionSpace::array_layout,
                  Kokkos::DefaultExecutionSpace, Kokkos::MemoryTraits<Kokkos::Atomic>,
                  std::conditional_t<is_host_exec, Kokkos::DefaultExecutionSpace, Kokkos::HostSpace>,
                  int, Kokkos::Impl::AtomicDataElement<Kokkos::ViewTraits<int, Kokkos::MemoryTraits<Kokkos::Atomic>>>>
                  (ViewParams<int, Kokkos::MemoryTraits<Kokkos::Atomic>>{}));
// clang-format on
}  // namespace
