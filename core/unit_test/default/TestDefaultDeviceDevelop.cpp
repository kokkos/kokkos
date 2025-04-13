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

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <TestDefaultDeviceType_Category.hpp>
#include <typeinfo>

namespace Foo {
struct SpecialScalar {};

template <class LayoutType, class DeviceType, class MemoryTraits>
auto mdspan_from_view_arguments(
    Kokkos::Impl::ViewArguments<SpecialScalar, LayoutType, DeviceType,
                                MemoryTraits>) {
  using mdspan_type_dummy =
      typename Kokkos::View<double*, LayoutType, DeviceType,
                            MemoryTraits>::mdspan_type;
  return Kokkos::mdspan<SpecialScalar,
                        Kokkos::extents<unsigned int, Kokkos::dynamic_extent>,
                        typename mdspan_type_dummy::layout_type,
                        Kokkos::Impl::SpaceAwareAccessor<
                            typename DeviceType::memory_space,
                            Kokkos::default_accessor<SpecialScalar>>>();
};

// This doesn't trigger because ADL doesn't find it
template <class LayoutType, class DeviceType, class MemoryTraits>
auto mdspan_from_view_argument(
    Kokkos::Impl::ViewArguments<double, LayoutType, DeviceType, MemoryTraits>) {
  using mdspan_type_dummy =
      typename Kokkos::View<double*, LayoutType, DeviceType,
                            MemoryTraits>::mdspan_type;
  return Kokkos::mdspan<SpecialScalar,
                        Kokkos::extents<unsigned int, Kokkos::dynamic_extent>,
                        typename mdspan_type_dummy::layout_type,
                        Kokkos::Impl::SpaceAwareAccessor<
                            typename DeviceType::memory_space,
                            Kokkos::default_accessor<SpecialScalar>>>();
};

}  // namespace Foo
namespace Test {

void foo() {
  static_assert(
      std::is_same_v<typename Kokkos::View<Foo::SpecialScalar>::index_type,
                     unsigned int>);
  static_assert(
      std::is_same_v<typename Kokkos::View<double>::index_type, size_t>);
}

TEST(defaultdevicetype, development_test) { foo(); }

}  // namespace Test
