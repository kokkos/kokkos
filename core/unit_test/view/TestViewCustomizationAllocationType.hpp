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

#include <gtest/gtest.h>

namespace Foo {

// ElementType which is actually just a tag type
struct Bar {};

// Reference type for Bar
struct BarRef {
  double* ptr{nullptr};
  size_t size{0ul};
  KOKKOS_FUNCTION
  double& operator[](size_t idx) const { return ptr[idx]; }
};

// A TestAccessor mimicking some of what Sacado does
// Specifically:
//   * returns a proxy reference
//   * the underlying storage is of some basic scalar type
//   * the size of the allocation does not actually come from
//   sizeof(ElementType)
template <class ElementType, class MemorySpace>
struct TestAccessor {
  static_assert(std::is_same_v<std::remove_cv_t<ElementType>, Bar>);
  using element_type     = ElementType;
  using reference        = BarRef;
  using data_handle_type = Kokkos::Impl::ReferenceCountedDataHandle<
      std::conditional_t<std::is_const_v<ElementType>, const double, double>,
      MemorySpace>;
  using offset_policy = TestAccessor;

  // View expects this from accessors right now
  using memory_space = MemorySpace;

  KOKKOS_DEFAULTED_FUNCTION
  constexpr TestAccessor() = default;

  template <class OtherElementType,
            std::enable_if_t<std::is_constructible_v<
                                 Kokkos::default_accessor<element_type>,
                                 Kokkos::default_accessor<OtherElementType>>,
                             int> = 0>
  KOKKOS_FUNCTION constexpr TestAccessor(
      const TestAccessor<OtherElementType, MemorySpace>& other) noexcept
      : size(other.size) {}

  KOKKOS_FUNCTION
  TestAccessor(const size_t val) : size(val) {}

  KOKKOS_FUNCTION
  constexpr reference access(
#ifndef KOKKOS_ENABLE_OPENACC
      const data_handle_type& p,
#else
      // FIXME OPENACC: illegal address when passing by reference
      data_handle_type p,
#endif
      size_t i) const noexcept {
    return BarRef{(p.get() + i * size), size};
  }

  KOKKOS_FUNCTION
  constexpr typename offset_policy::data_handle_type offset(
#ifndef KOKKOS_ENABLE_OPENACC
      const data_handle_type& p,
#else
      // FIXME OPENACC: illegal address when passing by reference
      data_handle_type p,
#endif
      size_t i) const noexcept {
    return p + i * size;
  }

  size_t size{0lu};
};

// Use the customization point to inject the custom accessor
template <class LayoutType, class DeviceType, class MemoryTraits>
constexpr auto customize_view_arguments(
    Kokkos::Impl::ViewArguments<Bar, LayoutType, DeviceType, MemoryTraits>) {
  return Kokkos::Impl::ViewCustomArguments<
      size_t, TestAccessor<Bar, typename DeviceType::memory_space>>{};
}

template <class LayoutType, class DeviceType, class MemoryTraits>
constexpr auto customize_view_arguments(
    Kokkos::Impl::ViewArguments<const Bar, LayoutType, DeviceType,
                                MemoryTraits>) {
  return Kokkos::Impl::ViewCustomArguments<
      size_t, TestAccessor<const Bar, typename DeviceType::memory_space>>{};
}

// Customization point to compute allocation sizes
template <class MappingType, class ElementType, class MemorySpace>
size_t allocation_size_from_mapping_and_accessor(
    const MappingType& map, const TestAccessor<ElementType, MemorySpace>& acc) {
  return map.required_span_size() * acc.size;
}
}  // namespace Foo

void test_allocation_type() {
  size_t ext0 = 10;
  size_t ext1 = 10;
  size_t size = 5;

  using view_t = Kokkos::View<Foo::Bar**, TEST_EXECSPACE>;

  // Make sure I got the accessor I expect
  static_assert(
      std::is_same_v<
          view_t::accessor_type,
          Foo::TestAccessor<Foo::Bar, typename TEST_EXECSPACE::memory_space>>);
  static_assert(std::is_same_v<view_t::pointer_type, double*>);

  using c_view_t = Kokkos::View<const Foo::Bar**, TEST_EXECSPACE>;
  static_assert(
      std::is_same_v<c_view_t::accessor_type,
                     Foo::TestAccessor<const Foo::Bar,
                                       typename TEST_EXECSPACE::memory_space>>);
  static_assert(std::is_same_v<c_view_t::pointer_type, const double*>);

  // accessor will be constructed from AccessorArg_t
  view_t a(Kokkos::view_alloc("A", Kokkos::Impl::AccessorArg_t{size}), ext0,
           ext1);
  ASSERT_EQ(a.accessor().size, size);
  static_assert(std::is_same_v<decltype(a.data()), double*>);

  // Test copy ctor to make sure the customize_view_arguments thing doesn't
  // interfere
  view_t a_copy = a;
  ASSERT_EQ(a_copy.accessor().size, size);

  c_view_t const_a = a;
  ASSERT_EQ(const_a.accessor().size, size);
  ASSERT_EQ(const_a.data(), a.data());

  view_t b(Kokkos::view_wrap(a.data(), Kokkos::Impl::AccessorArg_t{size}), ext0,
           ext1);
  ASSERT_EQ(b.accessor().size, size);

  // Get a compatible mapping for address calculation in the kernel
  using mapping_t = typename Kokkos::View<
      int**, typename TEST_EXECSPACE::memory_space>::mdspan_type::mapping_type;
  mapping_t map(Kokkos::dextents<size_t, 2>{ext0, ext1});

  // Test unmanaged ctors on GPU too (if GPU is enabled)
  int num_error = 0;
  Kokkos::parallel_reduce(
      "test_accessor_arg",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>, TEST_EXECSPACE>({0, 0},
                                                             {ext0, ext1}),
      KOKKOS_LAMBDA(int i, int j, int& errors) {
        view_t c(Kokkos::view_wrap(a.data(), Kokkos::Impl::AccessorArg_t{size}),
                 ext0, ext1);
        if (c.accessor().size != size) errors++;
        // Test copy ctor to make sure the customize_view_arguments thing
        // doesn't interfere
        view_t c_copy = c;
        if (c_copy.accessor().size != size) errors++;
        for (size_t k = 0; k < size; k++) {
          if (&a(i, j)[k] != a.data() + map(i, j) * size + k) errors++;
          if (&c_copy(i, j)[k] != a.data() + map(i, j) * size + k) errors++;
        }
      },
      num_error);
  ASSERT_EQ(num_error, 0);
}

TEST(TEST_CATEGORY, view_customization_allocation_type) {
  test_allocation_type();
}
