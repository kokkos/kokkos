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
template <class ElementType, class MemorySpace>
struct TestAccessor {
  using element_type = ElementType;
  using reference    = element_type&;
  using data_handle_type =
      Kokkos::Impl::ReferenceCountedDataHandle<ElementType, MemorySpace>;
  using offset_policy = TestAccessor;

  // View expects this from accessors right now
  using memory_space = MemorySpace;

  KOKKOS_DEFAULTED_FUNCTION
  constexpr TestAccessor() = default;

  template <class OtherElementType,
            std::enable_if_t<std::is_constructible_v<
                                 Kokkos::default_accessor<ElementType>,
                                 Kokkos::default_accessor<OtherElementType>>,
                             int> = 0>
  KOKKOS_FUNCTION constexpr TestAccessor(
      const TestAccessor<OtherElementType, MemorySpace>& other) noexcept
      : value(other.value) {}

  KOKKOS_FUNCTION
  TestAccessor(const size_t val) : value(val) {}

  KOKKOS_FUNCTION
  constexpr reference access(
#ifndef KOKKOS_ENABLE_OPENACC
      const data_handle_type& p,
#else
      // FIXME OpenACC: illegal address when passing by reference
      data_handle_type p,
#endif
      size_t i) const noexcept {
    return *(p + (i % value));
  }

  KOKKOS_FUNCTION
  constexpr typename offset_policy::data_handle_type offset(
#ifndef KOKKOS_ENABLE_OPENACC
      const data_handle_type& p,
#else
      // FIXME OpenACC: illegal address when passing by reference
      data_handle_type p,
#endif
      size_t i) const noexcept {
    return p + i;
  }

  size_t value{};
};

struct Bar {
  int val;
};

template <class LayoutType, class DeviceType, class MemoryTraits>
constexpr auto customize_view_arguments(
    Kokkos::Impl::ViewArguments<Bar, LayoutType, DeviceType, MemoryTraits>) {
  return Kokkos::Impl::ViewCustomArguments<
      size_t, TestAccessor<Bar, typename DeviceType::memory_space>>{};
}
}  // namespace Foo

void test_accessor_arg() {
  using view_t = Kokkos::View<Foo::Bar*, TEST_EXECSPACE>;
  static_assert(
      std::is_same_v<
          view_t::accessor_type,
          Foo::TestAccessor<Foo::Bar, typename TEST_EXECSPACE::memory_space>>);

  view_t a("A", 10);
  ASSERT_EQ(a.accessor().value, 0ul);

  view_t b(a.data(), 10);
  ASSERT_EQ(b.accessor().value, 0ul);

  view_t c(Kokkos::view_alloc("C", Kokkos::Impl::AccessorArg_t{5}), 10);
  ASSERT_EQ(c.accessor().value, 5ul);

  view_t c_copy = c;
  ASSERT_EQ(c_copy.accessor().value, 5ul);

  view_t d(Kokkos::view_wrap(c.data(), Kokkos::Impl::AccessorArg_t{7}), 10);
  ASSERT_EQ(d.accessor().value, 7ul);

  int num_error = 0;
  Kokkos::parallel_reduce(
      "test_accessor_arg", 1,
      KOKKOS_LAMBDA(int, int& errors) {
        view_t e(a.data(), 10);
        if (e.accessor().value != 0lu) errors++;
        view_t f(Kokkos::view_wrap(e.data(), Kokkos::Impl::AccessorArg_t{7}),
                 10);
        if (f.accessor().value != 7lu) errors++;
        view_t f_copy = f;
        if (f_copy.accessor().value != 7lu) errors++;
      },
      num_error);
  ASSERT_EQ(num_error, 0);
}

TEST(TEST_CATEGORY, view_customization_accessor_arg) { test_accessor_arg(); }
