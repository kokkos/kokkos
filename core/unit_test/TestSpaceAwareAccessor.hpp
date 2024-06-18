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
#include <type_traits>

#include <gtest/gtest.h>

template <class T, class ExecutionSpace>
void test_space_aware_accessor() {
  using memory_space_t = typename ExecutionSpace::memory_space;
  using value_type     = std::remove_const_t<T>;
  Kokkos::View<value_type*, ExecutionSpace> v("V", 100);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(0, v.extent(0)),
      KOKKOS_LAMBDA(int i) { v(i) = i; });

  int errors;
  using acc_t = Kokkos::Impl::SpaceAwareAccessor<memory_space_t,
                                                 Kokkos::default_accessor<T>>;
  acc_t acc{};
  typename acc_t::data_handle_type ptr = v.data();

  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<ExecutionSpace>(0, v.extent(0)),
      KOKKOS_LAMBDA(int i, int& error) {
        if (acc.access(ptr, i) != ptr[i]) error++;
        static_assert(std::is_same_v<typename acc_t::element_type, T>);
        static_assert(std::is_same_v<typename acc_t::reference, T&>);
        static_assert(std::is_same_v<typename acc_t::data_handle_type, T*>);
        static_assert(std::is_same_v<typename acc_t::offset_policy, acc_t>);
        static_assert(
            std::is_same_v<typename acc_t::memory_space, memory_space_t>);
#if defined(KOKKOS_ENABLE_CXX20) || defined(KOKKOS_ENABLE_CXX23) || \
    defined(KOKKOS_ENABLE_CXX26)
        static_assert(std::is_empty_v<acc_t>);
#endif
      },
      errors);
  ASSERT_EQ(errors, 0);
}

void test_space_aware_accessor_conversion() {
  using ExecutionSpace = TEST_EXECSPACE;
  using memory_space_t = typename ExecutionSpace::memory_space;
  using T              = float;
  using acc_t          = Kokkos::Impl::SpaceAwareAccessor<memory_space_t,
                                                 Kokkos::default_accessor<T>>;
  using const_acc_t =
      Kokkos::Impl::SpaceAwareAccessor<memory_space_t,
                                       Kokkos::default_accessor<const T>>;
  using int_acc_t =
      Kokkos::Impl::SpaceAwareAccessor<memory_space_t,
                                       Kokkos::default_accessor<int>>;
  using host_acc_t =
      Kokkos::Impl::SpaceAwareAccessor<Kokkos::HostSpace,
                                       Kokkos::default_accessor<T>>;
  using anon_acc_t =
      Kokkos::Impl::SpaceAwareAccessor<Kokkos::AnonymousSpace,
                                       Kokkos::default_accessor<T>>;

  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(0, 1), KOKKOS_LAMBDA(int) {
        static_assert(std::is_constructible_v<const_acc_t, acc_t>);
        static_assert(std::is_convertible_v<acc_t, const_acc_t>);
        static_assert(!std::is_constructible_v<acc_t, const_acc_t>);
        static_assert(!std::is_constructible_v<acc_t, int_acc_t>);
        static_assert(std::is_constructible_v<acc_t, host_acc_t> ==
                      std::is_same_v<memory_space_t, Kokkos::HostSpace>);
        static_assert(std::is_constructible_v<anon_acc_t, acc_t>);
        static_assert(std::is_constructible_v<acc_t, anon_acc_t>);
        static_assert(std::is_convertible_v<anon_acc_t, acc_t>);
        static_assert(std::is_convertible_v<acc_t, anon_acc_t>);
      });
}

TEST(TEST_CATEGORY, space_aware_accessor) {
  using ExecutionSpace = TEST_EXECSPACE;
  test_space_aware_accessor<int, ExecutionSpace>();
  test_space_aware_accessor<double, ExecutionSpace>();
  test_space_aware_accessor<const int, ExecutionSpace>();
  test_space_aware_accessor<const double, ExecutionSpace>();
  test_space_aware_accessor_conversion();
}
