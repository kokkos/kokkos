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
void test_atomic_accessor() {
  using memory_space_t = typename ExecutionSpace::memory_space;
  using value_type     = std::remove_const_t<T>;
  Kokkos::View<value_type*, ExecutionSpace> v("V", 100);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(0, v.extent(0)),
      KOKKOS_LAMBDA(int i) { v(i) = i; });

  int errors;
  using acc_t = Kokkos::Impl::AtomicAccessorRelaxed<T>;
  acc_t acc{};
  typename acc_t::data_handle_type ptr = v.data();

  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<ExecutionSpace>(0, v.extent(0)),
      KOKKOS_LAMBDA(int i, int& error) {
        if (acc.access(ptr, i) != ptr[i]) error++;
        if (acc.offset(ptr, i) != ptr + i) error++;
        static_assert(std::is_same_v<typename acc_t::element_type, T>);
        static_assert(std::is_same_v<typename acc_t::reference, desul::AtomicRef<T, desul::MemoryOrderRelaxed, desul::MemoryScopeDevice>>);
        static_assert(std::is_same_v<typename acc_t::data_handle_type, T*>);
        static_assert(std::is_same_v<typename acc_t::offset_policy, acc_t>);
        static_assert(std::is_same_v<decltype(acc.access(ptr, i)), typename acc_t::reference>);
        static_assert(std::is_same_v<decltype(acc.offset(ptr, i)), T*>);
        static_assert(std::is_nothrow_move_constructible_v<acc_t>);
        static_assert(std::is_nothrow_move_assignable_v<acc_t>);
        static_assert(std::is_nothrow_swappable_v<acc_t>);
#if defined(KOKKOS_ENABLE_CXX20) || defined(KOKKOS_ENABLE_CXX23) || \
    defined(KOKKOS_ENABLE_CXX26)
        static_assert(std::copyable<acc_t>);
        static_assert(std::is_empty_v<acc_t>);
#endif
      },
      errors);
  ASSERT_EQ(errors, 0);
}

TEST(TEST_CATEGORY, atomic_accessor) {
  using ExecutionSpace = TEST_EXECSPACE;
  test_atomic_accessor<int, ExecutionSpace>();
  test_atomic_accessor<double, ExecutionSpace>();
}
