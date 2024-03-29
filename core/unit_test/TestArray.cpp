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

#include <Kokkos_Array.hpp>

namespace {

KOKKOS_FUNCTION constexpr bool test_array() {
  constexpr Kokkos::Array<int, 3> a{{1, 2}};

  static_assert(!a.empty());
  static_assert(a.size() == 3);
  static_assert(a.max_size() == 3);

  static_assert(*a.data() == 1);
  static_assert(a[1] == 2);

  return true;
}

static_assert(test_array());

KOKKOS_FUNCTION constexpr bool test_array_structured_binding_support() {
  constexpr Kokkos::Array<float, 2> a{};
  auto& [xr, yr] = a;
  (void)xr;
  (void)yr;
  auto [x, y] = a;
  (void)x;
  (void)y;
  auto const& [xcr, ycr] = a;
  (void)xcr;
  (void)ycr;
  return true;
}

static_assert(test_array_structured_binding_support());

template <typename L, typename R>
KOKKOS_FUNCTION constexpr bool is_equal(L const& l, R const& r) {
  if (std::size(l) != std::size(r)) return false;

  for (size_t i = 0; i != std::size(l); ++i) {
    if (l[i] != r[i]) return false;
  }

  return true;
}

// Disable ctad test for intel versions < 2021, see issue #6702
#if !defined(KOKKOS_COMPILER_INTEL) || KOKKOS_COMPILER_INTEL >= 2021
KOKKOS_FUNCTION constexpr bool test_array_ctad() {
  constexpr int x = 10;
  constexpr Kokkos::Array a{1, 2, 3, 5, x};
  constexpr Kokkos::Array<int, 5> b{1, 2, 3, 5, x};

  return std::is_same_v<decltype(a), decltype(b)> && is_equal(a, b);
}

static_assert(test_array_ctad());
#endif

KOKKOS_FUNCTION constexpr bool test_array_aggregate_initialization() {
  // Initialize arrays from brace-init-list as for std::array.

  Kokkos::Array<float, 2> aggregate_initialization_syntax_1 = {1.41f, 3.14f};
  if ((aggregate_initialization_syntax_1[0] != 1.41f) ||
      (aggregate_initialization_syntax_1[1] != 3.14f))
    return false;

  Kokkos::Array<int, 3> aggregate_initialization_syntax_2{
      {0, 1, 2}};  // since C++11
  if ((aggregate_initialization_syntax_2[0] != 0) ||
      (aggregate_initialization_syntax_2[1] != 1) ||
      (aggregate_initialization_syntax_2[2] != 2))
    return false;

  // Note that this is a valid initialization.
  Kokkos::Array<double, 3> initialized_with_one_argument_missing = {{255, 255}};
  if ((initialized_with_one_argument_missing[0] != 255) ||
      (initialized_with_one_argument_missing[1] != 255) ||
      (initialized_with_one_argument_missing[2] != 0))
    return false;

  // But the following line would not compile
  //  Kokkos::Array< double, 3 > initialized_with_too_many{ { 1, 2, 3, 4 } };

  return true;
}

static_assert(test_array_aggregate_initialization());

// A few compilers, such as GCC 8.4, were erroring out when the function below
// appeared in a constant expression because
// Kokkos::Array<T, 0, Proxy>::operator[] is non-constexpr.  The issue
// disappears with GCC 9.1 (https://godbolt.org/z/TG4TEef1b).  As a workaround,
// the static_assert was dropped and the [[maybe_unused]] is used as an attempt
// to silent warnings that the function is never used.
[[maybe_unused]] KOKKOS_FUNCTION void test_array_zero_sized() {
  using T = float;

  // The code below must compile for zero-sized arrays.
  constexpr int N = 0;
  Kokkos::Array<T, N> a;
  for (int i = 0; i < N; ++i) {
    a[i] = T();
  }
}

}  // namespace
