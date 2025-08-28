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

#include <concepts>

// consteval specifier
consteval int sqr(int n) { return n * n; }
static_assert(sqr(100) == 10000);

// conditional explicit
struct S {
  explicit(sizeof(int) > 0) S(int) {}
};

// concepts library
constexpr std::floating_point auto x2(std::floating_point auto x) {
  return x + x;
}
constexpr std::integral auto x2(std::integral auto x) { return x << 1; }

int main() { return 0; }
