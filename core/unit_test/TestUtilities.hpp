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

#include <sstream>
#include <iostream>

#include <Kokkos_Core.hpp>

namespace Test {

void test_is_specialization_of() {
  using Kokkos::Impl::is_specialization_of;
  static_assert(is_specialization_of<Kokkos::pair<float, int>, Kokkos::pair>{},
                "");
  static_assert(!is_specialization_of<Kokkos::View<int*>, Kokkos::pair>{}, "");
  static_assert(is_specialization_of<Kokkos::View<int*>, Kokkos::View>{}, "");
  // NOTE Not removing cv-qualifiers
  static_assert(!is_specialization_of<Kokkos::View<int*> const, Kokkos::View>{},
                "");
  // NOTE Would not compile because Kokkos::Array takes a non-type template
  // parameter
  // static_assert(is_specialization_of<Kokkos::Array<int, 4>, Kokkos::Array>{},
  // "");
  // But this is fine of course
  static_assert(!is_specialization_of<Kokkos::Array<float, 2>, Kokkos::pair>{},
                "");
}

template <std::size_t... Idxs, class... Args>
std::size_t do_comma_emulation_test(std::integer_sequence<std::size_t, Idxs...>,
                                    Args... args) {
  // Count the bugs, since ASSERT_EQ is a statement and not an expression
  std::size_t bugs = 0;
  // Ensure in-order evaluation
  std::size_t i = 0;
  KOKKOS_IMPL_FOLD_COMMA_OPERATOR(bugs += std::size_t(Idxs != i++) /*, ...*/);
  // Ensure expansion of multiple packs works
  KOKKOS_IMPL_FOLD_COMMA_OPERATOR(bugs += std::size_t(Idxs != args) /*, ...*/);
  return bugs;
}

TEST(utilities, comma_operator_emulation) {
  ASSERT_EQ(0u, do_comma_emulation_test(std::make_index_sequence<5>{}, 0, 1, 2,
                                        3, 4));
}

}  // namespace Test
