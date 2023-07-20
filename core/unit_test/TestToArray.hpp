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
#include <numeric>

namespace {
struct SetOnMove {
  KOKKOS_INLINE_FUNCTION SetOnMove(int i_) : i(i_) {}
  KOKKOS_INLINE_FUNCTION operator int() const { return i; }

  KOKKOS_DEFAULTED_FUNCTION SetOnMove() = default;

  KOKKOS_DEFAULTED_FUNCTION SetOnMove(SetOnMove const&) = default;
  KOKKOS_DEFAULTED_FUNCTION SetOnMove& operator=(SetOnMove const&) = default;
  KOKKOS_DEFAULTED_FUNCTION ~SetOnMove()                           = default;

  KOKKOS_INLINE_FUNCTION SetOnMove(SetOnMove&& that) : i(that.i) {
    that.i = -1;
  }
  KOKKOS_INLINE_FUNCTION SetOnMove& operator=(SetOnMove&& that) {
    i      = that.i;
    that.i = -1;
    return *this;
  }

  int i = std::numeric_limits<int>::min();
};

TEST(TEST_CATEGORY, to_Array_lvalue) {
  int array[] = {
      2,
      3,
      5,
      7,
  };
  int a_sum   = 0;  // sum of array elements
  int som_sum = 0;  // sum of som_array elements
  int ka_sum  = 0;  // sum of Kokkos::Array elements

  Kokkos::parallel_reduce(
      1,
      KOKKOS_LAMBDA(int, int& asum, int& somsum, int& kasum) {
        SetOnMove som_array[std::size(array)];
        int i = 0;
        for (auto& v : array) som_array[i++] = v;

        auto ka = Kokkos::to_Array(som_array);
        static_assert(std::is_same_v<
                      Kokkos::Array<std::remove_extent_t<decltype(som_array)>,
                                    std::size(array)>,
                      decltype(ka)>);

        for (int j = 0; j != ka.size(); ++j) {
          asum += array[j];
          somsum += som_array[j];
          kasum += ka[j];
        }
      },
      a_sum, som_sum, ka_sum);

  ASSERT_EQ(som_sum, a_sum);
  ASSERT_EQ(ka_sum, a_sum);
}

TEST(TEST_CATEGORY, to_Array_rvalue) {
  int array[] = {
      2,
      3,
      5,
      7,
  };
  int a_sum   = 0;  // sum of array elements
  int som_sum = 0;  // sum of som_array elements
  int ka_sum  = 0;  // sum of Kokkos::Array elements

  Kokkos::parallel_reduce(
      1,
      KOKKOS_LAMBDA(int, int& asum, int& somsum, int& kasum) {
        SetOnMove som_array[std::size(array)];
        int i = 0;
        for (auto& v : array) som_array[i++] = v;

        auto ka = Kokkos::to_Array(std::move(som_array));
        static_assert(std::is_same_v<
                      Kokkos::Array<std::remove_extent_t<decltype(som_array)>,
                                    std::size(array)>,
                      decltype(ka)>);

        for (int j = 0; j != ka.size(); ++j) {
          asum += array[j];
          somsum += som_array[j];
          kasum += ka[j];
        }
      },
      a_sum, som_sum, ka_sum);

  ASSERT_EQ(som_sum, -1 * static_cast<int>(std::size(array)));
  ASSERT_EQ(ka_sum, a_sum);
}
}  // namespace

