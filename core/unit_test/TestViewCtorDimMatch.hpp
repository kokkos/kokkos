/*
//@HEADER
// ************************************************************************
//
// Kokkos v. 3.0
// Copyright (2020) National Technology & Engineering
// Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

namespace Test {

#define LIVE(EXPR, ARGS, DYNRANK, TOTALRANK) EXPECT_NO_THROW(EXPR)
#define DIE(EXPR, ARGS, DYNRANK, TOTALRANK)                                   \
  ASSERT_DEATH(                                                               \
      EXPR,                                                                   \
      "Constructor for Kokkos::View 'v' has mismatched number of arguments. " \
      "The number of arguments = " +                                          \
          std::to_string(ARGS) +                                              \
          " neither matches the dynamic rank = " + std::to_string(DYNRANK) +  \
          " nor the total rank = " + std::to_string(TOTALRANK))

template <int rank, int dynrank, template <int> class RankType,
          std::size_t... Is>
void test_matching_arguments_rank_helper(std::index_sequence<Is...>) {
  constexpr int nargs = sizeof...(Is);
  using view_type     = Kokkos::View<typename RankType<rank>::type>;
  if (nargs == rank || nargs == dynrank)
    LIVE({ view_type v("v", ((Is * 0) + 1)...); }, nargs, dynrank, rank);
  else
    DIE({ view_type v("v", ((Is * 0) + 1)...); }, nargs, dynrank, rank);
}

template <int rank, int dynrank, template <int> class RankType>
void test_matching_arguments_rank() {
  test_matching_arguments_rank_helper<rank, dynrank, RankType>(
      std::make_index_sequence<0>());
  test_matching_arguments_rank_helper<rank, dynrank, RankType>(
      std::make_index_sequence<1>());
  test_matching_arguments_rank_helper<rank, dynrank, RankType>(
      std::make_index_sequence<2>());
  test_matching_arguments_rank_helper<rank, dynrank, RankType>(
      std::make_index_sequence<3>());
  test_matching_arguments_rank_helper<rank, dynrank, RankType>(
      std::make_index_sequence<4>());
  test_matching_arguments_rank_helper<rank, dynrank, RankType>(
      std::make_index_sequence<5>());
  test_matching_arguments_rank_helper<rank, dynrank, RankType>(
      std::make_index_sequence<6>());
  test_matching_arguments_rank_helper<rank, dynrank, RankType>(
      std::make_index_sequence<7>());
  test_matching_arguments_rank_helper<rank, dynrank, RankType>(
      std::make_index_sequence<8>());
}

template <int rank>
struct DynamicRank {
  using type = typename DynamicRank<rank - 1>::type*;
};

template <>
struct DynamicRank<0> {
  using type = int;
};

// Skip test execution when KOKKOS_ENABLE_OPENMPTARGET is enabled until
// Kokkos::abort() aborts properly on that backend
// Skip test execution when KOKKOS_COMPILER_NVHPC until fixed in GTEST
#if defined(KOKKOS_ENABLE_OPENMPTARGET) || defined(KOKKOS_COMPILER_NVHPC)
#else
TEST(TEST_CATEGORY_DEATH, view_construction_with_wrong_params_dyn) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";

  test_matching_arguments_rank<0, 0, DynamicRank>();  // dim = 0, dynamic = 0
  test_matching_arguments_rank<1, 1, DynamicRank>();  // dim = 1, dynamic = 1
  test_matching_arguments_rank<2, 2, DynamicRank>();  // dim = 2, dynamic = 2
  test_matching_arguments_rank<3, 3, DynamicRank>();  // dim = 3, dynamic = 3
  test_matching_arguments_rank<4, 4, DynamicRank>();  // dim = 4, dynamic = 4
  test_matching_arguments_rank<5, 5, DynamicRank>();  // dim = 5, dynamic = 5
  test_matching_arguments_rank<6, 6, DynamicRank>();  // dim = 6, dynamic = 6
  test_matching_arguments_rank<7, 7, DynamicRank>();  // dim = 7, dynamic = 7
  test_matching_arguments_rank<8, 8, DynamicRank>();  // dim = 8, dynamic = 8
}

template <int rank>
struct StaticRank {
  using type = typename StaticRank<rank - 1>::type[1];
};

template <>
struct StaticRank<0> {
  using type = int;
};

TEST(TEST_CATEGORY_DEATH, view_construction_with_wrong_params_stat) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";

  test_matching_arguments_rank<0, 0, StaticRank>();  // dim = 0, dynamic = 0
  test_matching_arguments_rank<1, 0, StaticRank>();  // dim = 1, dynamic = 0
  test_matching_arguments_rank<2, 0, StaticRank>();  // dim = 2, dynamic = 0
  test_matching_arguments_rank<3, 0, StaticRank>();  // dim = 3, dynamic = 0
  test_matching_arguments_rank<4, 0, StaticRank>();  // dim = 4, dynamic = 0
  test_matching_arguments_rank<5, 0, StaticRank>();  // dim = 5, dynamic = 0
  test_matching_arguments_rank<6, 0, StaticRank>();  // dim = 6, dynamic = 0
  test_matching_arguments_rank<7, 0, StaticRank>();  // dim = 7, dynamic = 0
  test_matching_arguments_rank<8, 0, StaticRank>();  // dim = 8, dynamic = 0
}

template <int rank>
struct MixedRank {
  using type = typename DynamicRank<rank - 1>::type[1];
};

template <>
struct MixedRank<0> {
  using type = int;
};

TEST(TEST_CATEGORY_DEATH, view_construction_with_wrong_params_mix) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";

  test_matching_arguments_rank<0, 0, MixedRank>();  // dim = 0, dynamic = 0
  test_matching_arguments_rank<1, 0, MixedRank>();  // dim = 1, dynamic = 0
  test_matching_arguments_rank<2, 1, MixedRank>();  // dim = 2, dynamic = 1
  test_matching_arguments_rank<3, 2, MixedRank>();  // dim = 3, dynamic = 2
  test_matching_arguments_rank<4, 3, MixedRank>();  // dim = 4, dynamic = 3
  test_matching_arguments_rank<5, 4, MixedRank>();  // dim = 5, dynamic = 4
  test_matching_arguments_rank<6, 5, MixedRank>();  // dim = 6, dynamic = 5
  test_matching_arguments_rank<7, 6, MixedRank>();  // dim = 7, dynamic = 6
  test_matching_arguments_rank<8, 7, MixedRank>();  // dim = 8, dynamic = 7
}
#endif  // KOKKOS_ENABLE_OPENMPTARGET

#undef LIVE
#undef DIE

#define CHECK_DEATH(EXPR)                                                     \
  ASSERT_DEATH(EXPR,                                                          \
               "The specified run-time extent for Kokkos::View 'v' does not " \
               "match the compile-time extent in dimension 0. The given "     \
               "extent is 2 but should be 1.")

TEST(TEST_CATEGORY_DEATH, view_construction_with_wrong_static_extents) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";

  // clang-format off
  CHECK_DEATH({ Kokkos::View<typename StaticRank<1>::type> v("v", 2); });
  CHECK_DEATH({ Kokkos::View<typename StaticRank<2>::type> v("v", 2, 1); });
  CHECK_DEATH({ Kokkos::View<typename StaticRank<3>::type> v("v", 2, 1, 1); });
  CHECK_DEATH({ Kokkos::View<typename StaticRank<4>::type> v("v", 2, 1, 1, 1); });
  CHECK_DEATH({ Kokkos::View<typename StaticRank<5>::type> v("v", 2, 1, 1, 1, 1); });
  CHECK_DEATH({ Kokkos::View<typename StaticRank<6>::type> v("v", 2, 1, 1, 1, 1, 1); });
  CHECK_DEATH({ Kokkos::View<typename StaticRank<7>::type> v("v", 2, 1, 1, 1, 1, 1, 1); });
  CHECK_DEATH({ Kokkos::View<typename StaticRank<8>::type> v("v", 2, 1, 1, 1, 1, 1, 1, 1); });
  // clang-format on
}

#undef CHECK_DEATH
}  // namespace Test
