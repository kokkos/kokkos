// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <limits>
#include <regex>

namespace {

template <class IndexType>
void construct_mdrange_policy_variable_type() {
  (void)Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>>{
      Kokkos::Array<IndexType, 2>{}, Kokkos::Array<IndexType, 2>{}};

  (void)Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>>{
      {{IndexType(0), IndexType(0)}}, {{IndexType(2), IndexType(2)}}};

  (void)Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>>{
      {IndexType(0), IndexType(0)}, {IndexType(2), IndexType(2)}};
}

TEST(TEST_CATEGORY, md_range_policy_construction_from_arrays) {
  {
    // Check that construction from Kokkos::Array of the specified index type
    // works.
    using IndexType = unsigned long long;
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>,
                          Kokkos::IndexType<IndexType>>
        p1(Kokkos::Array<IndexType, 2>{{0, 1}},
           Kokkos::Array<IndexType, 2>{{2, 3}});
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>,
                          Kokkos::IndexType<IndexType>>
        p2(Kokkos::Array<IndexType, 2>{{0, 1}},
           Kokkos::Array<IndexType, 2>{{2, 3}});
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>,
                          Kokkos::IndexType<IndexType>>
        p3(Kokkos::Array<IndexType, 2>{{0, 1}},
           Kokkos::Array<IndexType, 2>{{2, 3}},
           Kokkos::Array<IndexType, 1>{{4}});
  }
  {
    // Check that construction from double-braced initializer list
    // works.
    using index_type = unsigned long long;
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>> p1({{0, 1}},
                                                              {{2, 3}});
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>,
                          Kokkos::IndexType<index_type>>
        p2({{0, 1}}, {{2, 3}});
  }
  {
    // Check that construction from Kokkos::Array of long compiles for backwards
    // compability.  This was broken in
    // https://github.com/kokkos/kokkos/pull/3527/commits/88ea8eec6567c84739d77bdd25fdbc647fae28bb#r512323639
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>> p1(
        Kokkos::Array<long, 2>{{0, 1}}, Kokkos::Array<long, 2>{{2, 3}});
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>> p2(
        Kokkos::Array<long, 2>{{0, 1}}, Kokkos::Array<long, 2>{{2, 3}});
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>> p3(
        Kokkos::Array<long, 2>{{0, 1}}, Kokkos::Array<long, 2>{{2, 3}},
        Kokkos::Array<long, 1>{{4}});
  }

  // Check that construction from various index types works.
  construct_mdrange_policy_variable_type<char>();
  construct_mdrange_policy_variable_type<int>();
  construct_mdrange_policy_variable_type<unsigned long>();
  construct_mdrange_policy_variable_type<std::int64_t>();
}

#ifndef KOKKOS_ENABLE_OPENMPTARGET  // FIXME_OPENMPTARGET
TEST(TEST_CATEGORY_DEATH, policy_bounds_unsafe_narrowing_conversions) {
  using Policy = Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>,
                                       Kokkos::IndexType<unsigned>>;

  std::string msg =
      "Kokkos::MDRangePolicy bound type error: an unsafe implicit conversion "
      "is "
      "performed on a bound (-1) in dimension (0), which may not preserve its "
      "original value.\n";
  std::string expected = std::regex_replace(msg, std::regex("\\(|\\)"), "\\$&");

  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_DEATH({ (void)Policy({-1, 0}, {2, 3}); }, expected);
}

TEST(TEST_CATEGORY_DEATH, policy_invalid_bounds) {
  using Policy = Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>>;

  ::testing::FLAGS_gtest_death_test_style = "threadsafe";

  auto [dim0, dim1] = (Policy::inner_direction == Kokkos::Iterate::Right)
                          ? std::make_pair(1, 0)
                          : std::make_pair(0, 1);
  std::string msg1 =
      "Kokkos::MDRangePolicy bounds error: The lower bound (100) is greater "
      "than its upper bound (90) in dimension " +
      std::to_string(dim0) + ".\n";

  std::string msg2 =
      "Kokkos::MDRangePolicy bounds error: The lower bound (100) is greater "
      "than its upper bound (90) in dimension " +
      std::to_string(dim1) + ".\n";

#if !defined(KOKKOS_ENABLE_DEPRECATED_CODE_4)
  // escape the parentheses in the regex to match the error message
  msg1 = std::regex_replace(msg1, std::regex("\\(|\\)"), "\\$&");
  (void)msg2;
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_DEATH({ (void)Policy({100, 100}, {90, 90}); }, msg1);
#else
  if (!Kokkos::show_warnings()) {
    GTEST_SKIP() << "Kokkos warning messages are disabled";
  }

  ::testing::internal::CaptureStderr();
  (void)Policy({100, 100}, {90, 90});
#ifdef KOKKOS_ENABLE_DEPRECATION_WARNINGS
  ASSERT_EQ(::testing::internal::GetCapturedStderr(), msg1 + msg2);
#else
  ASSERT_TRUE(::testing::internal::GetCapturedStderr().empty());
  (void)msg1;
  (void)msg2;
#endif

#endif
}
#endif

TEST(TEST_CATEGORY, policy_get_tile_size) {
  constexpr int rank = 3;
  using Policy    = Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<rank>>;
  using tile_type = typename Policy::tile_type;

  std::size_t last_rank =
      (Policy::inner_direction == Kokkos::Iterate::Right) ? rank - 1 : 0;

  auto default_size_properties =
      Kokkos::Impl::get_tile_size_properties(TEST_EXECSPACE());

  {
    int dim_length = 100;
    Policy policy_default({0, 0, 0}, {dim_length, dim_length, dim_length});

    auto rec_tile_sizes      = policy_default.tile_size_recommended();
    auto internal_tile_sizes = policy_default.m_tile;

    for (std::size_t i = 0; i < rank; ++i) {
      EXPECT_EQ(rec_tile_sizes[i], internal_tile_sizes[i])
          << " incorrect recommended tile size returned for rank " << i;
    }
  }
  {
    int dim_length = 100;
    Policy policy({0, 0, 0}, {dim_length, dim_length, dim_length},
                  tile_type{{2, 4, 16}});

    auto rec_tile_sizes = policy.tile_size_recommended();

    EXPECT_EQ(default_size_properties.max_total_tile_size,
              policy.max_total_tile_size());

    int prod_rec_tile_size = 1;
    for (std::size_t i = 0; i < rank; ++i) {
      EXPECT_GT(rec_tile_sizes[i], 0)
          << " invalid default tile size for rank " << i;

      if (default_size_properties.default_largest_tile_size == 0) {
        auto expected_rec_tile_size =
            (i == last_rank) ? dim_length
                             : default_size_properties.default_tile_size;
        EXPECT_EQ(expected_rec_tile_size, rec_tile_sizes[i])
            << " incorrect recommended tile size returned for rank " << i;
      } else {
        auto expected_rec_tile_size =
            (i == last_rank) ? default_size_properties.default_largest_tile_size
                             : default_size_properties.default_tile_size;
        EXPECT_EQ(expected_rec_tile_size, rec_tile_sizes[i])
            << " incorrect recommended tile size returned for rank " << i;
      }

      prod_rec_tile_size *= rec_tile_sizes[i];
    }
    EXPECT_LT(prod_rec_tile_size, policy.max_total_tile_size());
  }
}

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || \
    defined(KOKKOS_ENABLE_SYCL)

struct MDRangePolicyLimitsFunctor {
  KOKKOS_FUNCTION
  void operator()(const int, const int, const int, const int) const {}
};

TEST(TEST_CATEGORY_DEATH, md_range_policy_limits) {
  // test API limits
  // see #8103

  // get maximum number of threads per block for each backend
  int max_threads_per_block = std::numeric_limits<int>::max();
#if defined(KOKKOS_ENABLE_CUDA)
  if constexpr (std::is_same_v<TEST_EXECSPACE, Kokkos::Cuda>) {
    max_threads_per_block =
        Kokkos::Cuda().cuda_device_prop().maxThreadsPerBlock;
  } else {
    GTEST_SKIP() << "skipping for this backend";
  }
#elif defined(KOKKOS_ENABLE_HIP)
  if constexpr (std::is_same_v<TEST_EXECSPACE, Kokkos::HIP>) {
    max_threads_per_block = HIPTraits::MaxThreadsPerBlock;
  } else {
    GTEST_SKIP() << "skipping for this backend";
  }
#elif defined(KOKKOS_ENABLE_SYCL)
  if constexpr (std::is_same_v<TEST_EXECSPACE, Kokkos::SYCL>) {
    max_threads_per_block =
        Kokkos::SYCL().impl_internal_space_instance()->m_maxWorkGroupSize;
  } else {
    GTEST_SKIP() << "skipping for this backend";
  }
#endif

  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  const int N                             = 100;

  using range_type =
      typename Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<4>>;
  using range_type_bounds =
      typename Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<4>,
                                     Kokkos::LaunchBounds<32, 1>>;

  MDRangePolicyLimitsFunctor functor{};

  // request a very large tiling that exceeds tile product limits
  EXPECT_DEATH(
      {
        range_type range({0, 0, 0, 0}, {N, N, N, N},
                         {max_threads_per_block, max_threads_per_block, 1, 1});
        Kokkos::parallel_for("very large total tiling", range, functor);
        Kokkos::fence("wait very large total tiling");
      },
      "MDRange tile dims exceed maximum number of threads per block - choose "
      "smaller tile dims");  // TODO check if this is the error we want

  // request a very large tiling in one dimension
  EXPECT_DEATH(
      {
        range_type range({0, 0, 0, 0}, {N, N, N, N},
                         {2 * max_threads_per_block, 1, 1, 1});
        Kokkos::parallel_for("very large tiling", range, functor);
        Kokkos::fence("wait very large tiling");
      },
      "MDRange tile dims exceed maximum number of threads per block - choose "
      "smaller tile dims");  // TODO check if this is the error we want

  // request a slightly too large tiling in one dimension
  EXPECT_DEATH(
      {
        range_type range({0, 0, 0, 0}, {N, N, N, N},
                         {max_threads_per_block + 2, 1, 1, 1});
        Kokkos::parallel_for("slightly too large tiling", range, functor);
        Kokkos::fence("wait slightly too large tiling");
      },
      "Kokkos contract violation");  // TODO check if this is the error we
                                     // want

  // request an invalid tiling
  EXPECT_DEATH(
      {
        range_type_bounds range({0, 0, 0, 0}, {N, N, N, N}, {32, 2, 1, 1});
        Kokkos::parallel_for("invalid tiling", range, functor);
        Kokkos::fence("wait invalid tiling");
      },
      "invalid argument");
}
#endif

}  // namespace
