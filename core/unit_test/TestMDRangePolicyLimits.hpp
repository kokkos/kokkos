// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <limits>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

namespace {

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || \
    defined(KOKKOS_ENABLE_SYCL)

struct MDRangePolicyLimitsFunctor {
  KOKKOS_FUNCTION
  void operator()(const int, const int, const int, const int) const {}
};

using range_type =
    typename Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<4>>;
using range_type_bounds =
    typename Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<4>,
                                   Kokkos::LaunchBounds<32, 1>>;

// get maximum number of threads per block for each backend
int get_max_threads_per_block() {
#if defined(KOKKOS_ENABLE_CUDA)
  if constexpr (std::is_same_v<TEST_EXECSPACE, Kokkos::Cuda>) {
    return Kokkos::Cuda().cuda_device_prop().maxThreadsPerBlock;
  } else {
    return 0;
  }
#elif defined(KOKKOS_ENABLE_HIP)
  if constexpr (std::is_same_v<TEST_EXECSPACE, Kokkos::HIP>) {
    return Kokkos::Impl::HIPTraits::MaxThreadsPerBlock;
  } else {
    return 0;
  }
#elif defined(KOKKOS_ENABLE_SYCL)
  if constexpr (std::is_same_v<TEST_EXECSPACE, Kokkos::SYCL>) {
    return Kokkos::SYCL().impl_internal_space_instance()->m_maxWorkgroupSize;
  } else {
    return 0;
  }
#else
  return std::numeric_limits<int>::max();
#endif
}

TEST(TEST_CATEGORY_DEATH, md_range_policy_limits_large_tiling_total) {
  // test API limits
  // see #8103

  int max_threads_per_block = get_max_threads_per_block();
  if (!max_threads_per_block) {
    GTEST_SKIP() << "skipping for this backend";
  }

  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  const int N                             = 100;
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
}

TEST(TEST_CATEGORY_DEATH, md_range_policy_limits_large_tiling) {
  int max_threads_per_block = get_max_threads_per_block();
  if (!max_threads_per_block) {
    GTEST_SKIP() << "skipping for this backend";
  }

  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  const int N                             = 100;
  MDRangePolicyLimitsFunctor functor{};

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
}

TEST(TEST_CATEGORY_DEATH, md_range_policy_limits_slightly_large_tiling) {
  int max_threads_per_block = get_max_threads_per_block();
  if (!max_threads_per_block) {
    GTEST_SKIP() << "skipping for this backend";
  }

  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  const int N                             = 100;
  MDRangePolicyLimitsFunctor functor{};

  // request a slightly too large tiling in one dimension
  EXPECT_DEATH(
      {
        range_type range({0, 0, 0, 0}, {N, N, N, N},
                         {max_threads_per_block + 2, 1, 1, 1});
        Kokkos::parallel_for("slightly too large tiling", range, functor);
        Kokkos::fence("wait slightly too large tiling");
      },
      "MDRange tile dims exceed maximum number of threads per block - choose "
      "smaller tile dims");  // TODO check if this is the error we want
}

TEST(TEST_CATEGORY_DEATH, md_range_policy_limits_invalid_tiling) {
  int max_threads_per_block = get_max_threads_per_block();
  if (!max_threads_per_block) {
    GTEST_SKIP() << "skipping for this backend";
  }

  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  const int N                             = 100;
  MDRangePolicyLimitsFunctor functor{};

  // request an invalid tiling
  EXPECT_DEATH(
      {
        range_type_bounds range({0, 0, 0, 0}, {N, N, N, N}, {32, 2, 1, 1});
        Kokkos::parallel_for("invalid tiling", range, functor);
        Kokkos::fence("wait invalid tiling");
      },
      "invalid argument");
}

#endif  // if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) ||
        // defined(KOKKOS_ENABLE_SYCL)

}  // namespace
