// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

// The following tests check some KernelShape behavior for RangePolicy and
// TeamPolicy We do NOT guarantee this behavior officially in our semantics,
// however Trilinos's Sacado as well as some other specific projects rely on it.
// If changes to Kokkos's kernel launch behavior would break this test,
// we need to talk to the affected teams.
// This only really applies to CUDA, HIP and SYLC right now where Sacado
// relies on the specific block / nd_item shape. I also check that
// in CUDA we only use the x dimension of the grid.

namespace {

KOKKOS_INLINE_FUNCTION
bool error_range_policy_config() {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return (blockDim.x != 1 || blockDim.z != 1 || gridDim.y != 1 ||
          gridDim.z != 1);
#endif
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(KOKKOS_COMPILER_INTEL_LLVM) && \
    KOKKOS_COMPILER_INTEL_LLVM >= 20250000
  return sycl::ext::oneapi::this_work_item::get_nd_item<2>().get_local_range(
             1) != 1;
#else
  return sycl::ext::oneapi::experimental::this_nd_item<3>().get_local_range(
             1) != 1;
#endif
#endif
  return false;
}

void test_range_policy() {
  Kokkos::View<int> d_error("Errors");
  int h_error = 0;

  // Just doing some random large number
  // so we are not stuck in some weird corner case for kernel config
  Kokkos::RangePolicy<> policy(3, 102931);
  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(int i) {
        if (i == 0) {
          d_error() = error_range_policy_config() ? 1 : 0;
        }
      });
  Kokkos::deep_copy(h_error, d_error);
  ASSERT_EQ(h_error, 0);

  Kokkos::deep_copy(d_error, 0);

  int sum;
  Kokkos::parallel_reduce(
      policy,
      KOKKOS_LAMBDA(int i, int&) {
        if (i == 0) {
          d_error() = error_range_policy_config() ? 1 : 0;
        }
      },
      sum);
  Kokkos::deep_copy(h_error, d_error);
  ASSERT_EQ(h_error, 0);

  Kokkos::deep_copy(d_error, 0);

  Kokkos::parallel_scan(
      policy, KOKKOS_LAMBDA(int i, int&, bool) {
        if (i == 0) {
          d_error() = error_range_policy_config() ? 1 : 0;
        }
      });
  Kokkos::deep_copy(h_error, d_error);
  ASSERT_EQ(h_error, 0);
}

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || \
    defined(KOKKOS_ENABLE_SYCL)
static constexpr int team_size = 16;
#else
static constexpr int team_size = 1;
#endif
static constexpr int vector_length = 8;

KOKKOS_INLINE_FUNCTION
bool error_team_policy_config() {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return (blockDim.x != vector_length || blockDim.y != team_size ||
          blockDim.z != 1 || gridDim.y != 1 || gridDim.z != 1);
#endif
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(KOKKOS_COMPILER_INTEL_LLVM) && \
    KOKKOS_COMPILER_INTEL_LLVM >= 20250000
  return sycl::ext::oneapi::this_work_item::get_nd_item<2>().get_local_range(
             0) != team_size ||
         sycl::ext::oneapi::this_work_item::get_nd_item<2>().get_local_range(
             1) != vector_length;
#else
  return sycl::ext::oneapi::experimental::this_nd_item<3>().get_local_range(
             0) != team_size ||
         sycl::ext::oneapi::experimental::this_nd_item<3>().get_local_range(
             0) != vector_length;
#endif
#endif
  return false;
}

void test_team_policy() {
  Kokkos::View<int> d_error("Errors");
  int h_error = 0;

  // Just making sure we also use grid on CUDA/HIP because
  // and its not just 1x1x1
  Kokkos::TeamPolicy<> policy(1000, team_size, vector_length);

  Kokkos::parallel_for(
      policy,
      KOKKOS_LAMBDA(const typename Kokkos::TeamPolicy<>::member_type& team) {
        if (team.league_rank() == 0) {
          Kokkos::single(Kokkos::PerTeam(team), [&]() {
            d_error() = error_team_policy_config() ? 1 : 0;
          });
        }
      });
  Kokkos::deep_copy(h_error, d_error);
  ASSERT_EQ(h_error, 0);

  Kokkos::deep_copy(d_error, 0);

  int sum;
  Kokkos::parallel_reduce(
      policy,
      KOKKOS_LAMBDA(const typename Kokkos::TeamPolicy<>::member_type& team,
                    int&) {
        if (team.league_rank() == 0) {
          Kokkos::single(Kokkos::PerTeam(team), [&]() {
            d_error() = error_team_policy_config() ? 1 : 0;
          });
        }
      },
      sum);
  Kokkos::deep_copy(h_error, d_error);
  ASSERT_EQ(h_error, 0);
}

}  // namespace

TEST(defaultdevicetype, kernel_shape_check) {
#if !(defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || \
      defined(KOKKOS_ENABLE_SYCL))
  GTEST_SKIP() << "KernelConfig test is only meaningful for CUDA, HIP and SYCL";
#endif
  test_range_policy();
  test_team_policy();
}
