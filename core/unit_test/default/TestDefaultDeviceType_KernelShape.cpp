// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

// The following tests check some KernelShape behavior for RangePolicy and
// TeamPolicy We do NOT guarantee this behavior officially in our semantics,
// however Trilinos's Sacado as well as some other specific projects rely on it.
// If changes to Kokkos's kernel launch behavior would break this test,
// we need to talk to the affected teams.

void test_range_policy() {
  Kokkos::View<int> d_error("Errors");
  int h_error = 0;

  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(3, 102931), KOKKOS_LAMBDA(int i) {
        if (i == 0) {
          int err = 0;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          if (blockDim.x != 1 || blockDim.z != 1 || gridDim.y != 1 ||
              gridDim.z != 1)
            err = 1;
#endif
#if defined(__SYCL_DEVICE_ONLY__)
          if (sycl::ext::oneapi::this_work_item::get_nd_item<2>()
                  .get_local_range(1) != 1)
            err = 1;
#endif
          d_error() = err;
        }
      });
  Kokkos::deep_copy(h_error, d_error);
  ASSERT_EQ(h_error, 0);

  Kokkos::deep_copy(d_error, 0);

  int sum;
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<>(3, 102931),
      KOKKOS_LAMBDA(int i, int&) {
        if (i == 0) {
          int err = 0;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          if (blockDim.x != 1 || blockDim.z != 1 || gridDim.y != 1 ||
              gridDim.z != 1)
            err = 1;
#endif
#if defined(__SYCL_DEVICE_ONLY__)
          if (sycl::ext::oneapi::this_work_item::get_nd_item<2>()
                  .get_local_range(1) != 1)
            err = 1;
#endif
          d_error() = err;
        }
      },
      sum);
  Kokkos::deep_copy(h_error, d_error);
  ASSERT_EQ(h_error, 0);
}

void test_team_policy() {
  Kokkos::View<int> d_error("Errors");
  int h_error = 0;

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || \
    defined(KOKKOS_ENABLE_SYCL)
  int team_size = 16;
#else
  int team_size = 1;
#endif
  Kokkos::parallel_for(
      Kokkos::TeamPolicy<>(10000, team_size, 8),
      KOKKOS_LAMBDA(const typename Kokkos::TeamPolicy<>::member_type& team) {
        if (team.league_rank() == 0) {
          Kokkos::single(Kokkos::PerTeam(team), [&]() {
            int err = 0;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
            if (blockDim.x != 8 || blockDim.y != 16 || blockDim.z != 1 ||
                gridDim.y != 1 || gridDim.z != 1)
              err = 1;
#endif
#if defined(__SYCL_DEVICE_ONLY__)
            if (sycl::ext::oneapi::this_work_item::get_nd_item<2>()
                        .get_local_range(0) != 16 ||
                sycl::ext::oneapi::this_work_item::get_nd_item<2>()
                        .get_local_range(1) != 8)
              err = 1;
#endif
            d_error() = err;
          });
        }
      });
  Kokkos::deep_copy(h_error, d_error);
  ASSERT_EQ(h_error, 0);

  Kokkos::deep_copy(d_error, 0);

  int sum;
  Kokkos::parallel_reduce(
      Kokkos::TeamPolicy<>(10000, team_size, 8),
      KOKKOS_LAMBDA(const typename Kokkos::TeamPolicy<>::member_type& team,
                    int&) {
        if (team.league_rank() == 0) {
          Kokkos::single(Kokkos::PerTeam(team), [&]() {
            int err = 0;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
            if (blockDim.x != 8 || blockDim.y != 16 || blockDim.z != 1 ||
                gridDim.y != 1 || gridDim.z != 1)
              err = 1;
#endif
#if defined(__SYCL_DEVICE_ONLY__)
            if (sycl::ext::oneapi::this_work_item::get_nd_item<2>()
                        .get_local_range(0) != 16 ||
                sycl::ext::oneapi::this_work_item::get_nd_item<2>()
                        .get_local_range(1) != 8)
              err = 1;
#endif
            d_error() = err;
          });
        }
      },
      sum);
  Kokkos::deep_copy(h_error, d_error);
  ASSERT_EQ(h_error, 0);
}

TEST(defaultdevicetype, kernel_shape_check) {
  test_range_policy();
  test_team_policy();
}
