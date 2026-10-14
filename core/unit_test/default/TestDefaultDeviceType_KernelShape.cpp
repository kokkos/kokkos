// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

// The following tests check some KernelShape behavior for RangePolicy and
// TeamPolicy. We do NOT guarantee this behavior officially in our semantics,
// however Trilinos's Sacado as well as some other specific projects rely on it.
// If changes to Kokkos's kernel launch behavior would break this test,
// we need to talk to the affected teams.
// This only really applies to CUDA, HIP and SYCL right now where Sacado
// relies on the specific block / nd_item shape. I also check that
// in CUDA we only use the x dimension of the grid.

// To explain further Kokkos maps parallel iterations to the backend's native
// execution hierarchy:
// - CUDA/HIP: iterations are distributed across a grid of threads organized
// into blocks,
//   with the total work partitioned by the kernel launch configuration.
//   For RangePolicy we only use the y dimension of the block, the x dimension
//   and z dimension are 1. The product of the gridDim.x and the blockDim.y
//   spans the iteration range. Note that there is an option for parallel reduce
//   to do reuse where we limit the total number of blocks launched so that we
//   don't need to allocate almost unbounded memory for the scratch allocation.
//   For MDRangePolicy we do distribute iterations over all three dimensions of
//   the block and the grid. For that reason Sacado which reuses the x-dimension
//   of the block for FAD type concurrency, if Hierarchical parallelism is
//   enabled, is not compatible with MDRangePolicy. In TeamPolicy we launch a
//   grid with using the x dimension for the league_rank. The Team corresponds
//   to a CUDA/HIP block, with the y dimension being the threads, and the x
//   dimension the vector lane.
//   For all of it is important to note that warps are 32 consecutive threads,
//   starting with the x dimension before going to the y dimension of a block.
//   I.e. the warp lane id is
//   ((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x +
//   threadIdx.x)%WARPSIZE.
// - SYCL: iterations are expressed through an nd-range, where the work is
// divided into
//   a global range of items grouped into local work-groups, matching the
//   device's underlying parallel structure. We used to only use a rank-1
//   ndrange for RangePolicy but Sacado needs to leverage a specific dimension
//   for its internal concurrency. That dimension MUST be the same as vector
//   lane in a Kokkos team, and MUST be unused in RangePolicy. Hence we went to
//   using a rank-2 nd_range, where the 1st dimension is reserved for the vector
//   id, and Sacado style use.

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
  auto nd_item = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
#else
  auto nd_item = sycl::ext::oneapi::experimental::this_nd_item<2>();
#endif
  return nd_item.get_local_range(1) != 1;
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
        if (i == 4) {
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
        if (i == 4) {
          d_error() = error_range_policy_config() ? 1 : 0;
        }
      },
      sum);
  Kokkos::deep_copy(h_error, d_error);
  ASSERT_EQ(h_error, 0);

  Kokkos::deep_copy(d_error, 0);

  Kokkos::parallel_scan(
      policy, KOKKOS_LAMBDA(int i, int&, bool) {
        if (i == 4) {
          d_error() = error_range_policy_config() ? 1 : 0;
        }
      });
  Kokkos::deep_copy(h_error, d_error);
  ASSERT_EQ(h_error, 0);
}

static constexpr int team_size     = 16;
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
  auto nd_item = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
#else
  auto nd_item = sycl::ext::oneapi::experimental::this_nd_item<2>();
#endif
  return nd_item.get_local_range(0) != team_size ||
         nd_item.get_local_range(1) != vector_length;
#endif
  return false;
}

void test_team_policy() {
  Kokkos::View<int> d_error("Errors");
  int h_error = 0;

  // Just making sure we also use grid on CUDA/HIP because
  // its not just 1x1x1
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
  KOKKOS_IMPL_UNREACHABLE();
#endif

  test_range_policy();
  test_team_policy();
}
