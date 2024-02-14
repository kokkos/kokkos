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

#ifndef TESTLOCALDEEPCOPY_HPP_
#define TESTLOCALDEEPCOPY_HPP_

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

namespace Test {

//-------------------------------------------------------------------------------------------------------------
// Utility functions
//-------------------------------------------------------------------------------------------------------------

template <typename ViewType>
bool view_check_equals(const ViewType& lhs, const ViewType& rhs) {
  int result = 1;

  using exec_space = typename ViewType::execution_space;

  auto reducer = Kokkos::LAnd<int>(result);
  Kokkos::parallel_reduce(
      "view check equals", Kokkos::RangePolicy<exec_space>(0, lhs.span()),
      KOKKOS_LAMBDA(int i, int& local_result) {
        local_result = (lhs.data()[i] == rhs.data()[i]) && local_result;
      },
      reducer);

  Kokkos::fence();
  return (result);
}

template <typename ViewType>
void view_init(ViewType& view) {
  using exec_space = typename ViewType::execution_space;

  Kokkos::parallel_for(
      "initialize array", Kokkos::RangePolicy<exec_space>(0, view.span()),
      KOKKOS_LAMBDA(int i) { view.data()[i] = i; });
  Kokkos::fence();
}

// Create a view with a given label and dimensions
template <typename ViewType>
typename std::enable_if<(ViewType::rank == 1), ViewType>::type view_create(
    std::string label, const int N) {
  return ViewType(label, N);
}

template <typename ViewType>
typename std::enable_if<(ViewType::rank == 2), ViewType>::type view_create(
    std::string label, const int N) {
  return ViewType(label, N, N);
}

template <typename ViewType>
typename std::enable_if<(ViewType::rank == 3), ViewType>::type view_create(
    std::string label, const int N) {
  return ViewType(label, N, N, N);
}

template <typename ViewType>
typename std::enable_if<(ViewType::rank == 4), ViewType>::type view_create(
    std::string label, const int N) {
  return ViewType(label, N, N, N, N);
}

template <typename ViewType>
typename std::enable_if<(ViewType::rank == 5), ViewType>::type view_create(
    std::string label, const int N) {
  return ViewType(label, N, N, N, N, N);
}

template <typename ViewType>
typename std::enable_if<(ViewType::rank == 6), ViewType>::type view_create(
    std::string label, const int N) {
  return ViewType(label, N, N, N, N, N, N);
}

template <typename ViewType>
typename std::enable_if<(ViewType::rank == 7), ViewType>::type view_create(
    std::string label, const int N) {
  return ViewType(label, N, N, N, N, N, N, N);
}

template <typename ViewType>
typename std::enable_if<(ViewType::rank == 8), ViewType>::type view_create(
    std::string label, const int N) {
  return ViewType(label, N, N, N, N, N, N, N, N);
}

// Extract a subview from a view to run our tests
template <typename ViewType, typename Bounds>
KOKKOS_INLINE_FUNCTION auto extract_subview(
    ViewType& src, int lid, Bounds bounds,
    std::enable_if_t<unsigned(ViewType::rank) == 2>* = nullptr) {
  return Kokkos::subview(src, lid, bounds);
}

template <typename ViewType, typename Bounds>
KOKKOS_INLINE_FUNCTION auto extract_subview(
    ViewType& src, int lid, Bounds bounds,
    std::enable_if_t<unsigned(ViewType::rank) == 3>* = nullptr) {
  return Kokkos::subview(src, lid, bounds, Kokkos::ALL);
}

template <typename ViewType, typename Bounds>
KOKKOS_INLINE_FUNCTION auto extract_subview(
    ViewType& src, int lid, Bounds bounds,
    std::enable_if_t<unsigned(ViewType::rank) == 4>* = nullptr) {
  return Kokkos::subview(src, lid, bounds, Kokkos::ALL, Kokkos::ALL);
}

template <typename ViewType, typename Bounds>
KOKKOS_INLINE_FUNCTION auto extract_subview(
    ViewType& src, int lid, Bounds bounds,
    std::enable_if_t<unsigned(ViewType::rank) == 5>* = nullptr) {
  return Kokkos::subview(src, lid, bounds, Kokkos::ALL, Kokkos::ALL,
                         Kokkos::ALL);
}

template <typename ViewType, typename Bounds>
KOKKOS_INLINE_FUNCTION auto extract_subview(
    ViewType& src, int lid, Bounds bounds,
    std::enable_if_t<unsigned(ViewType::rank) == 6>* = nullptr) {
  return Kokkos::subview(src, lid, bounds, Kokkos::ALL, Kokkos::ALL,
                         Kokkos::ALL, Kokkos::ALL);
}

template <typename ViewType, typename Bounds>
KOKKOS_INLINE_FUNCTION auto extract_subview(
    ViewType& src, int lid, Bounds bounds,
    std::enable_if_t<unsigned(ViewType::rank) == 7>* = nullptr) {
  return Kokkos::subview(src, lid, bounds, Kokkos::ALL, Kokkos::ALL,
                         Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
}

template <typename ViewType, typename Bounds>
KOKKOS_INLINE_FUNCTION auto extract_subview(
    ViewType& src, int lid, Bounds bounds,
    std::enable_if_t<unsigned(ViewType::rank) == 8>* = nullptr) {
  return Kokkos::subview(src, lid, bounds, Kokkos::ALL, Kokkos::ALL,
                         Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
}

template <typename ViewType>
void reset(ViewType B) {
  Kokkos::deep_copy(B, 0);
}

template <typename ViewType>
bool check_sum(ViewType B, const int N,
               typename ViewType::value_type fill_value) {
  using exec_space = typename ViewType::execution_space;

  double sum_all = 0;
  Kokkos::parallel_reduce(
      "Check B", Kokkos::RangePolicy<exec_space>(0, B.span()),
      KOKKOS_LAMBDA(int i, double& lsum) { lsum += B.data()[i]; },
      Kokkos::Sum<double>(sum_all));

  auto correct_sum = fill_value;
  for (size_t i = 0; i < ViewType::rank; ++i) {
    correct_sum *= N;
  }

  return sum_all == correct_sum;
}

template <typename ViewType, typename ExecSpace>
void test_local_deepcopy_thread(ViewType A, ViewType B, const int N) {
  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

  // Test local_deep_copy_thread
  // Each thread copies a subview of A into B
  Kokkos::parallel_for(
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        int lid = teamMember.league_rank();  // returns a number between 0 and N

        // Compute the number of units of work per thread
        auto thread_number = teamMember.league_size();
        auto unitsOfWork   = N / thread_number;
        if (N % thread_number) {
          unitsOfWork += 1;
        }
        auto numberOfBatches = N / unitsOfWork;

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(teamMember, numberOfBatches),
            [=](const int indexWithinBatch) {
              const int idx = indexWithinBatch;

              auto start = idx * unitsOfWork;
              auto stop  = (idx + 1) * unitsOfWork;
              stop       = Kokkos::clamp(stop, 0, N);
              auto subSrc =
                  extract_subview(A, lid, Kokkos::make_pair(start, stop));
              auto subDst =
                  extract_subview(B, lid, Kokkos::make_pair(start, stop));
              Kokkos::Experimental::deep_copy(
                  teamMember, Kokkos::ThreadVectorRange(teamMember, 0), subDst,
                  subSrc);
              // No wait for local_deep_copy_thread
            });
      });

  Kokkos::fence();
  ASSERT_TRUE(view_check_equals(A, B));
}

template <typename ViewType, typename ExecSpace>
void test_local_deepcopy(ViewType A, ViewType B, const int N) {
  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  // Deep Copy
  Kokkos::parallel_for(
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        int lid = teamMember.league_rank();  // returns a number between 0 and N
        auto subSrc = extract_subview(A, lid, Kokkos::ALL);
        auto subDst = extract_subview(B, lid, Kokkos::ALL);
        Kokkos::Experimental::local_deep_copy(teamMember, subDst, subSrc);
      });

  Kokkos::fence();
  ASSERT_TRUE(view_check_equals(A, B));
}

template <typename ViewType, typename ExecSpace>
void test_local_deepcopy_range(ViewType A, ViewType B, const int N) {
  // Deep Copy
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(const int& lid) {
        auto subSrc = extract_subview(A, lid, Kokkos::ALL);
        auto subDst = extract_subview(B, lid, Kokkos::ALL);
        Kokkos::Experimental::local_deep_copy(subDst, subSrc);
      });

  Kokkos::fence();
  ASSERT_TRUE(view_check_equals(A, B));
}

template <typename ViewType, typename ExecSpace>
void test_local_deepcopy_scalar(ViewType B, const int N,
                                typename ViewType::value_type fill_value) {
  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

  Kokkos::parallel_for(
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        int lid = teamMember.league_rank();  // returns a number between 0 and N
        auto subDst = extract_subview(B, lid, Kokkos::ALL);
        Kokkos::Experimental::local_deep_copy(teamMember, subDst, fill_value);
      });

  Kokkos::fence();
  ASSERT_TRUE(check_sum(B, N, fill_value));
}

template <typename ViewType, typename ExecSpace>
void test_local_deepcopy_scalar_range(
    ViewType B, const int N, typename ViewType::value_type fill_value) {
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(const int& lid) {
        auto subDst = extract_subview(B, lid, Kokkos::ALL);
        Kokkos::Experimental::local_deep_copy(subDst, fill_value);
      });

  double sum_all = 0.0;
  Kokkos::parallel_reduce(
      "Check B", Kokkos::RangePolicy<ExecSpace>(0, B.span()),
      KOKKOS_LAMBDA(int i, double& lsum) { lsum += B.data()[i]; },
      Kokkos::Sum<double>(sum_all));

  Kokkos::fence();
  ASSERT_TRUE(check_sum(B, N, fill_value));
}

//-------------------------------------------------------------------------------------------------------------
// Tests scenarii
//-------------------------------------------------------------------------------------------------------------

template <typename ViewType, typename ExecSpace>
void run_team_policy(const int N) {
  auto A = view_create<ViewType>("A", N);
  auto B = view_create<ViewType>("B", N);

  // Initialize A matrix.
  view_init(A);

  test_local_deepcopy_thread<ViewType, ExecSpace>(A, B, N);
  reset(B);
  test_local_deepcopy<ViewType, ExecSpace>(A, B, N);
  reset(B);
  test_local_deepcopy_scalar<ViewType, ExecSpace>(B, N, 20);
}

template <typename ViewType, typename ExecSpace>
void run_range_policy(const int N) {
  auto A = view_create<ViewType>("A", N);
  auto B = view_create<ViewType>("B", N);

  // Initialize A matrix.
  view_init(A);

  test_local_deepcopy_range<ViewType, ExecSpace>(A, B, N);
  reset(B);
  test_local_deepcopy_scalar_range<ViewType, ExecSpace>(B, N, 20);
}

//-------------------------------------------------------------------------------------------------------------
// Test definitions
//-------------------------------------------------------------------------------------------------------------

#if defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)
TEST(TEST_CATEGORY, local_deepcopy_teampolicy_layoutleft) {
  using ExecSpace = TEST_EXECSPACE;
#if defined(KOKKOS_ENABLE_CUDA) && \
    defined(KOKKOS_COMPILER_NVHPC)  // FIXME_NVHPC 23.7
  if (std::is_same_v<ExecSpace, Kokkos::Cuda>)
    GTEST_SKIP()
        << "FIXME_NVHPC : Compiler bug affecting subviews of high rank Views";
#endif
  using Layout = Kokkos::LayoutLeft;

  {  // Rank-1
    run_team_policy<Kokkos::View<double**, Layout, ExecSpace>, ExecSpace>(8);
  }
  {  // Rank-2
    run_team_policy<Kokkos::View<double***, Layout, ExecSpace>, ExecSpace>(8);
  }
  {  // Rank-3
    run_team_policy<Kokkos::View<double****, Layout, ExecSpace>, ExecSpace>(8);
  }
  {  // Rank-4
    run_team_policy<Kokkos::View<double*****, Layout, ExecSpace>, ExecSpace>(8);
  }
  {  // Rank-5
    run_team_policy<Kokkos::View<double******, Layout, ExecSpace>, ExecSpace>(
        8);
  }
  {  // Rank-6
    run_team_policy<Kokkos::View<double*******, Layout, ExecSpace>, ExecSpace>(
        8);
  }
  {  // Rank-7
    run_team_policy<Kokkos::View<double********, Layout, ExecSpace>, ExecSpace>(
        8);
  }
}
//-------------------------------------------------------------------------------------------------------------
TEST(TEST_CATEGORY, local_deepcopy_rangepolicy_layoutleft) {
  using ExecSpace = TEST_EXECSPACE;
#if defined(KOKKOS_ENABLE_CUDA) && \
    defined(KOKKOS_COMPILER_NVHPC)  // FIXME_NVHPC 23.7
  if (std::is_same_v<ExecSpace, Kokkos::Cuda>)
    GTEST_SKIP()
        << "FIXME_NVHPC : Compiler bug affecting subviews of high rank Views";
#endif
  using Layout = Kokkos::LayoutLeft;

  {  // Rank-1
    run_range_policy<Kokkos::View<double**, Layout, ExecSpace>, ExecSpace>(8);
  }
  {  // Rank-2
    run_range_policy<Kokkos::View<double***, Layout, ExecSpace>, ExecSpace>(8);
  }
  {  // Rank-3
    run_range_policy<Kokkos::View<double****, Layout, ExecSpace>, ExecSpace>(8);
  }
  {  // Rank-4
    run_range_policy<Kokkos::View<double*****, Layout, ExecSpace>, ExecSpace>(
        8);
  }
  {  // Rank-5
    run_range_policy<Kokkos::View<double******, Layout, ExecSpace>, ExecSpace>(
        8);
  }
  {  // Rank-6
    run_range_policy<Kokkos::View<double*******, Layout, ExecSpace>, ExecSpace>(
        8);
  }
  {  // Rank-7
    run_range_policy<Kokkos::View<double********, Layout, ExecSpace>,
                     ExecSpace>(8);
  }
}
//-------------------------------------------------------------------------------------------------------------
TEST(TEST_CATEGORY, local_deepcopy_teampolicy_layoutright) {
  using ExecSpace = TEST_EXECSPACE;
#if defined(KOKKOS_ENABLE_CUDA) && \
    defined(KOKKOS_COMPILER_NVHPC)  // FIXME_NVHPC 23.7
  if (std::is_same_v<ExecSpace, Kokkos::Cuda>)
    GTEST_SKIP()
        << "FIXME_NVHPC : Compiler bug affecting subviews of high rank Views";
#endif
  using Layout = Kokkos::LayoutRight;

  {  // Rank-1
    run_team_policy<Kokkos::View<double**, Layout, ExecSpace>, ExecSpace>(8);
  }
  {  // Rank-2
    run_team_policy<Kokkos::View<double***, Layout, ExecSpace>, ExecSpace>(8);
  }
  {  // Rank-3
    run_team_policy<Kokkos::View<double****, Layout, ExecSpace>, ExecSpace>(8);
  }
  {  // Rank-4
    run_team_policy<Kokkos::View<double*****, Layout, ExecSpace>, ExecSpace>(8);
  }
  {  // Rank-5
    run_team_policy<Kokkos::View<double******, Layout, ExecSpace>, ExecSpace>(
        8);
  }
  {  // Rank-6
    run_team_policy<Kokkos::View<double*******, Layout, ExecSpace>, ExecSpace>(
        8);
  }
  {  // Rank-7
    run_team_policy<Kokkos::View<double********, Layout, ExecSpace>, ExecSpace>(
        8);
  }
}
//-------------------------------------------------------------------------------------------------------------
TEST(TEST_CATEGORY, local_deepcopy_rangepolicy_layoutright) {
  using ExecSpace = TEST_EXECSPACE;
#if defined(KOKKOS_ENABLE_CUDA) && \
    defined(KOKKOS_COMPILER_NVHPC)  // FIXME_NVHPC 23.7
  if (std::is_same_v<ExecSpace, Kokkos::Cuda>)
    GTEST_SKIP()
        << "FIXME_NVHPC : Compiler bug affecting subviews of high rank Views";
#endif
  using Layout = Kokkos::LayoutRight;

  {  // Rank-1
    run_range_policy<Kokkos::View<double**, Layout, ExecSpace>, ExecSpace>(8);
  }
  {  // Rank-2
    run_range_policy<Kokkos::View<double***, Layout, ExecSpace>, ExecSpace>(8);
  }
  {  // Rank-3
    run_range_policy<Kokkos::View<double****, Layout, ExecSpace>, ExecSpace>(8);
  }
  {  // Rank-4
    run_range_policy<Kokkos::View<double*****, Layout, ExecSpace>, ExecSpace>(
        8);
  }
  {  // Rank-5
    run_range_policy<Kokkos::View<double******, Layout, ExecSpace>, ExecSpace>(
        8);
  }
  {  // Rank-6
    run_range_policy<Kokkos::View<double*******, Layout, ExecSpace>, ExecSpace>(
        8);
  }
  {  // Rank-7
    run_range_policy<Kokkos::View<double********, Layout, ExecSpace>,
                     ExecSpace>(8);
  }
}
#endif

namespace Impl {
template <typename T, typename SHMEMTYPE>
using ShMemView =
    Kokkos::View<T, Kokkos::LayoutRight, SHMEMTYPE, Kokkos::MemoryUnmanaged>;

struct DeepCopyScratchFunctor {
  DeepCopyScratchFunctor(
      Kokkos::View<double*, TEST_EXECSPACE::memory_space> check_view_1,
      Kokkos::View<double*, TEST_EXECSPACE::memory_space> check_view_2)
      : check_view_1_(check_view_1),
        check_view_2_(check_view_2),
        N_(check_view_1.extent(0)) {}

  KOKKOS_INLINE_FUNCTION void operator()(
      Kokkos::TeamPolicy<TEST_EXECSPACE,
                         Kokkos::Schedule<Kokkos::Dynamic>>::member_type team)
      const {
    using ShmemType = TEST_EXECSPACE::scratch_memory_space;
    auto shview =
        Impl::ShMemView<double**, ShmemType>(team.team_scratch(1), N_, 1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, N_), KOKKOS_LAMBDA(const size_t& index) {
          auto thread_shview = Kokkos::subview(shview, index, Kokkos::ALL());
          Kokkos::Experimental::local_deep_copy(thread_shview, index);
        });
    Kokkos::Experimental::local_deep_copy(
        team, check_view_1_, Kokkos::subview(shview, Kokkos::ALL(), 0));

    Kokkos::Experimental::local_deep_copy(team, shview, 6.);
    Kokkos::Experimental::local_deep_copy(
        team, check_view_2_, Kokkos::subview(shview, Kokkos::ALL(), 0));
  }

  Kokkos::View<double*, TEST_EXECSPACE::memory_space> check_view_1_;
  Kokkos::View<double*, TEST_EXECSPACE::memory_space> check_view_2_;
  int const N_;
};
}  // namespace Impl

TEST(TEST_CATEGORY, deep_copy_scratch) {
  using TestDeviceTeamPolicy = Kokkos::TeamPolicy<TEST_EXECSPACE>;

  const int N = 8;
  const int bytes_per_team =
      Impl::ShMemView<double**,
                      TEST_EXECSPACE::scratch_memory_space>::shmem_size(N, 1);

  TestDeviceTeamPolicy policy(1, Kokkos::AUTO);
  auto team_exec = policy.set_scratch_size(1, Kokkos::PerTeam(bytes_per_team));

  Kokkos::View<double*, TEST_EXECSPACE::memory_space> check_view_1("check_1",
                                                                   N);
  Kokkos::View<double*, TEST_EXECSPACE::memory_space> check_view_2("check_2",
                                                                   N);

  Kokkos::parallel_for(
      team_exec, Impl::DeepCopyScratchFunctor{check_view_1, check_view_2});
  auto host_copy_1 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), check_view_1);
  auto host_copy_2 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), check_view_2);

  for (unsigned int i = 0; i < N; ++i) {
    ASSERT_EQ(host_copy_1(i), i);
    ASSERT_EQ(host_copy_2(i), 6.0);
  }
}
}  // namespace Test

#endif  // TESTLOCALDEEPCOPY_HPP_
