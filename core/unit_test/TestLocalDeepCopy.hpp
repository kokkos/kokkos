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

#ifndef KOKKOS_TEST_LOCAL_DEEP_COPY_HPP
#define KOKKOS_TEST_LOCAL_DEEP_COPY_HPP

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
  return result;
}

template <typename ViewType>
void view_init(ViewType& view) {
  using exec_space = typename ViewType::execution_space;

  Kokkos::parallel_for(
      "initialize array", Kokkos::RangePolicy<exec_space>(0, view.span()),
      KOKKOS_LAMBDA(int i) { view.data()[i] = i; });
  Kokkos::fence();
}

template <typename ViewType, std::size_t... Ints>
ViewType view_create(std::string label, const int N,
                     std::index_sequence<Ints...>) {
  return ViewType(label, ((void)Ints, N)...);
}

// Create a view with a given label and dimensions
template <typename ViewType>
ViewType view_create(std::string label, const int N) {
  return view_create<ViewType>(label, N,
                               std::make_index_sequence<ViewType::rank>{});
}

template <typename ViewType, typename Bounds, std::size_t... Ints>
KOKKOS_INLINE_FUNCTION auto extract_subview(ViewType& src, int lid,
                                            Bounds bounds,
                                            std::index_sequence<Ints...>) {
  return Kokkos::subview(src, lid, bounds, ((void)Ints, Kokkos::ALL)...);
}

// Extract a subview from a view to run our tests
template <typename ViewType, typename Bounds>
KOKKOS_INLINE_FUNCTION auto extract_subview(ViewType& src, int lid,
                                            Bounds bounds) {
  return extract_subview(src, lid, bounds,
                         std::make_index_sequence<ViewType::rank - 2>{});
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

  // Test deep_copy with ThreadVectorRange
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
                  Kokkos::ThreadVectorRange(teamMember, 0), subDst, subSrc);
            });
      });

  Kokkos::fence();
  ASSERT_TRUE(view_check_equals(A, B));
}

template <typename ViewType, typename ExecSpace>
void test_local_deepcopy_team(ViewType A, ViewType B, const int N) {
  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  // Deep Copy
  Kokkos::parallel_for(
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        int lid = teamMember.league_rank();  // returns a number between 0 and N
        auto subSrc = extract_subview(A, lid, Kokkos::ALL);
        auto subDst = extract_subview(B, lid, Kokkos::ALL);
        Kokkos::Experimental::deep_copy(Kokkos::TeamVectorRange(teamMember, 0),
                                        subDst, subSrc);
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
        Kokkos::Experimental::deep_copy(Kokkos::Experimental::copy_seq(),
                                        subDst, subSrc);
      });

  Kokkos::fence();
  ASSERT_TRUE(view_check_equals(A, B));
}

template <typename ViewType, typename ExecSpace>
void test_local_deepcopy_thread_extents_mismatch(ViewType A, ViewType B,
                                                 const int N) {
  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

  ::testing::FLAGS_gtest_death_test_style = "threadsafe";

  ASSERT_DEATH(
      {
        Kokkos::parallel_for(
            team_policy(N, Kokkos::AUTO),
            KOKKOS_LAMBDA(const member_type& teamMember) {
              int lid =
                  teamMember.league_rank();  // returns a number between 0 and N

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
                        extract_subview(B, lid, Kokkos::make_pair(0, stop));
                    Kokkos::Experimental::deep_copy(
                        Kokkos::ThreadVectorRange(teamMember, 0), subDst,
                        subSrc);
                  });
            });

        Kokkos::fence();
      },
      "Error: Kokkos::deep_copy extents of views don't match");
}

template <typename ViewType, typename ExecSpace>
void test_local_deepcopy_team_extents_mismatch(ViewType A, ViewType B,
                                               const int N) {
  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

  ::testing::FLAGS_gtest_death_test_style = "threadsafe";

  ASSERT_DEATH(
      {
        Kokkos::parallel_for(
            team_policy(N, Kokkos::AUTO),
            KOKKOS_LAMBDA(const member_type& teamMember) {
              int lid =
                  teamMember.league_rank();  // returns a number between 0 and N
              auto subSrc = extract_subview(A, lid, Kokkos::ALL);
              auto subDst =
                  extract_subview(B, lid, Kokkos::make_pair(0, N - 1));
              Kokkos::Experimental::deep_copy(
                  Kokkos::TeamVectorRange(teamMember, 0), subDst, subSrc);
            });

        Kokkos::fence();
      },
      "Error: Kokkos::deep_copy extents of views don't match");
}

template <typename ViewType, typename ExecSpace>
void test_local_deepcopy_range_extents_mismatch(ViewType A, ViewType B,
                                                const int N) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";

  ASSERT_DEATH(
      {
        Kokkos::parallel_for(
            Kokkos::RangePolicy<ExecSpace>(0, N),
            KOKKOS_LAMBDA(const int& lid) {
              auto subSrc = extract_subview(A, lid, Kokkos::ALL);
              auto subDst =
                  extract_subview(B, lid, Kokkos::make_pair(0, N - 1));
              Kokkos::Experimental::deep_copy(Kokkos::Experimental::copy_seq(),
                                              subDst, subSrc);
            });

        Kokkos::fence();
      },
      "Error: Kokkos::deep_copy extents of views don't match");
}

template <typename ViewType, typename ExecSpace>
void test_local_deepcopy_scalar_thread(
    ViewType B, const int N, typename ViewType::value_type fill_value) {
  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

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
              auto subDst =
                  extract_subview(B, lid, Kokkos::make_pair(start, stop));
              Kokkos::Experimental::deep_copy(
                  Kokkos::ThreadVectorRange(teamMember, 0), subDst, fill_value);
            });
      });

  Kokkos::fence();
  ASSERT_TRUE(check_sum(B, N, fill_value));
}

template <typename ViewType, typename ExecSpace>
void test_local_deepcopy_scalar_team(ViewType B, const int N,
                                     typename ViewType::value_type fill_value) {
  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

  Kokkos::parallel_for(
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        int lid = teamMember.league_rank();  // returns a number between 0 and N
        auto subDst = extract_subview(B, lid, Kokkos::ALL);
        Kokkos::Experimental::deep_copy(Kokkos::TeamVectorRange(teamMember, 0),
                                        subDst, fill_value);
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
        Kokkos::Experimental::deep_copy(Kokkos::Experimental::copy_seq(),
                                        subDst, fill_value);
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

#define KOKKOS_IMPL_LOCAL_DEEP_COPY_TEST(name)                                 \
  template <typename ViewType, typename ExecSpace>                             \
  void run_##name##_policy(const int N) {                                      \
    ViewType A = view_create<ViewType>("A", N);                                \
    ViewType B = view_create<ViewType>("B", N);                                \
                                                                               \
    view_init(A);                                                              \
                                                                               \
    test_local_deepcopy_##name<ViewType, ExecSpace>(A, B, N);                  \
    reset(B);                                                                  \
    test_local_deepcopy_scalar_##name<ViewType, ExecSpace>(B, N, 20);          \
  }                                                                            \
                                                                               \
  TEST(TEST_CATEGORY, local_deep_copy_##name##_layoutleft) {                   \
    using ExecSpace = TEST_EXECSPACE;                                          \
    using Layout    = Kokkos::LayoutLeft;                                      \
                                                                               \
    run_##name##_policy<Kokkos::View<double**, Layout, ExecSpace>, ExecSpace>( \
        8);                                                                    \
    run_##name##_policy<Kokkos::View<double***, Layout, ExecSpace>,            \
                        ExecSpace>(8);                                         \
    run_##name##_policy<Kokkos::View<double****, Layout, ExecSpace>,           \
                        ExecSpace>(8);                                         \
    run_##name##_policy<Kokkos::View<double*****, Layout, ExecSpace>,          \
                        ExecSpace>(8);                                         \
    run_##name##_policy<Kokkos::View<double******, Layout, ExecSpace>,         \
                        ExecSpace>(8);                                         \
    run_##name##_policy<Kokkos::View<double*******, Layout, ExecSpace>,        \
                        ExecSpace>(8);                                         \
    run_##name##_policy<Kokkos::View<double********, Layout, ExecSpace>,       \
                        ExecSpace>(8);                                         \
  }                                                                            \
  TEST(TEST_CATEGORY, local_deep_copy_##name##_layoutright) {                  \
    using ExecSpace = TEST_EXECSPACE;                                          \
    using Layout    = Kokkos::LayoutRight;                                     \
                                                                               \
    run_##name##_policy<Kokkos::View<double**, Layout, ExecSpace>, ExecSpace>( \
        8);                                                                    \
    run_##name##_policy<Kokkos::View<double***, Layout, ExecSpace>,            \
                        ExecSpace>(8);                                         \
    run_##name##_policy<Kokkos::View<double****, Layout, ExecSpace>,           \
                        ExecSpace>(8);                                         \
    run_##name##_policy<Kokkos::View<double*****, Layout, ExecSpace>,          \
                        ExecSpace>(8);                                         \
    run_##name##_policy<Kokkos::View<double******, Layout, ExecSpace>,         \
                        ExecSpace>(8);                                         \
    run_##name##_policy<Kokkos::View<double*******, Layout, ExecSpace>,        \
                        ExecSpace>(8);                                         \
    run_##name##_policy<Kokkos::View<double********, Layout, ExecSpace>,       \
                        ExecSpace>(8);                                         \
  }

KOKKOS_IMPL_LOCAL_DEEP_COPY_TEST(team)
KOKKOS_IMPL_LOCAL_DEEP_COPY_TEST(thread)
KOKKOS_IMPL_LOCAL_DEEP_COPY_TEST(range)

#if (defined(KOKKOS_ENABLE_SYCL) && defined(NDEBUG)) || \
    defined(KOKKOS_ENABLE_OPENMPTARGET) || defined(KOKKOS_ENABLE_OPENACC)
#define KOKKOS_IMPL_LOCAL_DEEP_COPY_DEATH_SKIP                         \
  GTEST_SKIP()                                                         \
      << "Kokkos::abort() does not terminate the program on sycl (in " \
         "release mode), openmptarget and openacc";
#else
#define KOKKOS_IMPL_LOCAL_DEEP_COPY_DEATH_SKIP
#endif

#define KOKKOS_IMPL_LOCAL_DEEP_COPY_DEATH_TEST(name)                         \
  template <typename ViewType, typename ExecSpace>                           \
  void run_##name##_policy_extents_mismatch(const int N) {                   \
    ViewType A = view_create<ViewType>("A", N);                              \
    ViewType B = view_create<ViewType>("B", N);                              \
                                                                             \
    test_local_deepcopy_##name##_extents_mismatch<ViewType, ExecSpace>(A, B, \
                                                                       N);   \
  }                                                                          \
                                                                             \
  TEST(TEST_CATEGORY, local_deep_copy_##name##_extents_mismatch) {           \
    using ExecSpace = TEST_EXECSPACE;                                        \
    using Layout    = Kokkos::LayoutRight;                                   \
    KOKKOS_IMPL_LOCAL_DEEP_COPY_DEATH_SKIP                                   \
    run_##name##_policy_extents_mismatch<                                    \
        Kokkos::View<double**, Layout, ExecSpace>, ExecSpace>(8);            \
    run_##name##_policy_extents_mismatch<                                    \
        Kokkos::View<double***, Layout, ExecSpace>, ExecSpace>(8);           \
    run_##name##_policy_extents_mismatch<                                    \
        Kokkos::View<double****, Layout, ExecSpace>, ExecSpace>(8);          \
    run_##name##_policy_extents_mismatch<                                    \
        Kokkos::View<double*****, Layout, ExecSpace>, ExecSpace>(8);         \
    run_##name##_policy_extents_mismatch<                                    \
        Kokkos::View<double******, Layout, ExecSpace>, ExecSpace>(8);        \
    run_##name##_policy_extents_mismatch<                                    \
        Kokkos::View<double*******, Layout, ExecSpace>, ExecSpace>(8);       \
    run_##name##_policy_extents_mismatch<                                    \
        Kokkos::View<double********, Layout, ExecSpace>, ExecSpace>(8);      \
  }

KOKKOS_IMPL_LOCAL_DEEP_COPY_DEATH_TEST(team)
KOKKOS_IMPL_LOCAL_DEEP_COPY_DEATH_TEST(thread)
KOKKOS_IMPL_LOCAL_DEEP_COPY_DEATH_TEST(range)

//-------------------------------------------------------------------------------------------------------------

#if defined(KOKKOS_ENABLE_DEPRECATED_CODE_4)

#if defined(KOKKOS_ENABLE_DEPRECATION_WARNINGS)
KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_PUSH()
#endif

template <typename ViewType, typename ExecSpace>
void test_local_deepcopy_team_member(ViewType A, ViewType B, const int N) {
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
void test_local_deepcopy_sequential(ViewType A, ViewType B, const int N) {
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
void test_local_deepcopy_scalar_team_member(
    ViewType B, const int N, typename ViewType::value_type fill_value) {
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
void test_local_deepcopy_scalar_sequential(
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

KOKKOS_IMPL_LOCAL_DEEP_COPY_TEST(team_member)
KOKKOS_IMPL_LOCAL_DEEP_COPY_TEST(sequential)

#if defined(KOKKOS_ENABLE_DEPRECATION_WARNINGS)
KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_POP()
#endif

#endif

namespace Impl {
template <typename T, typename ShMemType>
using ShMemView =
    Kokkos::View<T, Kokkos::LayoutRight, ShMemType, Kokkos::MemoryUnmanaged>;

struct DeepCopyScratchFunctor {
  DeepCopyScratchFunctor(
      Kokkos::View<double*, TEST_EXECSPACE::memory_space> check_view_1,
      Kokkos::View<double*, TEST_EXECSPACE::memory_space> check_view_2,
      int scratch_level)
      : check_view_1_(check_view_1),
        check_view_2_(check_view_2),
        N_(check_view_1.extent(0)),
        scratch_level_(scratch_level) {}

  KOKKOS_INLINE_FUNCTION void operator()(
      Kokkos::TeamPolicy<TEST_EXECSPACE,
                         Kokkos::Schedule<Kokkos::Dynamic>>::member_type team)
      const {
    using ShmemType = TEST_EXECSPACE::scratch_memory_space;
    auto shview     = Impl::ShMemView<double**, ShmemType>(
        team.team_scratch(scratch_level_), N_, 1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, N_), KOKKOS_LAMBDA(const size_t& index) {
          auto thread_shview = Kokkos::subview(shview, index, Kokkos::ALL());
          Kokkos::Experimental::deep_copy(Kokkos::ThreadVectorRange(team, 0),
                                          thread_shview, index);
        });

    if (scratch_level_ == 0) {
      Kokkos::Experimental::deep_copy(
          Kokkos::TeamThreadRange(team, 0), check_view_1_,
          Kokkos::subview(shview, Kokkos::ALL(), 0));

      Kokkos::Experimental::deep_copy(Kokkos::TeamThreadRange(team, 0), shview,
                                      6.);
      Kokkos::Experimental::deep_copy(
          Kokkos::TeamThreadRange(team, 0), check_view_2_,
          Kokkos::subview(shview, Kokkos::ALL(), 0));
    } else {
      Kokkos::Experimental::deep_copy(
          Kokkos::TeamVectorRange(team, 0), check_view_1_,
          Kokkos::subview(shview, Kokkos::ALL(), 0));

      Kokkos::Experimental::deep_copy(Kokkos::TeamVectorRange(team, 0), shview,
                                      6.);
      Kokkos::Experimental::deep_copy(
          Kokkos::TeamVectorRange(team, 0), check_view_2_,
          Kokkos::subview(shview, Kokkos::ALL(), 0));
    }
  }

  Kokkos::View<double*, TEST_EXECSPACE::memory_space> check_view_1_;
  Kokkos::View<double*, TEST_EXECSPACE::memory_space> check_view_2_;
  int const N_;
  int const scratch_level_;
};
}  // namespace Impl

TEST(TEST_CATEGORY, deep_copy_team_scratch) {
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
      team_exec, Impl::DeepCopyScratchFunctor{check_view_1, check_view_2, 1});
  auto host_copy_1 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), check_view_1);
  auto host_copy_2 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), check_view_2);

  for (unsigned int i = 0; i < N; ++i) {
    ASSERT_EQ(host_copy_1(i), i);
    ASSERT_EQ(host_copy_2(i), 6.0);
  }
}

TEST(TEST_CATEGORY, deep_copy_thread_scratch) {
  using TestDeviceTeamPolicy = Kokkos::TeamPolicy<TEST_EXECSPACE>;

  const int N = 8;
  const int bytes_per_team =
      Impl::ShMemView<double**,
                      TEST_EXECSPACE::scratch_memory_space>::shmem_size(N, 1);

  TestDeviceTeamPolicy policy(1, Kokkos::AUTO);
  auto team_exec = policy.set_scratch_size(0, Kokkos::PerTeam(bytes_per_team));

  Kokkos::View<double*, TEST_EXECSPACE::memory_space> check_view_1("check_1",
                                                                   N);
  Kokkos::View<double*, TEST_EXECSPACE::memory_space> check_view_2("check_2",
                                                                   N);

  Kokkos::parallel_for(
      team_exec, Impl::DeepCopyScratchFunctor{check_view_1, check_view_2, 0});
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

#undef KOKKOS_IMPL_LOCAL_DEEP_COPY_TEST
#undef KOKKOS_IMPL_LOCAL_DEEP_COPY_DEATH_TEST
#undef KOKKOS_IMPL_LOCAL_DEEP_COPY_DEATH_SKIP

#endif
