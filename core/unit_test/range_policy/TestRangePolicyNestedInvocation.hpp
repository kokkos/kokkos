// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>

// ThreadHandle team members are provided for the Cuda category only.
// TEST_CATEGORY_NUMBER 5 is Cuda.
#if TEST_CATEGORY_NUMBER == 5
#define KOKKOS_IMPL_THREAD_HANDLE_TESTS 1
#endif

namespace {

template <class ExecSpace>
using team_member_t = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

template <class ExecSpace>
using float_tensor4_t = Kokkos::View<float****, ExecSpace>;

template <class ExecSpace>
using float_tensor4_row_t = decltype(Kokkos::subview(
    std::declval<float_tensor4_t<ExecSpace>&>(), std::declval<int>(),
    std::declval<int>(), std::declval<int>(), Kokkos::ALL()));

template <class ExecSpace>
struct Tensor4 {
  static constexpr int leagues  = 4;
  static constexpr int threads  = 4;
  static constexpr int vectors  = 4;
  static constexpr int elements = 16;  // row
};

template <class Handle, class X>
KOKKOS_INLINE_FUNCTION std::enable_if_t<X::rank == 2> sum_views(
    const Handle& handle, const X& x, const float c) {
  Kokkos::parallel_for(
      Kokkos::RangePolicy(handle, 0, x.extent_int(0)),
      KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < x.extent_int(1); ++j) {
          x(i, j) += c;
        }
      });
}

template <class Handle, class X>
KOKKOS_INLINE_FUNCTION std::enable_if_t<X::rank == 3> sum_views(
    const Handle& handle, const X& x, const float c) {
  Kokkos::parallel_for(
      Kokkos::RangePolicy(handle, 0, x.extent_int(0)),
      KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < x.extent_int(1); ++j) {
          for (int k = 0; k < x.extent_int(2); ++k) {
            x(i, j, k) += c;
          }
        }
      });
}

template <class Handle, class X>
KOKKOS_INLINE_FUNCTION std::enable_if_t<X::rank == 4> sum_views(
    const Handle& handle, const X& x, const float c) {
  Kokkos::parallel_for(
      Kokkos::RangePolicy(handle, 0, x.extent_int(0)),
      KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < x.extent_int(1); ++j) {
          for (int k = 0; k < x.extent_int(2); ++k) {
            for (int l = 0; l < x.extent_int(3); ++l) {
              x(i, j, k, l) += c;
            }
          }
        }
      });
}

template <class ExecSpace>
void verify(const float_tensor4_t<ExecSpace>& M, const float expected,
            const char* label) {
  const int L      = M.extent_int(0);
  const int T      = M.extent_int(1);
  const int V      = M.extent_int(2);
  const int N      = M.extent_int(3);
  const auto count = static_cast<size_t>(L) * T * V * N;

  double sum = 0;
  // Flatten indices to avoid use of nesting and associated data structures
  Kokkos::parallel_reduce(
      label, Kokkos::RangePolicy<ExecSpace>(0, count),
      KOKKOS_LAMBDA(const size_t i, double& s) {
        const int n = static_cast<int>(i % static_cast<size_t>(N));
        const int v = static_cast<int>((i / N) % V);
        const int t = static_cast<int>((i / (static_cast<size_t>(N) * V)) % T);
        const int l = static_cast<int>(i / (static_cast<size_t>(N) * V * T));
        s += M(l, t, v, n);
      },
      sum);

  ASSERT_FLOAT_EQ(static_cast<float>(sum),
                  static_cast<float>(count) * expected);
}

template <class ExecSpace>
void allocate(float_tensor4_t<ExecSpace>& M) {
  using D = Tensor4<ExecSpace>;
  M       = float_tensor4_t<ExecSpace>("M", D::leagues, D::threads, D::vectors,
                                 D::elements);
}

template <class ExecSpace>
struct CheckRangePolicyWithExecSpace {
  void operator()() const {
    float_tensor4_t<ExecSpace> M;
    allocate<ExecSpace>(M);
    Kokkos::deep_copy(M, 0.f);

    const ExecSpace exec;
    // sum_views(exec, ...): RangePolicy(exec, 0, M.extent_int(0)) over the
    // execution space.
    sum_views(exec, M, 1.f);

    verify<ExecSpace>(M, 1.f, "check_range_policy_with_exec_space");
  }
};

template <class ExecSpace>
struct CheckRangePolicyWithTeamHandle {
  void operator()() const {
    float_tensor4_t<ExecSpace> M;
    allocate<ExecSpace>(M);
    Kokkos::deep_copy(M, 0.f);

    using team_t          = team_member_t<ExecSpace>;
    const int num_leagues = M.extent_int(0);
    Kokkos::parallel_for(
        "check_range_policy_with_team_handle",
        Kokkos::TeamPolicy<ExecSpace>(num_leagues, Kokkos::AUTO()),
        KOKKOS_LAMBDA(const team_t& team) {
          // sum_views(team, ...): RangePolicy(team, 0, M_sub.extent_int(0)) ->
          // TeamVectorRange.
          sum_views(team,
                    Kokkos::subview(M, team.league_rank(), Kokkos::ALL(),
                                    Kokkos::ALL(), Kokkos::ALL()),
                    2.f);
        });

    verify<ExecSpace>(M, 2.f, "check_range_policy_with_team_handle");
  }
};

#if KOKKOS_IMPL_THREAD_HANDLE_TESTS

template <class ExecSpace>
struct CheckRangePolicyWithThreadHandle {
  void operator()() const {
    float_tensor4_t<ExecSpace> M;
    allocate<ExecSpace>(M);
    Kokkos::deep_copy(M, 0.f);

    using team_t          = team_member_t<ExecSpace>;
    using thread_handle   = typename team_t::thread_handle;
    const int num_leagues = M.extent_int(0);
    const int num_threads = M.extent_int(1);
    Kokkos::parallel_for(
        "check_range_policy_with_thread_handle",
        Kokkos::TeamPolicy<ExecSpace>(num_leagues, Kokkos::AUTO()),
        KOKKOS_LAMBDA(const team_t& team) {
          auto M_sub = Kokkos::subview(M, team.league_rank(), Kokkos::ALL(),
                                       Kokkos::ALL(), Kokkos::ALL());
          // TeamThreadRange(team, num_threads) with (thread_handle, i).
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, num_threads),
                               [&](const thread_handle& th, int i) {
                                 auto M_sub_sub = Kokkos::subview(
                                     M_sub, i, Kokkos::ALL(), Kokkos::ALL());
                                 // Inner (sum_views): RangePolicy(th, 0,
                                 // M_sub_sub.extent_int(0)) with (int) ->
                                 // ThreadVectorRange.
                                 sum_views(th, M_sub_sub, 3.f);
                               });
        });

    verify<ExecSpace>(M, 3.f, "check_range_policy_with_thread_handle");
  }
};

template <class ExecSpace>
struct CheckRangePolicyWithConstructedThreadHandle {
  void operator()() const {
    float_tensor4_t<ExecSpace> M;
    allocate<ExecSpace>(M);
    Kokkos::deep_copy(M, 0.f);

    using team_t          = team_member_t<ExecSpace>;
    using thread_handle   = typename team_t::thread_handle;
    const int num_leagues = M.extent_int(0);
    Kokkos::parallel_for(
        "check_range_policy_with_constructed_thread_handle",
        Kokkos::TeamPolicy<ExecSpace>(num_leagues, Kokkos::AUTO()),
        KOKKOS_LAMBDA(const team_t& team) {
          auto M_sub = Kokkos::subview(M, team.league_rank(), Kokkos::ALL(),
                                       Kokkos::ALL(), Kokkos::ALL());
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 1), [&](int) {
            // Inner (sum_views): RangePolicy(thread_handle, 0,
            // M_sub.extent_int(0)) with (int) -> ThreadVectorRange.
            sum_views(thread_handle(team), M_sub, 4.f);
          });
        });

    verify<ExecSpace>(M, 4.f,
                      "check_range_policy_with_constructed_thread_handle");
  }
};

template <class ExecSpace>
struct CheckRangePolicyWithConstructedThreadHandleInSingle {
  void operator()() const {
    float_tensor4_t<ExecSpace> M;
    allocate<ExecSpace>(M);
    Kokkos::deep_copy(M, 0.f);

    using team_t          = team_member_t<ExecSpace>;
    using thread_handle   = typename team_t::thread_handle;
    const int num_leagues = M.extent_int(0);
    Kokkos::parallel_for(
        "check_range_policy_with_constructed_thread_handle_in_single",
        Kokkos::TeamPolicy<ExecSpace>(num_leagues, Kokkos::AUTO()),
        KOKKOS_LAMBDA(const team_t& team) {
          auto M_sub = Kokkos::subview(M, team.league_rank(), Kokkos::ALL(),
                                       Kokkos::ALL(), Kokkos::ALL());
          Kokkos::single(Kokkos::PerTeam(team), [&]() {
            // Inner (sum_views): RangePolicy(thread_handle, 0,
            // M_sub.extent_int(0)) with (int) -> ThreadVectorRange.
            sum_views(thread_handle(team), M_sub, 5.f);
          });
        });

    verify<ExecSpace>(
        M, 5.f, "check_range_policy_with_constructed_thread_handle_in_single");
  }
};

#endif  // KOKKOS_IMPL_THREAD_HANDLE_TESTS

}  // namespace

TEST(TEST_CATEGORY, self_similar_range_policy_with_execution_space) {
  CheckRangePolicyWithExecSpace<TEST_EXECSPACE>{}();
}

TEST(TEST_CATEGORY, self_similar_range_policy_with_team_handle) {
  CheckRangePolicyWithTeamHandle<TEST_EXECSPACE>{}();
}

#if KOKKOS_IMPL_THREAD_HANDLE_TESTS

TEST(TEST_CATEGORY, self_similar_range_policy_with_thread_handle) {
  CheckRangePolicyWithThreadHandle<TEST_EXECSPACE>{}();
}

TEST(TEST_CATEGORY, self_similar_range_policy_with_constructed_thread_handle) {
  CheckRangePolicyWithConstructedThreadHandle<TEST_EXECSPACE>{}();
}

TEST(TEST_CATEGORY,
     self_similar_range_policy_with_constructed_thread_handle_in_single) {
  CheckRangePolicyWithConstructedThreadHandleInSingle<TEST_EXECSPACE>{}();
}

#endif  // KOKKOS_IMPL_THREAD_HANDLE_TESTS

#undef KOKKOS_IMPL_THREAD_HANDLE_TESTS
