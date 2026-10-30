// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>

namespace {

template <class ExecSpace>
using team_member_t = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

template <class ExecSpace>
using float_tensor4_t = Kokkos::View<float[4][4][4][16], ExecSpace>;

//----------------------------------------------------------------------------
// sum_views: parallel_for(RangePolicy(handle, ...))
//----------------------------------------------------------------------------

template <class Handle, class X>
  requires(X::rank == 2)
KOKKOS_INLINE_FUNCTION void sum_views(const Handle& handle, const X& x,
                                      const float c) {
  Kokkos::parallel_for(
      Kokkos::RangePolicy(handle, 0, x.extent_int(0)),
      KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < x.extent_int(1); ++j) {
          x(i, j) += c;
        }
      });
}

template <class Handle, class X>
  requires(X::rank == 3)
KOKKOS_INLINE_FUNCTION void sum_views(const Handle& handle, const X& x,
                                      const float c) {
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
  requires(X::rank == 4)
KOKKOS_INLINE_FUNCTION void sum_views(const Handle& handle, const X& x,
                                      const float c) {
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

//----------------------------------------------------------------------------
// reduce_views: parallel_reduce(RangePolicy(handle, ...))
//----------------------------------------------------------------------------

template <class Handle, class X>
  requires(X::rank == 2)
KOKKOS_INLINE_FUNCTION void reduce_views(const Handle& handle, const X& x,
                                         float& result) {
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy(handle, 0, x.extent_int(0)),
      KOKKOS_LAMBDA(const int i, float& update) {
        for (int j = 0; j < x.extent_int(1); ++j) {
          update += x(i, j);
        }
      },
      result);
}

template <class Handle, class X>
  requires(X::rank == 3)
KOKKOS_INLINE_FUNCTION void reduce_views(const Handle& handle, const X& x,
                                         float& result) {
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy(handle, 0, x.extent_int(0)),
      KOKKOS_LAMBDA(const int i, float& update) {
        for (int j = 0; j < x.extent_int(1); ++j) {
          for (int k = 0; k < x.extent_int(2); ++k) {
            update += x(i, j, k);
          }
        }
      },
      result);
}

template <class Handle, class X>
  requires(X::rank == 4)
KOKKOS_INLINE_FUNCTION void reduce_views(const Handle& handle, const X& x,
                                         float& result) {
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy(handle, 0, x.extent_int(0)),
      KOKKOS_LAMBDA(const int i, float& update) {
        for (int j = 0; j < x.extent_int(1); ++j) {
          for (int k = 0; k < x.extent_int(2); ++k) {
            for (int l = 0; l < x.extent_int(3); ++l) {
              update += x(i, j, k, l);
            }
          }
        }
      },
      result);
}

//----------------------------------------------------------------------------

template <class ExecSpace>
void verify_sum(const float_tensor4_t<ExecSpace>& M, const float expected,
                const char* label) {
  const auto L     = M.extent_int(0);
  const auto T     = M.extent_int(1);
  const auto V     = M.extent_int(2);
  const auto N     = M.extent_int(3);
  const auto count = L * T * V * N;

  float sum = 0;
  Kokkos::parallel_reduce(
      label, Kokkos::RangePolicy<ExecSpace>(0, count),
      KOKKOS_LAMBDA(const size_t i, float& s) {
        const auto n = i % N;
        const auto v = (i / N) % V;
        const auto t = (i / (N * V)) % T;
        const auto l = i / (N * V * T);
        s += M(l, t, v, n);
      },
      sum);

  ASSERT_FLOAT_EQ(sum, count * expected);
}

//----------------------------------------------------------------------------
// parallel_for / sum_views nesting
//----------------------------------------------------------------------------

template <class ExecSpace>
struct CheckRangePolicyWithExecSpace {
  void operator()() const {
    float_tensor4_t<ExecSpace> M("M");
    Kokkos::deep_copy(M, 0.f);

    const ExecSpace exec;
    sum_views(exec, M, 1.f);

    verify_sum<ExecSpace>(M, 1.f, "check_range_policy_with_exec_space");
  }
};

template <class ExecSpace>
struct CheckRangePolicyWithTeamHandle {
  void operator()() const {
    float_tensor4_t<ExecSpace> M("M");
    Kokkos::deep_copy(M, 0.f);

    using team_t          = team_member_t<ExecSpace>;
    const int num_leagues = M.extent_int(0);
    Kokkos::parallel_for(
        "check_range_policy_with_team_handle",
        Kokkos::TeamPolicy<ExecSpace>(num_leagues, Kokkos::AUTO()),
        KOKKOS_LAMBDA(const team_t& team) {
          sum_views(team,
                    Kokkos::subview(M, team.league_rank(), Kokkos::ALL(),
                                    Kokkos::ALL(), Kokkos::ALL()),
                    2.f);
        });

    verify_sum<ExecSpace>(M, 2.f, "check_range_policy_with_team_handle");
  }
};

template <class ExecSpace>
struct CheckRangePolicyWithThreadHandle {
  void operator()() const {
    float_tensor4_t<ExecSpace> M("M");
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
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, num_threads),
                               [&](const thread_handle& th, int i) {
                                 auto M_sub_sub = Kokkos::subview(
                                     M_sub, i, Kokkos::ALL(), Kokkos::ALL());
                                 sum_views(th, M_sub_sub, 3.f);
                               });
        });

    verify_sum<ExecSpace>(M, 3.f, "check_range_policy_with_thread_handle");
  }
};

template <class ExecSpace>
struct CheckRangePolicyWithConstructedThreadHandle {
  void operator()() const {
    float_tensor4_t<ExecSpace> M("M");
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
            sum_views(thread_handle(team), M_sub, 4.f);
          });
        });

    verify_sum<ExecSpace>(M, 4.f,
                          "check_range_policy_with_constructed_thread_handle");
  }
};

template <class ExecSpace>
struct CheckRangePolicyWithConstructedThreadHandleInSingle {
  void operator()() const {
    float_tensor4_t<ExecSpace> M("M");
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
          Kokkos::single(Kokkos::PerTeam(team),
                         [&]() { sum_views(thread_handle(team), M_sub, 5.f); });
        });

    verify_sum<ExecSpace>(
        M, 5.f, "check_range_policy_with_constructed_thread_handle_in_single");
  }
};

//----------------------------------------------------------------------------
// parallel_reduce / reduce_views nesting
//----------------------------------------------------------------------------

template <class ExecSpace>
struct CheckReduceViewsWithExecSpace {
  void operator()() const {
    float_tensor4_t<ExecSpace> M("M");
    Kokkos::deep_copy(M, 1.f);

    float result = 0.f;
    reduce_views(ExecSpace{}, M, result);

    const float expected = static_cast<float>(
        M.extent_int(0) * M.extent_int(1) * M.extent_int(2) * M.extent_int(3));
    ASSERT_FLOAT_EQ(result, expected);
  }
};

template <class ExecSpace>
struct CheckReduceViewsWithTeamHandle {
  void operator()() const {
    float_tensor4_t<ExecSpace> M("M");
    Kokkos::deep_copy(M, 1.f);

    using team_t          = team_member_t<ExecSpace>;
    const int num_leagues = M.extent_int(0);
    const float expected  = static_cast<float>(
        M.extent_int(0) * M.extent_int(1) * M.extent_int(2) * M.extent_int(3));

    float result = 0.f;
    Kokkos::parallel_reduce(
        "check_reduce_views_with_team_handle",
        Kokkos::TeamPolicy<ExecSpace>(num_leagues, Kokkos::AUTO()),
        KOKKOS_LAMBDA(const team_t& team, float& update) {
          float team_sum = 0.f;
          reduce_views(team,
                       Kokkos::subview(M, team.league_rank(), Kokkos::ALL(),
                                       Kokkos::ALL(), Kokkos::ALL()),
                       team_sum);
          Kokkos::single(Kokkos::PerTeam(team), [&]() { update += team_sum; });
        },
        result);

    ASSERT_FLOAT_EQ(result, expected);
  }
};

template <class ExecSpace>
struct CheckReduceViewsWithThreadHandle {
  void operator()() const {
    float_tensor4_t<ExecSpace> M("M");
    Kokkos::deep_copy(M, 1.f);

    using team_t          = team_member_t<ExecSpace>;
    using thread_handle   = typename team_t::thread_handle;
    const int num_leagues = M.extent_int(0);
    const int num_threads = M.extent_int(1);
    const float expected  = static_cast<float>(
        M.extent_int(0) * M.extent_int(1) * M.extent_int(2) * M.extent_int(3));

    float result = 0.f;
    Kokkos::parallel_reduce(
        "check_reduce_views_with_thread_handle",
        Kokkos::TeamPolicy<ExecSpace>(num_leagues, Kokkos::AUTO()),
        KOKKOS_LAMBDA(const team_t& team, float& update) {
          auto M_sub = Kokkos::subview(M, team.league_rank(), Kokkos::ALL(),
                                       Kokkos::ALL(), Kokkos::ALL());
          // TeamThreadRange reduce with (thread_handle, i, update): nest
          // reduce_views(th, ...) for RangePolicy(ThreadHandle) reduce.
          float team_sum = 0.f;
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, num_threads),
              [&](const thread_handle& th, int i, float& thread_update) {
                auto M_sub_sub =
                    Kokkos::subview(M_sub, i, Kokkos::ALL(), Kokkos::ALL());
                float local = 0.f;
                reduce_views(th, M_sub_sub, local);
                thread_update += local;
              },
              team_sum);
          Kokkos::single(Kokkos::PerTeam(team), [&]() { update += team_sum; });
        },
        result);

    ASSERT_FLOAT_EQ(result, expected);
  }
};

template <class ExecSpace>
struct CheckReduceViewsWithConstructedThreadHandle {
  void operator()() const {
    float_tensor4_t<ExecSpace> M("M");
    Kokkos::deep_copy(M, 1.f);

    using team_t          = team_member_t<ExecSpace>;
    using thread_handle   = typename team_t::thread_handle;
    const int num_leagues = M.extent_int(0);
    const float expected  = static_cast<float>(
        M.extent_int(0) * M.extent_int(1) * M.extent_int(2) * M.extent_int(3));

    float result = 0.f;
    Kokkos::parallel_reduce(
        "check_reduce_views_with_constructed_thread_handle",
        Kokkos::TeamPolicy<ExecSpace>(num_leagues, Kokkos::AUTO()),
        KOKKOS_LAMBDA(const team_t& team, float& update) {
          auto M_sub  = Kokkos::subview(M, team.league_rank(), Kokkos::ALL(),
                                        Kokkos::ALL(), Kokkos::ALL());
          float local = 0.f;
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 1), [&](int) {
            reduce_views(thread_handle(team), M_sub, local);
          });
          Kokkos::single(Kokkos::PerTeam(team), [&]() { update += local; });
        },
        result);

    ASSERT_FLOAT_EQ(result, expected);
  }
};

}  // namespace

TEST(TEST_CATEGORY, self_similar_range_policy_with_execution_space) {
  CheckRangePolicyWithExecSpace<TEST_EXECSPACE>{}();
}

TEST(TEST_CATEGORY, self_similar_range_policy_with_team_handle) {
  CheckRangePolicyWithTeamHandle<TEST_EXECSPACE>{}();
}

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

TEST(TEST_CATEGORY, self_similar_reduce_views_with_execution_space) {
  CheckReduceViewsWithExecSpace<TEST_EXECSPACE>{}();
}

TEST(TEST_CATEGORY, self_similar_reduce_views_with_team_handle) {
  CheckReduceViewsWithTeamHandle<TEST_EXECSPACE>{}();
}

TEST(TEST_CATEGORY, self_similar_reduce_views_with_thread_handle) {
  CheckReduceViewsWithThreadHandle<TEST_EXECSPACE>{}();
}

TEST(TEST_CATEGORY, self_similar_reduce_views_with_constructed_thread_handle) {
  CheckReduceViewsWithConstructedThreadHandle<TEST_EXECSPACE>{}();
}
