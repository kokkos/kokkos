// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>

namespace Test {

template <class Policy>
KOKKOS_INLINE_FUNCTION int check_runtime_inputs(
    Policy& p, const typename Policy::index_type expected_begin,
    const typename Policy::index_type expected_end,
    const typename Policy::index_type chunk_size = 1) {
  int nerrs = 0;

  if (p.begin() != expected_begin) ++nerrs;
  if (p.end() != expected_end) ++nerrs;

  auto p2 = p.set_chunk_size(chunk_size);
  if (p2.chunk_size() != chunk_size) ++nerrs;

  return nerrs;
}

void check_runtime_values() {
  using IndexType = typename Kokkos::DefaultExecutionSpace::size_type;

  IndexType beg        = 5;
  IndexType end        = 15;
  IndexType chunk_size = 10;

  auto p_execspace =
      Kokkos::RangePolicy(Kokkos::DefaultExecutionSpace(), beg, end);
  auto nerrs_exec_space =
      check_runtime_inputs(p_execspace, beg, end, chunk_size);
  ASSERT_EQ(nerrs_exec_space, 0);

  int nerrs_team_handle;
  using team_t = typename Kokkos::TeamPolicy<>::member_type;
  Kokkos::parallel_reduce(
      "check_runtime", Kokkos::TeamPolicy(1, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const team_t& team, int& nerrs) {
        auto p_teamhandle = Kokkos::RangePolicy(team, beg, end);
        auto tvr          = Kokkos::TeamVectorRange(team, beg, end);
        nerrs = check_runtime_inputs(p_teamhandle, tvr.start, tvr.end);
      },
      nerrs_team_handle);
  ASSERT_EQ(nerrs_team_handle, 0);

  int nerrs_thread_handle;
  Kokkos::parallel_reduce(
      "check_runtime_thread", Kokkos::TeamPolicy(1, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const team_t& team, int& nerrs) {
        auto p_threadhandle =
            Kokkos::RangePolicy(Kokkos::ThreadHandle<team_t>(team), beg, end);
        auto tvr = Kokkos::ThreadVectorRange(team, beg, end);
        nerrs    = check_runtime_inputs(p_threadhandle, tvr.start, tvr.end);
      },
      nerrs_thread_handle);
  ASSERT_EQ(nerrs_thread_handle, 0);
}

template <class Exec, class X, class Y>
KOKKOS_INLINE_FUNCTION void sum_views(const Exec& exec, const X& x,
                                      const Y& y) {
  auto policy = Kokkos::RangePolicy(exec, 0, x.extent(0));
  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const int& i) { x(i) += y(i); });
}

// Two-level self-similar pattern: the same sum_views template is used at each
// nesting level; Kokkos::RangePolicy deduces the parallel layer from the
// handle:
//   lvl1: execution space -> RangePolicy(exec, ...) partitions over the space
//   lvl2: team member     -> RangePolicy(team, ...) matches TeamVectorRange
void self_similar_range_policy_sum_views_case1() {
  size_t N         = 7;
  size_t num_teams = 5;

  Kokkos::View<float*> v_x("v_x", N), v_y("v_y", N);
  Kokkos::View<float**> M_x("M_x", num_teams, N), M_y("M_y", num_teams, N);

  // Initialize v_x and v_y with values from 1 to N
  Kokkos::parallel_for(
      "init_v_x", Kokkos::RangePolicy<>(0, N),
      KOKKOS_LAMBDA(const int& i) { v_x(i) = static_cast<float>(i + 1); });
  Kokkos::parallel_for(
      "init_v_y", Kokkos::RangePolicy<>(0, N),
      KOKKOS_LAMBDA(const int& i) { v_y(i) = static_cast<float>(i + 1); });

  // Initialize M_x and M_y (element (i,j) = row-major linear index + 1)
  Kokkos::parallel_for(
      "init_M_x", Kokkos::RangePolicy<>(0, num_teams),
      KOKKOS_LAMBDA(const int& i) {
        for (size_t j = 0; j < N; j++) {
          M_x(i, j) = static_cast<float>(i * N + j + 1);
        }
      });
  Kokkos::parallel_for(
      "init_M_y", Kokkos::RangePolicy<>(0, num_teams),
      KOKKOS_LAMBDA(const int& i) {
        for (size_t j = 0; j < N; j++) {
          M_y(i, j) = static_cast<float>(i * N + j + 1);
        }
      });

  // Call sum_views(ExecSpace):
  sum_views(Kokkos::DefaultExecutionSpace(), v_x, v_y);

  // Call sum_views(TeamHandle)
  using team_t = typename Kokkos::TeamPolicy<>::member_type;
  Kokkos::parallel_for(
      "apxyFromTeam", Kokkos::TeamPolicy(num_teams, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const team_t& team) {
        sum_views(team, Kokkos::subview(M_x, team.league_rank(), Kokkos::ALL()),
                  Kokkos::subview(M_y, team.league_rank(), Kokkos::ALL()));
      });

  // Check v_x
  size_t result = 0;
  Kokkos::parallel_reduce(
      "Check1", v_x.extent(0),
      KOKKOS_LAMBDA(int i, size_t& val) { val += v_x(i); }, result);
  size_t expected_v_x = N * (N + 1);
  ASSERT_EQ(result, expected_v_x);

  // Check individual elements of v_x
  Kokkos::parallel_reduce(
      "Check1_elements", v_x.extent(0),
      KOKKOS_LAMBDA(int i, size_t& errors) {
        float expected = static_cast<float>(2 * (i + 1));
        if (v_x(i) != expected) ++errors;
      },
      result);
  ASSERT_EQ(result, size_t(0));

  // Check M_x
  result = 0;
  Kokkos::parallel_reduce(
      "Check2", M_x.extent(0),
      KOKKOS_LAMBDA(int i, size_t& val) {
        for (int j = 0; j < M_x.extent_int(1); j++) val += M_x(i, j);
      },
      result);
  size_t M_total      = num_teams * N;
  size_t expected_M_x = M_total * (M_total + 1);
  ASSERT_EQ(result, expected_M_x);

  // Check individual elements of M_x
  Kokkos::parallel_reduce(
      "Check2_elements", M_x.extent(0),
      KOKKOS_LAMBDA(int i, size_t& errors) {
        for (int j = 0; j < M_x.extent_int(1); j++) {
          float expected = static_cast<float>(2 * (i * N + j + 1));
          if (M_x(i, j) != expected) ++errors;
        }
      },
      result);
  ASSERT_EQ(result, size_t(0));
}

// Three-level self-similar pattern (execution space -> team -> thread):
//   lvl1: DefaultExecutionSpace -> sum_views(exec, ...) uses RangePolicy over
//         the whole execution space
//   lvl2: team member -> sum_views(team, ...) uses RangePolicy with
//   TeamVectorRange
//         semantics (partition across threads in the team)
//   lvl3: ThreadHandle -> sum_views(th, ...) uses RangePolicy(th, ...) with
//         ThreadVectorRange semantics (vector parallelism within the thread)
//
// After lvl2 (team-vector RangePolicy), lvl3 is entered with
// parallel_for(TeamThreadRange(team, 1), ...): one team-thread iteration with a
// thread_handle (and team-thread index when the closure accepts it), not nested
// RangePolicy(team, ...) / TeamVectorRange.

void self_similar_range_policy_sum_views_case2() {
  const size_t N         = 16;
  const size_t num_teams = 4;

  Kokkos::View<float*> v_x("v_x", N), v_y("v_y", N);
  Kokkos::View<float**> M_x("M_x", num_teams, N),
      M_add2("M_add2", num_teams, N), M_add4("M_add4", num_teams, N);

  Kokkos::parallel_for(
      "init_v", Kokkos::RangePolicy<>(0, N), KOKKOS_LAMBDA(const size_t i) {
        v_x(i) = 0.f;
        v_y(i) = 1.f;
      });
  Kokkos::parallel_for(
      "init_M", Kokkos::RangePolicy<>(0, num_teams),
      KOKKOS_LAMBDA(const size_t i) {
        for (size_t j = 0; j < N; j++) {
          M_x(i, j)    = 0.f;
          M_add2(i, j) = 2.f;
          M_add4(i, j) = 4.f;
        }
      });

  // lvl1: sum_views with ExecutionSpace -> RangePolicy<ExecSpace>
  sum_views(Kokkos::DefaultExecutionSpace(), v_x, v_y);

  // lvl2: sum_views with TeamHandle -> RangePolicy<TeamHandle>
  // (TeamVectorRange)
  using team_t        = typename Kokkos::TeamPolicy<>::member_type;
  using thread_handle = team_t::thread_handle;
  Kokkos::parallel_for(
      "nested_team", Kokkos::TeamPolicy(num_teams, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const team_t& team) {
        auto row_x = Kokkos::subview(M_x, team.league_rank(), Kokkos::ALL());
        auto row_add2 =
            Kokkos::subview(M_add2, team.league_rank(), Kokkos::ALL());
        sum_views(team, row_x, row_add2);

        // lvl3: TeamThreadRange — closure(thread_handle, i) exercises the
        // two-argument team-thread dispatch; sum_views uses RangePolicy(th,
        // ...).
        auto row_add4 =
            Kokkos::subview(M_add4, team.league_rank(), Kokkos::ALL());
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 1),
                             [&](const thread_handle& th, int i) {
                               (void)i;
                               sum_views(th, row_x, row_add4);
                             });
      });

  // Verify: v_x = v_y (each element = 1)
  size_t result = 0;
  Kokkos::parallel_reduce(
      "check_v", N,
      KOKKOS_LAMBDA(size_t i, size_t & s) { s += static_cast<size_t>(v_x(i)); },
      result);
  ASSERT_EQ(result, N);

  // Verify: M_x gets +2 (lvl2) +4 (lvl3) = 6 per element
  result = 0;
  Kokkos::parallel_reduce(
      "check_M", Kokkos::RangePolicy<>(0, num_teams * N),
      KOKKOS_LAMBDA(size_t i, size_t & s) {
        int row = i / N;
        int col = i % N;
        s += static_cast<size_t>(M_x(row, col));
      },
      result);
  ASSERT_EQ(result, num_teams * N * 6);
}

// Same pattern as self_similar_range_policy_sum_views_case2, but the
// inner parallel_for uses the index-only closure; ThreadHandle is built inside
// the lambda from the team member.
void self_similar_range_policy_sum_views_case3() {
  const size_t N         = 16;
  const size_t num_teams = 4;

  Kokkos::View<float*> v_x("v_x", N), v_y("v_y", N);
  Kokkos::View<float**> M_x("M_x", num_teams, N),
      M_add2("M_add2", num_teams, N), M_add4("M_add4", num_teams, N);

  Kokkos::parallel_for(
      "init_v", Kokkos::RangePolicy<>(0, N), KOKKOS_LAMBDA(const size_t i) {
        v_x(i) = 0.f;
        v_y(i) = 1.f;
      });
  Kokkos::parallel_for(
      "init_M", Kokkos::RangePolicy<>(0, num_teams),
      KOKKOS_LAMBDA(const size_t i) {
        for (size_t j = 0; j < N; j++) {
          M_x(i, j)    = 0.f;
          M_add2(i, j) = 2.f;
          M_add4(i, j) = 4.f;
        }
      });

  sum_views(Kokkos::DefaultExecutionSpace(), v_x, v_y);

  using team_t = typename Kokkos::TeamPolicy<>::member_type;
  Kokkos::parallel_for(
      "nested_team", Kokkos::TeamPolicy(num_teams, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const team_t& team) {
        auto row_x = Kokkos::subview(M_x, team.league_rank(), Kokkos::ALL());
        auto row_add2 =
            Kokkos::subview(M_add2, team.league_rank(), Kokkos::ALL());
        sum_views(team, row_x, row_add2);

        auto row_add4 =
            Kokkos::subview(M_add4, team.league_rank(), Kokkos::ALL());
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 1), [&](int i) {
          (void)i;
          sum_views(Kokkos::ThreadHandle<team_t>(team), row_x, row_add4);
        });
      });

  size_t result = 0;
  Kokkos::parallel_reduce(
      "check_v", N,
      KOKKOS_LAMBDA(size_t i, size_t & s) { s += static_cast<size_t>(v_x(i)); },
      result);
  ASSERT_EQ(result, N);

  result = 0;
  Kokkos::parallel_reduce(
      "check_M", Kokkos::RangePolicy<>(0, num_teams * N),
      KOKKOS_LAMBDA(size_t i, size_t & s) {
        int row = i / N;
        int col = i % N;
        s += static_cast<size_t>(M_x(row, col));
      },
      result);
  ASSERT_EQ(result, num_teams * N * 6);
}

void self_similar_range_policy_sum_views_case4() {
  const size_t N         = 16;
  const size_t num_teams = 4;

  Kokkos::View<float**> M_x("M_x", num_teams, N),
      M_add4("M_add4", num_teams, N);
  Kokkos::parallel_for(
      "init_M", Kokkos::RangePolicy<>(0, num_teams),
      KOKKOS_LAMBDA(const size_t i) {
        for (size_t j = 0; j < N; j++) {
          M_x(i, j)    = 0.f;
          M_add4(i, j) = 4.f;
        }
      });

  using team_t = typename Kokkos::TeamPolicy<>::member_type;
  Kokkos::parallel_for(
      "team_then_thread_handle", Kokkos::TeamPolicy(num_teams, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const team_t& team) {
        auto row_x = Kokkos::subview(M_x, team.league_rank(), Kokkos::ALL());
        auto row_add4 =
            Kokkos::subview(M_add4, team.league_rank(), Kokkos::ALL());
        Kokkos::single(Kokkos::PerTeam(team), [&]() {
          sum_views(Kokkos::ThreadHandle<team_t>(team), row_x, row_add4);
        });
      });

  size_t result = 0;
  Kokkos::parallel_reduce(
      "check_M_thread_handle", Kokkos::RangePolicy<>(0, num_teams * N),
      KOKKOS_LAMBDA(size_t i, size_t & s) {
        int row = i / N;
        int col = i % N;
        s += static_cast<size_t>(M_x(row, col));
      },
      result);
  ASSERT_EQ(result, num_teams * N * 4);
}

// Like self_similar_range_policy_sum_views_case2, but the inner
// TeamThreadRange closure takes only thread_handle (no team-thread index);
// exercises parallel_for(TeamThreadRange, ...) dispatch to closure(th).
void self_similar_range_policy_sum_views_case5() {
  const size_t N         = 16;
  const size_t num_teams = 4;

  Kokkos::View<float*> v_x("v_x", N), v_y("v_y", N);
  Kokkos::View<float**> M_x("M_x", num_teams, N),
      M_add2("M_add2", num_teams, N), M_add4("M_add4", num_teams, N);

  Kokkos::parallel_for(
      "init_v", Kokkos::RangePolicy<>(0, N), KOKKOS_LAMBDA(const size_t i) {
        v_x(i) = 0.f;
        v_y(i) = 1.f;
      });
  Kokkos::parallel_for(
      "init_M", Kokkos::RangePolicy<>(0, num_teams),
      KOKKOS_LAMBDA(const size_t i) {
        for (size_t j = 0; j < N; j++) {
          M_x(i, j)    = 0.f;
          M_add2(i, j) = 2.f;
          M_add4(i, j) = 4.f;
        }
      });

  sum_views(Kokkos::DefaultExecutionSpace(), v_x, v_y);

  using team_t        = typename Kokkos::TeamPolicy<>::member_type;
  using thread_handle = team_t::thread_handle;
  Kokkos::parallel_for(
      "nested_team_th_only", Kokkos::TeamPolicy(num_teams, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const team_t& team) {
        auto row_x = Kokkos::subview(M_x, team.league_rank(), Kokkos::ALL());
        auto row_add2 =
            Kokkos::subview(M_add2, team.league_rank(), Kokkos::ALL());
        sum_views(team, row_x, row_add2);

        auto row_add4 =
            Kokkos::subview(M_add4, team.league_rank(), Kokkos::ALL());
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, 1),
            [&](const thread_handle& th) { sum_views(th, row_x, row_add4); });
      });

  size_t result = 0;
  Kokkos::parallel_reduce(
      "check_v_th_only", N,
      KOKKOS_LAMBDA(size_t i, size_t & s) { s += static_cast<size_t>(v_x(i)); },
      result);
  ASSERT_EQ(result, N);

  result = 0;
  Kokkos::parallel_reduce(
      "check_M_th_only", Kokkos::RangePolicy<>(0, num_teams * N),
      KOKKOS_LAMBDA(size_t i, size_t & s) {
        int row = i / N;
        int col = i % N;
        s += static_cast<size_t>(M_x(row, col));
      },
      result);
  ASSERT_EQ(result, num_teams * N * 6);
}

// RangePolicy(team, ...) maps to TeamVectorRange. So no further concurrency is
// possible, and the closure must be invoked as closure(i) only. This test
// ensures we do not attempt to dispatch closure(thread_handle, i) (or
// closure(thread_handle)) from a TeamVectorRange.
void self_similar_range_policy_sum_views_case6() {
  const int N         = 32;
  const int num_teams = 4;

  using exec_space = Kokkos::DefaultExecutionSpace;
  using mem_space  = typename exec_space::memory_space;

  Kokkos::View<int, mem_space, Kokkos::MemoryTraits<Kokkos::Atomic>> count_i(
      "count_i");
  Kokkos::deep_copy(count_i, 0);

  using team_t        = typename Kokkos::TeamPolicy<>::member_type;
  using thread_handle = team_t::thread_handle;

  struct Closure {
    KOKKOS_INLINE_FUNCTION void operator()(const int) const {}

    KOKKOS_INLINE_FUNCTION void operator()(const thread_handle&,
                                           const int) const {
      Kokkos::abort(
          "RangePolicy(team, ...) maps to TeamVectorRange; "
          "closure(thread_handle, i) must not be used");
    }

    KOKKOS_INLINE_FUNCTION void operator()(const thread_handle&) const {
      Kokkos::abort(
          "RangePolicy(team, ...) maps to TeamVectorRange; "
          "closure(thread_handle) must not be used");
    }
  };

  Kokkos::parallel_for(
      "team_vector_index_only", Kokkos::TeamPolicy(num_teams, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const team_t& team) {
        Kokkos::parallel_for(Kokkos::RangePolicy(team, 0, N), Closure{});
      });
}

// Level-3 pattern from the self-similar RangePolicy interface: inside a team,
// parallel_for(RangePolicy(team, ...), closure(thread_handle)) obtains a
// thread_handle and uses RangePolicy(th, ...) for thread-vector work (e.g. via
// sum_views). Uses a degenerate team range [0, 1) so each team thread runs the
// closure once with its handle (TeamThreadRange dispatch).
void self_similar_range_policy_sum_views_case8() {
  const size_t N         = 16;
  const size_t num_teams = 4;

  Kokkos::View<float*> v_x("v_x", N), v_y("v_y", N);
  Kokkos::View<float**> M_x("M_x", num_teams, N),
      M_add2("M_add2", num_teams, N), M_add4("M_add4", num_teams, N);

  Kokkos::parallel_for(
      "init_v", Kokkos::RangePolicy<>(0, N), KOKKOS_LAMBDA(const size_t i) {
        v_x(i) = 0.f;
        v_y(i) = 1.f;
      });
  Kokkos::parallel_for(
      "init_M", Kokkos::RangePolicy<>(0, num_teams),
      KOKKOS_LAMBDA(const size_t i) {
        for (size_t j = 0; j < N; j++) {
          M_x(i, j)    = 0.f;
          M_add2(i, j) = 2.f;
          M_add4(i, j) = 4.f;
        }
      });

  sum_views(Kokkos::DefaultExecutionSpace(), v_x, v_y);

  using team_t        = typename Kokkos::TeamPolicy<>::member_type;
  using thread_handle = team_t::thread_handle;
  Kokkos::parallel_for(
      "nested_range_policy_thread_handle",
      Kokkos::TeamPolicy(num_teams, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const team_t& team) {
        auto row_x = Kokkos::subview(M_x, team.league_rank(), Kokkos::ALL());
        auto row_add2 =
            Kokkos::subview(M_add2, team.league_rank(), Kokkos::ALL());
        sum_views(team, row_x, row_add2);

        auto row_add4 =
            Kokkos::subview(M_add4, team.league_rank(), Kokkos::ALL());
        Kokkos::parallel_for(
            Kokkos::RangePolicy(team, 0, 1),
            [&](const thread_handle& th) { sum_views(th, row_x, row_add4); });
      });

  size_t result = 0;
  Kokkos::parallel_reduce(
      "check_v_case8", N,
      KOKKOS_LAMBDA(size_t i, size_t & s) { s += static_cast<size_t>(v_x(i)); },
      result);
  ASSERT_EQ(result, N);

  result = 0;
  Kokkos::parallel_reduce(
      "check_M_case8", Kokkos::RangePolicy<>(0, num_teams * N),
      KOKKOS_LAMBDA(size_t i, size_t & s) {
        int row = i / N;
        int col = i % N;
        s += static_cast<size_t>(M_x(row, col));
      },
      result);
  ASSERT_EQ(result, num_teams * N * 6);
}

void self_similar_range_policy_case7() {
  using team_t = typename Kokkos::TeamPolicy<>::member_type;
  int nerrs    = 0;
  Kokkos::parallel_reduce(
      "check_concurrency", Kokkos::TeamPolicy(1, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const team_t& team, int& errs) {
        int team_conc     = team.concurrency();
        int expected_team = team.team_size() * team.vector_length();
        if (team_conc != expected_team) ++errs;

        auto thread_handle = Kokkos::ThreadHandle<team_t>(team);
        if (thread_handle.concurrency() != team.vector_length()) ++errs;
      },
      nerrs);
  ASSERT_EQ(nerrs, 0);
}

TEST(TEST_CATEGORY, check_runtime_values) {
  // Check runtime values for RangePolicy constructed from ExecSpace,
  // TeamHandle (TeamVectorRange semantics), and ThreadHandle (ThreadVectorRange
  // semantics).
  check_runtime_values();
}

TEST(TEST_CATEGORY, self_similar_range_policy_sum_views_case1) {
  // Case 1: exec -> team. The same sum_views template is called at both levels.
  self_similar_range_policy_sum_views_case1();
}

TEST(TEST_CATEGORY, self_similar_range_policy_sum_views_case2) {
  // Case 2: exec -> team (RangePolicy(team, ...) / TeamVectorRange) ->
  // TeamThreadRange(team, 1) with closure(thread_handle, i); sum_views(th, ...)
  // uses RangePolicy(th, ...) (ThreadVectorRange).
  self_similar_range_policy_sum_views_case2();
}

TEST(TEST_CATEGORY, self_similar_range_policy_sum_views_case3) {
  // Case 3: same nesting as case 2, but TeamThreadRange uses an index-only
  // closure and constructs ThreadHandle(team) inside the lambda.
  self_similar_range_policy_sum_views_case3();
}

TEST(TEST_CATEGORY, self_similar_range_policy_sum_views_case4) {
  // Case 4: TeamPolicy outer, then single(PerTeam) calls
  // sum_views(ThreadHandle(team), ...) which uses RangePolicy(th, ...) inside.
  self_similar_range_policy_sum_views_case4();
}

TEST(TEST_CATEGORY, self_similar_range_policy_sum_views_case5) {
  // Case 5: TeamThreadRange(team, 1) with closure(thread_handle) (no index).
  self_similar_range_policy_sum_views_case5();
}

TEST(TEST_CATEGORY, self_similar_range_policy_sum_views_case6) {
  // Case 6: RangePolicy(team, ...) maps to TeamVectorRange and must invoke
  // closure(i) only (abort if closure(thread_handle, ...) is selected).
  self_similar_range_policy_sum_views_case6();
}

TEST(TEST_CATEGORY, self_similar_range_policy_case7) {
  // Case 7: handle concurrency queries. TeamHandle concurrency is
  // team_size*vector_length; ThreadHandle concurrency is vector_length.
  self_similar_range_policy_case7();
}

TEST(TEST_CATEGORY, self_similar_range_policy_sum_views_case8) {
  // Case 8: nested parallel_for(RangePolicy(team, 0, 1),
  // closure(thread_handle)) then sum_views(th, ...) using RangePolicy(th, ...)
  // (ThreadVectorRange).
  self_similar_range_policy_sum_views_case8();
}

}  // namespace Test
