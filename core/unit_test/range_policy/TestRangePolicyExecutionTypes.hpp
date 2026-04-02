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

void test_self_similar_range_policy_runtime() {
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

void test_handle_concurrency() {
  using team_t = typename Kokkos::TeamPolicy<>::member_type;
  int nerrs    = 0;
  Kokkos::parallel_reduce(
      "check_concurrency", Kokkos::TeamPolicy(1, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const team_t& team, int& errs) {
        // TeamHandle: concurrency = team_size * vector_length
        int team_conc     = team.concurrency();
        int expected_team = team.team_size() * team.vector_length();
        if (team_conc != expected_team) ++errs;

        // ThreadHandle: concurrency = team_size
        auto thread_handle = Kokkos::ThreadHandle<team_t>(team);
        if (thread_handle.concurrency() != team.team_size()) ++errs;
      },
      nerrs);
  ASSERT_EQ(nerrs, 0);
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
void test_self_similar_sum_views_exec_and_team() {
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
// Nesting lvl3 inside lvl2: parallel_for(RangePolicy(team, ...), f) binds like
// TeamVectorRange; if f is invocable with team_t::thread_handle, we pass that
//  handle (and the index when the closure accepts it).

void test_self_similar_sum_views_nested_exec_team_thread() {
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

        // lvl3: nested parallel_for over RangePolicy(team, 0, 1) — one
        // TeamVectorRange step; functor(const thread_handle&) so sum_views runs
        // with ThreadHandle and uses RangePolicy(th, ...) inside.
        auto row_add4 =
            Kokkos::subview(M_add4, team.league_rank(), Kokkos::ALL());
        Kokkos::parallel_for(
            Kokkos::RangePolicy(team, 0, 1),
            [&](const thread_handle& th) { sum_views(th, row_x, row_add4); });
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

TEST(TEST_CATEGORY, self_similar_range_policy_runtime) {
  test_self_similar_range_policy_runtime();
}

TEST(TEST_CATEGORY, self_similar_sum_views_nested_team_thread) {
  test_self_similar_sum_views_nested_exec_team_thread();
}

TEST(TEST_CATEGORY, handle_concurrency) { test_handle_concurrency(); }

TEST(TEST_CATEGORY, self_similar_sum_views_exec_and_team) {
  test_self_similar_sum_views_exec_and_team();
}

}  // namespace Test
