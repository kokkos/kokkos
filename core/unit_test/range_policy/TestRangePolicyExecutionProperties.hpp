// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>

namespace {

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

template <class ExecSpace>
using team_member_t = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

template <class ExecSpace>
struct CheckRuntimeValues {
  void operator()() const {
    using IndexType = typename ExecSpace::size_type;

    IndexType beg        = 5;
    IndexType end        = 15;
    IndexType chunk_size = 10;

    auto p_execspace = Kokkos::RangePolicy(ExecSpace(), beg, end);
    auto nerrs_exec_space =
        check_runtime_inputs(p_execspace, beg, end, chunk_size);
    ASSERT_EQ(nerrs_exec_space, 0);

    int nerrs_team_handle;
    Kokkos::parallel_reduce(
        "check_runtime", Kokkos::TeamPolicy<ExecSpace>(1, Kokkos::AUTO()),
        KOKKOS_LAMBDA(const team_member_t<ExecSpace>& team, int& nerrs) {
          auto p_teamhandle = Kokkos::RangePolicy(team, beg, end);
          auto tvr          = Kokkos::TeamVectorRange(team, beg, end);
          nerrs = check_runtime_inputs(p_teamhandle, tvr.start, tvr.end);
        },
        nerrs_team_handle);
    ASSERT_EQ(nerrs_team_handle, 0);

    int nerrs_thread_handle;
    Kokkos::parallel_reduce(
        "check_runtime", Kokkos::TeamPolicy<ExecSpace>(1, Kokkos::AUTO()),
        KOKKOS_LAMBDA(const team_member_t<ExecSpace>& team, int& nerrs) {
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, 1),
              [&](const int, int& thread_nerrs) {
                using thread_handle =
                    typename team_member_t<ExecSpace>::thread_handle;
                auto p_threadhandle =
                    Kokkos::RangePolicy(thread_handle(team), beg, end);
                auto thvr = Kokkos::ThreadVectorRange(team, beg, end);
                thread_nerrs +=
                    check_runtime_inputs(p_threadhandle, thvr.start, thvr.end);
              },
              nerrs);
        },
        nerrs_thread_handle);
    ASSERT_EQ(nerrs_thread_handle, 0);

    int nerrs_inline_handle;
    Kokkos::parallel_reduce(
        "check_runtime_inline",
        Kokkos::TeamPolicy<ExecSpace>(1, Kokkos::AUTO()),
        KOKKOS_LAMBDA(const team_member_t<ExecSpace>& team, int& nerrs) {
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, 1),
              [&](const int, int& thread_nerrs) {
                using thread_handle =
                    typename team_member_t<ExecSpace>::thread_handle;
                using inline_handle = typename thread_handle::inline_handle;
                auto p_inline       = Kokkos::RangePolicy(
                    inline_handle(thread_handle(team)), beg, end);
                thread_nerrs += check_runtime_inputs(p_inline, beg, end);
              },
              nerrs);
        },
        nerrs_inline_handle);
    ASSERT_EQ(nerrs_inline_handle, 0);
  }
};

template <class ExecSpace>
struct CheckInvocationOrder {
  void operator()() const {
    using thread_handle = typename team_member_t<ExecSpace>::thread_handle;

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
        "check_invocation_order",
        Kokkos::TeamPolicy<ExecSpace>(1, Kokkos::AUTO()),
        KOKKOS_LAMBDA(const team_member_t<ExecSpace>& team) {
          Kokkos::parallel_for(Kokkos::RangePolicy(team, 0, 1), Closure{});
        });
  }
};
}  // namespace

TEST(TEST_CATEGORY, range_policy_check_runtime_values) {
  CheckRuntimeValues<TEST_EXECSPACE>{}();
}

TEST(TEST_CATEGORY, range_policy_check_invocation_order) {
  CheckInvocationOrder<TEST_EXECSPACE>{}();
}
