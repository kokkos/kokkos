// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_PARALLEL_RANGEPOLICYHANDLES_HPP
#define KOKKOS_PARALLEL_RANGEPOLICYHANDLES_HPP

#include <Kokkos_ExecPolicy.hpp>
#include <Kokkos_Parallel_RangePolicyHandlesDispatch.hpp>
#include <impl/Kokkos_HostThreadTeam.hpp>
#include <type_traits>

// Included from Kokkos_Declare_* after backend team-range parallel_for
// overloads and KOKKOS_IMPL_PARALLEL_FOR_RANGE_POLICY_TEAM_DISPATCH
// specializations.

namespace Kokkos {
namespace Impl {

template <class iType, class HostExecSpace, class Closure>
struct ParallelForTeamVectorRangePolicyDispatch<
    iType, HostThreadTeamMember<HostExecSpace>, Closure> {
  static KOKKOS_INLINE_FUNCTION void apply(
      TeamVectorRangeBoundariesStruct<
          iType, HostThreadTeamMember<HostExecSpace>> const& bounds,
      Closure const& closure) {
    Kokkos::parallel_for(bounds, closure);
  }
};

template <class iType, class HostExecSpace, class Closure>
struct ParallelForTeamThreadRangePolicyDispatch<
    iType, HostThreadTeamMember<HostExecSpace>, Closure> {
  static KOKKOS_INLINE_FUNCTION void apply(
      TeamThreadRangeBoundariesStruct<
          iType, HostThreadTeamMember<HostExecSpace>> const& bounds,
      Closure const& closure) {
    Kokkos::parallel_for(bounds, closure);
  }
};

template <class iType, class Member, class Closure>
KOKKOS_INLINE_FUNCTION void parallel_for_team_vector_range_policy(
    TeamVectorRangeBoundariesStruct<iType, Member> const& bounds,
    Closure const& closure) {
  ParallelForTeamVectorRangePolicyDispatch<iType, Member, Closure>::apply(
      bounds, closure);
}

template <class iType, class Member, class Closure>
KOKKOS_INLINE_FUNCTION void parallel_for_team_thread_range_policy(
    TeamThreadRangeBoundariesStruct<iType, Member> const& bounds,
    Closure const& closure) {
  ParallelForTeamThreadRangePolicyDispatch<iType, Member, Closure>::apply(
      bounds, closure);
}

}  // namespace Impl

/** \brief Nested parallel_for for RangePolicy(team, ...).
 *
 * RangePolicy(team, ...) uses TeamVectorRange boundaries. When the closure is
 * callable with an index, dispatch to team-vector parallel_for (closure(i)
 * only). When the closure is callable with a thread_handle (or
 * thread_handle and index), dispatch to TeamThreadRange so the handle is
 * passed and inner RangePolicy(th, ...) can be used.
 */
template <class... Properties, class Closure>
  requires(
      Kokkos::TeamHandle<
          typename Kokkos::Impl::PolicyTraits<Properties...>::execution_type> &&
      !Kokkos::ExecutionSpace<
          typename Kokkos::Impl::PolicyTraits<Properties...>::execution_type>)
KOKKOS_INLINE_FUNCTION void parallel_for(
    Kokkos::RangePolicy<Properties...> const& policy, Closure const& closure) {
  using Member = typename Kokkos::RangePolicy<Properties...>::execution_type;
  using iType  = typename Kokkos::RangePolicy<Properties...>::index_type;
  using thread_handle_t = Kokkos::ThreadHandle<Member>;

  Member const& team = policy.space();

  using team_vector_bounds_t =
      Kokkos::Impl::TeamVectorRangeBoundariesStruct<iType, Member>;
  using team_thread_bounds_t =
      Kokkos::Impl::TeamThreadRangeBoundariesStruct<iType, Member>;

  if constexpr (std::is_invocable_v<Closure, iType> ||
                std::is_invocable_v<Closure, iType const&>) {
    team_vector_bounds_t const& bounds =
        static_cast<team_vector_bounds_t const&>(policy);
    Kokkos::Impl::parallel_for_team_vector_range_policy(bounds, closure);
  } else if constexpr (std::is_invocable_v<Closure, thread_handle_t const&,
                                           iType> ||
                       std::is_invocable_v<Closure, thread_handle_t const&>) {
    team_thread_bounds_t const bounds(team, policy.begin(), policy.end());
    Kokkos::Impl::parallel_for_team_thread_range_policy(bounds, closure);
  } else {
    team_vector_bounds_t const& bounds =
        static_cast<team_vector_bounds_t const&>(policy);
    Kokkos::Impl::parallel_for_team_vector_range_policy(bounds, closure);
  }
}

}  // namespace Kokkos

#endif /* #define KOKKOS_PARALLEL_RANGEPOLICYHANDLES_HPP */
