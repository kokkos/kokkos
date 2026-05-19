// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_PARALLEL_RANGEPOLICYHANDLES_HPP
#define KOKKOS_PARALLEL_RANGEPOLICYHANDLES_HPP

#include <Kokkos_ExecPolicy.hpp>
#include <type_traits>

// Included at the end of Kokkos_ExecPolicy.hpp after RangePolicy and team range
// types are defined.

namespace Kokkos {

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
  using Member          = typename Kokkos::RangePolicy<Properties...>::execution_type;
  using iType           = typename Kokkos::RangePolicy<Properties...>::index_type;
  using thread_handle_t = Kokkos::ThreadHandle<Member>;

  Member const& team = policy.space();

  if constexpr (std::is_invocable_v<Closure, iType> ||
                std::is_invocable_v<Closure, iType const&>) {
    Kokkos::parallel_for(
        static_cast<Kokkos::Impl::TeamVectorRangeBoundariesStruct<iType, Member> const&>(
            policy),
        closure);
  } else if constexpr (std::is_invocable_v<Closure, thread_handle_t const&, iType> ||
                       std::is_invocable_v<Closure, thread_handle_t const&>) {
    const Kokkos::Impl::TeamThreadRangeBoundariesStruct<iType, Member> bounds(
        team, policy.begin(), policy.end());
    Kokkos::parallel_for(bounds, closure);
  } else {
    Kokkos::parallel_for(
        static_cast<Kokkos::Impl::TeamVectorRangeBoundariesStruct<iType, Member> const&>(
            policy),
        closure);
  }
}

}  // namespace Kokkos

#endif /* #define KOKKOS_PARALLEL_RANGEPOLICYHANDLES_HPP */
