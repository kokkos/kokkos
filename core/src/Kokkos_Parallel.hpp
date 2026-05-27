// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

/// \file Kokkos_Parallel.hpp
/// \brief Declaration of parallel operators

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#include <Kokkos_Macros.hpp>
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif
#ifndef KOKKOS_PARALLEL_HPP
#define KOKKOS_PARALLEL_HPP

#include <Kokkos_Parallel_For.hpp>
#include <Kokkos_Parallel_Reduce.hpp>
#include <Kokkos_Parallel_Scan.hpp>
#include <impl/Kokkos_HostThreadTeam.hpp>

#include <type_traits>

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
    Kokkos::parallel_for(bounds, closure);
  } else if constexpr (std::is_invocable_v<Closure, thread_handle_t const&,
                                           iType> ||
                       std::is_invocable_v<Closure, thread_handle_t const&>) {
    team_thread_bounds_t const bounds(team, policy.begin(), policy.end());
    Kokkos::parallel_for(bounds, closure);
  } else {
    team_vector_bounds_t const& bounds =
        static_cast<team_vector_bounds_t const&>(policy);
    Kokkos::parallel_for(bounds, closure);
  }
}

}  // namespace Kokkos

#endif /* KOKKOS_PARALLEL_HPP */
