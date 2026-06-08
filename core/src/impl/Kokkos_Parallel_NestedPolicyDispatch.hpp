// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_IMPL_PARALLEL_NESTED_POLICY_DISPATCH_HPP
#define KOKKOS_IMPL_PARALLEL_NESTED_POLICY_DISPATCH_HPP

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#include <Kokkos_Macros.hpp>
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_ExecPolicy.hpp>

#include <type_traits>

namespace Kokkos {

/** \brief Nested parallel_for for RangePolicy(team, ...).
 *
 * RangePolicy(team, ...) uses TeamVectorRange boundaries. When the closure is
 * callable with an index, dispatch to team-vector parallel_for (closure(i)
 * only). When the closure is callable with a thread_handle (or
 * thread_handle and index), dispatch to TeamThreadRange so the handle is
 * passed and inner RangePolicy(th, ...) can be used.
 *
 * This header is included from Kokkos_Core after backend team implementations
 * so Kokkos::parallel_for(TeamVectorRangeBoundariesStruct, ...) overloads
 * exist.
 *
 * Additional handle-built policies (MDRangePolicy) and patterns
 * (parallel_reduce, parallel_scan) will be added here.
 */
template <class... Properties, class Closure>
  requires Kokkos::TeamHandle<
      typename Kokkos::Impl::PolicyTraits<Properties...>::execution_type>
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
    // policy.begin()/end() are per-thread TeamVectorRange slices;
    // TeamThreadRange must receive the full range and partition once (see
    // work_begin/end on ImplRangePolicy<TeamHandle>).
    team_thread_bounds_t const bounds(team, policy.work_begin(),
                                      policy.work_end());
    Kokkos::parallel_for(bounds, closure);
  } else {
    static_assert(Kokkos::Impl::always_false<Closure>::value,
                  "Kokkos::parallel_for(RangePolicy): closure must be "
                  "invocable with (iType), (ThreadHandle, iType), or "
                  "(ThreadHandle)");
  }
}

}  // namespace Kokkos

#endif /* KOKKOS_IMPL_PARALLEL_NESTED_POLICY_DISPATCH_HPP */
