// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#include <Kokkos_Macros.hpp>
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif
#ifndef KOKKOS_PARALLEL_RANGE_POLICY_HANDLES_HPP
#define KOKKOS_PARALLEL_RANGE_POLICY_HANDLES_HPP

/// \file Kokkos_Parallel_RangePolicyHandles.hpp
/// \brief `parallel_for` overloads for `RangePolicy` over team / thread
/// handles.
///
/// This header is included from `Kokkos_Core.hpp` after
/// `KokkosCore_Config_DeclareBackend.hpp` so enabled backends have already
/// declared `TeamVectorRange`, `ThreadVectorRange`, and the matching nested
/// `parallel_for` overloads. It must not be included from
/// `Kokkos_Parallel.hpp`: that file is pulled in by backend team headers before
/// those declarations exist.

#include <Kokkos_Parallel.hpp>

namespace Kokkos {

/** \brief parallel_for(RangePolicy over a team handle): same mapping as
 *  TeamVectorRange(team, begin, end); enables nested use from team kernels. */
template <class... Traits, class FunctorType>
  requires(TeamHandle<typename RangePolicy<Traits...>::execution_type>)
KOKKOS_INLINE_FUNCTION void parallel_for(RangePolicy<Traits...> const& policy,
                                         FunctorType const& functor) {
  auto const& handle = policy.space();
  Kokkos::parallel_for(
      Kokkos::TeamVectorRange(handle, policy.begin(), policy.end()), functor);
}

/** \brief parallel_for(RangePolicy over a thread handle): same mapping as
 *  ThreadVectorRange; enables nested use from team kernels. */
template <class... Traits, class FunctorType>
  requires(ThreadHandleType<typename RangePolicy<Traits...>::execution_type>)
KOKKOS_INLINE_FUNCTION void parallel_for(RangePolicy<Traits...> const& policy,
                                         FunctorType const& functor) {
  auto const& thread_handle = policy.space();
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(thread_handle.member,
                                                 policy.begin(), policy.end()),
                       functor);
}

}  // namespace Kokkos

#endif  // KOKKOS_PARALLEL_RANGE_POLICY_HANDLES_HPP
