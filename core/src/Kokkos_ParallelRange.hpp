//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#include <Kokkos_Macros.hpp>
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif
#ifndef KOKKOS_PARALLEL_RANGE_HPP
#define KOKKOS_PARALLEL_RANGE_HPP

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

/** \brief  Self-similar interface for work over a range of an integral type.
 *
 * If called with an ExecutionSpace, this returns a RangePolicy over work range.
 * If called with a TeamHandle, this returns a TeamVectorRange over work range.
 *
 * Allows for writing code like
 *
 * template <class Exec, class X, class Y>
 * KOKKOS_INLINE_FUNCTION void sum_views(const Exec& exec, const X& x, const Y&
 * y) { Kokkos::parallel_for( Kokkos::parallel_range(exec, 0, x.extent(0)),
 *     KOKKOS_LAMBDA(const int& i) { x(i) += y(i); });
 * }
 *
 * which can be called from host:
 *
 *   sum_views(exec_space, x, y);
 *
 * or inside a TeamPolicy loop
 *
 *   Kokkos::parallel_for(TeamPolicy, KOKKOS_LAMBDA(team) {
 *     sum_views(team, x, y);
 *   });
 * }
 */

template <class ExecType, class IndexType1, class IndexType2, class... Args>
KOKKOS_INLINE_FUNCTION auto parallel_range(const ExecType& exec,
                                           const IndexType1& begin,
                                           const IndexType2& end,
                                           const Args&... args) {
  if constexpr (Kokkos::is_execution_space_v<ExecType>) {
    KOKKOS_IF_ON_DEVICE(
        Kokkos::abort("Kokkos::parallel_range() called with an execution space "
                      "should only be called from host.");)

    // Get the RangePolicy. Supress warnings about host/device
    // function calling constructor which is __host__ only since
    // we are asserting above that we are on host.
#pragma nv_diag_suppress 20011, 20013, 20014, 20015
    auto policy = Kokkos::RangePolicy(exec, begin, end, args...);
#pragma nv_diag_default 20011, 20013, 20014, 20015

    return policy;
  } else if constexpr (Kokkos::is_team_handle_v<ExecType>) {
    return Kokkos::TeamVectorRange(exec, begin, end);
  } else {
    static_assert(
        Kokkos::is_execution_space_v<ExecType> ||
            Kokkos::is_team_handle_v<ExecType>,
        "Kokkos::parallel_range() must take execution space or team handle.\n");
  }
}

}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif  // KOKKOS_PARALLEL_RANGE_HPP
