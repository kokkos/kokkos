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

#ifndef KOKKOS_OPENMPTARGET_PARALLELSCAN_TEAM_HPP
#define KOKKOS_OPENMPTARGET_PARALLELSCAN_TEAM_HPP

#include <omp.h>
#include <sstream>
#include <Kokkos_Parallel.hpp>
#include <OpenMPTarget/Kokkos_OpenMPTarget_Parallel.hpp>

namespace Kokkos {

// This is largely the same code as in HIP and CUDA except for the member name
template <typename iType, class FunctorType, class ValueType>
KOKKOS_INLINE_FUNCTION void parallel_scan(
    const Impl::TeamThreadRangeBoundariesStruct<
        iType, Impl::OpenMPTargetExecTeamMember>& loop_bounds,
    const FunctorType& lambda, ValueType& return_val) {
  using Analysis = Impl::FunctorAnalysis<Impl::FunctorPatternInterface::SCAN,
                                         TeamPolicy<Experimental::OpenMPTarget>,
                                         FunctorType, void>;
  using analysis_value_type = typename Analysis::value_type;
  static_assert(std::is_same_v<analysis_value_type, ValueType>,
                "Non-matching value types of functor and return type");

  const auto start     = loop_bounds.start;
  const auto end       = loop_bounds.end;
  auto& member         = loop_bounds.member;
  const auto team_rank = member.team_rank();

  const auto team_size = member.team_size();
  const auto nchunk    = (end - start + team_size - 1) / team_size;
  ValueType accum      = {};
  // each team has to process one or
  //      more chunks of the prefix scan
  for (iType i = 0; i < nchunk; ++i) {
    auto ii = start + i * team_size + team_rank;
    // local accumulation for this chunk
    ValueType local_accum = {};
    // user updates value with prefix value
    if (ii < loop_bounds.end) lambda(ii, local_accum, false);
    // perform team scan
    local_accum = member.team_scan(local_accum);
    // add this blocks accum to total accumulation
    auto val = accum + local_accum;
    // user updates their data with total accumulation
    if (ii < loop_bounds.end) lambda(ii, val, true);
    // the last value needs to be propogated to next chunk
    if (team_rank == team_size - 1) accum = val;
    // broadcast last value to rest of the team
    member.team_broadcast(accum, team_size - 1);
  }
  return_val = accum;
}

template <typename iType, class FunctorType>
KOKKOS_INLINE_FUNCTION void parallel_scan(
    const Impl::TeamThreadRangeBoundariesStruct<
        iType, Impl::OpenMPTargetExecTeamMember>& loop_bounds,
    const FunctorType& lambda) {
  using Analysis   = Impl::FunctorAnalysis<Impl::FunctorPatternInterface::SCAN,
                                         TeamPolicy<Experimental::OpenMPTarget>,
                                         FunctorType, void>;
  using value_type = typename Analysis::value_type;
  value_type scan_val;
  parallel_scan(loop_bounds, lambda, scan_val);
}
}  // namespace Kokkos

namespace Kokkos {

/** \brief  Intra-thread vector parallel exclusive prefix sum. Executes
 * lambda(iType i, ValueType & val, bool final) for each i=0..N-1.
 *
 * The range i=0..N-1 is mapped to all vector lanes in the thread and a scan
 * operation is performed. Depending on the target execution space the operator
 * might be called twice: once with final=false and once with final=true. When
 * final==true val contains the prefix sum value. The contribution of this "i"
 * needs to be added to val no matter whether final==true or not. In a serial
 * execution (i.e. team_size==1) the operator is only called once with
 * final==true. Scan_val will be set to the final sum value over all vector
 * lanes.
 */
template <typename iType, class FunctorType, class ValueType>
KOKKOS_INLINE_FUNCTION void parallel_scan(
    const Impl::ThreadVectorRangeBoundariesStruct<
        iType, Impl::OpenMPTargetExecTeamMember>& loop_boundaries,
    const FunctorType& lambda, ValueType& return_val) {
  using Analysis = Impl::FunctorAnalysis<Impl::FunctorPatternInterface::SCAN,
                                         TeamPolicy<Experimental::OpenMPTarget>,
                                         FunctorType, void>;
  using analysis_value_type = typename Analysis::value_type;
  static_assert(std::is_same_v<analysis_value_type, ValueType>,
                "Non-matching value types of functor and return type");

  ValueType scan_val = {};

#ifdef KOKKOS_ENABLE_PRAGMA_IVDEP
#pragma ivdep
#endif
  for (iType i = loop_boundaries.start; i < loop_boundaries.end; ++i) {
    lambda(i, scan_val, true);
  }

  return_val = scan_val;
}

template <typename iType, class FunctorType>
KOKKOS_INLINE_FUNCTION void parallel_scan(
    const Impl::ThreadVectorRangeBoundariesStruct<
        iType, Impl::OpenMPTargetExecTeamMember>& loop_boundaries,
    const FunctorType& lambda) {
  using Analysis   = Impl::FunctorAnalysis<Impl::FunctorPatternInterface::SCAN,
                                         TeamPolicy<Experimental::OpenMPTarget>,
                                         FunctorType, void>;
  using value_type = typename Analysis::value_type;

  value_type scan_val = value_type();
  parallel_scan(loop_boundaries, lambda, scan_val);
}

}  // namespace Kokkos

#endif
