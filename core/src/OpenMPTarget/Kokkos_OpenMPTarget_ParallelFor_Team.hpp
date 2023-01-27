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

#ifndef KOKKOS_OPENMPTARGET_PARALLEL_FOR_TEAM_HPP
#define KOKKOS_OPENMPTARGET_PARALLEL_FOR_TEAM_HPP

#include <omp.h>
#include <sstream>
#include <Kokkos_Parallel.hpp>
#include <OpenMPTarget/Kokkos_OpenMPTarget_Parallel.hpp>

namespace Kokkos {

/** \brief  Inter-thread parallel_for. Executes lambda(iType i) for each
 * i=0..N-1.
 *
 * The range i=0..N-1 is mapped to all threads of the the calling thread team.
 */
template <typename iType, class Lambda>
KOKKOS_INLINE_FUNCTION void parallel_for(
    const Impl::TeamThreadRangeBoundariesStruct<
        iType, Impl::OpenMPTargetExecTeamMember>& loop_boundaries,
    const Lambda& lambda) {
#pragma omp for nowait schedule(static, 1)
  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++) lambda(i);
}

/** \brief  Intra-thread vector parallel_for. Executes lambda(iType i) for each
 * i=0..N-1.
 *
 * The range i=0..N-1 is mapped to all vector lanes of the the calling thread.
 */
template <typename iType, class Lambda>
KOKKOS_INLINE_FUNCTION void parallel_for(
    const Impl::ThreadVectorRangeBoundariesStruct<
        iType, Impl::OpenMPTargetExecTeamMember>& loop_boundaries,
    const Lambda& lambda) {
#pragma omp simd
  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++) lambda(i);
}

/** \brief  Intra-team vector parallel_for. Executes lambda(iType i) for each
 * i=0..N-1.
 *
 * The range i=0..N-1 is mapped to all vector lanes of the the calling team.
 */
template <typename iType, class Lambda>
KOKKOS_INLINE_FUNCTION void parallel_for(
    const Impl::TeamVectorRangeBoundariesStruct<
        iType, Impl::OpenMPTargetExecTeamMember>& loop_boundaries,
    const Lambda& lambda) {
#pragma omp for simd nowait schedule(static, 1)
  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++) lambda(i);
}

namespace Impl {

template <class FunctorType, class... Properties>
class ParallelFor<FunctorType, Kokkos::TeamPolicy<Properties...>,
                  Kokkos::Experimental::OpenMPTarget> {
 private:
  using Policy =
      Kokkos::Impl::TeamPolicyInternal<Kokkos::Experimental::OpenMPTarget,
                                       Properties...>;
  using WorkTag = typename Policy::work_tag;
  using Member  = typename Policy::member_type;

  const FunctorType m_functor;
  const Policy m_policy;
  const size_t m_shmem_size;

 public:
  void execute() const {
    OpenMPTargetExec::verify_is_process(
        "Kokkos::Experimental::OpenMPTarget parallel_for");
    OpenMPTargetExec::verify_initialized(
        "Kokkos::Experimental::OpenMPTarget parallel_for");
    execute_impl<WorkTag>();
  }

 private:
  template <class TagType>
  void execute_impl() const {
    OpenMPTargetExec::verify_is_process(
        "Kokkos::Experimental::OpenMPTarget parallel_for");
    OpenMPTargetExec::verify_initialized(
        "Kokkos::Experimental::OpenMPTarget parallel_for");
    const auto league_size   = m_policy.league_size();
    const auto team_size     = m_policy.team_size();
    const auto vector_length = m_policy.impl_vector_length();

    const size_t shmem_size_L0 = m_policy.scratch_size(0, team_size);
    const size_t shmem_size_L1 = m_policy.scratch_size(1, team_size);
    OpenMPTargetExec::resize_scratch(team_size, shmem_size_L0, shmem_size_L1,
                                     league_size);

    void* scratch_ptr = OpenMPTargetExec::get_scratch_ptr();
    FunctorType a_functor(m_functor);

    // FIXME_OPENMPTARGET - If the team_size is not a multiple of 32, the
    // scratch implementation does not work in the Release or RelWithDebugInfo
    // mode but works in the Debug mode.

    // Maximum active teams possible.
    int max_active_teams = OpenMPTargetExec::MAX_ACTIVE_THREADS / team_size;
    // nteams should not exceed the maximum in-flight teams possible.
    const auto nteams =
        league_size < max_active_teams ? league_size : max_active_teams;

    // If the league size is <=0, do not launch the kernel.
    if (nteams <= 0) return;

// Performing our own scheduling of teams to avoid separation of code between
// teams-distribute and parallel. Gave a 2x performance boost in test cases with
// the clang compiler. atomic_compare_exchange can be avoided since the standard
// guarantees that the number of teams specified in the `num_teams` clause is
// always less than or equal to the maximum concurrently running teams.
#pragma omp target teams num_teams(nteams) thread_limit(team_size) \
    map(to                                                         \
        : a_functor) is_device_ptr(scratch_ptr)
#pragma omp parallel
    {
      const int blockIdx = omp_get_team_num();
      const int gridDim  = omp_get_num_teams();

      // Iterate through the number of teams until league_size and assign the
      // league_id accordingly
      // Guarantee that the compilers respect the `num_teams` clause
      if (gridDim <= nteams) {
        for (int league_id = blockIdx; league_id < league_size;
             league_id += gridDim) {
          typename Policy::member_type team(
              league_id, league_size, team_size, vector_length, scratch_ptr,
              blockIdx, shmem_size_L0, shmem_size_L1);
          if constexpr (std::is_void<TagType>::value)
            m_functor(team);
          else
            m_functor(TagType(), team);
        }
      } else
        Kokkos::abort("`num_teams` clause was not respected.\n");
    }
  }

 public:
  ParallelFor(const FunctorType& arg_functor, const Policy& arg_policy)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_shmem_size(arg_policy.scratch_size(0) + arg_policy.scratch_size(1) +
                     FunctorTeamShmemSize<FunctorType>::value(
                         arg_functor, arg_policy.team_size())) {}
};

}  // namespace Impl
}  // namespace Kokkos

#endif
