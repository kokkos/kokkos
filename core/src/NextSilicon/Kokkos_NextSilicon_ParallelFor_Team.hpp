// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

/*! \brief Team Parallel constructs for NextSilicon.
FIXME_NEXTSILICON: Currently implemented by using one range-policy iteration to
execute each team, AND team size = 1 and vector size = 1. The hierarchical
connstructs are written to support variable-sized teams and vectors that will be
mapped onto a 1D iteration space, to serve as a signpost for future
implementation work.

Assuming for a requested team size M and vector size N, at least M * N
underlying "threads" will be launched.

The team rank of a hardware thread is T / N
The vector rank of a hardware thread is T % N
*/

#ifndef KOKKOS_NEXTSILICON_PARALLEL_FOR_TEAM_HPP
#define KOKKOS_NEXTSILICON_PARALLEL_FOR_TEAM_HPP

#include <Kokkos_Parallel.hpp>
#include <NextSilicon/Kokkos_NextSilicon.hpp>
#include <NextSilicon/Kokkos_NextSilicon_Team.hpp>
#include <NextSilicon/Kokkos_NextSilicon_Instance.hpp>
#include <mutex>

namespace Kokkos::Experimental::Impl {
template <typename Member, typename LeagueIndexType, typename Functor>
class NextSiliconParallelForTeamPolicyFunctor {
 public:
  NextSiliconParallelForTeamPolicyFunctor(Functor const& functor,
                                          const LeagueIndexType league_size,
                                          std::byte* league_scratch_buffer,
                                          const size_t L0_size,
                                          const size_t L1_size)
      : functor_(functor),
        m_league_size(league_size),
        m_league_scratch_buffer(league_scratch_buffer),
        m_L0_size(L0_size),
        m_L1_size(L1_size) {}

  KOKKOS_INLINE_FUNCTION void operator()(
      const LeagueIndexType league_rank) const {
    functor_(league_rank_to_member(league_rank));
  }

  template <typename Tag>
  KOKKOS_INLINE_FUNCTION void operator()(
      Tag, const LeagueIndexType league_rank) const {
    functor_(Tag{}, league_rank_to_member(league_rank));
  }

 private:
  KOKKOS_INLINE_FUNCTION Member
  league_rank_to_member(const LeagueIndexType league_rank) const {
    std::byte* team_scratch_buffer =
        m_league_scratch_buffer + league_rank * (m_L0_size + m_L1_size);
    using scratch_memory_space = typename Member::scratch_memory_space;
    return Member(
        league_rank, m_league_size,
        scratch_memory_space(team_scratch_buffer, m_L0_size,
                             team_scratch_buffer + m_L0_size, m_L1_size));
  }

  const Functor functor_;
  LeagueIndexType m_league_size;
  std::byte* m_league_scratch_buffer;
  size_t m_L0_size;
  size_t m_L1_size;
};
}  // namespace Kokkos::Experimental::Impl

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
// Hierarchical Parallelism -> Team level implementation
template <class FunctorType, class... Properties>
class Kokkos::Impl::ParallelFor<FunctorType, Kokkos::TeamPolicy<Properties...>,
                                Kokkos::Experimental::NextSilicon> {
 private:
  using Policy =
      Kokkos::Impl::TeamPolicyInternal<Kokkos::Experimental::NextSilicon,
                                       Properties...>;
  using Member = typename Policy::member_type;

  FunctorType m_functor;
  const Policy m_policy;

 public:
  // Invoke one RangePolicy parallel_for iteration per team
  inline void execute() const {
    // Acquire the device for potential handoff before kernel execution begins
    const std::lock_guard<std::recursive_mutex> device_lock =
        this->m_policy.space().impl_internal_space_instance()->lock_device();

    // FIXME_NEXTSILICON: when hierarchical parallel_for available in NextAPI,
    // use that here instead

    const auto league_size = m_policy.league_size();

    const auto L0_size =
        m_policy.scratch_size(0, 1 /*team size */) +
        FunctorTeamShmemSize<FunctorType>::value(m_functor, 1 /*team size */);

    const auto L1_size = m_policy.scratch_size(1, 1 /*team size */);

    // Make sure there's a scratch allocation big enough for all our teams
    // FIXME_NEXTSILICON: Move L0 to local allocation. L1 can stay in global.
    auto internal_instance = m_policy.space().impl_internal_space_instance();
    std::byte* league_scratch_buffer =
        internal_instance->resize_league_scratch_buffer(league_size *
                                                        (L0_size + L1_size));

    Experimental::Impl::NextSiliconParallelForTeamPolicyFunctor<
        Member, decltype(league_size), FunctorType>
        op(m_functor, league_size, league_scratch_buffer, L0_size, L1_size);
    Kokkos::parallel_for(
        RangePolicy<Properties...>(m_policy.space(), 0, league_size), op);
  }

  inline ParallelFor(const FunctorType& arg_functor, Policy arg_policy)
      : m_functor(arg_functor), m_policy(arg_policy) {}
};

namespace Kokkos {

// Hierarchical Parallelism -> Team thread level implementation
template <typename iType, class Lambda>
KOKKOS_INLINE_FUNCTION void parallel_for(
    const Impl::TeamThreadRangeBoundariesStruct<
        iType, Impl::NextSiliconTeamMember>& loop_boundaries,
    const Lambda& lambda) {
  iType j_start = loop_boundaries.start;
  iType j_end   = loop_boundaries.end;
  for (iType j = j_start; j < j_end; ++j /* team size 1 */) {
    lambda(j);
  }
}

// Hierarchical Parallelism -> Thread vector level implementation
template <typename iType, class Lambda>
KOKKOS_INLINE_FUNCTION void parallel_for(
    const Impl::ThreadVectorRangeBoundariesStruct<
        iType, Impl::NextSiliconTeamMember>& loop_boundaries,
    const Lambda& lambda) {
  iType j_start = loop_boundaries.start;
  iType j_end   = loop_boundaries.end;
  for (iType j = j_start; j < j_end; ++j /* vector size 1 */) {
    lambda(j);
  }
}

// Hierarchical Parallelism -> Team vector level implementation
template <typename iType, class Lambda>
KOKKOS_INLINE_FUNCTION void parallel_for(
    const Impl::TeamVectorRangeBoundariesStruct<
        iType, Impl::NextSiliconTeamMember>& loop_boundaries,
    const Lambda& lambda) {
  iType j_start = loop_boundaries.start;
  iType j_end   = loop_boundaries.end;  // team size 1: full range per thread
  for (iType j = j_start; j < j_end; ++j) {
    lambda(j);
  }
}

}  // namespace Kokkos

#endif /* #ifndef KOKKOS_NEXTSILICON_PARALLEL_FOR_TEAM_HPP */
