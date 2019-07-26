/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_HIP_PARALLEL_HPP
#define KOKKOS_HIP_PARALLEL_HPP

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_HIP

#include <HIP/Kokkos_HIP_Parallel_MDRange.hpp>
#include <HIP/Kokkos_HIP_Parallel_Range.hpp>
#include <Kokkos_Parallel.hpp>

namespace Kokkos {
namespace Impl {
template <class... Properties>
class TeamPolicyInternal<Kokkos::Experimental::HIP, Properties...>
    : public PolicyTraits<Properties...> {
 public:
  //! Tag this class as a kokkos execution policy
  using execution_policy = TeamPolicyInternal;

  using traits = PolicyTraits<Properties...>;

  template <class ExecSpace, class... OtherProperties>
  friend class TeamPolicyInternal;

  TeamPolicyInternal(int league_size, int team_size_request,
                     int vector_length_request = 1);

  TeamPolicyInternal(int league_size, Kokkos::AUTO_t const &team_size_request,
                     int vector_length_request = 1);

  /** \brief set per team scratch size for a specific level of the scratch
   * hierarchy */
  TeamPolicyInternal &set_scratch_size(int const level,
                                       PerTeamValue const &per_team);

  /** \brief set per thread scratch size for a specific level of the scratch
   * hierarchy */
  TeamPolicyInternal &set_scratch_size(int const level,
                                       PerThreadValue const &per_thread);

  /** \brief set per thread and per team scratch size for a specific level of
   * the scratch hierarchy */
  TeamPolicyInternal &set_scratch_size(const int level,
                                       const PerTeamValue &per_team,
                                       const PerThreadValue &per_thread);

  /** \brief set chunk_size to a discrete value */
  TeamPolicyInternal &set_chunk_size(typename traits::index_type chunk_size);

  int league_size() const { return m_league_size; }

  int team_size() const { return m_team_size; }

  int scratch_size(int level, int team_size = -1) const;

  int chunk_size() const { return m_chunk_size; }

 private:
  typename traits::execution_space m_space;
  int m_league_size;
  int m_team_size;
  int m_team_scratch_size[2];
  int m_thread_scratch_size[2];
  int m_chunk_size;
};

template <class... Properties>
TeamPolicyInternal<Kokkos::Experimental::HIP,
                   Properties...>::TeamPolicyInternal(int league_size,
                                                      int team_size_request,
                                                      int vector_length_request)
    : m_space(typename traits::execution_space()),
      m_league_size(league_size),
      m_team_size(team_size_request),
      m_team_scratch_size{0, 0},
      m_thread_scratch_size{0, 0},
      m_chunk_size(32) {
  // FIXME add a check that the league size is permissible
  // FIXME add a check that the block size is permissible
}

template <class... Properties>
TeamPolicyInternal<Kokkos::Experimental::HIP,
                   Properties...>::TeamPolicyInternal(int league_size,
                                                      Kokkos::AUTO_t const &,
                                                      int vector_length_request)
    : m_space(typename traits::execution_space()),
      m_league_size(league_size),
      m_team_size(-1),
      m_team_scratch_size{0, 0},
      m_thread_scratch_size{0, 0},
      m_chunk_size(32) {
  // FIXME add a check that the league size is permissible
}

template <class... Properties>
inline TeamPolicyInternal<Kokkos::Experimental::HIP, Properties...> &
TeamPolicyInternal<Kokkos::Experimental::HIP, Properties...>::set_scratch_size(
    int const level, PerTeamValue const &per_team) {
  // FIXME add a check that per_time.value is permissible

  m_team_scratch_size[level] = per_team.value;

  return *this;
}

template <class... Properties>
inline TeamPolicyInternal<Kokkos::Experimental::HIP, Properties...> &
TeamPolicyInternal<Kokkos::Experimental::HIP, Properties...>::set_scratch_size(
    int const level, PerThreadValue const &per_thread) {
  // FIXME add a check that per_thread.value is permissible

  m_thread_scratch_size[level] = per_thread.value;

  return *this;
}

template <class... Properties>
inline TeamPolicyInternal<Kokkos::Experimental::HIP, Properties...> &
TeamPolicyInternal<Kokkos::Experimental::HIP, Properties...>::set_scratch_size(
    const int level, const PerTeamValue &per_team,
    const PerThreadValue &per_thread) {
  // FIXME add a check that per_team.value is permissible
  // FIXME add a check that per_thread.value is permissible

  m_team_scratch_size[level]   = per_team.value;
  m_thread_scratch_size[level] = per_thread.value;

  return *this;
}

template <class... Properties>
inline TeamPolicyInternal<Kokkos::Experimental::HIP, Properties...> &
TeamPolicyInternal<Kokkos::Experimental::HIP, Properties...>::set_chunk_size(
    typename traits::index_type chunk_size) {
  // FIXME add a check that chunk_size is permissible

  m_chunk_size = chunk_size;

  return *this;
}

template <class... Properties>
inline int
TeamPolicyInternal<Kokkos::Experimental::HIP, Properties...>::scratch_size(
    int level, int team_size) const {
  if (team_size < 0) team_size = m_team_size;

  return m_team_scratch_size[level] + team_size * m_thread_scratch_size[level];
}

}  // namespace Impl
}  // namespace Kokkos

#endif
#endif
