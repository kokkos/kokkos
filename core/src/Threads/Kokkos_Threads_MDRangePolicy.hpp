// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_THREADS_MDRANGEPOLICY_HPP_
#define KOKKOS_THREADS_MDRANGEPOLICY_HPP_

#include <KokkosExp_MDRangePolicy.hpp>

namespace Kokkos {
namespace Impl {

template <>
struct TileSizeRecommended<Kokkos::Threads> {
  template <typename Policy>
  static auto get(Policy const& policy) {
    if constexpr (Policy::rank == 1) {
      using range_policy = typename Policy::impl_range_policy;
      typename Policy::tile_type tile{};
      range_policy range_policy_1d(policy.space(), policy.m_lower[0],
                                   policy.m_upper[0]);
      tile[0] = range_policy_1d.chunk_size();
      return tile;
    } else {
      return get_default_tile_size_recommended(policy);
    }
  }
};

// Settings for TeamMDRangePolicy
template <typename Rank, TeamMDRangeThreadAndVector ThreadAndVector>
struct ThreadAndVectorNestLevel<Rank, Threads, ThreadAndVector>
    : HostBasedNestLevel<Rank, ThreadAndVector> {};

}  // namespace Impl
}  // namespace Kokkos
#endif
