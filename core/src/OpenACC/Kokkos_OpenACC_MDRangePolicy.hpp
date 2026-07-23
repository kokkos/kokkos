// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_OPENACC_MDRANGE_POLICY_HPP_
#define KOKKOS_OPENACC_MDRANGE_POLICY_HPP_

#include <KokkosExp_MDRangePolicy.hpp>

template <>
struct Kokkos::default_outer_direction<Kokkos::Experimental::OpenACC> {
  using type                     = Iterate;
  static constexpr Iterate value = Iterate::Left;
};

template <>
struct Kokkos::default_inner_direction<Kokkos::Experimental::OpenACC> {
  using type                     = Iterate;
  static constexpr Iterate value = Iterate::Left;
};

namespace Kokkos {
namespace Impl {

template <typename Rank, TeamMDRangeThreadAndVector ThreadAndVector>
struct ThreadAndVectorNestLevel<Rank, Kokkos::Experimental::OpenACC,
                                ThreadAndVector>
    : AcceleratorBasedNestLevel<Rank, ThreadAndVector> {};

}  // namespace Impl
}  // namespace Kokkos

namespace Kokkos::Experimental::Impl {

struct OpenACCCollapse {};
struct OpenACCTile {};
using OpenACCIterateLeft  = std::integral_constant<Iterate, Iterate::Left>;
using OpenACCIterateRight = std::integral_constant<Iterate, Iterate::Right>;
template <int N>
using OpenACCMDRangeBegin = MDRangePolicy<OpenACC, Rank<N>>::point_type;
template <int N>
using OpenACCMDRangeEnd = MDRangePolicy<OpenACC, Rank<N>>::point_type;
template <int N>
using OpenACCMDRangeTile = MDRangePolicy<OpenACC, Rank<N>>::tile_type;

}  // namespace Kokkos::Experimental::Impl

#endif
