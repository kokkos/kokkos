// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_OPENACC_MDRANGE_POLICY_HPP_
#define KOKKOS_OPENACC_MDRANGE_POLICY_HPP_

#include <Kokkos_Array.hpp>
#include <KokkosExp_MDRangePolicy.hpp>
#include <OpenACC/Kokkos_OpenACCSpace.hpp>

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
template <int N,
          typename iType = Kokkos::Experimental::OpenACCSpace::index_type>
using OpenACCMDRangeBegin = Kokkos::Array<iType, N>;
template <int N,
          typename iType = Kokkos::Experimental::OpenACCSpace::index_type>
using OpenACCMDRangeEnd = Kokkos::Array<iType, N>;
template <int N,
          typename iType = Kokkos::Experimental::OpenACCSpace::index_type>
using OpenACCMDRangeTile = Kokkos::Array<iType, N>;
}  // namespace Kokkos::Experimental::Impl

#endif
