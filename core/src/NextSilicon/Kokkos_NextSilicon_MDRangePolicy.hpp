// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_NEXTSILICON_MDRANGE_POLICY_HPP
#define KOKKOS_NEXTSILICON_MDRANGE_POLICY_HPP

#include <NextSilicon/Kokkos_NextSilicon.hpp>
#include <KokkosExp_MDRangePolicy.hpp>

namespace Kokkos {

template <>
struct default_outer_direction<Experimental::NextSilicon> {
  using type                     = Iterate;
  static constexpr Iterate value = Iterate::Left;
};

template <>
struct default_inner_direction<Experimental::NextSilicon> {
  using type                     = Iterate;
  static constexpr Iterate value = Iterate::Left;
};

namespace Impl {

// Settings for TeamMDRangePolicy
template <typename Rank, TeamMDRangeThreadAndVector ThreadAndVector>
struct ThreadAndVectorNestLevel<Rank, Experimental::NextSilicon,
                                ThreadAndVector>
    : AcceleratorBasedNestLevel<Rank, ThreadAndVector> {};

}  // Namespace Impl
}  // namespace Kokkos
#endif
