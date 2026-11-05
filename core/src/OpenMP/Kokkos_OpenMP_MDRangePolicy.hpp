// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

// Copyright Contributors to the Kokkos project
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef KOKKOS_OPENMP_MDRANGEPOLICY_HPP_
#define KOKKOS_OPENMP_MDRANGEPOLICY_HPP_

#include <KokkosExp_MDRangePolicy.hpp>

namespace Kokkos {
namespace Impl {

// Settings for TeamMDRangePolicy
template <typename Rank, TeamMDRangeThreadAndVector ThreadAndVector>
struct ThreadAndVectorNestLevel<Rank, OpenMP, ThreadAndVector>
    : HostBasedNestLevel<Rank, ThreadAndVector> {};

}  // namespace Impl
}  // namespace Kokkos
#endif
