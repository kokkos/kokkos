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

#ifndef KOKKOS_STD_ALGORITHMS_MUSTUSEKOKKOSSINGLEINTEAM_HPP
#define KOKKOS_STD_ALGORITHMS_MUSTUSEKOKKOSSINGLEINTEAM_HPP

#include <Kokkos_Core.hpp>

namespace Kokkos {
namespace Experimental {
namespace Impl {

template <typename T>
struct stdalgo_team_level_must_use_kokkos_single : std::false_type {};

#if defined(KOKKOS_ENABLE_OPENACC)
template <>
struct stdalgo_team_level_must_use_kokkos_single<Kokkos::Experimental::OpenACC>
  : std::true_type{};
#endif

#if defined(KOKKOS_ENABLE_OPENMPTARGET)
template <>
struct stdalgo_team_level_must_use_kokkos_single<Kokkos::Experimental::OpenMPTarget>
  : std::true_type{};
#endif

#if defined(KOKKOS_ENABLE_HPX)
template <>
struct stdalgo_team_level_must_use_kokkos_single<Kokkos::Experimental::HPX>
  : std::true_type{};
#endif

#if defined(KOKKOS_ENABLE_THREADS)
template <>
struct stdalgo_team_level_must_use_kokkos_single<Kokkos::Threads>
  : std::true_type{};
#endif


// template <class TeamHandleType>
// KOKKOS_INLINE_FUNCTION
// constexpr bool team_level_must_use_kokkos_single(const TeamHandleType& /* th*/)
// {
//   return stdalgo_team_level_must_use_kokkos_single<typename TeamHandleType::execution_space>::value;
// }


}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#endif
