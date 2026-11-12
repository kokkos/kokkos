// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

#ifdef KOKKOS_HAS_SHARED_SPACE
static_assert(Kokkos::SpaceAccessibility<Kokkos::DefaultHostExecutionSpace,
                                         Kokkos::SharedSpace>::accessible);
static_assert(Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace,
                                         Kokkos::SharedSpace>::accessible);
#endif

#ifdef KOKKOS_HAS_SHARED_HOST_PINNED_SPACE
static_assert(
    Kokkos::SpaceAccessibility<Kokkos::DefaultHostExecutionSpace,
                               Kokkos::SharedHostPinnedSpace>::accessible);
static_assert(
    Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace,
                               Kokkos::SharedHostPinnedSpace>::accessible);
#endif
