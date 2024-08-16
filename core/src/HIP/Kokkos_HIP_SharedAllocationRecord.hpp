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

#ifndef KOKKOS_HIP_SHARED_ALLOCATION_RECORD_HPP
#define KOKKOS_HIP_SHARED_ALLOCATION_RECORD_HPP

#include <HIP/Kokkos_HIP_Space.hpp>
#include <impl/Kokkos_SharedAlloc.hpp>

#if defined(KOKKOS_ENABLE_IMPL_HIP_UNIFIED_MEMORY)
KOKKOS_IMPL_SHARED_ALLOCATION_SPECIALIZATION(Kokkos::HIPSpace);
#else
KOKKOS_IMPL_HOST_INACCESSIBLE_SHARED_ALLOCATION_SPECIALIZATION(
    Kokkos::HIPSpace);
#endif
KOKKOS_IMPL_SHARED_ALLOCATION_SPECIALIZATION(Kokkos::HIPHostPinnedSpace);
KOKKOS_IMPL_SHARED_ALLOCATION_SPECIALIZATION(Kokkos::HIPManagedSpace);

#endif
