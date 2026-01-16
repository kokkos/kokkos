// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#define KOKKOS_IMPL_PUBLIC_INCLUDE

#include <NextSilicon/Kokkos_NextSilicon_SharedAllocationRecord.hpp>
#include <NextSilicon/Kokkos_NextSilicon_DeepCopy.hpp>
#include <impl/Kokkos_SharedAlloc_timpl.hpp>

KOKKOS_IMPL_SHARED_ALLOCATION_RECORD_EXPLICIT_INSTANTIATION(
    Kokkos::Experimental::NextSiliconSpace);
KOKKOS_IMPL_SHARED_ALLOCATION_RECORD_EXPLICIT_INSTANTIATION(
    Kokkos::Experimental::NextSiliconSharedSpace);
