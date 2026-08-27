// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_DECLARE_NEXTSILICON_HPP
#define KOKKOS_DECLARE_NEXTSILICON_HPP

#if defined(KOKKOS_ENABLE_NEXTSILICON)
#include <NextSilicon/Kokkos_NextSilicon.hpp>
#include <NextSilicon/Kokkos_NextSiliconSpace.hpp>
#include <NextSilicon/Kokkos_NextSilicon_DeepCopy.hpp>
#include <NextSilicon/Kokkos_NextSilicon_ZeroMemset.hpp>
#include <NextSilicon/Kokkos_NextSilicon_SharedAllocationRecord.hpp>
#include <NextSilicon/Kokkos_NextSilicon_MDRangePolicy.hpp>
#include <NextSilicon/Kokkos_NextSilicon_Parallel_MDRange.hpp>
#include <NextSilicon/Kokkos_NextSilicon_ParallelFor_Range.hpp>
#include <NextSilicon/Kokkos_NextSilicon_ParallelFor_Team.hpp>
#include <NextSilicon/Kokkos_NextSilicon_ParallelReduce_Range.hpp>
#include <NextSilicon/Kokkos_NextSilicon_ParallelReduce_Team.hpp>
#include <NextSilicon/Kokkos_NextSilicon_ParallelScan_Range.hpp>
#include <NextSilicon/Kokkos_NextSilicon_ParallelScan_Team.hpp>
#endif

#endif
