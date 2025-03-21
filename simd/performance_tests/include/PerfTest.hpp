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

#ifndef KOKKOS_SIMD_PERF_TEST_HPP
#define KOKKOS_SIMD_PERF_TEST_HPP

#include "PerfTest_Host.hpp"
#include "PerfTest_Device.hpp"

inline void register_benchmarks() {
#if defined(KOKKOS_SIMD_PERFTEST_HOST) || defined(KOKKOS_SIMD_PERFTEST_HOSTFORCESERIAL)
  register_host_benchmarks();
#endif
#if defined(KOKKOS_SIMD_PERFTEST_DEVICE)
  register_device_benchmarks();
#endif
}

#endif
