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

#ifndef KOKKOS_DECLARE_SYCL_HPP
#define KOKKOS_DECLARE_SYCL_HPP

#if defined(KOKKOS_ENABLE_SYCL)
#include <Kokkos_SYCL.hpp>
#include <SYCL/Kokkos_SYCL_Half_Impl_Type.hpp>
#include <SYCL/Kokkos_SYCL_Half_Conversion.hpp>
#include <SYCL/Kokkos_SYCL_DeepCopy.hpp>
#include <SYCL/Kokkos_SYCL_MDRangePolicy.hpp>
#include <SYCL/Kokkos_SYCL_Parallel_Range.hpp>
#include <SYCL/Kokkos_SYCL_Parallel_Reduce.hpp>
#include <SYCL/Kokkos_SYCL_Parallel_Scan.hpp>
#include <SYCL/Kokkos_SYCL_Parallel_Team.hpp>
#include <SYCL/Kokkos_SYCL_UniqueToken.hpp>
#endif

#endif
