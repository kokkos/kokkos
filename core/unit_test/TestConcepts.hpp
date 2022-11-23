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
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <Kokkos_Core.hpp>

namespace TestConcept {

using ExecutionSpace = TEST_EXECSPACE;
using MemorySpace    = typename ExecutionSpace::memory_space;
using DeviceType     = typename ExecutionSpace::device_type;

static_assert(Kokkos::is_execution_space<ExecutionSpace>{}, "");
static_assert(Kokkos::is_execution_space<ExecutionSpace const>{}, "");
static_assert(!Kokkos::is_execution_space<ExecutionSpace &>{}, "");
static_assert(!Kokkos::is_execution_space<ExecutionSpace const &>{}, "");

static_assert(Kokkos::is_memory_space<MemorySpace>{}, "");
static_assert(Kokkos::is_memory_space<MemorySpace const>{}, "");
static_assert(!Kokkos::is_memory_space<MemorySpace &>{}, "");
static_assert(!Kokkos::is_memory_space<MemorySpace const &>{}, "");

static_assert(Kokkos::is_device<DeviceType>{}, "");
static_assert(Kokkos::is_device<DeviceType const>{}, "");
static_assert(!Kokkos::is_device<DeviceType &>{}, "");
static_assert(!Kokkos::is_device<DeviceType const &>{}, "");

static_assert(!Kokkos::is_device<ExecutionSpace>{}, "");
static_assert(!Kokkos::is_device<MemorySpace>{}, "");

static_assert(Kokkos::is_space<ExecutionSpace>{}, "");
static_assert(Kokkos::is_space<MemorySpace>{}, "");
static_assert(Kokkos::is_space<DeviceType>{}, "");
static_assert(Kokkos::is_space<ExecutionSpace const>{}, "");
static_assert(Kokkos::is_space<MemorySpace const>{}, "");
static_assert(Kokkos::is_space<DeviceType const>{}, "");
static_assert(!Kokkos::is_space<ExecutionSpace &>{}, "");
static_assert(!Kokkos::is_space<MemorySpace &>{}, "");
static_assert(!Kokkos::is_space<DeviceType &>{}, "");

static_assert(Kokkos::is_execution_space_v<ExecutionSpace>, "");
static_assert(!Kokkos::is_execution_space_v<ExecutionSpace &>, "");

static_assert(
    std::is_same<float, Kokkos::Impl::remove_cvref_t<float const &>>{}, "");
static_assert(std::is_same<int, Kokkos::Impl::remove_cvref_t<int &>>{}, "");
static_assert(std::is_same<int, Kokkos::Impl::remove_cvref_t<int const>>{}, "");
static_assert(std::is_same<float, Kokkos::Impl::remove_cvref_t<float>>{}, "");

}  // namespace TestConcept
