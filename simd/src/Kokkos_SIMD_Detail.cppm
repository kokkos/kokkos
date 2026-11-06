// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

module;

#include <Kokkos_SIMD.hpp>
#include <Kokkos_SIMD_Modules_Macros.hpp>
#include <Kokkos_SIMD_Impl_Modules_Macros.hpp>

// This module exists so that kokkos.simd and kokkos.simd_impl don't both
// include Kokkos_SIMD.hpp which turns out to be problematic when including
// intrinsic headers
export module kokkos.simd_detail;

export {
  KOKKOS_SIMD_MODULES_SYMBOLS
  KOKKOS_SIMD_MODULES_IMPL_SYMBOLS
}
