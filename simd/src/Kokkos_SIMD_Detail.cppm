// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

module;

#include <Kokkos_SIMD.hpp>
#include <Kokkos_SIMD_Modules_Macros.hpp>

// Architecture-specific SIMD intrinsic headers (e.g., <immintrin.h> and
// <arm_neon.h>) are traditionally included by the SIMD implementation
// headers. When these headers are included from multiple C++ module
// translation units, their declarations can become problematic because
// the headers are not necessarily written with C++ modules in mind,
// also see https://github.com/llvm/llvm-project/issues/195661.
//
// In particular, declarations and definitions originating from these
// headers must be consistently placed in the global module. Including
// them in the global module fragment avoids inadvertently associating
// their contents with the Kokkos SIMD module.
//
// Keep the architecture-specific intrinsic header inclusion centralized
// here so that the module interface and implementation units use a
// consistent set of declarations.
//
// Kokkos_SIMD.cppm and Kokkos_SIMD_Impl.cppm will re-export the respective
// symbols from this module.
export module kokkos.simd_detail;

export {
  KOKKOS_SIMD_MODULES_SYMBOLS
  KOKKOS_SIMD_MODULES_IMPL_SYMBOLS
}
