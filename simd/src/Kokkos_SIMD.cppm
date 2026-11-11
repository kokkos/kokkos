// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

module;

// See Kokkos_SIMD_Detail.cppm for an explanation for
// re-exporting symbols from kokkos.simd_detail.

#include <Kokkos_SIMD_Modules_Macros.hpp>
import kokkos.simd_detail;

export module kokkos.simd;

export { KOKKOS_SIMD_MODULES_SYMBOLS }
