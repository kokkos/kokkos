// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_NEON_SVE_BRIDGE_HPP
#define KOKKOS_NEON_SVE_BRIDGE_HPP

#include <arm_neon.h>
#include <arm_sve.h>

#include <Kokkos_Macros.hpp>

/**
 * See https://clang.llvm.org/doxygen/arm__neon__sve__bridge_8h_source.html
 * for documentation on the Neon-SVE bridge
 */

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION static uint64x2_t svget_neonq_u64(
    svuint64_t z) {
  uint64x2_t res;

  res[0] = svlastb(svptrue_pat_b64(SV_VL1), z);
  res[1] = svlastb(svptrue_pat_b64(SV_VL2), z);

  return res;
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION static int32x4_t svget_neonq_s32(
    svint32_t z) {
  int32x4_t res;

  res[0] = svlastb(svptrue_pat_b32(SV_VL1), z);
  res[1] = svlastb(svptrue_pat_b32(SV_VL2), z);
  res[2] = svlastb(svptrue_pat_b32(SV_VL3), z);
  res[3] = svlastb(svptrue_pat_b32(SV_VL4), z);

  return res;
}

#endif  // KOKKOS_NEON_SVE_BRIDGE_HPP
