// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

module;

#include <Kokkos_SIMD.hpp>

export module kokkos.simd;

export {
  namespace Kokkos {
  using ::Kokkos::abs;
  using ::Kokkos::acos;
  using ::Kokkos::acosh;
  using ::Kokkos::asin;
  using ::Kokkos::asinh;
  using ::Kokkos::atan;
  using ::Kokkos::atan2;
  using ::Kokkos::atanh;
  using ::Kokkos::cbrt;
  using ::Kokkos::ceil;
  using ::Kokkos::copysign;
  using ::Kokkos::cos;
  using ::Kokkos::cosh;
  using ::Kokkos::erf;
  using ::Kokkos::erfc;
  using ::Kokkos::exp;
  using ::Kokkos::exp2;
  using ::Kokkos::floor;
  using ::Kokkos::fma;
  using ::Kokkos::hypot;
  using ::Kokkos::lgamma;
  using ::Kokkos::log;
  using ::Kokkos::log10;
  using ::Kokkos::log2;
  using ::Kokkos::pow;
  using ::Kokkos::round;
  using ::Kokkos::sin;
  using ::Kokkos::sinh;
  using ::Kokkos::sqrt;
  using ::Kokkos::tan;
  using ::Kokkos::tanh;
  using ::Kokkos::tgamma;
  using ::Kokkos::trunc;

  using ::Kokkos::max;
  using ::Kokkos::min;
  }  // namespace Kokkos

  namespace Kokkos::Experimental {
  using ::Kokkos::Experimental::all_of;
  using ::Kokkos::Experimental::any_of;
  using ::Kokkos::Experimental::basic_simd;
  using ::Kokkos::Experimental::basic_simd_mask;
  using ::Kokkos::Experimental::condition;
  using ::Kokkos::Experimental::none_of;
  using ::Kokkos::Experimental::partial_gather_from;
  using ::Kokkos::Experimental::partial_scatter_to;
  using ::Kokkos::Experimental::reduce;
  using ::Kokkos::Experimental::reduce_max;
  using ::Kokkos::Experimental::reduce_min;
  using ::Kokkos::Experimental::round_half_to_nearest_even;
  using ::Kokkos::Experimental::simd;
  using ::Kokkos::Experimental::simd_flag_aligned;
  using ::Kokkos::Experimental::simd_flag_default;
  using ::Kokkos::Experimental::simd_mask;
  using ::Kokkos::Experimental::simd_partial_load;
  using ::Kokkos::Experimental::simd_partial_store;
  using ::Kokkos::Experimental::simd_unchecked_load;
  using ::Kokkos::Experimental::simd_unchecked_store;
  using ::Kokkos::Experimental::unchecked_gather_from;
  using ::Kokkos::Experimental::unchecked_scatter_to;

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || \
    defined(KOKKOS_ENABLE_EXPERIMENTAL_SIMD_AMX)
  using ::Kokkos::Experimental::accumulator;
  using ::Kokkos::Experimental::accumulator_extents;
  using ::Kokkos::Experimental::fill_fragment;
  using ::Kokkos::Experimental::fragment;
  using ::Kokkos::Experimental::FragmentDType;
  using ::Kokkos::Experimental::FragmentUse;
  using ::Kokkos::Experimental::load_matrix_sync;
  using ::Kokkos::Experimental::matrix_a;
  using ::Kokkos::Experimental::matrix_a_extents;
  using ::Kokkos::Experimental::matrix_b;
  using ::Kokkos::Experimental::matrix_b_extents;
  using ::Kokkos::Experimental::mma_policy;
  using ::Kokkos::Experimental::mma_shape;
  using ::Kokkos::Experimental::mma_sync;
  using ::Kokkos::Experimental::PrecisionType;
  using ::Kokkos::Experimental::store_matrix_sync;
#endif

  using ::Kokkos::Experimental::operator+=;
  using ::Kokkos::Experimental::operator*=;
  using ::Kokkos::Experimental::operator-=;
  using ::Kokkos::Experimental::operator/=;
  using ::Kokkos::Experimental::operator+;
  using ::Kokkos::Experimental::operator*;
  using ::Kokkos::Experimental::operator-;
  using ::Kokkos::Experimental::operator/;
  using ::Kokkos::Experimental::operator&=;
  using ::Kokkos::Experimental::operator|=;
  using ::Kokkos::Experimental::operator^=;
  using ::Kokkos::Experimental::operator>>=;
  using ::Kokkos::Experimental::operator<<=;

  namespace simd_abi {
#if defined(KOKKOS_ARCH_AVX2)
  using ::Kokkos::Experimental::simd_abi::avx2_fixed_size;
#endif
#if defined(KOKKOS_ARCH_AVX512XEON)
  using ::Kokkos::Experimental::simd_abi::avx512_fixed_size;
#endif
#if defined(KOKKOS_ARCH_ARM_NEON)
  using ::Kokkos::Experimental::simd_abi::neon_fixed_size;
#endif
  using ::Kokkos::Experimental::simd_abi::scalar;
#if defined(KOKKOS_ARCH_ARM_SVE)
  using ::Kokkos::Experimental::simd_abi::sve_fixed_size;
#endif
  }  // namespace simd_abi
  }  // namespace Kokkos::Experimental
}
