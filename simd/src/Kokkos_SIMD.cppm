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
  using ::Kokkos::Experimental::reduce;
  using ::Kokkos::Experimental::reduce_max;
  using ::Kokkos::Experimental::reduce_min;
  using ::Kokkos::Experimental::round_half_to_nearest_even;
  using ::Kokkos::Experimental::simd;
  using ::Kokkos::Experimental::simd_flag_aligned;
  using ::Kokkos::Experimental::simd_flag_default;
  using ::Kokkos::Experimental::simd_mask;
  using ::Kokkos::Experimental::where;

  using ::Kokkos::Experimental::operator+=;
  using ::Kokkos::Experimental::operator*=;
  using ::Kokkos::Experimental::operator-=;
  using ::Kokkos::Experimental::operator/=;
  using ::Kokkos::Experimental::operator+;
  using ::Kokkos::Experimental::operator*;
  using ::Kokkos::Experimental::operator-;
  using ::Kokkos::Experimental::operator/;
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

  namespace simd_abi::Impl {  // FIXME
  using ::Kokkos::Experimental::simd_abi::Impl::native_abi;
  using ::Kokkos::Experimental::simd_abi::Impl::native_fixed_abi;
  }  // namespace simd_abi::Impl

  namespace Impl {  // FIXME
  using ::Kokkos::Experimental::Impl::abi_set;
  using ::Kokkos::Experimental::Impl::data_type_set;
  using ::Kokkos::Experimental::Impl::data_types;
  using ::Kokkos::Experimental::Impl::device_abi_set;
  using ::Kokkos::Experimental::Impl::host_abi_set;
  using ::Kokkos::Experimental::Impl::Identity;
  }  // namespace Impl
  }  // namespace Kokkos::Experimental
}
