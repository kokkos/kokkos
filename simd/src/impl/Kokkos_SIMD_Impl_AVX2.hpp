// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_SIMD_IMPL_AVX2_HPP
#define KOKKOS_SIMD_IMPL_AVX2_HPP

#include <immintrin.h>
#include <Kokkos_Array.hpp>
#include <Kokkos_SIMD_Common.hpp>

namespace Kokkos::Experimental {

namespace simd_abi {

template <Impl::simd_size_t N>
class avx2_fixed_size {};

}  // namespace simd_abi

namespace Impl {

struct simd_host_tag {};
struct simd_device_tag {};

#ifdef KOKKOS_SIMD_IMPL_DEVICE_SIMD
using simd_backend_t = simd_device_tag;
#else
using simd_backend_t = simd_host_tag;
#endif

template <typename T, typename Abi, typename Tag>
struct simd_vector_impl;

template <>
struct simd_vector_impl<double, simd_abi::avx2_fixed_size<4>, simd_device_tag> {
  using type = Kokkos::Array<char, sizeof(__m256d)>;
};

template <>
struct simd_vector_impl<float, simd_abi::avx2_fixed_size<4>, simd_device_tag> {
  using type = Kokkos::Array<char, sizeof(__m128)>;
};

template <>
struct simd_vector_impl<float, simd_abi::avx2_fixed_size<8>, simd_device_tag> {
  using type = Kokkos::Array<char, sizeof(__m256)>;
};

template <>
struct simd_vector_impl<std::int32_t, simd_abi::avx2_fixed_size<4>, simd_device_tag> {
  using type = Kokkos::Array<char, sizeof(__m128i)>;
};

template <>
struct simd_vector_impl<std::int32_t, simd_abi::avx2_fixed_size<8>, simd_device_tag> {
  using type = Kokkos::Array<char, sizeof(__m256i)>;
};

template <>
struct simd_vector_impl<std::int64_t, simd_abi::avx2_fixed_size<4>, simd_device_tag> {
  using type = Kokkos::Array<char, sizeof(__m256i)>;
};

template <>
struct simd_vector_impl<std::uint64_t, simd_abi::avx2_fixed_size<4>, simd_device_tag> {
  using type = Kokkos::Array<char, sizeof(__m256i)>;
};

#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
template <>
struct simd_vector_impl<double, simd_abi::avx2_fixed_size<4>, simd_host_tag> {
  using type = __m256d;
};

template <>
struct simd_vector_impl<float, simd_abi::avx2_fixed_size<4>, simd_host_tag> {
  using type = __m128;
};

template <>
struct simd_vector_impl<float, simd_abi::avx2_fixed_size<8>, simd_host_tag> {
  using type = __m256;
};

template <>
struct simd_vector_impl<std::int32_t, simd_abi::avx2_fixed_size<4>, simd_host_tag> {
  using type = __m128i;
};

template <>
struct simd_vector_impl<std::int32_t, simd_abi::avx2_fixed_size<8>, simd_host_tag> {
  using type = __m256i;
};

template <>
struct simd_vector_impl<std::int64_t, simd_abi::avx2_fixed_size<4>, simd_host_tag> {
  using type = __m256i;
};

template <>
struct simd_vector_impl<std::uint64_t, simd_abi::avx2_fixed_size<4>, simd_host_tag> {
  using type = __m256i;
};
#endif

template <typename T, typename Abi, typename Tag>
using simd_vector_t = typename simd_vector_impl<T, Abi, Tag>::type;

template <typename T, typename Abi, typename Tag>
struct simd_mask_native_ops;

// FIXME
// device
template <>
struct simd_mask_native_ops<double, simd_abi::avx2_fixed_size<4>, simd_device_tag> {
  using vector_type = simd_vector_t<double, simd_abi::avx2_fixed_size<4>, simd_device_tag>;
  using value_type = bool;

  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type set1([[maybe_unused]] value_type value) { return vector_type{}; }

  template <typename G>
  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type gen([[maybe_unused]]G&& gen) { return vector_type{}; }

  template <typename T, typename Abi>
  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type convert_from(simd_vector_t<T, Abi, simd_device_tag>) { return vector_type{}; }

  KOKKOS_FORCEINLINE_FUNCTION
  static value_type extract(vector_type v, simd_size_t i) { return v[i]; }

  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type lnot([[maybe_unused]]vector_type v) { return vector_type{}; }

  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type bnot([[maybe_unused]]vector_type v) { return lnot(v); }

  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type land([[maybe_unused]]vector_type lhs, [[maybe_unused]] vector_type rhs) { return vector_type{}; }

  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type lor([[maybe_unused]]vector_type lhs, [[maybe_unused]] vector_type rhs) { return vector_type{}; }

  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type band([[maybe_unused]]vector_type lhs, [[maybe_unused]] vector_type rhs) { return vector_type{}; }

  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type bor([[maybe_unused]]vector_type lhs, [[maybe_unused]] vector_type rhs) { return vector_type{}; }

  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type bxor([[maybe_unused]]vector_type lhs, [[maybe_unused]] vector_type rhs) { return vector_type{}; }

  KOKKOS_FORCEINLINE_FUNCTION
  static value_type eq([[maybe_unused]]vector_type lhs, [[maybe_unused]] vector_type rhs) { return value_type{}; }

  KOKKOS_FORCEINLINE_FUNCTION
  static value_type neq(vector_type lhs, vector_type rhs) { return !eq(lhs,rhs); }
};

// host
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
template <>
struct simd_mask_native_ops<double, simd_abi::avx2_fixed_size<4>, simd_host_tag> {
  using vector_type = simd_vector_t<double, simd_abi::avx2_fixed_size<4>, simd_host_tag>;
  using value_type = bool;

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type set1(value_type value) { return _mm256_castsi256_pd(_mm256_set1_epi64x(-std::int64_t(value))); }

  template <typename G>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type gen(G&& gen) {
    return _mm256_castsi256_pd(_mm256_setr_epi64x(
            -std::int64_t(gen(std::integral_constant<simd_size_t, 0>())),
            -std::int64_t(gen(std::integral_constant<simd_size_t, 1>())),
            -std::int64_t(gen(std::integral_constant<simd_size_t, 2>())),
            -std::int64_t(gen(std::integral_constant<simd_size_t, 3>()))));
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type convert_from(simd_vector_t<float, simd_abi::avx2_fixed_size<4>, simd_host_tag> v) { return _mm256_cvtps_pd(v); }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type convert_from(simd_vector_t<std::int32_t, simd_abi::avx2_fixed_size<4>, simd_host_tag> v) { return _mm256_cvtepi32_pd(v); }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type convert_from(simd_vector_t<std::int64_t, simd_abi::avx2_fixed_size<4>, simd_host_tag> v) { return _mm256_castsi256_pd(v); }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static value_type extract(vector_type v, simd_size_t i) { return (_mm256_movemask_pd(v) & (1 << i)) != 0; }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type lnot(vector_type v) { return _mm256_andnot_pd(v, set1(true)); }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type bnot(vector_type v) { return lnot(v); }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type land(vector_type lhs, vector_type rhs) { return band(lhs, rhs); }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type lor(vector_type lhs, vector_type rhs) { return bor(lhs, rhs); }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type band(vector_type lhs, vector_type rhs) {  return _mm256_and_pd(lhs, rhs); }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type bor(vector_type lhs, vector_type rhs) {  return _mm256_or_pd(lhs, rhs); }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type bxor(vector_type lhs, vector_type rhs) { return _mm256_xor_pd(lhs, rhs); }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static value_type eq(vector_type lhs, vector_type rhs) {
    return (_mm256_movemask_pd(lhs) == _mm256_movemask_pd(rhs));
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static value_type neq(vector_type lhs, vector_type rhs) {
    return !eq(lhs, rhs);
  }
};
#endif

template <typename T, typename Abi, typename Tag>
struct simd_native_ops;

// device
// FIXME
template <>
struct simd_native_ops<double, simd_abi::avx2_fixed_size<4>, simd_device_tag> {
  using vector_type = simd_vector_t<double, simd_abi::avx2_fixed_size<4>, simd_device_tag>;
  using value_type = double;

  template <typename T>
  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type set1([[maybe_unused]] T&& value) { return vector_type{}; }

  template <typename G>
  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type gen([[maybe_unused]]G&& gen) { return vector_type{}; }

  template <typename FlagType>
  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type load([[maybe_unused]]const value_type* ptr, FlagType) { return vector_type{}; }

  template <typename FlagType>
  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type masked_load([[maybe_unused]]const value_type* ptr, [[maybe_unused]]vector_type mask, FlagType) { return vector_type{}; }

  template <typename FlagType>
  KOKKOS_FORCEINLINE_FUNCTION
  static void store([[maybe_unused]]value_type* ptr, [[maybe_unused]]vector_type v, FlagType) {}

  template <typename FlagType>
  KOKKOS_FORCEINLINE_FUNCTION
  static void masked_store([[maybe_unused]]value_type* ptr, [[maybe_unused]]vector_type v, [[maybe_unused]]vector_type mask, FlagType) {}

  template <typename T, typename Abi>
  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type convert_from(simd_vector_t<T, Abi, simd_device_tag>) { return vector_type{}; }

  KOKKOS_FORCEINLINE_FUNCTION
  static value_type extract(vector_type v, simd_size_t i) { return v[i]; }

  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type neg([[maybe_unused]]vector_type v) { return vector_type{}; }

  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type plus([[maybe_unused]]vector_type lhs, [[maybe_unused]] vector_type rhs) { return vector_type{}; }

  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type minus([[maybe_unused]]vector_type lhs, [[maybe_unused]] vector_type rhs) { return vector_type{}; }

  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type multiply([[maybe_unused]]vector_type lhs, [[maybe_unused]] vector_type rhs) { return vector_type{}; }

  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type divide([[maybe_unused]]vector_type lhs, [[maybe_unused]] vector_type rhs) { return vector_type{}; }

  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type eq([[maybe_unused]]vector_type lhs, [[maybe_unused]] vector_type rhs) { return vector_type{}; }

  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type neq([[maybe_unused]]vector_type lhs, [[maybe_unused]] vector_type rhs) { return vector_type{}; }

  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type ge([[maybe_unused]]vector_type lhs, [[maybe_unused]] vector_type rhs) { return vector_type{}; }

  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type le([[maybe_unused]]vector_type lhs, [[maybe_unused]] vector_type rhs) { return vector_type{}; }

  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type gt([[maybe_unused]]vector_type lhs, [[maybe_unused]] vector_type rhs) { return vector_type{}; }

  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type lt([[maybe_unused]]vector_type lhs, [[maybe_unused]] vector_type rhs) { return vector_type{}; }

  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type copysign([[maybe_unused]]vector_type a, [[maybe_unused]]vector_type b) { return vector_type{}; }

  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type abs([[maybe_unused]]vector_type v) { return vector_type{}; }

  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type floor([[maybe_unused]]vector_type v) { return vector_type{}; }

  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type ceil([[maybe_unused]]vector_type v) { return vector_type{}; }

  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type round([[maybe_unused]]vector_type v) { return vector_type{}; }

  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type trunc([[maybe_unused]]vector_type v) { return vector_type{}; }

  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type sqrt([[maybe_unused]]vector_type v) { return vector_type{}; }

#ifdef KOKKOS_HAVE_INTEL_SVML
  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type cbrt([[maybe_unused]]vector_type v) { return vector_type{}; }

  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type exp([[maybe_unused]]vector_type v) { return vector_type{}; }

  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type log([[maybe_unused]]vector_type v) { return vector_type{}; }
#endif

  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type fma([[maybe_unused]]vector_type a, [[maybe_unused]]vector_type b, [[maybe_unused]]vector_type c) { return vector_type{}; }

  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type max([[maybe_unused]]vector_type a, [[maybe_unused]]vector_type b) { return vector_type{}; }

  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type min([[maybe_unused]]vector_type a, [[maybe_unused]]vector_type b) { return vector_type{}; }

  KOKKOS_FORCEINLINE_FUNCTION
  static vector_type condition([[maybe_unused]]vector_type a, [[maybe_unused]]vector_type b, [[maybe_unused]]vector_type c) { return vector_type{}; }
};

// host
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
template <>
struct simd_native_ops<double, simd_abi::avx2_fixed_size<4>, simd_host_tag> {
  using vector_type = simd_vector_t<double, simd_abi::avx2_fixed_size<4>, simd_host_tag>;
  using value_type = double;

  template <typename T>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type set1(T&& value) { return _mm256_set1_pd(value_type(value)); }

  template <typename G>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type gen(G&& gen) {
    return _mm256_setr_pd(
            gen(std::integral_constant<simd_size_t, 0>()),
            gen(std::integral_constant<simd_size_t, 1>()),
            gen(std::integral_constant<simd_size_t, 2>()),
            gen(std::integral_constant<simd_size_t, 3>()));
  }

  template <typename FlagType>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type load(const value_type* ptr, FlagType) {
    if constexpr (std::is_same_v<FlagType,
                                 simd_flags<simd_alignment_vector_aligned>>) {
      return _mm256_load_pd(ptr);
    } else {
      return _mm256_loadu_pd(ptr);
    }
  }

  template <typename FlagType>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type masked_load(const value_type* ptr, vector_type mask, FlagType) {
    return _mm256_maskload_pd(ptr, _mm256_castpd_si256(mask));
  }

  template <typename FlagType>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static void store(value_type* ptr, vector_type v, FlagType) {
    if constexpr (std::is_same_v<FlagType,
                                simd_flags<simd_alignment_vector_aligned>>) {
      _mm256_store_pd(ptr, v);
    } else {
      _mm256_storeu_pd(ptr, v);
    }
  }

  template <typename FlagType>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static void masked_store(value_type* ptr, vector_type v, vector_type mask, FlagType) {
    _mm256_maskstore_pd(ptr, _mm256_castpd_si256(mask), v);
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type convert_from(simd_vector_t<float, simd_abi::avx2_fixed_size<4>, simd_host_tag> v) { return _mm256_cvtps_pd(v); }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type convert_from(simd_vector_t<std::int32_t, simd_abi::avx2_fixed_size<4>, simd_host_tag> v) { return _mm256_cvtepi32_pd(v); }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static value_type extract(vector_type v, simd_size_t i) {
    constexpr auto size = sizeof(vector_type) / sizeof(value_type);
    value_type tmp[size];
    _mm256_storeu_pd(tmp, v);
    return tmp[i];
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type neg(vector_type v) { return _mm256_sub_pd(_mm256_set1_pd(0.0), v); }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type plus(vector_type lhs, vector_type rhs) { return _mm256_add_pd(lhs, rhs); }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type minus(vector_type lhs, vector_type rhs) { return _mm256_sub_pd(lhs, rhs); }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type multiply(vector_type lhs, vector_type rhs) { return _mm256_mul_pd(lhs, rhs); }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type divide(vector_type lhs, vector_type rhs) { return _mm256_div_pd(lhs, rhs); }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type eq(vector_type lhs, vector_type rhs) { return _mm256_cmp_pd(lhs, rhs, _CMP_EQ_OS); }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type neq(vector_type lhs, vector_type rhs) { return _mm256_cmp_pd(lhs, rhs, _CMP_NEQ_OS); }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type ge(vector_type lhs, vector_type rhs) { return _mm256_cmp_pd(lhs, rhs, _CMP_GE_OS); }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type le(vector_type lhs, vector_type rhs) { return _mm256_cmp_pd(lhs, rhs, _CMP_LE_OS); }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type gt(vector_type lhs, vector_type rhs) { return _mm256_cmp_pd(lhs, rhs, _CMP_GT_OS); }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type lt(vector_type lhs, vector_type rhs) { return _mm256_cmp_pd(lhs, rhs, _CMP_LT_OS); }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type copysign(vector_type lhs, vector_type rhs) {
    auto sign_mask = _mm256_set1_pd(-0.0);
    return _mm256_xor_pd(_mm256_andnot_pd(sign_mask, static_cast<vector_type>(lhs)),
                    _mm256_and_pd(sign_mask, static_cast<vector_type>(rhs)));
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type abs(vector_type v) {
    auto sign_mask = _mm256_set1_pd(-0.0);
    return _mm256_andnot_pd(sign_mask, static_cast<vector_type>(v));
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type floor(vector_type v) {
    return _mm256_round_pd(static_cast<vector_type>(v),
                      (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type ceil(vector_type v) {
    return _mm256_round_pd(static_cast<vector_type>(v),
                      (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type round(vector_type v) {
    return _mm256_round_pd(static_cast<vector_type>(v),
                      (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type trunc(vector_type v) {
    return _mm256_round_pd(static_cast<vector_type>(v),
                      (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type sqrt(vector_type v) {
    return _mm256_sqrt_pd(static_cast<vector_type>(v));
  }

#ifdef KOKKOS_HAVE_INTEL_SVML
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type cbrt(vector_type v) {
    return _mm256_cbrt_pd(static_cast<vector_type>(v));
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type exp(vector_type v) {
    return _mm256_exp_pd(static_cast<vector_type>(v));
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type log(vector_type v) {
    return _mm256_log_pd(static_cast<vector_type>(v));
  }
#endif

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type fma(vector_type a, vector_type b, vector_type c) {
    return _mm256_fmadd_pd(static_cast<vector_type>(a), static_cast<vector_type>(b),
                           static_cast<vector_type>(c));
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type max(vector_type a, vector_type b) {
    return _mm256_max_pd(static_cast<vector_type>(a), static_cast<vector_type>(b));
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type min(vector_type a, vector_type b) {
    return _mm256_min_pd(static_cast<vector_type>(a), static_cast<vector_type>(b));
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static vector_type condition(vector_type a, vector_type b, vector_type c) {
    return _mm256_blendv_pd(static_cast<vector_type>(c), static_cast<vector_type>(b),
                       static_cast<vector_type>(a));
  }
};
#endif

}
}

#endif