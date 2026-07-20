// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_SIMD_IMPL_AVX2_HPP
#define KOKKOS_SIMD_IMPL_AVX2_HPP

#include <immintrin.h>
#include <Kokkos_Array.hpp>
#include <Kokkos_SIMD_Common.hpp>
#include <Kokkos_BitManipulation.hpp>  // bit_cast

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

KOKKOS_SIMD_IMPL_DEFINE_HOST_VECTOR(double, simd_abi::avx2_fixed_size<4>, __m256d)
KOKKOS_SIMD_IMPL_DEFINE_HOST_VECTOR(float, simd_abi::avx2_fixed_size<4>, __m128)
KOKKOS_SIMD_IMPL_DEFINE_HOST_VECTOR(float, simd_abi::avx2_fixed_size<8>, __m256)
KOKKOS_SIMD_IMPL_DEFINE_HOST_VECTOR(std::int32_t, simd_abi::avx2_fixed_size<4>, __m128i)
KOKKOS_SIMD_IMPL_DEFINE_HOST_VECTOR(std::int32_t, simd_abi::avx2_fixed_size<8>, __m256i)
KOKKOS_SIMD_IMPL_DEFINE_HOST_VECTOR(std::int64_t, simd_abi::avx2_fixed_size<4>, __m256i)
KOKKOS_SIMD_IMPL_DEFINE_HOST_VECTOR(std::uint64_t, simd_abi::avx2_fixed_size<4>, __m256i)

template <typename T, typename Abi>
  requires requires {
    typename simd_vector_impl<T, Abi, simd_host_tag>::type;
  }
struct simd_vector_impl<T, Abi, simd_device_tag> {
  using host_type = typename simd_vector_impl<T, Abi, simd_host_tag>::type;
  using type = Kokkos::Array<char, sizeof(host_type)>;
};

template <typename T, typename Abi, typename Tag>
  requires requires {
    typename simd_vector_impl<T, Abi, Tag>::type;
  }
using simd_vector_t = typename simd_vector_impl<T, Abi, Tag>::type;

template <typename T, typename Abi, typename Tag>
struct simd_mask_native_ops;

// FIXME
// device
template <>
struct simd_mask_native_ops<double, simd_abi::avx2_fixed_size<4>, simd_device_tag> {
  using abi_type = simd_abi::avx2_fixed_size<4>;
  using host_vector_type = simd_vector_t<double, abi_type, simd_host_tag>;
  using vector_type = simd_vector_t<double, abi_type, simd_device_tag>;
  using value_type = bool;

  KOKKOS_SIMD_IMPL_DEFINE_FN_DEVICE(vector_type, set1, value_type, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_DEVICE(vector_type, gen, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_DEVICE(vector_type, convert_from, T, Abi, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DECL_DEVICE(abi_type)

  KOKKOS_SIMD_IMPL_DEFINE_EXTRACT_FN_DEVICE(value_type, extract, return v[i])

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_DEVICE(vector_type, lnot, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_DEVICE(vector_type, bnot, lnot(v))

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, land, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, lor, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, band, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, bor, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, bxor, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, eq, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, neq, vector_type{})
};

template <>
struct simd_mask_native_ops<float, simd_abi::avx2_fixed_size<4>, simd_device_tag> {
  using abi_type = simd_abi::avx2_fixed_size<4>;
  using host_vector_type = simd_vector_t<float, abi_type, simd_host_tag>;
  using vector_type = simd_vector_t<float, abi_type, simd_device_tag>;
  using value_type = bool;

  KOKKOS_SIMD_IMPL_DEFINE_FN_DEVICE(vector_type, set1, value_type, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_DEVICE(vector_type, gen, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_DEVICE(vector_type, convert_from, T, Abi, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DECL_DEVICE(abi_type)

  KOKKOS_SIMD_IMPL_DEFINE_EXTRACT_FN_DEVICE(value_type, extract, return v[i])

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_DEVICE(vector_type, lnot, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_DEVICE(vector_type, bnot, lnot(v))

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, land, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, lor, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, band, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, bor, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, bxor, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, eq, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, neq, vector_type{})
};

template <>
struct simd_mask_native_ops<float, simd_abi::avx2_fixed_size<8>, simd_device_tag> {
  using abi_type = simd_abi::avx2_fixed_size<8>;
  using host_vector_type = simd_vector_t<float, abi_type, simd_host_tag>;
  using vector_type = simd_vector_t<float, abi_type, simd_device_tag>;
  using value_type = bool;

  KOKKOS_SIMD_IMPL_DEFINE_FN_DEVICE(vector_type, set1, value_type, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_DEVICE(vector_type, gen, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_DEVICE(vector_type, convert_from, T, Abi, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DECL_DEVICE(abi_type)

  KOKKOS_SIMD_IMPL_DEFINE_EXTRACT_FN_DEVICE(value_type, extract, return v[i])

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_DEVICE(vector_type, lnot, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_DEVICE(vector_type, bnot, lnot(v))

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, land, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, lor, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, band, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, bor, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, bxor, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, eq, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, neq, vector_type{})
};

template <>
struct simd_mask_native_ops<std::int32_t, simd_abi::avx2_fixed_size<4>, simd_device_tag> {
  using abi_type = simd_abi::avx2_fixed_size<4>;
  using host_vector_type = simd_vector_t<std::int32_t, abi_type, simd_host_tag>;
  using vector_type = simd_vector_t<std::int32_t, abi_type, simd_device_tag>;
  using value_type = bool;

  KOKKOS_SIMD_IMPL_DEFINE_FN_DEVICE(vector_type, set1, value_type, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_DEVICE(vector_type, gen, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_DEVICE(vector_type, convert_from, T, Abi, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DECL_DEVICE(abi_type)

  KOKKOS_SIMD_IMPL_DEFINE_EXTRACT_FN_DEVICE(value_type, extract, return v[i])

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_DEVICE(vector_type, lnot, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_DEVICE(vector_type, bnot, lnot(v))

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, land, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, lor, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, band, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, bor, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, bxor, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, eq, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, neq, vector_type{})
};

template <>
struct simd_mask_native_ops<std::int32_t, simd_abi::avx2_fixed_size<8>, simd_device_tag> {
  using abi_type = simd_abi::avx2_fixed_size<8>;
  using host_vector_type = simd_vector_t<std::int32_t, abi_type, simd_host_tag>;
  using vector_type = simd_vector_t<std::int32_t, abi_type, simd_device_tag>;
  using value_type = bool;

  KOKKOS_SIMD_IMPL_DEFINE_FN_DEVICE(vector_type, set1, value_type, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_DEVICE(vector_type, gen, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_DEVICE(vector_type, convert_from, T, Abi, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DECL_DEVICE(abi_type)

  KOKKOS_SIMD_IMPL_DEFINE_EXTRACT_FN_DEVICE(value_type, extract, return v[i])

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_DEVICE(vector_type, lnot, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_DEVICE(vector_type, bnot, lnot(v))

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, land, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, lor, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, band, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, bor, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, bxor, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, eq, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, neq, vector_type{})
};

template <>
struct simd_mask_native_ops<std::int64_t, simd_abi::avx2_fixed_size<4>, simd_device_tag> {
  using abi_type = simd_abi::avx2_fixed_size<4>;
  using host_vector_type = simd_vector_t<std::int64_t, abi_type, simd_host_tag>;
  using vector_type = simd_vector_t<std::int64_t, abi_type, simd_device_tag>;
  using value_type = bool;

  KOKKOS_SIMD_IMPL_DEFINE_FN_DEVICE(vector_type, set1, value_type, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_DEVICE(vector_type, gen, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_DEVICE(vector_type, convert_from, T, Abi, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DECL_DEVICE(abi_type)

  KOKKOS_SIMD_IMPL_DEFINE_EXTRACT_FN_DEVICE(value_type, extract, return v[i])

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_DEVICE(vector_type, lnot, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_DEVICE(vector_type, bnot, lnot(v))

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, land, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, lor, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, band, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, bor, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, bxor, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, eq, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, neq, vector_type{})
};

template <>
struct simd_mask_native_ops<std::uint64_t, simd_abi::avx2_fixed_size<4>, simd_device_tag> {
  using abi_type = simd_abi::avx2_fixed_size<4>;
  using host_vector_type = simd_vector_t<std::uint64_t, abi_type, simd_host_tag>;
  using vector_type = simd_vector_t<std::uint64_t, abi_type, simd_device_tag>;
  using value_type = bool;

  KOKKOS_SIMD_IMPL_DEFINE_FN_DEVICE(vector_type, set1, value_type, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_DEVICE(vector_type, gen, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_DEVICE(vector_type, convert_from, T, Abi, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DECL_DEVICE(abi_type)

  KOKKOS_SIMD_IMPL_DEFINE_EXTRACT_FN_DEVICE(value_type, extract, return v[i])

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_DEVICE(vector_type, lnot, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_DEVICE(vector_type, bnot, lnot(v))

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, land, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, lor, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, band, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, bor, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, bxor, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, eq, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, neq, vector_type{})
};

KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DEFN_DEVICE(simd_mask_native_ops, double, simd_abi::avx2_fixed_size<4>)
KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DEFN_DEVICE(simd_mask_native_ops, float, simd_abi::avx2_fixed_size<4>)
KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DEFN_DEVICE(simd_mask_native_ops, float, simd_abi::avx2_fixed_size<8>)
KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DEFN_DEVICE(simd_mask_native_ops, std::int32_t, simd_abi::avx2_fixed_size<4>)
KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DEFN_DEVICE(simd_mask_native_ops, std::int32_t, simd_abi::avx2_fixed_size<8>)
KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DEFN_DEVICE(simd_mask_native_ops, std::int64_t, simd_abi::avx2_fixed_size<4>)
KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DEFN_DEVICE(simd_mask_native_ops, std::uint64_t, simd_abi::avx2_fixed_size<4>)

// host
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
template <>
struct simd_mask_native_ops<double, simd_abi::avx2_fixed_size<4>, simd_host_tag> {
  using abi_type = simd_abi::avx2_fixed_size<4>;
  using host_vector_type = simd_vector_t<double, abi_type, simd_host_tag>;
  using vector_type = host_vector_type;
  using value_type = bool;

  KOKKOS_SIMD_IMPL_DEFINE_FN_HOST(vector_type, set1, value_type, _mm256_castsi256_pd(_mm256_set1_epi64x(-std::int64_t(v))))

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_HOST(vector_type, gen, _mm256_castsi256_pd(_mm256_setr_epi64x(
                                                            -std::int64_t(v(std::integral_constant<simd_size_t, 0>())),
                                                            -std::int64_t(v(std::integral_constant<simd_size_t, 1>())),
                                                            -std::int64_t(v(std::integral_constant<simd_size_t, 2>())),
                                                            -std::int64_t(v(std::integral_constant<simd_size_t, 3>())))))

  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_HOST(vector_type, convert_from, float, abi_type, _mm256_cvtps_pd(v))
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_HOST(vector_type, convert_from, std::int32_t, abi_type, _mm256_cvtepi32_pd(v))
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_HOST(vector_type, convert_from, std::int64_t, abi_type, _mm256_castsi256_pd(v))
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DECL_HOST(abi_type)

  KOKKOS_SIMD_IMPL_DEFINE_EXTRACT_FN_HOST(value_type, extract, return (_mm256_movemask_pd(v) & (1 << i)) != 0)

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_HOST(vector_type, lnot, _mm256_andnot_pd(v, set1(true)))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_HOST(vector_type, bnot, lnot(v);)

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, land, band(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, lor, bor(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, band, _mm256_and_pd(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, bor, _mm256_or_pd(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, bxor, _mm256_xor_pd(lhs, rhs))

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(value_type, eq, (_mm256_movemask_pd(lhs) == _mm256_movemask_pd(rhs)))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(value_type, neq, !eq(lhs, rhs))
};

template <>
struct simd_mask_native_ops<float, simd_abi::avx2_fixed_size<4>, simd_host_tag> {
  using abi_type = simd_abi::avx2_fixed_size<4>;
  using host_vector_type = simd_vector_t<float, abi_type, simd_host_tag>;
  using vector_type = host_vector_type;
  using value_type = bool;

  KOKKOS_SIMD_IMPL_DEFINE_FN_HOST(vector_type, set1, value_type, _mm_castsi128_ps(_mm_set1_epi32(-std::int32_t(v))))

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_HOST(vector_type, gen, _mm_castsi128_ps(_mm_setr_epi32(
                                                                -std::int32_t(v(std::integral_constant<simd_size_t, 0>())),
                                                                -std::int32_t(v(std::integral_constant<simd_size_t, 1>())),
                                                                -std::int32_t(v(std::integral_constant<simd_size_t, 2>())),
                                                                -std::int32_t(v(std::integral_constant<simd_size_t, 3>())))))

  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_HOST(vector_type, convert_from, std::int32_t, abi_type, _mm_cvtepi32_ps(v))
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DECL_HOST(abi_type)

  KOKKOS_SIMD_IMPL_DEFINE_EXTRACT_FN_HOST(value_type, extract, return (_mm_movemask_ps(v) & (1 << i)) != 0)

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_HOST(vector_type, lnot, _mm_andnot_ps(v, set1(true)))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_HOST(vector_type, bnot, lnot(v);)

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, land, band(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, lor, bor(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, band, _mm_and_ps(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, bor, _mm_or_ps(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, bxor, _mm_xor_ps(lhs, rhs))

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(value_type, eq, (_mm_movemask_ps(lhs) == _mm_movemask_ps(rhs)))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(value_type, neq, !eq(lhs, rhs))
};

template <>
struct simd_mask_native_ops<float, simd_abi::avx2_fixed_size<8>, simd_host_tag> {
  using abi_type = simd_abi::avx2_fixed_size<8>;
  using host_vector_type = simd_vector_t<float, abi_type, simd_host_tag>;
  using vector_type = host_vector_type;
  using value_type = bool;

  KOKKOS_SIMD_IMPL_DEFINE_FN_HOST(vector_type, set1, value_type, _mm256_castsi256_ps(_mm256_set1_epi32(-std::int32_t(v))))

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_HOST(vector_type, gen, _mm256_castsi256_ps(_mm256_setr_epi32(
                                                          -std::int32_t(v(std::integral_constant<simd_size_t, 0>())),
                                                          -std::int32_t(v(std::integral_constant<simd_size_t, 1>())),
                                                          -std::int32_t(v(std::integral_constant<simd_size_t, 2>())),
                                                          -std::int32_t(v(std::integral_constant<simd_size_t, 3>())),
                                                          -std::int32_t(v(std::integral_constant<simd_size_t, 4>())),
                                                          -std::int32_t(v(std::integral_constant<simd_size_t, 5>())),
                                                          -std::int32_t(v(std::integral_constant<simd_size_t, 6>())),
                                                          -std::int32_t(v(std::integral_constant<simd_size_t, 7>())))))

  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_HOST(vector_type, convert_from, std::int32_t, abi_type, _mm256_castsi256_ps(v))
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DECL_HOST(abi_type)

  KOKKOS_SIMD_IMPL_DEFINE_EXTRACT_FN_HOST(value_type, extract, return (_mm256_movemask_ps(v) & (1 << i)) != 0)

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_HOST(vector_type, lnot, _mm256_andnot_ps(v, set1(true)))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_HOST(vector_type, bnot, lnot(v);)

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, land, band(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, lor, bor(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, band, _mm256_and_ps(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, bor, _mm256_or_ps(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, bxor, _mm256_xor_ps(lhs, rhs))

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(value_type, eq, (_mm256_movemask_ps(lhs) == _mm256_movemask_ps(rhs)))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(value_type, neq, !eq(lhs, rhs))
};

template <>
struct simd_mask_native_ops<std::int32_t, simd_abi::avx2_fixed_size<4>, simd_host_tag> {
  using abi_type = simd_abi::avx2_fixed_size<4>;
  using host_vector_type = simd_vector_t<std::int32_t, abi_type, simd_host_tag>;
  using vector_type = host_vector_type;
  using value_type = bool;

  KOKKOS_SIMD_IMPL_DEFINE_FN_HOST(vector_type, set1, value_type, _mm_set1_epi32(-std::int32_t(v)))

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_HOST(vector_type, gen, _mm_setr_epi32(
                                                          -std::int32_t(v(std::integral_constant<simd_size_t, 0>())),
                                                          -std::int32_t(v(std::integral_constant<simd_size_t, 1>())),
                                                          -std::int32_t(v(std::integral_constant<simd_size_t, 2>())),
                                                          -std::int32_t(v(std::integral_constant<simd_size_t, 3>()))))

  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_HOST(vector_type, convert_from, float, abi_type, _mm_castps_si128(v))
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DECL_HOST(abi_type)

  KOKKOS_SIMD_IMPL_DEFINE_EXTRACT_FN_HOST(value_type, extract, return (_mm_movemask_ps(_mm_castsi128_ps(v)) & (1 << i)) != 0)

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_HOST(vector_type, lnot, _mm_andnot_si128(v, set1(true)))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_HOST(vector_type, bnot, lnot(v);)

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, land, band(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, lor, bor(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, band, _mm_and_si128(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, bor, _mm_or_si128(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, bxor, _mm_xor_si128(lhs, rhs))

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(value_type, eq, (_mm_movemask_ps(_mm_castsi128_ps(lhs)) ==
                                                          _mm_movemask_ps(_mm_castsi128_ps(rhs))))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(value_type, neq, !eq(lhs, rhs))
};

template <>
struct simd_mask_native_ops<std::int32_t, simd_abi::avx2_fixed_size<8>, simd_host_tag> {
  using abi_type = simd_abi::avx2_fixed_size<8>;
  using host_vector_type = simd_vector_t<std::int32_t, abi_type, simd_host_tag>;
  using vector_type = host_vector_type;
  using value_type = bool;

  KOKKOS_SIMD_IMPL_DEFINE_FN_HOST(vector_type, set1, value_type, _mm256_set1_epi32(-std::int32_t(v)))

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_HOST(vector_type, gen, _mm256_setr_epi32(
                                                          -std::int32_t(v(std::integral_constant<simd_size_t, 0>())),
                                                          -std::int32_t(v(std::integral_constant<simd_size_t, 1>())),
                                                          -std::int32_t(v(std::integral_constant<simd_size_t, 2>())),
                                                          -std::int32_t(v(std::integral_constant<simd_size_t, 3>())),
                                                          -std::int32_t(v(std::integral_constant<simd_size_t, 4>())),
                                                          -std::int32_t(v(std::integral_constant<simd_size_t, 5>())),
                                                          -std::int32_t(v(std::integral_constant<simd_size_t, 6>())),
                                                          -std::int32_t(v(std::integral_constant<simd_size_t, 7>()))))

  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_HOST(vector_type, convert_from, float, abi_type, _mm256_castps_si256(v))
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DECL_HOST(abi_type)

  KOKKOS_SIMD_IMPL_DEFINE_EXTRACT_FN_HOST(value_type, extract, return (_mm256_movemask_ps(_mm256_castsi256_ps(v)) & (1 << i)) != 0)

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_HOST(vector_type, lnot, _mm256_andnot_si256(v, set1(true)))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_HOST(vector_type, bnot, lnot(v);)

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, land, band(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, lor, bor(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, band, _mm256_and_si256(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, bor, _mm256_or_si256(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, bxor, _mm256_xor_si256(lhs, rhs))

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(value_type, eq, (_mm256_movemask_ps(_mm256_castsi256_ps(lhs)) == _mm256_movemask_ps(_mm256_castsi256_ps(rhs))))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(value_type, neq, !eq(lhs, rhs))
};

template <>
struct simd_mask_native_ops<std::int64_t, simd_abi::avx2_fixed_size<4>, simd_host_tag> {
  using abi_type = simd_abi::avx2_fixed_size<4>;
  using host_vector_type = simd_vector_t<std::int64_t, abi_type, simd_host_tag>;
  using vector_type = host_vector_type;
  using value_type = bool;

  KOKKOS_SIMD_IMPL_DEFINE_FN_HOST(vector_type, set1, value_type, _mm256_set1_epi64x(-std::int64_t(v)))

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_HOST(vector_type, gen, _mm256_setr_epi64x(
                                                          -std::int64_t(v(std::integral_constant<simd_size_t, 0>())),
                                                          -std::int64_t(v(std::integral_constant<simd_size_t, 1>())),
                                                          -std::int64_t(v(std::integral_constant<simd_size_t, 2>())),
                                                          -std::int64_t(v(std::integral_constant<simd_size_t, 3>()))))

  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_HOST(vector_type, convert_from, std::int32_t, abi_type, _mm256_cvtepi32_epi64(v))
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_HOST(vector_type, convert_from, double, abi_type, _mm256_castpd_si256(v))
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_HOST(vector_type, convert_from, std::uint64_t, abi_type, v)
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DECL_HOST(abi_type)

  KOKKOS_SIMD_IMPL_DEFINE_EXTRACT_FN_HOST(value_type, extract, return (_mm256_movemask_pd(_mm256_castsi256_pd(v)) & (1 << i)) != 0)

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_HOST(vector_type, lnot, _mm256_andnot_si256(v, set1(true)))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_HOST(vector_type, bnot, lnot(v);)

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, land, band(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, lor, bor(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, band, _mm256_and_si256(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, bor, _mm256_or_si256(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, bxor, _mm256_xor_si256(lhs, rhs))

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(value_type, eq, (_mm256_movemask_pd(_mm256_castsi256_pd(lhs)) == _mm256_movemask_pd(_mm256_castsi256_pd(rhs))))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(value_type, neq, !eq(lhs, rhs))
};

template <>
struct simd_mask_native_ops<std::uint64_t, simd_abi::avx2_fixed_size<4>, simd_host_tag> {
  using abi_type = simd_abi::avx2_fixed_size<4>;
  using host_vector_type = simd_vector_t<std::uint64_t, abi_type, simd_host_tag>;
  using vector_type = host_vector_type;
  using value_type = bool;

  KOKKOS_SIMD_IMPL_DEFINE_FN_HOST(vector_type, set1, value_type, _mm256_set1_epi64x(-std::int64_t(v)))

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_HOST(vector_type, gen, _mm256_setr_epi64x(
                                                          -std::int64_t(v(std::integral_constant<simd_size_t, 0>())),
                                                          -std::int64_t(v(std::integral_constant<simd_size_t, 1>())),
                                                          -std::int64_t(v(std::integral_constant<simd_size_t, 2>())),
                                                          -std::int64_t(v(std::integral_constant<simd_size_t, 3>()))))

  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_HOST(vector_type, convert_from, std::int32_t, abi_type, _mm256_cvtepi32_epi64(v))
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_HOST(vector_type, convert_from, double, abi_type, _mm256_castpd_si256(v))
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_HOST(vector_type, convert_from, std::int64_t, abi_type, v)
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DECL_HOST(abi_type)

  KOKKOS_SIMD_IMPL_DEFINE_EXTRACT_FN_HOST(value_type, extract, return (_mm256_movemask_pd(_mm256_castsi256_pd(v)) & (1 << i)) != 0)

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_HOST(vector_type, lnot, _mm256_andnot_si256(v, set1(true)))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_HOST(vector_type, bnot, lnot(v);)

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, land, band(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, lor, bor(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, band, _mm256_and_si256(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, bor, _mm256_or_si256(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, bxor, _mm256_xor_si256(lhs, rhs))

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(value_type, eq, (_mm256_movemask_pd(_mm256_castsi256_pd(lhs)) == _mm256_movemask_pd(_mm256_castsi256_pd(rhs))))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(value_type, neq, !eq(lhs, rhs))
};

KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DEFN_HOST(simd_mask_native_ops, double, simd_abi::avx2_fixed_size<4>)
KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DEFN_HOST(simd_mask_native_ops, float, simd_abi::avx2_fixed_size<4>)
KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DEFN_HOST(simd_mask_native_ops, float, simd_abi::avx2_fixed_size<8>)
KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DEFN_HOST(simd_mask_native_ops, std::int32_t, simd_abi::avx2_fixed_size<4>)
KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DEFN_HOST(simd_mask_native_ops, std::int32_t, simd_abi::avx2_fixed_size<8>)
KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DEFN_HOST(simd_mask_native_ops, std::int64_t, simd_abi::avx2_fixed_size<4>)
KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DEFN_HOST(simd_mask_native_ops, std::uint64_t, simd_abi::avx2_fixed_size<4>)

#endif

template <typename T, typename Abi, typename Tag>
struct simd_native_ops;

// FIXME
// device
template <>
struct simd_native_ops<double, simd_abi::avx2_fixed_size<4>, simd_device_tag> {
  using abi_type = simd_abi::avx2_fixed_size<4>;
  using host_vector_type = simd_vector_t<double, abi_type, simd_host_tag>;
  using vector_type = simd_vector_t<double, abi_type, simd_device_tag>;
  using value_type = double;

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_DEVICE(vector_type, set1, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_DEVICE(vector_type, gen, vector_type{})
  
  KOKKOS_SIMD_IMPL_DEFINE_LOAD_DEVICE(vector_type, load, const value_type*, return vector_type{}; )
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_LOAD_DEVICE(vector_type, masked_load, const value_type*, vector_type, return vector_type{}; )
  KOKKOS_SIMD_IMPL_DEFINE_STORE_DEVICE(store, value_type*, vector_type, {})
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_STORE_DEVICE(masked_store, value_type*, vector_type, vector_type, {})

  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_DEVICE(vector_type, convert_from, T, Abi, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DECL_DEVICE(abi_type)

  KOKKOS_SIMD_IMPL_DEFINE_EXTRACT_FN_DEVICE(value_type, extract, return v[i])

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_DEVICE(vector_type, neg, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, plus, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, minus, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, multiply, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, divide, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, eq, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, neq, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, ge, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, le, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, gt, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, lt, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, copysign, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(vector_type, abs, vector_type{})
  using rounded_data_type = value_type;
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(vector_type, floor, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(vector_type, ceil, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(vector_type, round, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(vector_type, trunc, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(vector_type, sqrt, vector_type{})

#ifdef KOKKOS_HAVE_INTEL_SVML
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(vector_type, cbrt, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(vector_type, exp, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(vector_type, log, vector_type{})
#endif

  KOKKOS_SIMD_IMPL_DEFINE_TERNARY_MATH_OP_DEVICE(vector_type, fma, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_MATH_OP_DEVICE(vector_type, max, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_MATH_OP_DEVICE(vector_type, min, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_TERNARY_MATH_OP_DEVICE(vector_type, condition, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_GATHER_FROM_FN_DEVICE(vector_type, unchecked, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_GATHER_FROM_FN_DEVICE(vector_type, unchecked, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_GATHER_FROM_FN_DEVICE(vector_type, partial, unchecked_gather_from(in, indices, flag))
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_GATHER_FROM_FN_DEVICE(vector_type, partial, unchecked_gather_from(in, indices, mmask, flag))
};

template <>
struct simd_native_ops<float, simd_abi::avx2_fixed_size<4>, simd_device_tag> {
  using abi_type = simd_abi::avx2_fixed_size<4>;
  using host_vector_type = simd_vector_t<float, abi_type, simd_host_tag>;
  using vector_type = simd_vector_t<float, abi_type, simd_device_tag>;
  using value_type = float;

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_DEVICE(vector_type, set1, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_DEVICE(vector_type, gen, vector_type{})
  
  KOKKOS_SIMD_IMPL_DEFINE_LOAD_DEVICE(vector_type, load, const value_type*, return vector_type{}; )
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_LOAD_DEVICE(vector_type, masked_load, const value_type*, vector_type, return vector_type{}; )
  KOKKOS_SIMD_IMPL_DEFINE_STORE_DEVICE(store, value_type*, vector_type, {})
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_STORE_DEVICE(masked_store, value_type*, vector_type, vector_type, {})

  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_DEVICE(vector_type, convert_from, T, Abi, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DECL_DEVICE(abi_type)

  KOKKOS_SIMD_IMPL_DEFINE_EXTRACT_FN_DEVICE(value_type, extract, return v[i])

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_DEVICE(vector_type, neg, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, plus, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, minus, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, multiply, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, divide, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, eq, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, neq, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, ge, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, le, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, gt, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, lt, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, copysign, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(vector_type, abs, vector_type{})
  using rounded_data_type = value_type;
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(vector_type, floor, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(vector_type, ceil, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(vector_type, round, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(vector_type, trunc, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(vector_type, sqrt, vector_type{})

#ifdef KOKKOS_HAVE_INTEL_SVML
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(vector_type, cbrt, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(vector_type, exp, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(vector_type, log, vector_type{})
#endif

  KOKKOS_SIMD_IMPL_DEFINE_TERNARY_MATH_OP_DEVICE(vector_type, fma, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_MATH_OP_DEVICE(vector_type, max, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_MATH_OP_DEVICE(vector_type, min, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_TERNARY_MATH_OP_DEVICE(vector_type, condition, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_GATHER_FROM_FN_DEVICE(vector_type, unchecked, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_GATHER_FROM_FN_DEVICE(vector_type, unchecked, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_GATHER_FROM_FN_DEVICE(vector_type, partial, unchecked_gather_from(in, indices, flag))
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_GATHER_FROM_FN_DEVICE(vector_type, partial, unchecked_gather_from(in, indices, mmask, flag))
};

template <>
struct simd_native_ops<float, simd_abi::avx2_fixed_size<8>, simd_device_tag> {
  using abi_type = simd_abi::avx2_fixed_size<8>;
  using host_vector_type = simd_vector_t<float, abi_type, simd_host_tag>;
  using vector_type = simd_vector_t<float, abi_type, simd_device_tag>;
  using value_type = float;

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_DEVICE(vector_type, set1, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_DEVICE(vector_type, gen, vector_type{})
  
  KOKKOS_SIMD_IMPL_DEFINE_LOAD_DEVICE(vector_type, load, const value_type*, return vector_type{}; )
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_LOAD_DEVICE(vector_type, masked_load, const value_type*, vector_type, return vector_type{}; )
  KOKKOS_SIMD_IMPL_DEFINE_STORE_DEVICE(store, value_type*, vector_type, {})
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_STORE_DEVICE(masked_store, value_type*, vector_type, vector_type, {})

  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_DEVICE(vector_type, convert_from, T, Abi, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DECL_DEVICE(abi_type)

  KOKKOS_SIMD_IMPL_DEFINE_EXTRACT_FN_DEVICE(value_type, extract, return v[i])

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_DEVICE(vector_type, neg, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, plus, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, minus, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, multiply, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, divide, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, eq, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, neq, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, ge, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, le, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, gt, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, lt, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, copysign, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(vector_type, abs, vector_type{})
  using rounded_data_type = value_type;
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(vector_type, floor, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(vector_type, ceil, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(vector_type, round, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(vector_type, trunc, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(vector_type, sqrt, vector_type{})

#ifdef KOKKOS_HAVE_INTEL_SVML
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(vector_type, cbrt, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(vector_type, exp, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(vector_type, log, vector_type{})
#endif

  KOKKOS_SIMD_IMPL_DEFINE_TERNARY_MATH_OP_DEVICE(vector_type, fma, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_MATH_OP_DEVICE(vector_type, max, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_MATH_OP_DEVICE(vector_type, min, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_TERNARY_MATH_OP_DEVICE(vector_type, condition, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_GATHER_FROM_FN_DEVICE(vector_type, unchecked, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_GATHER_FROM_FN_DEVICE(vector_type, unchecked, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_GATHER_FROM_FN_DEVICE(vector_type, partial, unchecked_gather_from(in, indices, flag))
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_GATHER_FROM_FN_DEVICE(vector_type, partial, unchecked_gather_from(in, indices, mmask, flag))
};

template <>
struct simd_native_ops<std::int32_t, simd_abi::avx2_fixed_size<4>, simd_device_tag> {
  using abi_type = simd_abi::avx2_fixed_size<4>;
  using host_vector_type = simd_vector_t<std::int32_t, abi_type, simd_host_tag>;
  using vector_type = simd_vector_t<std::int32_t, abi_type, simd_device_tag>;
  using value_type = std::int32_t;

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_DEVICE(vector_type, set1, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_DEVICE(vector_type, gen, vector_type{})
  
  KOKKOS_SIMD_IMPL_DEFINE_LOAD_DEVICE(vector_type, load, const value_type*, return vector_type{}; )
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_LOAD_DEVICE(vector_type, masked_load, const value_type*, vector_type, return vector_type{}; )
  KOKKOS_SIMD_IMPL_DEFINE_STORE_DEVICE(store, value_type*, vector_type, {})
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_STORE_DEVICE(masked_store, value_type*, vector_type, vector_type, {})

  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_DEVICE(vector_type, convert_from, T, Abi, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DECL_DEVICE(abi_type)

  KOKKOS_SIMD_IMPL_DEFINE_EXTRACT_FN_DEVICE(value_type, extract, return v[i])

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_DEVICE(vector_type, neg, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_DEVICE(vector_type, bnot, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, plus, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, minus, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, multiply, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, band, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, bor, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, bxor, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, sll, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, sra, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_SHIFT_SCALAR_DEVICE(vector_type, sll, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_SHIFT_SCALAR_DEVICE(vector_type, sra, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, eq, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, neq, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, ge, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, le, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, gt, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, lt, vector_type{})
  // KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, copysign, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(vector_type, abs, vector_type{})

  using rounded_data_type = double;
  using rounded_type = simd_vector_t<rounded_data_type, abi_type, simd_device_tag>;
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(rounded_type, floor, rounded_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(rounded_type, ceil, rounded_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(rounded_type, round, rounded_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(rounded_type, trunc, rounded_type{})

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_MATH_OP_DEVICE(vector_type, max, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_MATH_OP_DEVICE(vector_type, min, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_TERNARY_MATH_OP_DEVICE(vector_type, condition, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_GATHER_FROM_FN_DEVICE(vector_type, unchecked, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_GATHER_FROM_FN_DEVICE(vector_type, unchecked, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_GATHER_FROM_FN_DEVICE(vector_type, partial, unchecked_gather_from(in, indices, flag))
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_GATHER_FROM_FN_DEVICE(vector_type, partial, unchecked_gather_from(in, indices, mmask, flag))
};

template <>
struct simd_native_ops<std::int32_t, simd_abi::avx2_fixed_size<8>, simd_device_tag> {
  using abi_type = simd_abi::avx2_fixed_size<8>;
  using host_vector_type = simd_vector_t<std::int32_t, abi_type, simd_host_tag>;
  using vector_type = simd_vector_t<std::int32_t, abi_type, simd_device_tag>;
  using value_type = std::int32_t;

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_DEVICE(vector_type, set1, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_DEVICE(vector_type, gen, vector_type{})
  
  KOKKOS_SIMD_IMPL_DEFINE_LOAD_DEVICE(vector_type, load, const value_type*, return vector_type{}; )
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_LOAD_DEVICE(vector_type, masked_load, const value_type*, vector_type, return vector_type{}; )
  KOKKOS_SIMD_IMPL_DEFINE_STORE_DEVICE(store, value_type*, vector_type, {})
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_STORE_DEVICE(masked_store, value_type*, vector_type, vector_type, {})

  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_DEVICE(vector_type, convert_from, T, Abi, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DECL_DEVICE(abi_type)

  KOKKOS_SIMD_IMPL_DEFINE_EXTRACT_FN_DEVICE(value_type, extract, return v[i])

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_DEVICE(vector_type, neg, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_DEVICE(vector_type, bnot, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, plus, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, minus, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, multiply, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, band, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, bor, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, bxor, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, sll, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, sra, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_SHIFT_SCALAR_DEVICE(vector_type, sll, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_SHIFT_SCALAR_DEVICE(vector_type, sra, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, eq, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, neq, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, ge, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, le, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, gt, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, lt, vector_type{})
  // KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, copysign, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(vector_type, abs, vector_type{})

  using rounded_data_type = float;
  using rounded_type = simd_vector_t<rounded_data_type, abi_type, simd_device_tag>;
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(rounded_type, floor, rounded_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(rounded_type, ceil, rounded_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(rounded_type, round, rounded_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(rounded_type, trunc, rounded_type{})

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_MATH_OP_DEVICE(vector_type, max, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_MATH_OP_DEVICE(vector_type, min, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_TERNARY_MATH_OP_DEVICE(vector_type, condition, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_GATHER_FROM_FN_DEVICE(vector_type, unchecked, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_GATHER_FROM_FN_DEVICE(vector_type, unchecked, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_GATHER_FROM_FN_DEVICE(vector_type, partial, unchecked_gather_from(in, indices, flag))
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_GATHER_FROM_FN_DEVICE(vector_type, partial, unchecked_gather_from(in, indices, mmask, flag))
};

template <>
struct simd_native_ops<std::int64_t, simd_abi::avx2_fixed_size<4>, simd_device_tag> {
  using abi_type = simd_abi::avx2_fixed_size<4>;
  using host_vector_type = simd_vector_t<std::int64_t, abi_type, simd_host_tag>;
  using vector_type = simd_vector_t<std::int64_t, abi_type, simd_device_tag>;
  using value_type = std::int64_t;

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_DEVICE(vector_type, set1, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_DEVICE(vector_type, gen, vector_type{})
  
  KOKKOS_SIMD_IMPL_DEFINE_LOAD_DEVICE(vector_type, load, const value_type*, return vector_type{}; )
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_LOAD_DEVICE(vector_type, masked_load, const value_type*, vector_type, return vector_type{}; )
  KOKKOS_SIMD_IMPL_DEFINE_STORE_DEVICE(store, value_type*, vector_type, {})
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_STORE_DEVICE(masked_store, value_type*, vector_type, vector_type, {})

  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_DEVICE(vector_type, convert_from, T, Abi, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DECL_DEVICE(abi_type)

  KOKKOS_SIMD_IMPL_DEFINE_EXTRACT_FN_DEVICE(value_type, extract, return v[i])

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_DEVICE(vector_type, neg, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_DEVICE(vector_type, bnot, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, plus, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, minus, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, multiply, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, band, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, bor, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, bxor, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, sll, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, sra, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_SHIFT_SCALAR_DEVICE(vector_type, sll, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_SHIFT_SCALAR_DEVICE(vector_type, sra, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, eq, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, neq, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, ge, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, le, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, gt, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, lt, vector_type{})
  // KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, copysign, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(vector_type, abs, vector_type{})

  using rounded_data_type = double;
  using rounded_type = simd_vector_t<rounded_data_type, abi_type, simd_device_tag>;
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(rounded_type, floor, rounded_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(rounded_type, ceil, rounded_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(rounded_type, round, rounded_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(rounded_type, trunc, rounded_type{})

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_MATH_OP_DEVICE(vector_type, max, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_MATH_OP_DEVICE(vector_type, min, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_TERNARY_MATH_OP_DEVICE(vector_type, condition, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_GATHER_FROM_FN_DEVICE(vector_type, unchecked, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_GATHER_FROM_FN_DEVICE(vector_type, unchecked, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_GATHER_FROM_FN_DEVICE(vector_type, partial, unchecked_gather_from(in, indices, flag))
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_GATHER_FROM_FN_DEVICE(vector_type, partial, unchecked_gather_from(in, indices, mmask, flag))
};

template <>
struct simd_native_ops<std::uint64_t, simd_abi::avx2_fixed_size<4>, simd_device_tag> {
  using abi_type = simd_abi::avx2_fixed_size<4>;
  using host_vector_type = simd_vector_t<std::uint64_t, abi_type, simd_host_tag>;
  using vector_type = simd_vector_t<std::uint64_t, abi_type, simd_device_tag>;
  using value_type = std::uint64_t;

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_DEVICE(vector_type, set1, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_DEVICE(vector_type, gen, vector_type{})
  
  KOKKOS_SIMD_IMPL_DEFINE_LOAD_DEVICE(vector_type, load, const value_type*, return vector_type{}; )
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_LOAD_DEVICE(vector_type, masked_load, const value_type*, vector_type, return vector_type{}; )
  KOKKOS_SIMD_IMPL_DEFINE_STORE_DEVICE(store, value_type*, vector_type, {})
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_STORE_DEVICE(masked_store, value_type*, vector_type, vector_type, {})

  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_DEVICE(vector_type, convert_from, T, Abi, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DECL_DEVICE(abi_type)

  KOKKOS_SIMD_IMPL_DEFINE_EXTRACT_FN_DEVICE(value_type, extract, return v[i])

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_DEVICE(vector_type, neg, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_DEVICE(vector_type, bnot, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, plus, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, minus, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, multiply, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, band, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, bor, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, bxor, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, sll, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, sra, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_SHIFT_SCALAR_DEVICE(vector_type, sll, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_SHIFT_SCALAR_DEVICE(vector_type, sra, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, eq, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, neq, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, ge, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, le, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, gt, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, lt, vector_type{})
  // KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(vector_type, copysign, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(vector_type, abs, vector_type{})

  using rounded_data_type = double;
  using rounded_type = simd_vector_t<rounded_data_type, abi_type, simd_device_tag>;
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(rounded_type, floor, rounded_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(rounded_type, ceil, rounded_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(rounded_type, round, rounded_type{})
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(rounded_type, trunc, rounded_type{})

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_MATH_OP_DEVICE(vector_type, max, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_MATH_OP_DEVICE(vector_type, min, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_TERNARY_MATH_OP_DEVICE(vector_type, condition, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_GATHER_FROM_FN_DEVICE(vector_type, unchecked, vector_type{})
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_GATHER_FROM_FN_DEVICE(vector_type, unchecked, vector_type{})

  KOKKOS_SIMD_IMPL_DEFINE_GATHER_FROM_FN_DEVICE(vector_type, partial, unchecked_gather_from(in, indices, flag))
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_GATHER_FROM_FN_DEVICE(vector_type, partial, unchecked_gather_from(in, indices, mmask, flag))
};

KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DEFN_DEVICE(simd_native_ops, double, simd_abi::avx2_fixed_size<4>)
KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DEFN_DEVICE(simd_native_ops, float, simd_abi::avx2_fixed_size<4>)
KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DEFN_DEVICE(simd_native_ops, float, simd_abi::avx2_fixed_size<8>)
KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DEFN_DEVICE(simd_native_ops, std::int32_t, simd_abi::avx2_fixed_size<4>)
KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DEFN_DEVICE(simd_native_ops, std::int32_t, simd_abi::avx2_fixed_size<8>)
KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DEFN_DEVICE(simd_native_ops, std::int64_t, simd_abi::avx2_fixed_size<4>)
KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DEFN_DEVICE(simd_native_ops, std::uint64_t, simd_abi::avx2_fixed_size<4>)

// host
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
template <>
struct simd_native_ops<double, simd_abi::avx2_fixed_size<4>, simd_host_tag> {
  using abi_type = simd_abi::avx2_fixed_size<4>;
  using host_vector_type = simd_vector_t<double, abi_type, simd_host_tag>;
  using vector_type = host_vector_type;
  using value_type = double;

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_HOST(vector_type, set1, _mm256_set1_pd(value_type(v)))

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_HOST(vector_type, gen, _mm256_setr_pd(
                                                          v(std::integral_constant<simd_size_t, 0>()),
                                                          v(std::integral_constant<simd_size_t, 1>()),
                                                          v(std::integral_constant<simd_size_t, 2>()),
                                                          v(std::integral_constant<simd_size_t, 3>())))

  KOKKOS_SIMD_IMPL_DEFINE_LOAD_HOST(vector_type, load, const value_type*, if constexpr (std::is_same_v<simd_flags<Flags...>,
                                 simd_flags<simd_alignment_vector_aligned>>) {
      return _mm256_load_pd(ptr);
    } else {
      return _mm256_loadu_pd(ptr);
    })

  KOKKOS_SIMD_IMPL_DEFINE_MASKED_LOAD_HOST(vector_type, masked_load, const value_type*, vector_type, return _mm256_maskload_pd(ptr, _mm256_castpd_si256(mask));)

  KOKKOS_SIMD_IMPL_DEFINE_STORE_HOST(store, value_type*, vector_type, 
    if constexpr (std::is_same_v<simd_flags<Flags...>,
                                simd_flags<simd_alignment_vector_aligned>>) {
      _mm256_store_pd(ptr, v);
    } else {
      _mm256_storeu_pd(ptr, v);
    }
  )

  KOKKOS_SIMD_IMPL_DEFINE_MASKED_STORE_HOST(masked_store, value_type*, vector_type, vector_type,
    _mm256_maskstore_pd(ptr, _mm256_castpd_si256(mask), v);
  )

  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_HOST(vector_type, convert_from, float, abi_type, _mm256_cvtps_pd(v))
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_HOST(vector_type, convert_from, std::int32_t, abi_type, _mm256_cvtepi32_pd(v))
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DECL_HOST(abi_type)

  KOKKOS_SIMD_IMPL_DEFINE_EXTRACT_FN_HOST(value_type, extract, constexpr auto size = sizeof(vector_type) / sizeof(value_type);
                                                                                  value_type tmp[size];
                                                                                  _mm256_storeu_pd(tmp, v);
                                                                                  return tmp[i];
                                          )

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_HOST(vector_type, neg, _mm256_sub_pd(_mm256_set1_pd(0.0), v))

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, plus, _mm256_add_pd(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, minus, _mm256_sub_pd(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, multiply, _mm256_mul_pd(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, divide, _mm256_div_pd(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, eq, _mm256_cmp_pd(lhs, rhs, _CMP_EQ_OS))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, neq, _mm256_cmp_pd(lhs, rhs, _CMP_NEQ_OS))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, ge, _mm256_cmp_pd(lhs, rhs, _CMP_GE_OS))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, le, _mm256_cmp_pd(lhs, rhs, _CMP_LE_OS))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, gt, _mm256_cmp_pd(lhs, rhs, _CMP_GT_OS))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, lt, _mm256_cmp_pd(lhs, rhs, _CMP_LT_OS))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, copysign, _mm256_xor_pd(_mm256_andnot_pd(_mm256_set1_pd(-0.0), lhs),
                    _mm256_and_pd(_mm256_set1_pd(-0.0), rhs)))

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(vector_type, abs, _mm256_andnot_pd(_mm256_set1_pd(-0.0), v))
  using rounded_data_type = value_type;
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(vector_type, floor, _mm256_round_pd(v,
                                                                (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(vector_type, ceil, _mm256_round_pd(v,
                                                                (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(vector_type, round, _mm256_round_pd(v,
                                                                (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(vector_type, trunc, _mm256_round_pd(v,
                                                                (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(vector_type, sqrt, _mm256_sqrt_pd(v))

#ifdef KOKKOS_HAVE_INTEL_SVML
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(vector_type, cbrt, _mm256_cbrt_pd(v))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(vector_type, exp, _mm256_exp_pd(v))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(vector_type, log, _mm256_log_pd(v))
#endif

  KOKKOS_SIMD_IMPL_DEFINE_TERNARY_MATH_OP_HOST(vector_type, fma, _mm256_fmadd_pd(a, b,
                                                                                 c))

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_MATH_OP_HOST(vector_type, max, _mm256_max_pd(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_MATH_OP_HOST(vector_type, min, _mm256_min_pd(lhs, rhs))

  KOKKOS_SIMD_IMPL_DEFINE_TERNARY_MATH_OP_HOST(vector_type, condition, _mm256_blendv_pd(c, b,
                                                                                        a))

  KOKKOS_SIMD_IMPL_DEFINE_GATHER_FROM_FN_HOST(vector_type, unchecked, _mm256_i32gather_pd(Ranges::data(in), indices, 8))
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_GATHER_FROM_FN_HOST(vector_type, unchecked, _mm256_mask_i32gather_pd(_mm256_set1_pd(value_type{}), Ranges::data(in), indices, mmask, 8))

  KOKKOS_SIMD_IMPL_DEFINE_GATHER_FROM_FN_HOST(vector_type, partial, unchecked_gather_from(in, indices, flag))
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_GATHER_FROM_FN_HOST(vector_type, partial, unchecked_gather_from(in, indices, mmask, flag))
};

template <>
struct simd_native_ops<float, simd_abi::avx2_fixed_size<4>, simd_host_tag> {
  using abi_type = simd_abi::avx2_fixed_size<4>;
  using host_vector_type = simd_vector_t<float, abi_type, simd_host_tag>;
  using vector_type = host_vector_type;
  using value_type = float;

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_HOST(vector_type, set1, _mm_set1_ps(value_type(v)))

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_HOST(vector_type, gen, _mm_setr_ps(v(std::integral_constant<simd_size_t, 0>()),
                                                          v(std::integral_constant<simd_size_t, 1>()),
                                                          v(std::integral_constant<simd_size_t, 2>()),
                                                          v(std::integral_constant<simd_size_t, 3>())))

  KOKKOS_SIMD_IMPL_DEFINE_LOAD_HOST(vector_type, load, const value_type*, 
    if constexpr (std::is_same_v<simd_flags<Flags...>,
                                 simd_flags<simd_alignment_vector_aligned>>) {
      return _mm_load_ps(ptr);
    } else {
      return _mm_loadu_ps(ptr);
    }
  )

  KOKKOS_SIMD_IMPL_DEFINE_MASKED_LOAD_HOST(vector_type, masked_load, const value_type*, vector_type, return _mm_maskload_ps(ptr, _mm_castps_si128(mask));)

  KOKKOS_SIMD_IMPL_DEFINE_STORE_HOST(store, value_type*, vector_type, 
    if constexpr (std::is_same_v<simd_flags<Flags...>,
                                simd_flags<simd_alignment_vector_aligned>>) {
      _mm_store_ps(ptr, v);
    } else {
      _mm_storeu_ps(ptr, v);
    }
  )

  KOKKOS_SIMD_IMPL_DEFINE_MASKED_STORE_HOST(masked_store, value_type*, vector_type, vector_type,
    _mm_maskstore_ps(ptr, _mm_castps_si128(mask), v);
  )

  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_HOST(vector_type, convert_from, double, abi_type, _mm256_cvtpd_ps(v))
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_HOST(vector_type, convert_from, std::int32_t, abi_type, _mm_cvtepi32_ps(v))
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DECL_HOST(abi_type)

  KOKKOS_SIMD_IMPL_DEFINE_EXTRACT_FN_HOST(value_type, extract,
    auto index = _mm_cvtsi32_si128(i);
    auto tmp   = _mm_permutevar_ps(v, index);
    return _mm_cvtss_f32(tmp);
                                          )

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_HOST(vector_type, neg, _mm_sub_ps(_mm_set1_ps(0.0), v))

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, plus, _mm_add_ps(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, minus, _mm_sub_ps(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, multiply, _mm_mul_ps(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, divide, _mm_div_ps(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, eq, _mm_cmpeq_ps(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, neq, _mm_cmpneq_ps(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, ge, _mm_cmpge_ps(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, le, _mm_cmple_ps(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, gt, _mm_cmpgt_ps(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, lt, _mm_cmplt_ps(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, copysign, _mm_xor_ps(_mm_andnot_ps(_mm_set1_ps(-0.0), lhs),
                 _mm_and_ps(_mm_set1_ps(-0.0), rhs)))

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(vector_type, abs, _mm_andnot_ps(_mm_set1_ps(-0.0), v))
  using rounded_data_type = value_type;
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(vector_type, floor, _mm_round_ps(v,
                                                                (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(vector_type, ceil, _mm_round_ps(v,
                                                                (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(vector_type, round, _mm_round_ps(v,
                                                                (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(vector_type, trunc, _mm_round_ps(v,
                                                                (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(vector_type, sqrt, _mm_sqrt_ps(v))

#ifdef KOKKOS_HAVE_INTEL_SVML
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(vector_type, cbrt, _mm_cbrt_ps(v))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(vector_type, exp, _mm_exp_ps(v))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(vector_type, log, _mm_log_ps(v))
#endif

  KOKKOS_SIMD_IMPL_DEFINE_TERNARY_MATH_OP_HOST(vector_type, fma, _mm_fmadd_ps(a, b,
                                                                                 c))

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_MATH_OP_HOST(vector_type, max, _mm_max_ps(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_MATH_OP_HOST(vector_type, min, _mm_min_ps(lhs, rhs))

  KOKKOS_SIMD_IMPL_DEFINE_TERNARY_MATH_OP_HOST(vector_type, condition, _mm_blendv_ps(c, b,
                                                                                        a))

  KOKKOS_SIMD_IMPL_DEFINE_GATHER_FROM_FN_HOST(vector_type, unchecked, _mm_i32gather_ps(Ranges::data(in), indices, 4))
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_GATHER_FROM_FN_HOST(vector_type, unchecked, _mm_mask_i32gather_ps(_mm_set1_ps(value_type{}), Ranges::data(in), indices, mmask, 4))

  KOKKOS_SIMD_IMPL_DEFINE_GATHER_FROM_FN_HOST(vector_type, partial, unchecked_gather_from(in, indices, flag))
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_GATHER_FROM_FN_HOST(vector_type, partial, unchecked_gather_from(in, indices, mmask, flag))
};

template <>
struct simd_native_ops<float, simd_abi::avx2_fixed_size<8>, simd_host_tag> {
  using abi_type = simd_abi::avx2_fixed_size<8>;
  using host_vector_type = simd_vector_t<float, abi_type, simd_host_tag>;
  using vector_type = host_vector_type;
  using value_type = float;

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_HOST(vector_type, set1, _mm256_set1_ps(value_type(v)))

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_HOST(vector_type, gen, _mm256_setr_ps(
                                                          v(std::integral_constant<simd_size_t, 0>()),
                                                          v(std::integral_constant<simd_size_t, 1>()),
                                                          v(std::integral_constant<simd_size_t, 2>()),
                                                          v(std::integral_constant<simd_size_t, 3>()),
                                                          v(std::integral_constant<simd_size_t, 4>()),
                                                          v(std::integral_constant<simd_size_t, 5>()),
                                                          v(std::integral_constant<simd_size_t, 6>()),
                                                          v(std::integral_constant<simd_size_t, 7>())))

  KOKKOS_SIMD_IMPL_DEFINE_LOAD_HOST(vector_type, load, const value_type*, 
    if constexpr (std::is_same_v<simd_flags<Flags...>,
                                 simd_flags<simd_alignment_vector_aligned>>) {
      return _mm256_load_ps(ptr);
    } else {
      return _mm256_loadu_ps(ptr);
    }
  )

  KOKKOS_SIMD_IMPL_DEFINE_MASKED_LOAD_HOST(vector_type, masked_load, const value_type*, vector_type, return _mm256_maskload_ps(ptr, _mm256_castps_si256(mask));)

  KOKKOS_SIMD_IMPL_DEFINE_STORE_HOST(store, value_type*, vector_type, 
    if constexpr (std::is_same_v<simd_flags<Flags...>,
                                simd_flags<simd_alignment_vector_aligned>>) {
      _mm256_store_ps(ptr, v);
    } else {
      _mm256_storeu_ps(ptr, v);
    }
  )

  KOKKOS_SIMD_IMPL_DEFINE_MASKED_STORE_HOST(masked_store, value_type*, vector_type, vector_type,
    _mm256_maskstore_ps(ptr, _mm256_castps_si256(mask), v);
  )

  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_HOST(vector_type, convert_from, std::int32_t, abi_type, _mm256_cvtepi32_ps(v))
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DECL_HOST(abi_type)

  KOKKOS_SIMD_IMPL_DEFINE_EXTRACT_FN_HOST(value_type, extract,
    auto index = _mm256_set1_epi32(i);
    auto tmp   = _mm256_permutevar8x32_ps(v, index);
    return _mm256_cvtss_f32(tmp);
                                          )

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_HOST(vector_type, neg, _mm256_sub_ps(_mm256_set1_ps(0.0), v))

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, plus, _mm256_add_ps(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, minus, _mm256_sub_ps(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, multiply, _mm256_mul_ps(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, divide, _mm256_div_ps(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, eq, _mm256_cmp_ps(lhs, rhs, _CMP_EQ_OS))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, neq, _mm256_cmp_ps(lhs, rhs, _CMP_NEQ_OS))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, ge, _mm256_cmp_ps(lhs, rhs, _CMP_GE_OS))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, le, _mm256_cmp_ps(lhs, rhs, _CMP_LE_OS))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, gt, _mm256_cmp_ps(lhs, rhs, _CMP_GT_OS))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, lt, _mm256_cmp_ps(lhs, rhs, _CMP_LT_OS))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, copysign, _mm256_xor_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.0), lhs),
                    _mm256_and_ps(_mm256_set1_ps(-0.0), rhs)))

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(vector_type, abs, _mm256_andnot_ps(_mm256_set1_ps(-0.0), v))
  using rounded_data_type = value_type;
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(vector_type, floor,_mm256_round_ps(v,
                                                                  (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(vector_type, ceil, _mm256_round_ps(v,
                                                                  (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(vector_type, round, _mm256_round_ps(v,
                                                                  (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(vector_type, trunc, _mm256_round_ps(v,
                                                                  (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(vector_type, sqrt, _mm256_sqrt_ps(v))

#ifdef KOKKOS_HAVE_INTEL_SVML
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(vector_type, cbrt, _mm256_cbrt_ps(v))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(vector_type, exp, _mm256_exp_ps(v))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(vector_type, log, _mm256_log_ps(v))
#endif

  KOKKOS_SIMD_IMPL_DEFINE_TERNARY_MATH_OP_HOST(vector_type, fma, _mm256_fmadd_ps(a, b,
                                                                                 c))

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_MATH_OP_HOST(vector_type, max, _mm256_max_ps(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_MATH_OP_HOST(vector_type, min, _mm256_min_ps(lhs, rhs))

  KOKKOS_SIMD_IMPL_DEFINE_TERNARY_MATH_OP_HOST(vector_type, condition, _mm256_blendv_ps(c, b,
                                                                                        a))

  KOKKOS_SIMD_IMPL_DEFINE_GATHER_FROM_FN_HOST(vector_type, unchecked, _mm256_i32gather_ps(Ranges::data(in), indices, 4))
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_GATHER_FROM_FN_HOST(vector_type, unchecked, _mm256_mask_i32gather_ps(_mm256_set1_ps(value_type{}),
                                        Ranges::data(in), indices, mmask, 4))

  KOKKOS_SIMD_IMPL_DEFINE_GATHER_FROM_FN_HOST(vector_type, partial, unchecked_gather_from(in, indices, flag))
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_GATHER_FROM_FN_HOST(vector_type, partial, unchecked_gather_from(in, indices, mmask, flag))
};

template <>
struct simd_native_ops<std::int32_t, simd_abi::avx2_fixed_size<4>, simd_host_tag> {
  using abi_type = simd_abi::avx2_fixed_size<4>;
  using host_vector_type = simd_vector_t<std::int32_t, abi_type, simd_host_tag>;
  using vector_type = host_vector_type;
  using value_type = std::int32_t;

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_HOST(vector_type, set1, _mm_set1_epi32(value_type(v)))

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_HOST(vector_type, gen, _mm_setr_epi32(
                                                          v(std::integral_constant<simd_size_t, 0>()),
                                                          v(std::integral_constant<simd_size_t, 1>()),
                                                          v(std::integral_constant<simd_size_t, 2>()),
                                                          v(std::integral_constant<simd_size_t, 3>())))

  KOKKOS_SIMD_IMPL_DEFINE_LOAD_HOST(vector_type, load, const value_type*, 
    if constexpr (std::is_same_v<simd_flags<Flags...>,
                                 simd_flags<simd_alignment_vector_aligned>>) {
      return _mm_load_si128(reinterpret_cast<const vector_type*>(ptr));
    } else {
      return _mm_loadu_si128(reinterpret_cast<const vector_type*>(ptr));
    }
  )

  KOKKOS_SIMD_IMPL_DEFINE_MASKED_LOAD_HOST(vector_type, masked_load, const value_type*, vector_type, return _mm_maskload_epi32(ptr, mask);)

  KOKKOS_SIMD_IMPL_DEFINE_STORE_HOST(store, value_type*, vector_type, 
  if constexpr (std::is_same_v<simd_flags<Flags...>,
                               simd_flags<simd_alignment_vector_aligned>>) {
    _mm_store_si128(reinterpret_cast<vector_type*>(ptr), v);
  } else {
    _mm_storeu_si128(reinterpret_cast<vector_type*>(ptr), v);
  }
  )

  KOKKOS_SIMD_IMPL_DEFINE_MASKED_STORE_HOST(masked_store, value_type*, vector_type, vector_type,
    _mm_maskstore_epi32(ptr, mask, v);
  )

  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_HOST(vector_type, convert_from, float, abi_type, _mm_cvtps_epi32(v))
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_HOST(vector_type, convert_from, double, abi_type, _mm256_cvtpd_epi32(v))
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DECL_HOST(abi_type)

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static value_type extract(vector_type v, simd_size_t i) {
    switch (i) {
      case 0: return _mm_extract_epi32(v, 0x0);
      case 1: return _mm_extract_epi32(v, 0x1);
      case 2: return _mm_extract_epi32(v, 0x2);
      case 3: return _mm_extract_epi32(v, 0x3);
      default: Kokkos::abort("Index out of bound"); break;
    }
// missing return statement warning with cuda >= 12.9
#if defined(KOKKOS_COMPILER_NVCC) && (KOKKOS_COMPILER_NVCC >= 1290) && \
    defined(KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK)
    return value_type{};
#endif
  }

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_HOST(vector_type, neg, minus(set1(0), v))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_HOST(vector_type, bnot, _mm_andnot_si128(v, set1(~value_type(0))))

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, plus, _mm_add_epi32(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, minus, _mm_sub_epi32(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, multiply, _mm_mullo_epi32(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, band, _mm_and_si128(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, bor, _mm_or_si128(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, bxor, _mm_xor_si128(lhs, rhs))

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, sll, _mm_sllv_epi32(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, sra, _mm_srav_epi32(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_SHIFT_SCALAR_HOST(vector_type, sll, _mm_slli_epi32(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_SHIFT_SCALAR_HOST(vector_type, sra, _mm_srai_epi32(lhs, rhs))

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, eq, _mm_cmpeq_epi32(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, neq, bnot(eq(lhs, rhs)))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, gt, _mm_cmpgt_epi32(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, lt, _mm_cmplt_epi32(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, ge, bor(gt(lhs, rhs), eq(lhs, rhs)))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, le, bor(lt(lhs, rhs), eq(lhs, rhs)))
  // KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, copysign, _mm256_xor_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.0), static_cast<vector_type>(lhs)),
  //                   _mm256_and_ps(_mm256_set1_ps(-0.0), static_cast<vector_type>(rhs))))

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(vector_type, abs, _mm_abs_epi32(v))

  using rounded_data_type = double;
  using rounded_type = simd_vector_t<rounded_data_type, abi_type, simd_host_tag>;
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(rounded_type, floor, _mm256_cvtepi32_pd(v))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(rounded_type, ceil, _mm256_cvtepi32_pd(v))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(rounded_type, round, _mm256_cvtepi32_pd(v))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(rounded_type, trunc, _mm256_cvtepi32_pd(v))

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_MATH_OP_HOST(vector_type, max, _mm_max_epi32(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_MATH_OP_HOST(vector_type, min, _mm_min_epi32(lhs, rhs))

  KOKKOS_SIMD_IMPL_DEFINE_TERNARY_MATH_OP_HOST(vector_type, condition, _mm_castps_si128(
                                                                        _mm_blendv_ps(_mm_castsi128_ps(c),
                                                                                      _mm_castsi128_ps(b),
                                                                                      _mm_castsi128_ps(a))))

  KOKKOS_SIMD_IMPL_DEFINE_GATHER_FROM_FN_HOST(vector_type, unchecked, _mm_i32gather_epi32(Ranges::data(in), indices, 4))
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_GATHER_FROM_FN_HOST(vector_type, unchecked, _mm_mask_i32gather_epi32(_mm_set1_epi32(value_type{}),
                                        Ranges::data(in), indices, mmask, 4))

  KOKKOS_SIMD_IMPL_DEFINE_GATHER_FROM_FN_HOST(vector_type, partial, unchecked_gather_from(in, indices, flag))
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_GATHER_FROM_FN_HOST(vector_type, partial, unchecked_gather_from(in, indices, mmask, flag))
};

template <>
struct simd_native_ops<std::int32_t, simd_abi::avx2_fixed_size<8>, simd_host_tag> {
  using abi_type = simd_abi::avx2_fixed_size<8>;
  using host_vector_type = simd_vector_t<std::int32_t, abi_type, simd_host_tag>;
  using vector_type = host_vector_type;
  using value_type = std::int32_t;

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_HOST(vector_type, set1, _mm256_set1_epi32(value_type(v)))

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_HOST(vector_type, gen, _mm256_setr_epi32(
                                                          v(std::integral_constant<simd_size_t, 0>()),
                                                          v(std::integral_constant<simd_size_t, 1>()),
                                                          v(std::integral_constant<simd_size_t, 2>()),
                                                          v(std::integral_constant<simd_size_t, 3>()),
                                                          v(std::integral_constant<simd_size_t, 4>()),
                                                          v(std::integral_constant<simd_size_t, 5>()),
                                                          v(std::integral_constant<simd_size_t, 6>()),
                                                          v(std::integral_constant<simd_size_t, 7>())))

  KOKKOS_SIMD_IMPL_DEFINE_LOAD_HOST(vector_type, load, const value_type*, 
    if constexpr (std::is_same_v<simd_flags<Flags...>,
                                 simd_flags<simd_alignment_vector_aligned>>) {
      return _mm256_load_si256(reinterpret_cast<const vector_type*>(ptr));
    } else {
      return _mm256_loadu_si256(reinterpret_cast<const vector_type*>(ptr));
    }
  )

  KOKKOS_SIMD_IMPL_DEFINE_MASKED_LOAD_HOST(vector_type, masked_load, const value_type*, vector_type, return _mm256_maskload_epi32(ptr, mask);)

  KOKKOS_SIMD_IMPL_DEFINE_STORE_HOST(store, value_type*, vector_type, 
  if constexpr (std::is_same_v<simd_flags<Flags...>,
                               simd_flags<simd_alignment_vector_aligned>>) {
    _mm256_store_si256(reinterpret_cast<vector_type*>(ptr), v);
  } else {
    _mm256_storeu_si256(reinterpret_cast<vector_type*>(ptr), v);
  }
  )

  KOKKOS_SIMD_IMPL_DEFINE_MASKED_STORE_HOST(masked_store, value_type*, vector_type, vector_type,
    _mm256_maskstore_epi32(ptr, mask, v);
  )

  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_HOST(vector_type, convert_from, float, abi_type, _mm256_cvtps_epi32(v))
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DECL_HOST(abi_type)

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static value_type extract(vector_type v, simd_size_t i) {
// _mm256_cvtsi256_si32 was not added in GCC until 11
#if defined(KOKKOS_COMPILER_GNU) && (KOKKOS_COMPILER_GNU < 1100)
    value_type tmp[size()];
    _mm256_maskstore_epi32(tmp, static_cast<__m256i>(mask_type(true)), m_value);
    return tmp[i];
#else
    auto index = _mm256_set1_epi32(i);
    auto tmp   = _mm256_permutevar8x32_epi32(v, index);
    return _mm256_cvtsi256_si32(tmp);
#endif
  }

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_HOST(vector_type, neg, minus(set1(0), v))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_HOST(vector_type, bnot, _mm256_andnot_si256(v, set1(~value_type(0))))

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, plus, _mm256_add_epi32(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, minus, _mm256_sub_epi32(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, multiply, _mm256_mullo_epi32(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, band, _mm256_and_si256(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, bor, _mm256_or_si256(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, bxor, _mm256_xor_si256(lhs, rhs))

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, sll, _mm256_sllv_epi32(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, sra, _mm256_srav_epi32(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_SHIFT_SCALAR_HOST(vector_type, sll, _mm256_slli_epi32(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_SHIFT_SCALAR_HOST(vector_type, sra, _mm256_srai_epi32(lhs, rhs))

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, eq, _mm256_cmpeq_epi32(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, neq, bnot(eq(lhs, rhs)))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, gt, _mm256_cmpgt_epi32(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, ge, bor(gt(lhs, rhs), eq(lhs, rhs)))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, lt, bnot(ge(lhs, rhs)))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, le, bor(lt(lhs, rhs), eq(lhs, rhs)))
  // KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, copysign, _mm256_xor_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.0), static_cast<vector_type>(a)),
  //                   _mm256_and_ps(_mm256_set1_ps(-0.0), static_cast<vector_type>(b))))

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(vector_type, abs, _mm256_abs_epi32(v))

  using rounded_data_type = float;
  using rounded_type = simd_vector_t<rounded_data_type, abi_type, simd_host_tag>;
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(rounded_type, floor, _mm256_cvtepi32_ps(v))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(rounded_type, ceil, _mm256_cvtepi32_ps(v))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(rounded_type, round, _mm256_cvtepi32_ps(v))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(rounded_type, trunc, _mm256_cvtepi32_ps(v))

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_MATH_OP_HOST(vector_type, max, _mm256_max_epi32(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_MATH_OP_HOST(vector_type, min, _mm256_min_epi32(lhs, rhs))

  KOKKOS_SIMD_IMPL_DEFINE_TERNARY_MATH_OP_HOST(vector_type, condition, _mm256_castps_si256(
                                                                        _mm256_blendv_ps(_mm256_castsi256_ps(c),
                                                                                      _mm256_castsi256_ps(b),
                                                                                      _mm256_castsi256_ps(a))))
  KOKKOS_SIMD_IMPL_DEFINE_GATHER_FROM_FN_HOST(vector_type, unchecked, _mm256_i32gather_epi32(Ranges::data(in), indices, 4))
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_GATHER_FROM_FN_HOST(vector_type, unchecked, _mm256_mask_i32gather_epi32(_mm256_set1_epi32(value_type{}),
                                           Ranges::data(in), indices, mmask,
                                           4))

  KOKKOS_SIMD_IMPL_DEFINE_GATHER_FROM_FN_HOST(vector_type, partial, unchecked_gather_from(in, indices, flag))
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_GATHER_FROM_FN_HOST(vector_type, partial, unchecked_gather_from(in, indices, mmask, flag))
};

template <>
struct simd_native_ops<std::int64_t, simd_abi::avx2_fixed_size<4>, simd_host_tag> {
  using abi_type = simd_abi::avx2_fixed_size<4>;
  using host_vector_type = simd_vector_t<std::int64_t, abi_type, simd_host_tag>;
  using vector_type = host_vector_type;
  using value_type = std::int64_t;

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_HOST(vector_type, set1, _mm256_set1_epi64x(value_type(v)))

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_HOST(vector_type, gen, _mm256_setr_epi64x(
                                                          v(std::integral_constant<simd_size_t, 0>()),
                                                          v(std::integral_constant<simd_size_t, 1>()),
                                                          v(std::integral_constant<simd_size_t, 2>()),
                                                          v(std::integral_constant<simd_size_t, 3>())))

  KOKKOS_SIMD_IMPL_DEFINE_LOAD_HOST(vector_type, load, const value_type*, 
    if constexpr (std::is_same_v<simd_flags<Flags...>,
                                 simd_flags<simd_alignment_vector_aligned>>) {
      return _mm256_load_si256(reinterpret_cast<const vector_type*>(ptr));
    } else {
      return _mm256_loadu_si256(reinterpret_cast<const vector_type*>(ptr));
    }
  )

  KOKKOS_SIMD_IMPL_DEFINE_MASKED_LOAD_HOST(vector_type, masked_load, const value_type*, vector_type, return _mm256_maskload_epi64(reinterpret_cast<long long const*>(ptr), mask);)

  KOKKOS_SIMD_IMPL_DEFINE_STORE_HOST(store, value_type*, vector_type, 
  if constexpr (std::is_same_v<simd_flags<Flags...>,
                               simd_flags<simd_alignment_vector_aligned>>) {
    _mm256_store_si256(reinterpret_cast<vector_type*>(ptr), v);
  } else {
    _mm256_storeu_si256(reinterpret_cast<vector_type*>(ptr), v);
  }
  )

  KOKKOS_SIMD_IMPL_DEFINE_MASKED_STORE_HOST(masked_store, value_type*, vector_type, vector_type,
    _mm256_maskstore_epi64(reinterpret_cast<long long int*>(ptr), mask, v);
  )

  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_HOST(vector_type, convert_from, std::int32_t, abi_type, _mm256_cvtepi32_epi64(v))
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_HOST(vector_type, convert_from, std::uint64_t, abi_type, v)
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DECL_HOST(abi_type)

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static value_type extract(vector_type v, simd_size_t i) {
    switch (i) {
      case 0: return _mm256_extract_epi64(v, 0x0);
      case 1: return _mm256_extract_epi64(v, 0x1);
      case 2: return _mm256_extract_epi64(v, 0x2);
      case 3: return _mm256_extract_epi64(v, 0x3);
      default: Kokkos::abort("Index out of bound"); break;
    }
#if defined(KOKKOS_COMPILER_NVCC) && (KOKKOS_COMPILER_NVCC >= 1290) && \
    defined(KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK)
    return value_type{};
#endif
  }

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_HOST(vector_type, neg, _mm256_sub_epi64(set1(0), v))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_HOST(vector_type, bnot, _mm256_andnot_si256(v, set1(~value_type(0))))

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, plus, _mm256_add_epi64(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, minus, _mm256_sub_epi64(lhs, rhs))

  // fallback basic_simd multiplication using generator constructor
  // multiplying vectors of 64-bit signed integers is not available in AVX2
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, multiply, gen([&](simd_size_t i) { return extract(lhs, i) * extract(rhs, i); }))

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, band, _mm256_and_si256(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, bor, _mm256_or_si256(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, bxor, _mm256_xor_si256(lhs, rhs))

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, sll, _mm256_sllv_epi64(lhs, rhs))

  // fallback basic_simd shift right arithmetic using generator constructor
  // Shift right arithmetic for 64bit packed ints is not availalbe in AVX2
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, sra, gen([&](simd_size_t i) { return extract(lhs, i) >> extract(rhs, i); }))
  KOKKOS_SIMD_IMPL_DEFINE_SHIFT_SCALAR_HOST(vector_type, sll, _mm256_slli_epi64(lhs, rhs))

  // fallback basic_simd shift right arithmetic using generator constructor
  // Shift right arithmetic for 64bit packed ints is not availalbe in AVX2
  KOKKOS_SIMD_IMPL_DEFINE_SHIFT_SCALAR_HOST(vector_type, sra, gen([&](simd_size_t i) { return extract(lhs, i) >> rhs; }))

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, eq, _mm256_cmpeq_epi64(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, neq, bnot(eq(lhs, rhs)))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, gt, _mm256_cmpgt_epi64(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, ge, bor(gt(lhs, rhs), eq(lhs, rhs)))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, lt, gt(rhs, lhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, le, bor(lt(lhs, rhs), eq(lhs, rhs)))
  // KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, copysign, _mm256_xor_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.0), static_cast<vector_type>(a)),
  //                   _mm256_and_ps(_mm256_set1_ps(-0.0), static_cast<vector_type>(b))))

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(vector_type, abs, 
    gen([&](simd_size_t i) {
      auto a = extract(v, i);
      return (a < 0) ? -a : a;
    })
  )

  using rounded_data_type = double;
  using rounded_type = simd_vector_t<rounded_data_type, simd_abi::avx2_fixed_size<4>, simd_host_tag>;
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(rounded_type, floor, _mm256_setr_pd(extract(v,0),extract(v,1),extract(v,2),extract(v,3)))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(rounded_type, ceil, _mm256_setr_pd(extract(v,0),extract(v,1),extract(v,2),extract(v,3)))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(rounded_type, round, _mm256_setr_pd(extract(v,0),extract(v,1),extract(v,2),extract(v,3)))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(rounded_type, trunc, _mm256_setr_pd(extract(v,0),extract(v,1),extract(v,2),extract(v,3)))

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_MATH_OP_HOST(vector_type, max, _mm256_blendv_epi8(lhs, rhs, gt(rhs, lhs)))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_MATH_OP_HOST(vector_type, min, _mm256_blendv_epi8(lhs, rhs, lt(rhs, lhs)))

  KOKKOS_SIMD_IMPL_DEFINE_TERNARY_MATH_OP_HOST(vector_type, condition, _mm256_castpd_si256(
                                                                        _mm256_blendv_pd(_mm256_castsi256_pd(c),
                                                                                      _mm256_castsi256_pd(b),
                                                                                      _mm256_castsi256_pd(a))))

  KOKKOS_SIMD_IMPL_DEFINE_GATHER_FROM_FN_HOST(vector_type, unchecked, _mm256_i32gather_epi64(
          reinterpret_cast<long long const*>(Ranges::data(in)), indices, 8))
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_GATHER_FROM_FN_HOST(vector_type, unchecked, _mm256_mask_i32gather_epi64(
          _mm256_set1_epi64x(value_type{}),
          reinterpret_cast<long long const*>(Ranges::data(in)), indices,
          mmask, 8))

  KOKKOS_SIMD_IMPL_DEFINE_GATHER_FROM_FN_HOST(vector_type, partial, unchecked_gather_from(in, indices, flag))
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_GATHER_FROM_FN_HOST(vector_type, partial, unchecked_gather_from(in, indices, mmask, flag))
};

template <>
struct simd_native_ops<std::uint64_t, simd_abi::avx2_fixed_size<4>, simd_host_tag> {
  using abi_type = simd_abi::avx2_fixed_size<4>;
  using host_vector_type = simd_vector_t<std::uint64_t, abi_type, simd_host_tag>;
  using vector_type = host_vector_type;
  using value_type = std::uint64_t;

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_HOST(vector_type, set1, _mm256_set1_epi64x(Kokkos::bit_cast<std::int64_t>(value_type(v))))

  KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_HOST(vector_type, gen, _mm256_setr_epi64x(
                                                          v(std::integral_constant<simd_size_t, 0>()),
                                                          v(std::integral_constant<simd_size_t, 1>()),
                                                          v(std::integral_constant<simd_size_t, 2>()),
                                                          v(std::integral_constant<simd_size_t, 3>())))

  KOKKOS_SIMD_IMPL_DEFINE_LOAD_HOST(vector_type, load, const value_type*, 
    if constexpr (std::is_same_v<simd_flags<Flags...>,
                                 simd_flags<simd_alignment_vector_aligned>>) {
      return _mm256_load_si256(reinterpret_cast<const vector_type*>(ptr));
    } else {
      return _mm256_loadu_si256(reinterpret_cast<const vector_type*>(ptr));
    }
  )

  KOKKOS_SIMD_IMPL_DEFINE_MASKED_LOAD_HOST(vector_type, masked_load, const value_type*, vector_type, return _mm256_maskload_epi64(reinterpret_cast<long long const*>(ptr), mask);)

  KOKKOS_SIMD_IMPL_DEFINE_STORE_HOST(store, value_type*, vector_type, 
  if constexpr (std::is_same_v<simd_flags<Flags...>,
                               simd_flags<simd_alignment_vector_aligned>>) {
    _mm256_store_si256(reinterpret_cast<vector_type*>(ptr), v);
  } else {
    _mm256_storeu_si256(reinterpret_cast<vector_type*>(ptr), v);
  }
  )

  KOKKOS_SIMD_IMPL_DEFINE_MASKED_STORE_HOST(masked_store, value_type*, vector_type, vector_type,
    _mm256_maskstore_epi64(reinterpret_cast<long long int*>(ptr), mask, v);
  )

  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_HOST(vector_type, convert_from, std::int32_t, abi_type, _mm256_cvtepi32_epi64(v))
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_HOST(vector_type, convert_from, std::int64_t, abi_type, v)
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DECL_HOST(abi_type)

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  static value_type extract(vector_type v, simd_size_t i) {
    switch (i) {
      case 0: return _mm256_extract_epi64(v, 0x0);
      case 1: return _mm256_extract_epi64(v, 0x1);
      case 2: return _mm256_extract_epi64(v, 0x2);
      case 3: return _mm256_extract_epi64(v, 0x3);
      default: Kokkos::abort("Index out of bound"); break;
    }
#if defined(KOKKOS_COMPILER_NVCC) && (KOKKOS_COMPILER_NVCC >= 1290) && \
    defined(KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK)
    return value_type{};
#endif
  }

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_HOST(vector_type, neg, _mm256_sub_epi64(set1(0), v))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_HOST(vector_type, bnot, _mm256_andnot_si256(v, set1(~value_type(0))))

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, plus, _mm256_add_epi64(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, minus, _mm256_sub_epi64(lhs, rhs))

  // fallback basic_simd multiplication using generator constructor
  // multiplying vectors of 64-bit signed integers is not available in AVX2
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, multiply, gen([&](simd_size_t i) { return extract(lhs, i) * extract(rhs, i); }))

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, band, _mm256_and_si256(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, bor, _mm256_or_si256(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, bxor, _mm256_xor_si256(lhs, rhs))

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, sll, _mm256_sllv_epi64(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, sra, _mm256_srlv_epi64(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_SHIFT_SCALAR_HOST(vector_type, sll, _mm256_slli_epi64(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_SHIFT_SCALAR_HOST(vector_type, sra, _mm256_srli_epi64(lhs, rhs))

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, eq, _mm256_cmpeq_epi64(lhs, rhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, neq, bxor(eq(lhs, rhs), set1(-1));)

  using signed_t = simd_native_ops<std::int64_t, abi_type, simd_host_tag>;
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, gt, 
      // We use the following bit trick to compute the unsigned comparison from
    // the signed values since there is no intrinsic for unsigned int comparison
    // in AVX2: (a < 0) ^ (b < 0) ^ (a > b)
    // If a and b have the same sign, the signed and unsigned comparison will
    // give the same result. In this case:
    //  (a < 0) == (b < 0) => (a < 0) ^ (b < 0) = 0, and 0 ^ (a > b) == (a > b)
    // If they have different signs, the signed and unsigned comparisons will
    // give opposite results (negative values are higher than positive values
    // when interpreting them as unsigned). In that case:
    //  (a < 0) != (b < 0) => (a < 0) ^ (b < 0) = 1, and 1 ^ (a > b) == !(a > b)
    signed_t::bxor(signed_t::bxor(signed_t::lt(lhs, signed_t::set1(0)), signed_t::lt(rhs, signed_t::set1(0))), signed_t::gt(lhs, rhs));
  )

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, lt, gt(rhs, lhs))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, ge, bnot(lt(lhs, rhs)))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, le, bnot(gt(lhs, rhs)))
  // KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(vector_type, copysign, _mm256_xor_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.0), static_cast<vector_type>(a)),
  //                   _mm256_and_ps(_mm256_set1_ps(-0.0), static_cast<vector_type>(b))))

  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(vector_type, abs, v)

  using rounded_data_type = double;
  using rounded_type = simd_vector_t<rounded_data_type, abi_type, simd_host_tag>;
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(rounded_type, floor, _mm256_setr_pd(extract(v,0),extract(v,1),extract(v,2),extract(v,3)))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(rounded_type, ceil, _mm256_setr_pd(extract(v,0),extract(v,1),extract(v,2),extract(v,3)))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(rounded_type, round, _mm256_setr_pd(extract(v,0),extract(v,1),extract(v,2),extract(v,3)))
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(rounded_type, trunc, _mm256_setr_pd(extract(v,0),extract(v,1),extract(v,2),extract(v,3)))

  KOKKOS_SIMD_IMPL_DEFINE_BINARY_MATH_OP_HOST(vector_type, max, _mm256_blendv_epi8(lhs, rhs, gt(rhs, lhs)))
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_MATH_OP_HOST(vector_type, min, _mm256_blendv_epi8(lhs, rhs, lt(rhs, lhs)))

  KOKKOS_SIMD_IMPL_DEFINE_TERNARY_MATH_OP_HOST(vector_type, condition, _mm256_castpd_si256(
                                                                        _mm256_blendv_pd(_mm256_castsi256_pd(c),
                                                                                      _mm256_castsi256_pd(b),
                                                                                      _mm256_castsi256_pd(a))))

  KOKKOS_SIMD_IMPL_DEFINE_GATHER_FROM_FN_HOST(vector_type, unchecked, _mm256_i32gather_epi64(
          reinterpret_cast<long long const*>(Ranges::data(in)), indices, 8))
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_GATHER_FROM_FN_HOST(vector_type, unchecked, _mm256_mask_i32gather_epi64(
          _mm256_set1_epi64x(value_type{}),
          reinterpret_cast<long long const*>(Ranges::data(in)), indices,
          mmask, 8))

  KOKKOS_SIMD_IMPL_DEFINE_GATHER_FROM_FN_HOST(vector_type, partial, unchecked_gather_from(in, indices, flag))
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_GATHER_FROM_FN_HOST(vector_type, partial, unchecked_gather_from(in, indices, mmask, flag))
};

KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DEFN_HOST(simd_native_ops, double, simd_abi::avx2_fixed_size<4>)
KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DEFN_HOST(simd_native_ops, float, simd_abi::avx2_fixed_size<4>)
KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DEFN_HOST(simd_native_ops, float, simd_abi::avx2_fixed_size<8>)
KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DEFN_HOST(simd_native_ops, std::int32_t, simd_abi::avx2_fixed_size<4>)
KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DEFN_HOST(simd_native_ops, std::int32_t, simd_abi::avx2_fixed_size<8>)
KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DEFN_HOST(simd_native_ops, std::int64_t, simd_abi::avx2_fixed_size<4>)
KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DEFN_HOST(simd_native_ops, std::uint64_t, simd_abi::avx2_fixed_size<4>)

#endif

}
}

#endif