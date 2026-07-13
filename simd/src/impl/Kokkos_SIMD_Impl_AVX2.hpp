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

KOKKOS_SIMD_IMPL_SIMD_HOST_VECTOR_IMPL(double, simd_abi::avx2_fixed_size<4>, __m256d)
KOKKOS_SIMD_IMPL_SIMD_HOST_VECTOR_IMPL(float, simd_abi::avx2_fixed_size<4>, __m128)
KOKKOS_SIMD_IMPL_SIMD_HOST_VECTOR_IMPL(std::int32_t, simd_abi::avx2_fixed_size<4>, __m128i)
KOKKOS_SIMD_IMPL_SIMD_HOST_VECTOR_IMPL(std::int32_t, simd_abi::avx2_fixed_size<8>, __m256i)
KOKKOS_SIMD_IMPL_SIMD_HOST_VECTOR_IMPL(std::int64_t, simd_abi::avx2_fixed_size<4>, __m256i)
KOKKOS_SIMD_IMPL_SIMD_HOST_VECTOR_IMPL(std::uint64_t, simd_abi::avx2_fixed_size<4>, __m256i)

template <typename T, typename Abi>
struct simd_vector_impl<T, Abi, simd_device_tag> {
  using host_type = typename simd_vector_impl<T, Abi, simd_host_tag>::type;
  using type = Kokkos::Array<char, sizeof(host_type)>;
};

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

  KOKKOS_SIMD_IMPL_NATIVE_FN_DEVICE(vector_type, set1, value_type, vector_type{})

  KOKKOS_SIMD_IMPL_NATIVE_GEN_FN_DEVICE(vector_type, gen, vector_type{})

  template <typename T, typename Abi>
  KOKKOS_SIMD_IMPL_NATIVE_CONVERSION_FN_DEVICE(vector_type, convert_from, T, Abi, vector_type{})

  KOKKOS_SIMD_IMPL_NATIVE_EXTRACT_FN_DEVICE(value_type, extract, return v[i])

  KOKKOS_SIMD_IMPL_NATIVE_UNARY_OP_DEVICE(vector_type, lnot, vector_type{})
  KOKKOS_SIMD_IMPL_NATIVE_UNARY_OP_DEVICE(vector_type, bnot, lnot(v))

  KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_DEVICE(vector_type, land, vector_type{})
  KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_DEVICE(vector_type, lor, vector_type{})
  KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_DEVICE(vector_type, band, vector_type{})
  KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_DEVICE(vector_type, bor, vector_type{})
  KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_DEVICE(vector_type, bxor, vector_type{})

  KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_DEVICE(vector_type, eq, vector_type{})
  KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_DEVICE(vector_type, neq, vector_type{})
};

// host
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
template <>
struct simd_mask_native_ops<double, simd_abi::avx2_fixed_size<4>, simd_host_tag> {
  using vector_type = simd_vector_t<double, simd_abi::avx2_fixed_size<4>, simd_host_tag>;
  using value_type = bool;

  KOKKOS_SIMD_IMPL_NATIVE_FN_HOST(vector_type, set1, value_type, _mm256_castsi256_pd(_mm256_set1_epi64x(-std::int64_t(v))))

  KOKKOS_SIMD_IMPL_NATIVE_GEN_FN_HOST(vector_type, gen, _mm256_castsi256_pd(_mm256_setr_epi64x(
                                                            -std::int64_t(v(std::integral_constant<simd_size_t, 0>())),
                                                            -std::int64_t(v(std::integral_constant<simd_size_t, 1>())),
                                                            -std::int64_t(v(std::integral_constant<simd_size_t, 2>())),
                                                            -std::int64_t(v(std::integral_constant<simd_size_t, 3>())))))

  KOKKOS_SIMD_IMPL_NATIVE_CONVERSION_FN_HOST(vector_type, convert_from, float, simd_abi::avx2_fixed_size<4>, _mm256_cvtps_pd(v))
  KOKKOS_SIMD_IMPL_NATIVE_CONVERSION_FN_HOST(vector_type, convert_from, std::int32_t, simd_abi::avx2_fixed_size<4>, _mm256_cvtepi32_pd(v))
  KOKKOS_SIMD_IMPL_NATIVE_CONVERSION_FN_HOST(vector_type, convert_from, std::int64_t, simd_abi::avx2_fixed_size<4>, _mm256_castsi256_pd(v))

  KOKKOS_SIMD_IMPL_NATIVE_EXTRACT_FN_HOST(value_type, extract, return (_mm256_movemask_pd(v) & (1 << i)) != 0)

  KOKKOS_SIMD_IMPL_NATIVE_UNARY_OP_HOST(vector_type, lnot, _mm256_andnot_pd(v, set1(true)))
  KOKKOS_SIMD_IMPL_NATIVE_UNARY_OP_HOST(vector_type, bnot, lnot(v);)

  KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_HOST(vector_type, land, band(lhs, rhs))
  KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_HOST(vector_type, lor, bor(lhs, rhs))
  KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_HOST(vector_type, band, _mm256_and_pd(lhs, rhs))
  KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_HOST(vector_type, bor, _mm256_or_pd(lhs, rhs))
  KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_HOST(vector_type, bxor, _mm256_xor_pd(lhs, rhs))

  KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_HOST(value_type, eq, (_mm256_movemask_pd(lhs) == _mm256_movemask_pd(rhs)))
  KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_HOST(value_type, neq, !eq(lhs, rhs))
};
#endif

template <typename T, typename Abi, typename Tag>
struct simd_native_ops;

// FIXME
// device
template <>
struct simd_native_ops<double, simd_abi::avx2_fixed_size<4>, simd_device_tag> {
  using vector_type = simd_vector_t<double, simd_abi::avx2_fixed_size<4>, simd_device_tag>;
  using value_type = double;

  KOKKOS_SIMD_IMPL_NATIVE_GEN_FN_DEVICE(vector_type, set1, vector_type{})

  KOKKOS_SIMD_IMPL_NATIVE_GEN_FN_DEVICE(vector_type, gen, vector_type{})
  
  KOKKOS_SIMD_IMPL_NATIVE_LOAD_DEVICE(vector_type, load, const value_type*, return vector_type{}; )
  KOKKOS_SIMD_IMPL_NATIVE_MASKED_LOAD_DEVICE(vector_type, masked_load, const value_type*, vector_type, return vector_type{}; )
  KOKKOS_SIMD_IMPL_NATIVE_STORE_DEVICE(store, value_type*, vector_type, {})
  KOKKOS_SIMD_IMPL_NATIVE_MASKED_STORE_DEVICE(masked_store, value_type*, vector_type, vector_type, {})

  template <typename T, typename Abi>
  KOKKOS_SIMD_IMPL_NATIVE_CONVERSION_FN_DEVICE(vector_type, convert_from, T, Abi, vector_type{})

  KOKKOS_SIMD_IMPL_NATIVE_EXTRACT_FN_DEVICE(value_type, extract, return v[i])

  KOKKOS_SIMD_IMPL_NATIVE_UNARY_OP_DEVICE(vector_type, neg, vector_type{})

  KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_DEVICE(vector_type, plus, vector_type{})
  KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_DEVICE(vector_type, minus, vector_type{})
  KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_DEVICE(vector_type, multiply, vector_type{})
  KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_DEVICE(vector_type, divide, vector_type{})
  KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_DEVICE(vector_type, eq, vector_type{})
  KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_DEVICE(vector_type, neq, vector_type{})
  KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_DEVICE(vector_type, ge, vector_type{})
  KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_DEVICE(vector_type, le, vector_type{})
  KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_DEVICE(vector_type, gt, vector_type{})
  KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_DEVICE(vector_type, lt, vector_type{})
  KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_DEVICE(vector_type, copysign, vector_type{})

  KOKKOS_SIMD_IMPL_NATIVE_UNARY_MATH_OP_DEVICE(vector_type, abs, vector_type{})
  KOKKOS_SIMD_IMPL_NATIVE_UNARY_MATH_OP_DEVICE(vector_type, floor, vector_type{})
  KOKKOS_SIMD_IMPL_NATIVE_UNARY_MATH_OP_DEVICE(vector_type, ceil, vector_type{})
  KOKKOS_SIMD_IMPL_NATIVE_UNARY_MATH_OP_DEVICE(vector_type, round, vector_type{})
  KOKKOS_SIMD_IMPL_NATIVE_UNARY_MATH_OP_DEVICE(vector_type, trunc, vector_type{})
  KOKKOS_SIMD_IMPL_NATIVE_UNARY_MATH_OP_DEVICE(vector_type, sqrt, vector_type{})

#ifdef KOKKOS_HAVE_INTEL_SVML
  KOKKOS_SIMD_IMPL_NATIVE_UNARY_MATH_OP_DEVICE(vector_type, cbrt, vector_type{})
  KOKKOS_SIMD_IMPL_NATIVE_UNARY_MATH_OP_DEVICE(vector_type, exp, vector_type{})
  KOKKOS_SIMD_IMPL_NATIVE_UNARY_MATH_OP_DEVICE(vector_type, log, vector_type{})
#endif

  KOKKOS_SIMD_IMPL_NATIVE_TERNARY_MATH_OP_DEVICE(vector_type, fma, vector_type{})
  KOKKOS_SIMD_IMPL_NATIVE_BINARY_MATH_OP_DEVICE(vector_type, max, vector_type{})
  KOKKOS_SIMD_IMPL_NATIVE_BINARY_MATH_OP_DEVICE(vector_type, min, vector_type{})
  KOKKOS_SIMD_IMPL_NATIVE_TERNARY_MATH_OP_DEVICE(vector_type, condition, vector_type{})
};

// host
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
template <>
struct simd_native_ops<double, simd_abi::avx2_fixed_size<4>, simd_host_tag> {
  using vector_type = simd_vector_t<double, simd_abi::avx2_fixed_size<4>, simd_host_tag>;
  using value_type = double;

  KOKKOS_SIMD_IMPL_NATIVE_GEN_FN_HOST(vector_type, set1, _mm256_set1_pd(value_type(v)))

  KOKKOS_SIMD_IMPL_NATIVE_GEN_FN_HOST(vector_type, gen, _mm256_setr_pd(
                                                          v(std::integral_constant<simd_size_t, 0>()),
                                                          v(std::integral_constant<simd_size_t, 1>()),
                                                          v(std::integral_constant<simd_size_t, 2>()),
                                                          v(std::integral_constant<simd_size_t, 3>())))

  KOKKOS_SIMD_IMPL_NATIVE_LOAD_HOST(vector_type, load, const value_type*, if constexpr (std::is_same_v<simd_flags<Flags...>,
                                 simd_flags<simd_alignment_vector_aligned>>) {
      return _mm256_load_pd(ptr);
    } else {
      return _mm256_loadu_pd(ptr);
    })

  KOKKOS_SIMD_IMPL_NATIVE_MASKED_LOAD_HOST(vector_type, masked_load, const value_type*, vector_type, return _mm256_maskload_pd(ptr, _mm256_castpd_si256(mask));)

  KOKKOS_SIMD_IMPL_NATIVE_STORE_HOST(store, value_type*, vector_type, 
    if constexpr (std::is_same_v<simd_flags<Flags...>,
                                simd_flags<simd_alignment_vector_aligned>>) {
      _mm256_store_pd(ptr, v);
    } else {
      _mm256_storeu_pd(ptr, v);
    }
  )

  KOKKOS_SIMD_IMPL_NATIVE_MASKED_STORE_HOST(masked_store, value_type*, vector_type, vector_type,
    _mm256_maskstore_pd(ptr, _mm256_castpd_si256(mask), v);
  )

  KOKKOS_SIMD_IMPL_NATIVE_CONVERSION_FN_HOST(vector_type, convert_from, float, simd_abi::avx2_fixed_size<4>, _mm256_cvtps_pd(v))
  KOKKOS_SIMD_IMPL_NATIVE_CONVERSION_FN_HOST(vector_type, convert_from, std::int32_t, simd_abi::avx2_fixed_size<4>, _mm256_cvtepi32_pd(v))


  KOKKOS_SIMD_IMPL_NATIVE_EXTRACT_FN_HOST(value_type, extract, constexpr auto size = sizeof(vector_type) / sizeof(value_type);
                                                                                  value_type tmp[size];
                                                                                  _mm256_storeu_pd(tmp, v);
                                                                                  return tmp[i];
                                          )

  KOKKOS_SIMD_IMPL_NATIVE_UNARY_OP_HOST(vector_type, neg, _mm256_sub_pd(_mm256_set1_pd(0.0), v))

  KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_HOST(vector_type, plus, _mm256_add_pd(lhs, rhs))
  KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_HOST(vector_type, minus, _mm256_sub_pd(lhs, rhs))
  KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_HOST(vector_type, multiply, _mm256_mul_pd(lhs, rhs))
  KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_HOST(vector_type, divide, _mm256_div_pd(lhs, rhs))
  KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_HOST(vector_type, eq, _mm256_cmp_pd(lhs, rhs, _CMP_EQ_OS))
  KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_HOST(vector_type, neq, _mm256_cmp_pd(lhs, rhs, _CMP_NEQ_OS))
  KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_HOST(vector_type, ge, _mm256_cmp_pd(lhs, rhs, _CMP_GE_OS))
  KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_HOST(vector_type, le, _mm256_cmp_pd(lhs, rhs, _CMP_LE_OS))
  KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_HOST(vector_type, gt, _mm256_cmp_pd(lhs, rhs, _CMP_GT_OS))
  KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_HOST(vector_type, lt, _mm256_cmp_pd(lhs, rhs, _CMP_LT_OS))
  KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_HOST(vector_type, copysign, _mm256_xor_pd(_mm256_andnot_pd(_mm256_set1_pd(-0.0), static_cast<vector_type>(lhs)),
                    _mm256_and_pd(_mm256_set1_pd(-0.0), static_cast<vector_type>(rhs))))

  KOKKOS_SIMD_IMPL_NATIVE_UNARY_MATH_OP_HOST(vector_type, abs, _mm256_andnot_pd(_mm256_set1_pd(-0.0), static_cast<vector_type>(v)))
  KOKKOS_SIMD_IMPL_NATIVE_UNARY_MATH_OP_HOST(vector_type, floor, _mm256_round_pd(static_cast<vector_type>(v),
                                                                (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)))
  KOKKOS_SIMD_IMPL_NATIVE_UNARY_MATH_OP_HOST(vector_type, ceil, _mm256_round_pd(static_cast<vector_type>(v),
                                                                (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)))
  KOKKOS_SIMD_IMPL_NATIVE_UNARY_MATH_OP_HOST(vector_type, round, _mm256_round_pd(static_cast<vector_type>(v),
                                                                (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)))
  KOKKOS_SIMD_IMPL_NATIVE_UNARY_MATH_OP_HOST(vector_type, trunc, _mm256_round_pd(static_cast<vector_type>(v),
                                                                (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)))
  KOKKOS_SIMD_IMPL_NATIVE_UNARY_MATH_OP_HOST(vector_type, sqrt, _mm256_sqrt_pd(static_cast<vector_type>(v)))

#ifdef KOKKOS_HAVE_INTEL_SVML
  KOKKOS_SIMD_IMPL_NATIVE_UNARY_MATH_OP_HOST(vector_type, cbrt, _mm256_cbrt_pd(static_cast<vector_type>(v)))
  KOKKOS_SIMD_IMPL_NATIVE_UNARY_MATH_OP_HOST(vector_type, exp, _mm256_exp_pd(static_cast<vector_type>(v)))
  KOKKOS_SIMD_IMPL_NATIVE_UNARY_MATH_OP_HOST(vector_type, log, _mm256_log_pd(static_cast<vector_type>(v)))
#endif

  KOKKOS_SIMD_IMPL_NATIVE_TERNARY_MATH_OP_HOST(vector_type, fma, _mm256_fmadd_pd(static_cast<vector_type>(a), static_cast<vector_type>(b),
                                                                                 static_cast<vector_type>(c)))

  KOKKOS_SIMD_IMPL_NATIVE_BINARY_MATH_OP_HOST(vector_type, max, _mm256_max_pd(static_cast<vector_type>(lhs), static_cast<vector_type>(rhs)))
  KOKKOS_SIMD_IMPL_NATIVE_BINARY_MATH_OP_HOST(vector_type, min, _mm256_min_pd(static_cast<vector_type>(lhs), static_cast<vector_type>(rhs)))

  KOKKOS_SIMD_IMPL_NATIVE_TERNARY_MATH_OP_HOST(vector_type, condition, _mm256_blendv_pd(static_cast<vector_type>(c), static_cast<vector_type>(b),
                                                                                        static_cast<vector_type>(a)))
};
#endif

}
}

#endif