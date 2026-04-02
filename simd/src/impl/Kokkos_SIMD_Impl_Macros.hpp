// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_SIMD_IMPL_MACROS_HPP
#define KOKKOS_SIMD_IMPL_MACROS_HPP

#define KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_GATHER_FROM(PREFIX, DATA_TYPE,    \
                                                    ABI_TYPE, EXPR)       \
  template <Impl::SimdVecType V, std::ranges::contiguous_range R,         \
            Impl::SimdIntegral I, typename... Flags>                      \
    requires std::ranges::sized_range<R> &&                               \
             std::same_as<V, basic_simd<DATA_TYPE, ABI_TYPE>>             \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr V PREFIX##_gather_from( \
      R&& in, const I& indices,                                           \
      [[maybe_unused]] simd_flags<Flags...> flag = simd_flag_default) {   \
    EXPR                                                                  \
  }

#define KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_UNCHECKED_GATHER_FROM(DATA_TYPE,      \
                                                              ABI_TYPE, EXPR) \
  KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_GATHER_FROM(unchecked, DATA_TYPE, ABI_TYPE, \
                                              EXPR)

#define KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_PARTIAL_GATHER_FROM(DATA_TYPE,      \
                                                            ABI_TYPE, EXPR) \
  KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_GATHER_FROM(partial, DATA_TYPE, ABI_TYPE, \
                                              EXPR)

#define KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_GATHER_FROM_WITH_MASK(            \
    PREFIX, DATA_TYPE, ABI_TYPE, EXPR)                                    \
  template <Impl::SimdVecType V, std::ranges::contiguous_range R,         \
            Impl::SimdIntegral I, typename... Flags>                      \
    requires std::ranges::sized_range<R> &&                               \
             std::same_as<V, basic_simd<DATA_TYPE, ABI_TYPE>>             \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr V PREFIX##_gather_from( \
      R&& in, const typename I::mask_type& mask, const I& indices,        \
      [[maybe_unused]] simd_flags<Flags...> flag = simd_flag_default) {   \
    EXPR                                                                  \
  }

#define KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_UNCHECKED_GATHER_FROM_WITH_MASK(      \
    DATA_TYPE, ABI_TYPE, EXPR)                                                \
  KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_GATHER_FROM_WITH_MASK(unchecked, DATA_TYPE, \
                                                        ABI_TYPE, EXPR)

#define KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_PARTIAL_GATHER_FROM_WITH_MASK(      \
    DATA_TYPE, ABI_TYPE, EXPR)                                              \
  KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_GATHER_FROM_WITH_MASK(partial, DATA_TYPE, \
                                                        ABI_TYPE, EXPR)

#define KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_SCATTER_TO(PREFIX, DATA_TYPE,       \
                                                   ABI_TYPE, EXPR)          \
  template <Impl::SimdVecType V, std::ranges::contiguous_range R,           \
            Impl::SimdIntegral I, typename... Flags>                        \
    requires std::ranges::sized_range<R> &&                                 \
             std::same_as<V, basic_simd<DATA_TYPE, ABI_TYPE>>               \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr void PREFIX##_scatter_to( \
      const V& v, R&& out, const I& indices,                                \
      [[maybe_unused]] simd_flags<Flags...> flag = simd_flag_default) {     \
    EXPR                                                                    \
  }

#define KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_UNCHECKED_SCATTER_TO(DATA_TYPE,      \
                                                             ABI_TYPE, EXPR) \
  KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_SCATTER_TO(unchecked, DATA_TYPE, ABI_TYPE, \
                                             EXPR)

#define KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_PARTIAL_SCATTER_TO(DATA_TYPE,      \
                                                           ABI_TYPE, EXPR) \
  KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_SCATTER_TO(partial, DATA_TYPE, ABI_TYPE, EXPR)

#define KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_SCATTER_TO_WITH_MASK(               \
    PREFIX, DATA_TYPE, ABI_TYPE, EXPR)                                      \
  template <Impl::SimdVecType V, std::ranges::contiguous_range R,           \
            Impl::SimdIntegral I, typename... Flags>                        \
    requires std::ranges::sized_range<R> &&                                 \
             std::same_as<V, basic_simd<DATA_TYPE, ABI_TYPE>>               \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr void PREFIX##_scatter_to( \
      const V& v, R&& out, const typename I::mask_type& mask,               \
      const I& indices,                                                     \
      [[maybe_unused]] simd_flags<Flags...> flag = simd_flag_default) {     \
    EXPR                                                                    \
  }

#define KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_UNCHECKED_SCATTER_TO_WITH_MASK(      \
    DATA_TYPE, ABI_TYPE, EXPR)                                               \
  KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_SCATTER_TO_WITH_MASK(unchecked, DATA_TYPE, \
                                                       ABI_TYPE, EXPR)

#define KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_PARTIAL_SCATTER_TO_WITH_MASK(      \
    DATA_TYPE, ABI_TYPE, EXPR)                                             \
  KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_SCATTER_TO_WITH_MASK(partial, DATA_TYPE, \
                                                       ABI_TYPE, EXPR)

// These operators could be defined once as template functions over the simd
// type. However, defining them as non-template functions inside each of the
// simd classes allows passing a scalar of type T as second argument and relying
// on implicit conversions from T to simd<T> to get operator@=(simd<T>, T)
// (which is convenient, but not part of the C++ standard's API) for "free".
#define KOKKOS_SIMD_IMPL_COMPOUND_OPERATORS_IMPLEMENTATION             \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend basic_simd& operator+=( \
      basic_simd& lhs, basic_simd const& rhs) {                        \
    lhs = lhs + rhs;                                                   \
    return lhs;                                                        \
  }                                                                    \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend basic_simd& operator-=( \
      basic_simd& lhs, basic_simd const& rhs) {                        \
    lhs = lhs - rhs;                                                   \
    return lhs;                                                        \
  }                                                                    \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend basic_simd& operator*=( \
      basic_simd& lhs, basic_simd const& rhs) {                        \
    lhs = lhs * rhs;                                                   \
    return lhs;                                                        \
  }                                                                    \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend basic_simd& operator/=( \
      basic_simd& lhs, basic_simd const& rhs) {                        \
    lhs = lhs / rhs;                                                   \
    return lhs;                                                        \
  }

#define KOKKOS_SIMD_IMPL_INTEGRAL_COMPOUND_OPERATORS_IMPLEMENTATION     \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend basic_simd& operator&=(  \
      basic_simd& lhs, basic_simd const& rhs) {                         \
    lhs = lhs & rhs;                                                    \
    return lhs;                                                         \
  }                                                                     \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend basic_simd& operator|=(  \
      basic_simd& lhs, basic_simd const& rhs) {                         \
    lhs = lhs | rhs;                                                    \
    return lhs;                                                         \
  }                                                                     \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend basic_simd& operator^=(  \
      basic_simd& lhs, basic_simd const& rhs) {                         \
    lhs = lhs ^ rhs;                                                    \
    return lhs;                                                         \
  }                                                                     \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend basic_simd& operator>>=( \
      basic_simd& lhs, basic_simd const& rhs) {                         \
    lhs = lhs >> rhs;                                                   \
    return lhs;                                                         \
  }                                                                     \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend basic_simd& operator<<=( \
      basic_simd& lhs, basic_simd const& rhs) {                         \
    lhs = lhs << rhs;                                                   \
    return lhs;                                                         \
  }

#endif
