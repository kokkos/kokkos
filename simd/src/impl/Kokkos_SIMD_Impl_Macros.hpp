// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_SIMD_IMPL_MACROS_HPP
#define KOKKOS_SIMD_IMPL_MACROS_HPP

#define KOKKOS_SIMD_IMPL_FN_PREFIX_FN(fn) impl_##fn

#if (defined(KOKKOS_ENABLE_CUDA) && defined(__CUDA_ARCH__)) ||         \
    (defined(KOKKOS_ENABLE_HIP) && defined(__HIP_DEVICE_COMPILE__)) || \
    (defined(KOKKOS_ENABLE_SYCL) && defined(__SYCL_DEVICE_ONLY__))
#define KOKKOS_SIMD_IMPL_DEVICE_SIMD
#endif

#define KOKKOS_SIMD_IMPL_UNARY_OPERATOR(OP, FN_NAME)                           \
  KOKKOS_FORCEINLINE_FUNCTION constexpr auto operator OP() const noexcept {    \
    KOKKOS_IF_ON_HOST((if constexpr (requires(Derived d) {                     \
                                       d.KOKKOS_SIMD_IMPL_FN_PREFIX_FN(        \
                                           FN_NAME)();                         \
                                     }) {                                      \
      return static_cast<Derived const*>(this)->KOKKOS_SIMD_IMPL_FN_PREFIX_FN( \
          FN_NAME)();                                                          \
    }))                                                                        \
    return Derived{};                                                          \
  }

#define KOKKOS_SIMD_IMPL_SUBSCRIPT_OPERATOR(OP, FN_NAME, ARG_TYPE)             \
  KOKKOS_FORCEINLINE_FUNCTION constexpr auto operator OP(                      \
      [[maybe_unused]] ARG_TYPE arg) const {                                   \
    KOKKOS_IF_ON_HOST((if constexpr (requires(Derived d) {                     \
                                       d.KOKKOS_SIMD_IMPL_FN_PREFIX_FN(        \
                                           FN_NAME)(arg);                      \
                                     }) {                                      \
      return static_cast<Derived const*>(this)->KOKKOS_SIMD_IMPL_FN_PREFIX_FN( \
          FN_NAME)(arg);                                                       \
    }))                                                                        \
    return typename Derived::value_type{};                                     \
  }

#define KOKKOS_SIMD_IMPL_BINARY_OPERATOR(OP, FN_NAME, ARG_TYPE1, ARG_TYPE2) \
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr auto operator OP(            \
      [[maybe_unused]] ARG_TYPE1 lhs,                                       \
      [[maybe_unused]] ARG_TYPE2 rhs) noexcept {                            \
    KOKKOS_IF_ON_HOST((                                                     \
        if constexpr (requires {                                            \
                        lhs.KOKKOS_SIMD_IMPL_FN_PREFIX_FN(FN_NAME)(rhs);    \
                      }) {                                                  \
          return lhs.KOKKOS_SIMD_IMPL_FN_PREFIX_FN(FN_NAME)(rhs);           \
        } else {                                                            \
          return Derived([&](simd_size_t i) { return lhs[i] OP rhs[i]; });  \
        }))                                                                 \
    return Derived{};                                                       \
  }

#define KOKKOS_SIMD_IMPL_SHIFT_SCALAR_OPERATOR(OP, FN_NAME, ARG_TYPE1,   \
                                               ARG_TYPE2)                \
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr auto operator OP(         \
      [[maybe_unused]] ARG_TYPE1 lhs,                                    \
      [[maybe_unused]] ARG_TYPE2 rhs) noexcept {                         \
    KOKKOS_IF_ON_HOST((                                                  \
        if constexpr (requires {                                         \
                        lhs.KOKKOS_SIMD_IMPL_FN_PREFIX_FN(FN_NAME)(rhs); \
                      }) {                                               \
          return lhs.KOKKOS_SIMD_IMPL_FN_PREFIX_FN(FN_NAME)(rhs);        \
        } else {                                                         \
          return Derived([&](simd_size_t i) { return lhs[i] OP rhs; });  \
        }))                                                              \
    return Derived{};                                                    \
  }

#define KOKKOS_SIMD_IMPL_COMPOUND_ASSIGNMENT_OPERATOR(OP, FN_NAME, ARG_TYPE)   \
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr auto& operator OP(              \
      [[maybe_unused]] ARG_TYPE lhs, [[maybe_unused]] ARG_TYPE rhs) noexcept { \
    KOKKOS_IF_ON_HOST((                                                        \
        if constexpr (requires {                                               \
                        lhs.KOKKOS_SIMD_IMPL_FN_PREFIX_FN(FN_NAME)(rhs);       \
                      }) {                                                     \
          return lhs.KOKKOS_SIMD_IMPL_FN_PREFIX_FN(FN_NAME)(rhs);              \
        } else {                                                               \
          lhs = lhs OP rhs;                                                    \
          return lhs;                                                          \
        }))                                                                    \
    return Derived{};                                                          \
  }

#define KOKKOS_SIMD_IMPL_SHIFT_SCALAR_ASSIGN_OPERATOR(OP, FN_NAME, ARG_TYPE1, \
                                                      ARG_TYPE2)              \
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr auto operator OP(              \
      [[maybe_unused]] ARG_TYPE1 lhs,                                         \
      [[maybe_unused]] ARG_TYPE2 rhs) noexcept {                              \
    KOKKOS_IF_ON_HOST((                                                       \
        if constexpr (requires {                                              \
                        lhs.KOKKOS_SIMD_IMPL_FN_PREFIX_FN(FN_NAME)(rhs);      \
                      }) {                                                    \
          return lhs.KOKKOS_SIMD_IMPL_FN_PREFIX_FN(FN_NAME)(rhs);             \
        } else {                                                              \
          lhs = lhs OP std::forward<ARG_TYPE2>(rhs);                          \
          return lhs;                                                         \
        }))                                                                   \
    return Derived{};                                                         \
  }

#define KOKKOS_SIMD_IMPL_COMPARISON_OPERATOR(OP, FN_NAME, ARG_TYPE)            \
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr auto operator OP(               \
      [[maybe_unused]] ARG_TYPE lhs, [[maybe_unused]] ARG_TYPE rhs) noexcept { \
    KOKKOS_IF_ON_HOST((if constexpr (requires {                                \
                                       lhs.KOKKOS_SIMD_IMPL_FN_PREFIX_FN(      \
                                           FN_NAME)(rhs);                      \
                                     }) {                                      \
      return lhs.KOKKOS_SIMD_IMPL_FN_PREFIX_FN(FN_NAME)(rhs);                  \
    }))                                                                        \
    if constexpr (std::is_same_v<typename Derived::value_type, bool>) {        \
      return Derived{};                                                        \
    } else {                                                                   \
      return typename Derived::mask_type{};                                    \
    }                                                                          \
  }

#define KOKKOS_SIMD_IMPL_UNARY_MATH_FUNCTION(FN_NAME, DATA_TYPE, ABI_TYPE,  \
                                             EXPR)                          \
  KOKKOS_FORCEINLINE_FUNCTION                                               \
  Experimental::basic_simd<DATA_TYPE, ABI_TYPE> FN_NAME(                    \
      [[maybe_unused]] Experimental::basic_simd<DATA_TYPE, ABI_TYPE> const& \
          a) noexcept {                                                     \
    KOKKOS_IF_ON_HOST((EXPR))                                               \
    KOKKOS_IF_ON_DEVICE(                                                    \
        (return Experimental::basic_simd<DATA_TYPE, ABI_TYPE>{};))          \
  }

#define KOKKOS_SIMD_IMPL_ROUNDING_FUNCTION(FN_NAME, RETURN_DATA_TYPE,          \
                                           INPUT_DATA_TYPE, ABI_TYPE, EXPR)    \
  KOKKOS_FORCEINLINE_FUNCTION                                                  \
  Experimental::basic_simd<RETURN_DATA_TYPE, ABI_TYPE> FN_NAME(                \
      [[maybe_unused]] Experimental::basic_simd<INPUT_DATA_TYPE,               \
                                                ABI_TYPE> const& a) noexcept { \
    KOKKOS_IF_ON_HOST((EXPR))                                                  \
    KOKKOS_IF_ON_DEVICE(                                                       \
        (return Experimental::basic_simd<RETURN_DATA_TYPE, ABI_TYPE>{};))      \
  }

#define KOKKOS_SIMD_IMPL_BINARY_MATH_FUNCTION(FN_NAME, DATA_TYPE, ABI_TYPE,    \
                                              EXPR)                            \
  KOKKOS_FORCEINLINE_FUNCTION                                                  \
  Experimental::basic_simd<DATA_TYPE, ABI_TYPE> FN_NAME(                       \
      [[maybe_unused]] Experimental::basic_simd<DATA_TYPE, ABI_TYPE> const& a, \
      [[maybe_unused]] Experimental::basic_simd<DATA_TYPE, ABI_TYPE> const&    \
          b) noexcept {                                                        \
    KOKKOS_IF_ON_HOST((EXPR))                                                  \
    KOKKOS_IF_ON_DEVICE(                                                       \
        (return Experimental::basic_simd<DATA_TYPE, ABI_TYPE>{};))             \
  }

#define KOKKOS_SIMD_IMPL_MASKED_BINARY_MATH_FUNCTION(FN_NAME, DATA_TYPE,       \
                                                     ABI_TYPE, EXPR)           \
  KOKKOS_FORCEINLINE_FUNCTION                                                  \
  Experimental::basic_simd<DATA_TYPE, ABI_TYPE> FN_NAME(                       \
      [[maybe_unused]] Experimental::basic_simd_mask<DATA_TYPE,                \
                                                     ABI_TYPE> const& a,       \
      [[maybe_unused]] Experimental::basic_simd<DATA_TYPE, ABI_TYPE> const& b, \
      [[maybe_unused]] Experimental::basic_simd<DATA_TYPE, ABI_TYPE> const&    \
          c) noexcept {                                                        \
    KOKKOS_IF_ON_HOST((EXPR))                                                  \
    KOKKOS_IF_ON_DEVICE(                                                       \
        (return Experimental::basic_simd<DATA_TYPE, ABI_TYPE>{};))             \
  }

#define KOKKOS_SIMD_IMPL_TERNARY_MATH_FUNCTION(FN_NAME, DATA_TYPE, ABI_TYPE,   \
                                               EXPR)                           \
  KOKKOS_FORCEINLINE_FUNCTION                                                  \
  Experimental::basic_simd<DATA_TYPE, ABI_TYPE> FN_NAME(                       \
      [[maybe_unused]] Experimental::basic_simd<DATA_TYPE, ABI_TYPE> const& a, \
      [[maybe_unused]] Experimental::basic_simd<DATA_TYPE, ABI_TYPE> const& b, \
      [[maybe_unused]] Experimental::basic_simd<DATA_TYPE, ABI_TYPE> const&    \
          c) noexcept {                                                        \
    KOKKOS_IF_ON_HOST((EXPR))                                                  \
    KOKKOS_IF_ON_DEVICE(                                                       \
        (return Experimental::basic_simd<DATA_TYPE, ABI_TYPE>{};))             \
  }

#define KOKKOS_SIMD_IMPL_LOAD_STORE_NATIVE_LOAD(PREFIX, DATA_TYPE, ABI_TYPE, \
                                                EXPR)                        \
  template <typename SimdType, typename... Flags>                            \
    requires std::same_as<typename SimdType::abi_type, ABI_TYPE>             \
  KOKKOS_FORCEINLINE_FUNCTION basic_simd<DATA_TYPE, ABI_TYPE>                \
      simd_##PREFIX##_load(                                                  \
          [[maybe_unused]] const DATA_TYPE* ptr,                             \
          [[maybe_unused]] simd_flags<Flags...> flag = simd_flag_default) {  \
    KOKKOS_IF_ON_HOST((EXPR))                                                \
    KOKKOS_IF_ON_DEVICE((return basic_simd<DATA_TYPE, ABI_TYPE>{};))         \
  }

#define KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_LOAD(PREFIX, DATA_TYPE, ABI_TYPE, \
                                                EXPR)                        \
  template <typename... Flags>                                               \
  KOKKOS_FORCEINLINE_FUNCTION basic_simd<DATA_TYPE, ABI_TYPE>                \
      simd_##PREFIX##_load(                                                  \
          [[maybe_unused]] const DATA_TYPE* ptr,                             \
          [[maybe_unused]] basic_simd_mask<DATA_TYPE, ABI_TYPE> const& mask, \
          [[maybe_unused]] simd_flags<Flags...> flag = simd_flag_default) {  \
    KOKKOS_IF_ON_HOST((EXPR))                                                \
    KOKKOS_IF_ON_DEVICE((return basic_simd<DATA_TYPE, ABI_TYPE>{};))         \
  }

#define KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_NATIVE_LOAD(PREFIX, DATA_TYPE,    \
                                                       ABI_TYPE, EXPR)       \
  template <typename SimdType, typename... Flags>                            \
    requires std::same_as<typename SimdType::abi_type, ABI_TYPE>             \
  KOKKOS_FORCEINLINE_FUNCTION basic_simd<DATA_TYPE, ABI_TYPE>                \
      simd_##PREFIX##_load(                                                  \
          [[maybe_unused]] const DATA_TYPE* ptr,                             \
          [[maybe_unused]] basic_simd_mask<DATA_TYPE, ABI_TYPE> const& mask, \
          [[maybe_unused]] simd_flags<Flags...> flag = simd_flag_default) {  \
    KOKKOS_IF_ON_HOST((EXPR))                                                \
    KOKKOS_IF_ON_DEVICE((return basic_simd<DATA_TYPE, ABI_TYPE>{};))         \
  }

#define KOKKOS_SIMD_IMPL_LOAD_STORE_NATIVE_STORE(PREFIX, DATA_TYPE, ABI_TYPE, \
                                                 EXPR)                        \
  template <typename... Flags>                                                \
  KOKKOS_FORCEINLINE_FUNCTION void simd_##PREFIX##_store(                     \
      [[maybe_unused]] basic_simd<DATA_TYPE, ABI_TYPE> const& simd,           \
      [[maybe_unused]] DATA_TYPE* ptr,                                        \
      [[maybe_unused]] simd_flags<Flags...> flag = simd_flag_default) {       \
    KOKKOS_IF_ON_HOST((EXPR))                                                 \
  }

#define KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_STORE(PREFIX, DATA_TYPE, ABI_TYPE, \
                                                 EXPR)                        \
  template <typename... Flags>                                                \
  KOKKOS_FORCEINLINE_FUNCTION void simd_##PREFIX##_store(                     \
      [[maybe_unused]] basic_simd<DATA_TYPE, ABI_TYPE> const& simd,           \
      [[maybe_unused]] DATA_TYPE* ptr,                                        \
      [[maybe_unused]] basic_simd_mask<DATA_TYPE, ABI_TYPE> const& mask,      \
      [[maybe_unused]] simd_flags<Flags...> flag = simd_flag_default) {       \
    KOKKOS_IF_ON_HOST((EXPR))                                                 \
  }

#define KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_GATHER_FROM(PREFIX, DATA_TYPE,    \
                                                    ABI_TYPE, EXPR)       \
  template <Impl::SimdVecType V, Impl::Ranges::contiguous_range R,        \
            Impl::SimdIntegral I, typename... Flags>                      \
    requires Impl::Ranges::sized_range<R> &&                              \
             std::same_as<V, basic_simd<DATA_TYPE, ABI_TYPE>>             \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr V PREFIX##_gather_from( \
      [[maybe_unused]] R&& in, [[maybe_unused]] const I& indices,         \
      [[maybe_unused]] simd_flags<Flags...> flag = simd_flag_default) {   \
    KOKKOS_IF_ON_HOST((EXPR))                                             \
    KOKKOS_IF_ON_DEVICE((return V{};))                                    \
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
  template <Impl::SimdVecType V, Impl::Ranges::contiguous_range R,        \
            Impl::SimdIntegral I, typename... Flags>                      \
    requires Impl::Ranges::sized_range<R> &&                              \
             std::same_as<V, basic_simd<DATA_TYPE, ABI_TYPE>>             \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr V PREFIX##_gather_from( \
      [[maybe_unused]] R&& in,                                            \
      [[maybe_unused]] const typename I::mask_type& mask,                 \
      [[maybe_unused]] const I& indices,                                  \
      [[maybe_unused]] simd_flags<Flags...> flag = simd_flag_default) {   \
    KOKKOS_IF_ON_HOST((EXPR))                                             \
    KOKKOS_IF_ON_DEVICE((return V{};))                                    \
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
  template <Impl::SimdVecType V, Impl::Ranges::contiguous_range R,          \
            Impl::SimdIntegral I, typename... Flags>                        \
    requires Impl::Ranges::sized_range<R> &&                                \
             std::same_as<V, basic_simd<DATA_TYPE, ABI_TYPE>>               \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr void PREFIX##_scatter_to( \
      [[maybe_unused]] const V& v, [[maybe_unused]] R&& out,                \
      [[maybe_unused]] const I& indices,                                    \
      [[maybe_unused]] simd_flags<Flags...> flag = simd_flag_default) {     \
    KOKKOS_IF_ON_HOST((EXPR))                                               \
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
  template <Impl::SimdVecType V, Impl::Ranges::contiguous_range R,          \
            Impl::SimdIntegral I, typename... Flags>                        \
    requires Impl::Ranges::sized_range<R> &&                                \
             std::same_as<V, basic_simd<DATA_TYPE, ABI_TYPE>>               \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr void PREFIX##_scatter_to( \
      [[maybe_unused]] const V& v, [[maybe_unused]] R&& out,                \
      [[maybe_unused]] const typename I::mask_type& mask,                   \
      [[maybe_unused]] const I& indices,                                    \
      [[maybe_unused]] simd_flags<Flags...> flag = simd_flag_default) {     \
    KOKKOS_IF_ON_HOST((EXPR))                                               \
  }

#define KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_UNCHECKED_SCATTER_TO_WITH_MASK(      \
    DATA_TYPE, ABI_TYPE, EXPR)                                               \
  KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_SCATTER_TO_WITH_MASK(unchecked, DATA_TYPE, \
                                                       ABI_TYPE, EXPR)

#define KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_PARTIAL_SCATTER_TO_WITH_MASK(      \
    DATA_TYPE, ABI_TYPE, EXPR)                                             \
  KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_SCATTER_TO_WITH_MASK(partial, DATA_TYPE, \
                                                       ABI_TYPE, EXPR)

#endif
