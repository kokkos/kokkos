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

#define KOKKOS_SIMD_IMPL_UNARY_OPERATOR(OP, FN_NAME)                      \
  KOKKOS_FORCEINLINE_FUNCTION constexpr auto operator OP() const noexcept \
    requires requires(Derived d) {                                        \
      { d.KOKKOS_SIMD_IMPL_FN_PREFIX_FN(FN_NAME)() };                     \
    }                                                                     \
  {                                                                       \
    KOKKOS_IF_ON_HOST((return static_cast<Derived const*>(this)           \
                           ->KOKKOS_SIMD_IMPL_FN_PREFIX_FN(FN_NAME)();))  \
    KOKKOS_IF_ON_DEVICE((return Derived{};))                              \
  }

#define KOKKOS_SIMD_IMPL_SUBSCRIPT_OPERATOR(OP, FN_NAME, ARG_TYPE)           \
  KOKKOS_FORCEINLINE_FUNCTION constexpr auto operator OP(ARG_TYPE arg) const \
    requires requires(Derived d) {                                           \
      { d.KOKKOS_SIMD_IMPL_FN_PREFIX_FN(FN_NAME)(arg) };                     \
    }                                                                        \
  {                                                                          \
    KOKKOS_IF_ON_HOST((return static_cast<Derived const*>(this)              \
                           ->KOKKOS_SIMD_IMPL_FN_PREFIX_FN(FN_NAME)(arg);))  \
    KOKKOS_IF_ON_DEVICE((return typename Derived::value_type{};))            \
  }

#define KOKKOS_SIMD_IMPL_BINARY_OPERATOR(OP, FN_NAME, ARG_TYPE1, ARG_TYPE2) \
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr auto operator OP(            \
      ARG_TYPE1 lhs, ARG_TYPE2 rhs) noexcept                                \
    requires requires {                                                     \
      { lhs.KOKKOS_SIMD_IMPL_FN_PREFIX_FN(FN_NAME)(rhs) };                  \
    }                                                                       \
  {                                                                         \
    KOKKOS_IF_ON_HOST(                                                      \
        (return lhs.KOKKOS_SIMD_IMPL_FN_PREFIX_FN(FN_NAME)(rhs);))          \
    KOKKOS_IF_ON_DEVICE((return Derived{};))                                \
  }

#define KOKKOS_SIMD_IMPL_COMPOUND_ASSIGNMENT_OPERATOR(OP, FN_NAME, ARG_TYPE) \
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr auto& operator OP(            \
      ARG_TYPE lhs, ARG_TYPE rhs) noexcept                                   \
    requires requires {                                                      \
      { lhs.KOKKOS_SIMD_IMPL_FN_PREFIX_FN(FN_NAME)(rhs) };                   \
    }                                                                        \
  {                                                                          \
    KOKKOS_IF_ON_HOST(                                                       \
        (return lhs.KOKKOS_SIMD_IMPL_FN_PREFIX_FN(FN_NAME)(rhs);))           \
    KOKKOS_IF_ON_DEVICE((return Derived{};))                                 \
  }

#define KOKKOS_SIMD_IMPL_COMPARISON_OPERATOR(OP, FN_NAME, ARG_TYPE)         \
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr auto operator OP(            \
      ARG_TYPE lhs, ARG_TYPE rhs) noexcept                                  \
    requires requires {                                                     \
      { lhs.KOKKOS_SIMD_IMPL_FN_PREFIX_FN(FN_NAME)(rhs) };                  \
    }                                                                       \
  {                                                                         \
    KOKKOS_IF_ON_HOST(                                                      \
        (return lhs.KOKKOS_SIMD_IMPL_FN_PREFIX_FN(FN_NAME)(rhs);))          \
    KOKKOS_IF_ON_DEVICE((                                                   \
        if constexpr (std::is_same_v<typename Derived::value_type, bool>) { \
          return Derived{};                                                 \
        } else { return typename Derived::mask_type{}; }))                  \
  }

#define KOKKOS_SIMD_IMPL_UNARY_MATH_FUNCTION(FN_NAME, DATA_TYPE, ABI_TYPE, \
                                             EXPR)                         \
  KOKKOS_FORCEINLINE_FUNCTION                                              \
  Experimental::basic_simd<DATA_TYPE, ABI_TYPE> FN_NAME(                   \
      Experimental::basic_simd<DATA_TYPE, ABI_TYPE> const& a) noexcept {   \
    KOKKOS_IF_ON_HOST((EXPR))                                              \
    KOKKOS_IF_ON_DEVICE(                                                   \
        (return Experimental::basic_simd<DATA_TYPE, ABI_TYPE>{};))         \
  }

#define KOKKOS_SIMD_IMPL_ROUNDING_FUNCTION(FN_NAME, RETURN_DATA_TYPE,          \
                                           INPUT_DATA_TYPE, ABI_TYPE, EXPR)    \
  KOKKOS_FORCEINLINE_FUNCTION                                                  \
  Experimental::basic_simd<RETURN_DATA_TYPE, ABI_TYPE> FN_NAME(                \
      Experimental::basic_simd<INPUT_DATA_TYPE, ABI_TYPE> const& a) noexcept { \
    KOKKOS_IF_ON_HOST((EXPR))                                                  \
    KOKKOS_IF_ON_DEVICE(                                                       \
        (return Experimental::basic_simd<RETURN_DATA_TYPE, ABI_TYPE>{};))      \
  }

#define KOKKOS_SIMD_IMPL_BINARY_MATH_FUNCTION(FN_NAME, DATA_TYPE, ABI_TYPE, \
                                              EXPR)                         \
  KOKKOS_FORCEINLINE_FUNCTION                                               \
  Experimental::basic_simd<DATA_TYPE, ABI_TYPE> FN_NAME(                    \
      Experimental::basic_simd<DATA_TYPE, ABI_TYPE> const& a,               \
      Experimental::basic_simd<DATA_TYPE, ABI_TYPE> const& b) noexcept {    \
    KOKKOS_IF_ON_HOST((EXPR))                                               \
    KOKKOS_IF_ON_DEVICE(                                                    \
        (return Experimental::basic_simd<DATA_TYPE, ABI_TYPE>{};))          \
  }

#define KOKKOS_SIMD_IMPL_MASKED_BINARY_MATH_FUNCTION(FN_NAME, DATA_TYPE, \
                                                     ABI_TYPE, EXPR)     \
  KOKKOS_FORCEINLINE_FUNCTION                                            \
  Experimental::basic_simd<DATA_TYPE, ABI_TYPE> FN_NAME(                 \
      Experimental::basic_simd_mask<DATA_TYPE, ABI_TYPE> const& a,       \
      Experimental::basic_simd<DATA_TYPE, ABI_TYPE> const& b,            \
      Experimental::basic_simd<DATA_TYPE, ABI_TYPE> const& c) noexcept { \
    KOKKOS_IF_ON_HOST((EXPR))                                            \
    KOKKOS_IF_ON_DEVICE(                                                 \
        (return Experimental::basic_simd<DATA_TYPE, ABI_TYPE>{};))       \
  }

#define KOKKOS_SIMD_IMPL_TERNARY_MATH_FUNCTION(FN_NAME, DATA_TYPE, ABI_TYPE, \
                                               EXPR)                         \
  KOKKOS_FORCEINLINE_FUNCTION                                                \
  Experimental::basic_simd<DATA_TYPE, ABI_TYPE> FN_NAME(                     \
      Experimental::basic_simd<DATA_TYPE, ABI_TYPE> const& a,                \
      Experimental::basic_simd<DATA_TYPE, ABI_TYPE> const& b,                \
      Experimental::basic_simd<DATA_TYPE, ABI_TYPE> const& c) noexcept {     \
    KOKKOS_IF_ON_HOST((EXPR))                                                \
    KOKKOS_IF_ON_DEVICE(                                                     \
        (return Experimental::basic_simd<DATA_TYPE, ABI_TYPE>{};))           \
  }

#define KOKKOS_SIMD_IMPL_LOAD_STORE_NATIVE_LOAD(PREFIX, DATA_TYPE, ABI_TYPE, \
                                                EXPR)                        \
  template <typename SimdType, typename... Flags>                            \
    requires std::same_as<typename SimdType::abi_type, ABI_TYPE>             \
  KOKKOS_FORCEINLINE_FUNCTION basic_simd<DATA_TYPE, ABI_TYPE>                \
      simd_##PREFIX##_load(const DATA_TYPE* ptr,                             \
                           simd_flags<Flags...> flag = simd_flag_default) {  \
    KOKKOS_IF_ON_HOST((EXPR))                                                \
    KOKKOS_IF_ON_DEVICE((return basic_simd<DATA_TYPE, ABI_TYPE>{};))         \
  }

#define KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_LOAD(PREFIX, DATA_TYPE, ABI_TYPE, \
                                                EXPR)                        \
  template <typename... Flags>                                               \
  KOKKOS_FORCEINLINE_FUNCTION basic_simd<DATA_TYPE, ABI_TYPE>                \
      simd_##PREFIX##_load(const DATA_TYPE* ptr,                             \
                           basic_simd_mask<DATA_TYPE, ABI_TYPE> const& mask, \
                           simd_flags<Flags...> flag = simd_flag_default) {  \
    KOKKOS_IF_ON_HOST((EXPR))                                                \
    KOKKOS_IF_ON_DEVICE((return basic_simd<DATA_TYPE, ABI_TYPE>{};))         \
  }

#define KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_NATIVE_LOAD(PREFIX, DATA_TYPE,    \
                                                       ABI_TYPE, EXPR)       \
  template <typename SimdType, typename... Flags>                            \
    requires std::same_as<typename SimdType::abi_type, ABI_TYPE>             \
  KOKKOS_FORCEINLINE_FUNCTION basic_simd<DATA_TYPE, ABI_TYPE>                \
      simd_##PREFIX##_load(const DATA_TYPE* ptr,                             \
                           basic_simd_mask<DATA_TYPE, ABI_TYPE> const& mask, \
                           simd_flags<Flags...> flag = simd_flag_default) {  \
    KOKKOS_IF_ON_HOST((EXPR))                                                \
    KOKKOS_IF_ON_DEVICE((return basic_simd<DATA_TYPE, ABI_TYPE>{};))         \
  }

#define KOKKOS_SIMD_IMPL_LOAD_STORE_NATIVE_STORE(PREFIX, DATA_TYPE, ABI_TYPE, \
                                                 EXPR)                        \
  template <typename... Flags>                                                \
  KOKKOS_FORCEINLINE_FUNCTION void simd_##PREFIX##_store(                     \
      basic_simd<DATA_TYPE, ABI_TYPE> const& simd, DATA_TYPE* ptr,            \
      [[maybe_unused]] simd_flags<Flags...> flag = simd_flag_default) {       \
    KOKKOS_IF_ON_HOST((EXPR))                                                 \
  }

#define KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_STORE(PREFIX, DATA_TYPE, ABI_TYPE, \
                                                 EXPR)                        \
  template <typename... Flags>                                                \
  KOKKOS_FORCEINLINE_FUNCTION void simd_##PREFIX##_store(                     \
      basic_simd<DATA_TYPE, ABI_TYPE> const& simd, DATA_TYPE* ptr,            \
      basic_simd_mask<DATA_TYPE, ABI_TYPE> const& mask,                       \
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
      R&& in, const I& indices,                                           \
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
      R&& in, const typename I::mask_type& mask, const I& indices,        \
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
      const V& v, R&& out, const I& indices,                                \
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
      const V& v, R&& out, const typename I::mask_type& mask,               \
      const I& indices,                                                     \
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
