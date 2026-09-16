// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_SIMD_IMPL_MACROS_HPP
#define KOKKOS_SIMD_IMPL_MACROS_HPP

#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
// FIXME Temporarily disabling for OpenACC; there isn't a
// reliable, portable compile-time flag to detect the device compilation context
// to gate the device-only path
#if !defined(KOKKOS_ENABLE_OPENACC) &&                                  \
    ((defined(KOKKOS_ENABLE_CUDA) && defined(__CUDA_ARCH__)) ||         \
     (defined(KOKKOS_ENABLE_HIP) && defined(__HIP_DEVICE_COMPILE__)) || \
     (defined(KOKKOS_ENABLE_SYCL) && defined(__SYCL_DEVICE_ONLY__)))
#define KOKKOS_SIMD_IMPL_DEVICE_SIMD
#endif
#endif

#define KOKKOS_SIMD_IMPL_DEFINE_FN(RET_TYPE, FN, ARG_TYPE, ...) \
  static RET_TYPE FN(ARG_TYPE v) { return __VA_ARGS__; }

#define KOKKOS_SIMD_IMPL_DEFINE_FN_2ARGS(RET_TYPE, FN, ARG_TYPE, ...) \
  static RET_TYPE FN(ARG_TYPE lhs, ARG_TYPE rhs) { return __VA_ARGS__; }

#define KOKKOS_SIMD_IMPL_DEFINE_FN_SHIFT_SCALAR(RET_TYPE, FN, ARG_TYPE, ...) \
  static RET_TYPE FN(ARG_TYPE lhs, [[maybe_unused]] simd_size_t rhs) {       \
    return __VA_ARGS__;                                                      \
  }

#define KOKKOS_SIMD_IMPL_DEFINE_FN_3ARGS(RET_TYPE, FN, ARG_TYPE, ...) \
  static RET_TYPE FN(ARG_TYPE a, ARG_TYPE b, ARG_TYPE c) { return __VA_ARGS__; }

#define KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN(RET_TYPE, FN, FROM, ABI, TAG, \
                                              ...)                          \
  static RET_TYPE FN([[maybe_unused]] simd_vector_t<FROM, ABI, TAG> v) {    \
    return __VA_ARGS__;                                                     \
  }

#define KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DECL_FN(ABI, TAG) \
  static vector_type convert_from(simd_vector_t<U, ABI, TAG> v);

#define KOKKOS_SIMD_IMPL_DEFINE_LOAD(RET_TYPE, FN, SRC_TYPE, ...) \
  static RET_TYPE FN(SRC_TYPE ptr, simd_flags<Flags...> = {}) { __VA_ARGS__ }

#define KOKKOS_SIMD_IMPL_DEFINE_MASKED_LOAD(RET_TYPE, FN, SRC_TYPE, MASK_TYPE, \
                                            ...)                               \
  static RET_TYPE FN(SRC_TYPE ptr, MASK_TYPE mask,                             \
                     simd_flags<Flags...> = {}) {                              \
    __VA_ARGS__                                                                \
  }

#define KOKKOS_SIMD_IMPL_DEFINE_STORE(FN, DST_TYPE, SRC_TYPE, ...)      \
  static void FN(DST_TYPE ptr, SRC_TYPE v, simd_flags<Flags...> = {}) { \
    __VA_ARGS__                                                         \
  }

#define KOKKOS_SIMD_IMPL_DEFINE_MASKED_STORE(FN, DST_TYPE, SRC_TYPE,           \
                                             MASK_TYPE, ...)                   \
  static void FN([[maybe_unused]] DST_TYPE ptr, [[maybe_unused]] SRC_TYPE v,   \
                 [[maybe_unused]] MASK_TYPE mask, simd_flags<Flags...> = {}) { \
    __VA_ARGS__                                                                \
  }

#define KOKKOS_SIMD_IMPL_DEFINE_EXTRACT_FN(RET_TYPE, FN, ...) \
  static RET_TYPE FN(vector_type v, simd_size_t i) { __VA_ARGS__; }

#define KOKKOS_SIMD_IMPL_DEFINE_FN_HOST(RET_TYPE, FN, ARG_TYPE, ...) \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION                              \
  KOKKOS_SIMD_IMPL_DEFINE_FN(RET_TYPE, FN, ARG_TYPE, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_DEFINE_FN_DEVICE(RET_TYPE, FN, ARG_TYPE, ...) \
  KOKKOS_FORCEINLINE_FUNCTION                                          \
  KOKKOS_SIMD_IMPL_DEFINE_FN(RET_TYPE, FN, [[maybe_unused]] ARG_TYPE,  \
                             __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_HOST(RET_TYPE, FN, ...)      \
  template <typename G>                                             \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION KOKKOS_SIMD_IMPL_DEFINE_FN( \
      RET_TYPE, FN, G&&, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_DEFINE_GEN_FN_DEVICE(RET_TYPE, FN, ...) \
  template <typename G>                                          \
  KOKKOS_FORCEINLINE_FUNCTION KOKKOS_SIMD_IMPL_DEFINE_FN(        \
      RET_TYPE, FN, [[maybe_unused]] G&&, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_HOST(RET_TYPE, FN, FROM, ABI, \
                                                   ...)                     \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION                                     \
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN(RET_TYPE, FN, FROM, ABI,            \
                                        simd_host_tag, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN_DEVICE(RET_TYPE, FN, FROM, ABI, \
                                                     ...)                     \
  template <typename T, typename Abi>                                         \
  KOKKOS_FORCEINLINE_FUNCTION KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FN(          \
      RET_TYPE, FN, FROM, ABI, simd_device_tag, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DECL_HOST(ABI) \
  template <typename U>                                            \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION                            \
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DECL_FN(ABI, simd_host_tag)

#define KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DECL_DEVICE(ABI) \
  template <typename U>                                              \
  KOKKOS_FORCEINLINE_FUNCTION                                        \
  KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DECL_FN(ABI, simd_device_tag)

#define KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DEFN_HOST(IMPL_OPS, TO, \
                                                              ABI)          \
  template <typename From>                                                  \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION auto                                \
  IMPL_OPS<TO, ABI, simd_host_tag>::convert_from(                           \
      simd_vector_t<From, ABI, simd_host_tag> v)                            \
      ->typename IMPL_OPS<TO, ABI, simd_host_tag>::vector_type {            \
    using from_native_ops = IMPL_OPS<From, ABI, simd_host_tag>;             \
    return gen([&](simd_size_t i) {                                         \
      return static_cast<TO>(from_native_ops::extract(v, i));               \
    });                                                                     \
  }

#define KOKKOS_SIMD_IMPL_DEFINE_CONVERSION_FALLBACK_DEFN_DEVICE(IMPL_OPS, TO, \
                                                                ABI)          \
  template <typename From>                                                    \
  KOKKOS_FORCEINLINE_FUNCTION auto                                            \
  IMPL_OPS<TO, ABI, simd_device_tag>::convert_from(                           \
      simd_vector_t<From, ABI, simd_device_tag> v)                            \
      ->typename IMPL_OPS<TO, ABI, simd_device_tag>::vector_type {            \
    using from_native_ops = IMPL_OPS<From, ABI, simd_device_tag>;             \
    return gen(KOKKOS_LAMBDA(simd_size_t i) {                                 \
      return static_cast<TO>(from_native_ops::extract(v, i));                 \
    });                                                                       \
  }

#define KOKKOS_SIMD_IMPL_DEFINE_GATHER_FROM_FN(RET_TYPE, PREFIX, ...)       \
  static RET_TYPE PREFIX##_gather_from(                                     \
      [[maybe_unused]] R&& in, [[maybe_unused]] const IndicesType& indices, \
      [[maybe_unused]] simd_flags<Flags...> flag = {}) {                    \
    return __VA_ARGS__;                                                     \
  }

#define KOKKOS_SIMD_IMPL_DEFINE_MASKED_GATHER_FROM_FN(RET_TYPE, PREFIX, ...) \
  static RET_TYPE PREFIX##_gather_from(                                      \
      [[maybe_unused]] R&& in, [[maybe_unused]] IndicesType const& indices,  \
      [[maybe_unused]] MaskType const& mmask,                                \
      [[maybe_unused]] simd_flags<Flags...> flag = {}) {                     \
    return __VA_ARGS__;                                                      \
  }

#define KOKKOS_SIMD_IMPL_DEFINE_LOAD_HOST(RET_TYPE, FN, SRC_TYPE, ...) \
  template <typename... Flags>                                         \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION KOKKOS_SIMD_IMPL_DEFINE_LOAD(  \
      RET_TYPE, FN, SRC_TYPE, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_DEFINE_LOAD_DEVICE(RET_TYPE, FN, SRC_TYPE, ...) \
  template <typename... Flags>                                           \
  KOKKOS_FORCEINLINE_FUNCTION KOKKOS_SIMD_IMPL_DEFINE_LOAD(              \
      RET_TYPE, FN, [[maybe_unused]] SRC_TYPE, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_DEFINE_MASKED_LOAD_HOST(RET_TYPE, FN, SRC_TYPE,     \
                                                 MASK_TYPE, ...)             \
  template <typename... Flags>                                               \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION KOKKOS_SIMD_IMPL_DEFINE_MASKED_LOAD( \
      RET_TYPE, FN, SRC_TYPE, MASK_TYPE, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_DEFINE_MASKED_LOAD_DEVICE(RET_TYPE, FN, SRC_TYPE, \
                                                   MASK_TYPE, ...)         \
  template <typename... Flags>                                             \
  KOKKOS_FORCEINLINE_FUNCTION KOKKOS_SIMD_IMPL_DEFINE_MASKED_LOAD(         \
      RET_TYPE, FN, [[maybe_unused]] SRC_TYPE, [[maybe_unused]] MASK_TYPE, \
      __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_DEFINE_STORE_HOST(FN, DST_TYPE, SRC_TYPE, ...) \
  template <typename... Flags>                                          \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION KOKKOS_SIMD_IMPL_DEFINE_STORE(  \
      FN, DST_TYPE, SRC_TYPE, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_DEFINE_STORE_DEVICE(FN, DST_TYPE, SRC_TYPE, ...) \
  template <typename... Flags>                                            \
  KOKKOS_FORCEINLINE_FUNCTION KOKKOS_SIMD_IMPL_DEFINE_STORE(              \
      FN, [[maybe_unused]] DST_TYPE, [[maybe_unused]] SRC_TYPE, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_DEFINE_MASKED_STORE_HOST(FN, DST_TYPE, SRC_TYPE,     \
                                                  MASK_TYPE, ...)             \
  template <typename... Flags>                                                \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION KOKKOS_SIMD_IMPL_DEFINE_MASKED_STORE( \
      FN, DST_TYPE, SRC_TYPE, MASK_TYPE, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_DEFINE_MASKED_STORE_DEVICE(FN, DST_TYPE, SRC_TYPE, \
                                                    MASK_TYPE, ...)         \
  template <typename... Flags>                                              \
  KOKKOS_FORCEINLINE_FUNCTION KOKKOS_SIMD_IMPL_DEFINE_MASKED_STORE(         \
      FN, DST_TYPE, SRC_TYPE, MASK_TYPE, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_DEFINE_EXTRACT_FN_HOST(RET_TYPE, FN, ...) \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION                            \
  KOKKOS_SIMD_IMPL_DEFINE_EXTRACT_FN(RET_TYPE, FN, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_DEFINE_EXTRACT_FN_DEVICE(RET_TYPE, FN, ...) \
  KOKKOS_FORCEINLINE_FUNCTION                                        \
  KOKKOS_SIMD_IMPL_DEFINE_EXTRACT_FN(RET_TYPE, FN, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_HOST(RET_TYPE, FN, ...) \
  KOKKOS_SIMD_IMPL_DEFINE_FN_HOST(RET_TYPE, FN, vector_type, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_DEVICE(RET_TYPE, FN, ...) \
  KOKKOS_SIMD_IMPL_DEFINE_FN_DEVICE(RET_TYPE, FN,                  \
                                    [[maybe_unused]] vector_type, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_HOST(RET_TYPE, FN, ...) \
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_HOST(RET_TYPE, FN, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_DEFINE_UNARY_MATH_OP_DEVICE(RET_TYPE, FN, ...) \
  KOKKOS_SIMD_IMPL_DEFINE_UNARY_OP_DEVICE(RET_TYPE, FN, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_DEFINE_SHIFT_SCALAR_HOST(RET_TYPE, FN, ...) \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION                              \
  KOKKOS_SIMD_IMPL_DEFINE_FN_SHIFT_SCALAR(RET_TYPE, FN, vector_type, \
                                          __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_DEFINE_SHIFT_SCALAR_DEVICE(RET_TYPE, FN, ...) \
  KOKKOS_FORCEINLINE_FUNCTION                                          \
  KOKKOS_SIMD_IMPL_DEFINE_FN_SHIFT_SCALAR(                             \
      RET_TYPE, FN, [[maybe_unused]] vector_type, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(RET_TYPE, FN, ...) \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION                           \
  KOKKOS_SIMD_IMPL_DEFINE_FN_2ARGS(RET_TYPE, FN, vector_type, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(RET_TYPE, FN, ...)            \
  KOKKOS_FORCEINLINE_FUNCTION                                                  \
  KOKKOS_SIMD_IMPL_DEFINE_FN_2ARGS(RET_TYPE, FN, [[maybe_unused]] vector_type, \
                                   __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_DEFINE_BINARY_MATH_OP_HOST(RET_TYPE, FN, ...) \
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_HOST(RET_TYPE, FN, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_DEFINE_BINARY_MATH_OP_DEVICE(RET_TYPE, FN, ...) \
  KOKKOS_SIMD_IMPL_DEFINE_BINARY_OP_DEVICE(RET_TYPE, FN, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_DEFINE_TERNARY_MATH_OP_HOST(RET_TYPE, FN, ...) \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION                                 \
  KOKKOS_SIMD_IMPL_DEFINE_FN_3ARGS(RET_TYPE, FN, vector_type, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_DEFINE_TERNARY_MATH_OP_DEVICE(RET_TYPE, FN, ...)      \
  KOKKOS_FORCEINLINE_FUNCTION                                                  \
  KOKKOS_SIMD_IMPL_DEFINE_FN_3ARGS(RET_TYPE, FN, [[maybe_unused]] vector_type, \
                                   __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_DEFINE_GATHER_FROM_FN_HOST(RET_TYPE, PREFIX, ...) \
  template <Impl::Ranges::contiguous_range R, typename IndicesType,        \
            typename... Flags>                                             \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION                                    \
  KOKKOS_SIMD_IMPL_DEFINE_GATHER_FROM_FN(RET_TYPE, PREFIX, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_DEFINE_GATHER_FROM_FN_DEVICE(RET_TYPE, PREFIX, ...) \
  template <Impl::Ranges::contiguous_range R, typename IndicesType,          \
            typename... Flags>                                               \
  KOKKOS_FORCEINLINE_FUNCTION KOKKOS_SIMD_IMPL_DEFINE_GATHER_FROM_FN(        \
      RET_TYPE, PREFIX, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_DEFINE_MASKED_GATHER_FROM_FN_HOST(RET_TYPE, PREFIX, \
                                                           ...)              \
  template <Impl::Ranges::contiguous_range R, typename IndicesType,          \
            typename MaskType, typename... Flags>                            \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION                                      \
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_GATHER_FROM_FN(RET_TYPE, PREFIX, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_DEFINE_MASKED_GATHER_FROM_FN_DEVICE(RET_TYPE, PREFIX, \
                                                             ...)              \
  template <Impl::Ranges::contiguous_range R, typename IndicesType,            \
            typename MaskType, typename... Flags>                              \
  KOKKOS_FORCEINLINE_FUNCTION KOKKOS_SIMD_IMPL_DEFINE_MASKED_GATHER_FROM_FN(   \
      RET_TYPE, PREFIX, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_DEFINE_HOST_VECTOR(DATA_TYPE, ABI, IMPL_TYPE) \
  template <>                                                          \
  struct simd_vector_impl<DATA_TYPE, ABI, simd_host_tag> {             \
    using type = IMPL_TYPE;                                            \
  };

#define KOKKOS_SIMD_IMPL_BASE_DERIVED() \
  KOKKOS_FORCEINLINE_FUNCTION           \
  const Derived& derived() const { return static_cast<const Derived&>(*this); }

#define KOKKOS_SIMD_IMPL_BASE_SUBSCRIPT_OP()                           \
  KOKKOS_FORCEINLINE_FUNCTION                                          \
  constexpr auto operator[](simd_size_t lane) const                    \
    requires requires { Derived::impl_ops::extract(derived(), lane); } \
  {                                                                    \
    return Derived::impl_ops::extract(derived(), lane);                \
  }

#define KOKKOS_SIMD_IMPL_BASE_UNARY_OP(OP, IMPL_FN)              \
  KOKKOS_FORCEINLINE_FUNCTION                                    \
  constexpr Derived operator OP() const noexcept                 \
    requires requires { Derived::impl_ops::IMPL_FN(derived()); } \
  {                                                              \
    return Derived(Derived::impl_ops::IMPL_FN(derived()));       \
  }

#define KOKKOS_SIMD_IMPL_BASE_BINARY_OP(OP, IMPL_FN)                \
  KOKKOS_FORCEINLINE_FUNCTION                                       \
  constexpr friend Derived operator OP(Derived const& lhs,          \
                                       Derived const& rhs) noexcept \
    requires requires { Derived::impl_ops::IMPL_FN(lhs, rhs); }     \
  {                                                                 \
    return Derived(Derived::impl_ops::IMPL_FN(lhs, rhs));           \
  }

#define KOKKOS_SIMD_IMPL_BASE_COMPOUND_OP(OP, IMPL_FN)                  \
  KOKKOS_FORCEINLINE_FUNCTION                                           \
  constexpr friend Derived& operator OP(Derived& lhs,                   \
                                        Derived const& rhs) noexcept    \
    requires requires { Derived::impl_ops::IMPL_FN(lhs.m_value, rhs); } \
  {                                                                     \
    Derived::impl_ops::IMPL_FN(lhs.m_value, rhs);                       \
    return lhs;                                                         \
  }

#define KOKKOS_SIMD_IMPL_BASE_MASK_COMPARISON_OP(OP, IMPL_FN) \
  KOKKOS_SIMD_IMPL_BASE_BINARY_OP(OP, IMPL_FN)

#define KOKKOS_SIMD_IMPL_BASE_COMPARISON_OP(OP, IMPL_FN)                      \
  KOKKOS_FORCEINLINE_FUNCTION                                                 \
  constexpr friend auto operator OP(Derived const& lhs,                       \
                                    Derived const& rhs) noexcept              \
    requires requires { Derived::impl_ops::IMPL_FN(lhs, rhs); }               \
  {                                                                           \
    return typename Derived::mask_type(Derived::impl_ops::IMPL_FN(lhs, rhs)); \
  }

#define KOKKOS_SIMD_IMPL_BASE_SHIFT_OP(OP, IMPL_FN, RHS_TYPE)   \
  KOKKOS_FORCEINLINE_FUNCTION                                   \
  constexpr friend Derived operator OP(Derived const& lhs,      \
                                       RHS_TYPE rhs) noexcept   \
    requires requires { Derived::impl_ops::IMPL_FN(lhs, rhs); } \
  {                                                             \
    return Derived(Derived::impl_ops::IMPL_FN(lhs, rhs));       \
  }

#define KOKKOS_SIMD_IMPL_BASE_COMPOUND_SHIFT_OP(OP, IMPL_FN, RHS_TYPE)       \
  KOKKOS_FORCEINLINE_FUNCTION                                                \
  constexpr friend Derived& operator OP(Derived& lhs, RHS_TYPE rhs) noexcept \
    requires requires { Derived::impl_ops::IMPL_FN(lhs.m_value, rhs); }      \
  {                                                                          \
    Derived::impl_ops::IMPL_FN(lhs.m_value, rhs);                            \
    return lhs;                                                              \
  }

#define KOKKOS_SIMD_IMPL_DEFINE_MATH_UNARY_FN(FN, DATA_TYPE, ABI)          \
  template <typename ImplOps = Experimental::Impl::simd_native_ops<        \
                DATA_TYPE, ABI, Experimental::Impl::simd_backend_t>,       \
            typename... Flags>                                             \
  KOKKOS_FORCEINLINE_FUNCTION Experimental::basic_simd<DATA_TYPE, ABI> FN( \
      Experimental::basic_simd<DATA_TYPE, ABI> const& a,                   \
      Experimental::simd_flags<Flags...> = {})                             \
    requires requires { ImplOps::FN(a); }                                  \
  {                                                                        \
    using simd_type = Experimental::basic_simd<DATA_TYPE, ABI>;            \
    return simd_type(ImplOps::FN(a));                                      \
  }

#define KOKKOS_SIMD_IMPL_DEFINE_MATH_ROUNDING_FN(FN, DATA_TYPE, ABI)    \
  template <typename ImplOps = Experimental::Impl::simd_native_ops<     \
                DATA_TYPE, ABI, Experimental::Impl::simd_backend_t>,    \
            typename... Flags>                                          \
  KOKKOS_FORCEINLINE_FUNCTION auto FN(                                  \
      Experimental::basic_simd<DATA_TYPE, ABI> const& a,                \
      Experimental::simd_flags<Flags...> = {})                          \
    requires requires { ImplOps::FN(a); }                               \
  {                                                                     \
    using ret_data_type = typename ImplOps::rounded_data_type;          \
    using ret_type      = Experimental::basic_simd<ret_data_type, ABI>; \
    return ret_type(ImplOps::FN(a));                                    \
  }

#define KOKKOS_SIMD_IMPL_DEFINE_MATH_BINARY_FN(FN, DATA_TYPE, ABI)         \
  template <typename ImplOps = Experimental::Impl::simd_native_ops<        \
                DATA_TYPE, ABI, Experimental::Impl::simd_backend_t>,       \
            typename... Flags>                                             \
  KOKKOS_FORCEINLINE_FUNCTION Experimental::basic_simd<DATA_TYPE, ABI> FN( \
      Experimental::basic_simd<DATA_TYPE, ABI> const& a,                   \
      Experimental::basic_simd<DATA_TYPE, ABI> const& b,                   \
      Experimental::simd_flags<Flags...> = {})                             \
    requires requires { ImplOps::FN(a, b); }                               \
  {                                                                        \
    using simd_type = Experimental::basic_simd<DATA_TYPE, ABI>;            \
    return simd_type(ImplOps::FN(a, b));                                   \
  }

#define KOKKOS_SIMD_IMPL_DEFINE_MATH_TERNARY_FN(FN, DATA_TYPE, ABI)        \
  template <typename ImplOps = Experimental::Impl::simd_native_ops<        \
                DATA_TYPE, ABI, Experimental::Impl::simd_backend_t>,       \
            typename... Flags>                                             \
  KOKKOS_FORCEINLINE_FUNCTION Experimental::basic_simd<DATA_TYPE, ABI> FN( \
      Experimental::basic_simd<DATA_TYPE, ABI> const& a,                   \
      Experimental::basic_simd<DATA_TYPE, ABI> const& b,                   \
      Experimental::basic_simd<DATA_TYPE, ABI> const& c,                   \
      Experimental::simd_flags<Flags...> = {})                             \
    requires requires { ImplOps::FN(a, b, c); }                            \
  {                                                                        \
    using simd_type = Experimental::basic_simd<DATA_TYPE, ABI>;            \
    return simd_type(ImplOps::FN(a, b, c));                                \
  }

#define KOKKOS_SIMD_IMPL_DEFINE_MATH_FN(DATA_TYPE, ABI)            \
  KOKKOS_SIMD_IMPL_DEFINE_MATH_BINARY_FN(copysign, DATA_TYPE, ABI) \
  KOKKOS_SIMD_IMPL_DEFINE_MATH_UNARY_FN(abs, DATA_TYPE, ABI)       \
  KOKKOS_SIMD_IMPL_DEFINE_MATH_ROUNDING_FN(floor, DATA_TYPE, ABI)  \
  KOKKOS_SIMD_IMPL_DEFINE_MATH_ROUNDING_FN(ceil, DATA_TYPE, ABI)   \
  KOKKOS_SIMD_IMPL_DEFINE_MATH_ROUNDING_FN(round, DATA_TYPE, ABI)  \
  KOKKOS_SIMD_IMPL_DEFINE_MATH_ROUNDING_FN(trunc, DATA_TYPE, ABI)  \
  KOKKOS_SIMD_IMPL_DEFINE_MATH_UNARY_FN(cbrt, DATA_TYPE, ABI)      \
  KOKKOS_SIMD_IMPL_DEFINE_MATH_UNARY_FN(exp, DATA_TYPE, ABI)       \
  KOKKOS_SIMD_IMPL_DEFINE_MATH_UNARY_FN(log, DATA_TYPE, ABI)       \
  KOKKOS_SIMD_IMPL_DEFINE_MATH_UNARY_FN(sqrt, DATA_TYPE, ABI)      \
  KOKKOS_SIMD_IMPL_DEFINE_MATH_TERNARY_FN(fma, DATA_TYPE, ABI)     \
  KOKKOS_SIMD_IMPL_DEFINE_MATH_BINARY_FN(max, DATA_TYPE, ABI)      \
  KOKKOS_SIMD_IMPL_DEFINE_MATH_BINARY_FN(min, DATA_TYPE, ABI)

// Workaround for NVHPC/OpenACC constraint checking issue.
// Referring directly to the function parameters in the requires expression
// causes the constrained overload to be rejected during compilation. Using the
// requires parameter list instead seems to resolve this issue.
#define KOKKOS_SIMD_IMPL_DEFINE_MASKED_FN_AVX2(FN)                            \
  template <typename T, Experimental::Impl::simd_size_t N, typename... Args>  \
  KOKKOS_FORCEINLINE_FUNCTION                                                 \
      Experimental::basic_simd<T, Experimental::simd_abi::avx2_fixed_size<N>> \
      FN(Experimental::basic_simd_mask<                                       \
             T, Experimental::simd_abi::avx2_fixed_size<N>> const& a,         \
         Args const&... args)                                                 \
    requires requires(                                                        \
        Experimental::basic_simd_mask<                                        \
            T, Experimental::simd_abi::avx2_fixed_size<N>> const& m,          \
        Args const&... xs) {                                                  \
      Experimental::Impl::simd_native_ops<                                    \
          T, Experimental::simd_abi::avx2_fixed_size<N>,                      \
          Experimental::Impl::simd_backend_t>::FN(m, xs...);                  \
    }                                                                         \
  {                                                                           \
    using impl_ops = Experimental::Impl::simd_native_ops<                     \
        T, Experimental::simd_abi::avx2_fixed_size<N>,                        \
        Experimental::Impl::simd_backend_t>;                                  \
    using simd_type =                                                         \
        Experimental::basic_simd<T,                                           \
                                 Experimental::simd_abi::avx2_fixed_size<N>>; \
    return simd_type(impl_ops::FN(a, args...));                               \
  }

#define KOKKOS_SIMD_IMPL_DEFINE_LOAD_FN(PREFIX, DATA_TYPE, ABI)                \
  template <typename SimdType, typename... Flags>                              \
    requires std::same_as<typename SimdType::abi_type, ABI>                    \
  KOKKOS_FORCEINLINE_FUNCTION basic_simd<DATA_TYPE, ABI> simd_##PREFIX##_load( \
      const DATA_TYPE* ptr, simd_flags<Flags...> flag = simd_flag_default) {   \
    return basic_simd<DATA_TYPE, ABI>(ptr, flag);                              \
  }

#define KOKKOS_SIMD_IMPL_DEFINE_MASKED_LOAD_FN(PREFIX, DATA_TYPE, ABI)         \
  template <typename... Flags>                                                 \
  KOKKOS_FORCEINLINE_FUNCTION basic_simd<DATA_TYPE, ABI> simd_##PREFIX##_load( \
      const DATA_TYPE* ptr, basic_simd_mask<DATA_TYPE, ABI> const& mask,       \
      simd_flags<Flags...> flag = simd_flag_default) {                         \
    return basic_simd<DATA_TYPE, ABI>(ptr, mask, flag);                        \
  }

#define KOKKOS_SIMD_IMPL_DEFINE_EXPLICIT_MASKED_LOAD_FN(PREFIX, DATA_TYPE,     \
                                                        ABI)                   \
  template <typename SimdType, typename... Flags>                              \
    requires std::same_as<typename SimdType::abi_type, ABI>                    \
  KOKKOS_FORCEINLINE_FUNCTION basic_simd<DATA_TYPE, ABI> simd_##PREFIX##_load( \
      const DATA_TYPE* ptr, basic_simd_mask<DATA_TYPE, ABI> const& mask,       \
      simd_flags<Flags...> flag = simd_flag_default) {                         \
    return basic_simd<DATA_TYPE, ABI>(ptr, mask, flag);                        \
  }

#define KOKKOS_SIMD_IMPL_DEFINE_STORE_FN(PREFIX, DATA_TYPE, ABI, IMPL_FN)   \
  template <typename... Flags>                                              \
  KOKKOS_FORCEINLINE_FUNCTION void simd_##PREFIX##_store(                   \
      basic_simd<DATA_TYPE, ABI> const& simd, DATA_TYPE* ptr,               \
      simd_flags<Flags...> flag = {})                                       \
    requires requires {                                                     \
      Impl::simd_native_ops<DATA_TYPE, ABI, Impl::simd_backend_t>::IMPL_FN( \
          ptr, simd, flag);                                                 \
    }                                                                       \
  {                                                                         \
    using impl_ops =                                                        \
        Impl::simd_native_ops<DATA_TYPE, ABI, Impl::simd_backend_t>;        \
    impl_ops::IMPL_FN(ptr, simd, flag);                                     \
  }

#define KOKKOS_SIMD_IMPL_DEFINE_MASKED_STORE_FN(PREFIX, DATA_TYPE, ABI,     \
                                                IMPL_FN)                    \
  template <typename... Flags>                                              \
  KOKKOS_FORCEINLINE_FUNCTION void simd_##PREFIX##_store(                   \
      basic_simd<DATA_TYPE, ABI> const& simd, DATA_TYPE* ptr,               \
      basic_simd_mask<DATA_TYPE, ABI> const& mask,                          \
      simd_flags<Flags...> flag = {})                                       \
    requires requires {                                                     \
      Impl::simd_native_ops<DATA_TYPE, ABI, Impl::simd_backend_t>::IMPL_FN( \
          ptr, simd, mask, flag);                                           \
    }                                                                       \
  {                                                                         \
    using impl_ops =                                                        \
        Impl::simd_native_ops<DATA_TYPE, ABI, Impl::simd_backend_t>;        \
    impl_ops::IMPL_FN(ptr, simd, mask, flag);                               \
  }

#define KOKKOS_SIMD_IMPL_DEFINE_LOAD_STORE_FN_AVX2(DATA_TYPE, ABI)           \
  KOKKOS_SIMD_IMPL_DEFINE_LOAD_FN(unchecked, DATA_TYPE, ABI)                 \
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_LOAD_FN(unchecked, DATA_TYPE, ABI)          \
  KOKKOS_SIMD_IMPL_DEFINE_EXPLICIT_MASKED_LOAD_FN(unchecked, DATA_TYPE, ABI) \
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_LOAD_FN(partial, DATA_TYPE, ABI)            \
  KOKKOS_SIMD_IMPL_DEFINE_EXPLICIT_MASKED_LOAD_FN(partial, DATA_TYPE, ABI)   \
  KOKKOS_SIMD_IMPL_DEFINE_STORE_FN(unchecked, DATA_TYPE, ABI, store)         \
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_STORE_FN(unchecked, DATA_TYPE, ABI,         \
                                          masked_store)                      \
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_STORE_FN(partial, DATA_TYPE, ABI, masked_store)

#define KOKKOS_SIMD_IMPL_DEFINE_GATHER_FROM(PREFIX, DATA_TYPE, ABI)     \
  template <Impl::SimdVecType V, Impl::Ranges::contiguous_range R,      \
            Impl::SimdIntegral I, typename... Flags>                    \
    requires Impl::Ranges::sized_range<R> &&                            \
             std::same_as<V, basic_simd<DATA_TYPE, ABI>>                \
  KOKKOS_FORCEINLINE_FUNCTION constexpr V PREFIX##_gather_from(         \
      R&& in, const I& indices,                                         \
      simd_flags<Flags...> flag = simd_flag_default) {                  \
    using impl_ops =                                                    \
        Impl::simd_native_ops<DATA_TYPE, ABI, Impl::simd_backend_t>;    \
    using indices_type = basic_simd<std::int32_t, ABI>;                 \
    using native_indices_type =                                         \
        Impl::simd_vector_t<std::int32_t, ABI, Impl::simd_backend_t>;   \
    auto idx = static_cast<native_indices_type>(indices_type{indices}); \
                                                                        \
    return V(impl_ops::PREFIX##_gather_from(in, idx, flag));            \
  }

#define KOKKOS_SIMD_IMPL_DEFINE_MASKED_GATHER_FROM(PREFIX, DATA_TYPE,     \
                                                   MASK_DATA_TYPE, ABI)   \
  template <Impl::SimdVecType V, Impl::Ranges::contiguous_range R,        \
            Impl::SimdIntegral I, typename... Flags>                      \
    requires Impl::Ranges::sized_range<R> &&                              \
             std::same_as<V, basic_simd<DATA_TYPE, ABI>>                  \
  KOKKOS_FORCEINLINE_FUNCTION constexpr V PREFIX##_gather_from(           \
      R&& in, typename I::mask_type const& mask, const I& indices,        \
      simd_flags<Flags...> flag = simd_flag_default) {                    \
    using impl_ops =                                                      \
        Impl::simd_native_ops<DATA_TYPE, ABI, Impl::simd_backend_t>;      \
    using indices_type = basic_simd<std::int32_t, ABI>;                   \
    using mask_type    = basic_simd_mask<MASK_DATA_TYPE, ABI>;            \
    using native_indices_type =                                           \
        Impl::simd_vector_t<std::int32_t, ABI, Impl::simd_backend_t>;     \
    using native_mask_type =                                              \
        Impl::simd_vector_t<DATA_TYPE, ABI, Impl::simd_backend_t>;        \
    auto idx   = static_cast<native_indices_type>(indices_type{indices}); \
    auto mmask = static_cast<native_mask_type>(mask_type{mask});          \
                                                                          \
    return V(impl_ops::PREFIX##_gather_from(in, idx, mmask, flag));       \
  }

#define KOKKOS_SIMD_IMPL_DEFINE_SCATTER_TO_AVX2(PREFIX, DATA_TYPE, ABI) \
  template <Impl::SimdVecType V, Impl::Ranges::contiguous_range R,      \
            Impl::SimdIntegral I, typename... Flags>                    \
    requires Impl::Ranges::sized_range<R> &&                            \
             std::same_as<V, basic_simd<DATA_TYPE, ABI>>                \
  KOKKOS_FORCEINLINE_FUNCTION constexpr void PREFIX##_scatter_to(       \
      const V& v, R&& out, const I& indices,                            \
      [[maybe_unused]] simd_flags<Flags...> flag = simd_flag_default) { \
    for (Impl::simd_size_t lane = 0; lane < v.size(); ++lane) {         \
      out[indices[lane]] = v[lane];                                     \
    }                                                                   \
  }

#define KOKKOS_SIMD_IMPL_DEFINE_MASKED_SCATTER_TO_AVX2(PREFIX, DATA_TYPE, ABI) \
  template <Impl::SimdVecType V, Impl::Ranges::contiguous_range R,             \
            Impl::SimdIntegral I, typename... Flags>                           \
    requires Impl::Ranges::sized_range<R> &&                                   \
             std::same_as<V, basic_simd<DATA_TYPE, ABI>>                       \
  KOKKOS_FORCEINLINE_FUNCTION constexpr void PREFIX##_scatter_to(              \
      const V& v, R&& out, const typename I::mask_type& mask,                  \
      const I& indices,                                                        \
      [[maybe_unused]] simd_flags<Flags...> flag = simd_flag_default) {        \
    for (Impl::simd_size_t lane = 0; lane < v.size(); ++lane) {                \
      if (mask[lane]) out[indices[lane]] = v[lane];                            \
    }                                                                          \
  }

#define KOKKOS_SIMD_IMPL_DEFINE_MEMORY_PERMUTE_FN_AVX2(DATA_TYPE,           \
                                                       MASK_DATA_TYPE, ABI) \
  KOKKOS_SIMD_IMPL_DEFINE_GATHER_FROM(unchecked, DATA_TYPE, ABI)            \
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_GATHER_FROM(unchecked, DATA_TYPE,          \
                                             MASK_DATA_TYPE, ABI)           \
  KOKKOS_SIMD_IMPL_DEFINE_GATHER_FROM(partial, DATA_TYPE, ABI)              \
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_GATHER_FROM(partial, DATA_TYPE,            \
                                             MASK_DATA_TYPE, ABI)           \
  KOKKOS_SIMD_IMPL_DEFINE_SCATTER_TO_AVX2(unchecked, DATA_TYPE, ABI)        \
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_SCATTER_TO_AVX2(unchecked, DATA_TYPE, ABI) \
  KOKKOS_SIMD_IMPL_DEFINE_SCATTER_TO_AVX2(partial, DATA_TYPE, ABI)          \
  KOKKOS_SIMD_IMPL_DEFINE_MASKED_SCATTER_TO_AVX2(partial, DATA_TYPE, ABI)

#define KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_GATHER_FROM(PREFIX, DATA_TYPE,    \
                                                    ABI_TYPE, EXPR)       \
  template <Impl::SimdVecType V, Impl::Ranges::contiguous_range R,        \
            Impl::SimdIntegral I, typename... Flags>                      \
    requires Impl::Ranges::sized_range<R> &&                              \
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
  template <Impl::SimdVecType V, Impl::Ranges::contiguous_range R,        \
            Impl::SimdIntegral I, typename... Flags>                      \
    requires Impl::Ranges::sized_range<R> &&                              \
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
  template <Impl::SimdVecType V, Impl::Ranges::contiguous_range R,          \
            Impl::SimdIntegral I, typename... Flags>                        \
    requires Impl::Ranges::sized_range<R> &&                                \
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
  template <Impl::SimdVecType V, Impl::Ranges::contiguous_range R,          \
            Impl::SimdIntegral I, typename... Flags>                        \
    requires Impl::Ranges::sized_range<R> &&                                \
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

#endif
