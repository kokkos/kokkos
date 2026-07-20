// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_SIMD_IMPL_MACROS_HPP
#define KOKKOS_SIMD_IMPL_MACROS_HPP

#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
#if (defined(KOKKOS_ENABLE_CUDA) && defined(__CUDA_ARCH__)) ||         \
    (defined(KOKKOS_ENABLE_HIP) && defined(__HIP_DEVICE_COMPILE__)) || \
    (defined(KOKKOS_ENABLE_SYCL) && defined(__SYCL_DEVICE_ONLY__))
#define KOKKOS_SIMD_IMPL_DEVICE_SIMD
#endif
#endif

// SIMD_IMPL_NATIVE_OP/FN
#define KOKKOS_SIMD_IMPL_NATIVE_FN(RET_TYPE, FN, ARG_TYPE, ...) \
  static RET_TYPE FN(ARG_TYPE v) { return __VA_ARGS__; }

#define KOKKOS_SIMD_IMPL_NATIVE_FN_2ARGS(RET_TYPE, FN, ARG_TYPE, ...) \
  static RET_TYPE FN(ARG_TYPE lhs, ARG_TYPE rhs) { return __VA_ARGS__; }

#define KOKKOS_SIMD_IMPL_NATIVE_FN_SHIFT_SCALAR(RET_TYPE, FN, ARG_TYPE, ...) \
  static RET_TYPE FN(ARG_TYPE lhs, [[maybe_unused]] simd_size_t rhs) { return __VA_ARGS__; }

#define KOKKOS_SIMD_IMPL_NATIVE_FN_3ARGS(RET_TYPE, FN, ARG_TYPE, ...) \
  static RET_TYPE FN(ARG_TYPE a, ARG_TYPE b, ARG_TYPE c) { return __VA_ARGS__; }

#define KOKKOS_SIMD_IMPL_NATIVE_CONVERSION_FN(RET_TYPE, FN, FROM, ABI, TAG, ...) \
  static RET_TYPE FN(simd_vector_t<FROM, ABI, TAG> v) { return __VA_ARGS__; }

#define KOKKOS_SIMD_IMPL_NATIVE_CONVERSION_FALLBACK_DECL_FN(ABI, TAG) \
  static vector_type convert_from(                                             \
      simd_vector_t<U, ABI, TAG> v);

#define KOKKOS_SIMD_IMPL_NATIVE_LOAD(RET_TYPE, FN, SRC_TYPE, ...) \
  static RET_TYPE FN(SRC_TYPE ptr, simd_flags<Flags...> = {}) {  \
    __VA_ARGS__ \
  }

#define KOKKOS_SIMD_IMPL_NATIVE_MASKED_LOAD(RET_TYPE, FN, SRC_TYPE, MASK_TYPE, ...) \
  static RET_TYPE FN(SRC_TYPE ptr, MASK_TYPE mask, simd_flags<Flags...> = {}) {  \
    __VA_ARGS__ \
  }

#define KOKKOS_SIMD_IMPL_NATIVE_STORE(FN, DST_TYPE, SRC_TYPE, ...) \
  static void FN(DST_TYPE ptr, SRC_TYPE v, simd_flags<Flags...> = {}) {  \
    __VA_ARGS__ \
  }

#define KOKKOS_SIMD_IMPL_NATIVE_MASKED_STORE(FN, DST_TYPE, SRC_TYPE, MASK_TYPE, ...) \
  static void FN(DST_TYPE ptr, SRC_TYPE v, MASK_TYPE mask, simd_flags<Flags...> = {}) {  \
    __VA_ARGS__ \
  }

#define KOKKOS_SIMD_IMPL_NATIVE_EXTRACT_FN(RET_TYPE, FN, ...) \
  static RET_TYPE FN(vector_type v, simd_size_t i) { __VA_ARGS__; }

#define KOKKOS_SIMD_IMPL_NATIVE_FN_HOST(RET_TYPE, FN, ARG_TYPE, ...) \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION \
  KOKKOS_SIMD_IMPL_NATIVE_FN(RET_TYPE, FN, ARG_TYPE, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_NATIVE_FN_DEVICE(RET_TYPE, FN, ARG_TYPE, ...) \
  KOKKOS_FORCEINLINE_FUNCTION \
  KOKKOS_SIMD_IMPL_NATIVE_FN(RET_TYPE, FN, [[maybe_unused]] ARG_TYPE, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_NATIVE_GEN_FN_HOST(RET_TYPE, FN, ...) \
  template <typename G> \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION \
  KOKKOS_SIMD_IMPL_NATIVE_FN(RET_TYPE, FN, G&&, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_NATIVE_GEN_FN_DEVICE(RET_TYPE, FN, ...) \
  template <typename G> \
  KOKKOS_FORCEINLINE_FUNCTION \
  KOKKOS_SIMD_IMPL_NATIVE_FN(RET_TYPE, FN, [[maybe_unused]] G&&, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_NATIVE_CONVERSION_FN_HOST(RET_TYPE, FN, FROM, ABI, ...) \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION \
  KOKKOS_SIMD_IMPL_NATIVE_CONVERSION_FN(RET_TYPE, FN, FROM, ABI, simd_host_tag, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_NATIVE_CONVERSION_FN_DEVICE(RET_TYPE, FN, FROM, ABI, ...) \
  template <typename T, typename Abi> \
  KOKKOS_FORCEINLINE_FUNCTION \
  KOKKOS_SIMD_IMPL_NATIVE_CONVERSION_FN(RET_TYPE, FN, FROM, ABI, simd_device_tag, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_NATIVE_CONVERSION_FALLBACK_DECL_HOST(ABI)                 \
  template <typename U>                                                        \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION                                        \
  KOKKOS_SIMD_IMPL_NATIVE_CONVERSION_FALLBACK_DECL_FN(ABI, simd_host_tag)

#define KOKKOS_SIMD_IMPL_NATIVE_CONVERSION_FALLBACK_DECL_DEVICE(ABI)                 \
  template <typename U>                                                        \
  KOKKOS_FORCEINLINE_FUNCTION                                                  \
  KOKKOS_SIMD_IMPL_NATIVE_CONVERSION_FALLBACK_DECL_FN(ABI, simd_device_tag)

#define KOKKOS_SIMD_IMPL_NATIVE_CONVERSION_FALLBACK_DEFN_HOST(IMPL_OPS, TO, ABI)         \
  template <typename From>                                                     \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION                                        \
  auto IMPL_OPS<TO, ABI, simd_host_tag>::convert_from(   \
      simd_vector_t<From, ABI, simd_host_tag> v)                          \
      -> typename IMPL_OPS<TO, ABI, simd_host_tag>::vector_type \
  {                                                                            \
    using from_native_ops = IMPL_OPS<From, ABI, simd_host_tag>;     \
    return gen([&](simd_size_t i) {                                            \
      return static_cast<TO>(from_native_ops::extract(v, i));             \
    });                                                                        \
  }

#define KOKKOS_SIMD_IMPL_NATIVE_CONVERSION_FALLBACK_DEFN_DEVICE(IMPL_OPS, TO, ABI)         \
  template <typename From>                                                     \
  KOKKOS_FORCEINLINE_FUNCTION                                                  \
  auto IMPL_OPS<TO, ABI, simd_device_tag>::convert_from(   \
      simd_vector_t<From, ABI, simd_device_tag> v)                          \
      -> typename IMPL_OPS<TO, ABI, simd_device_tag>::vector_type \
  {                                                                            \
    using from_native_ops = IMPL_OPS<From, ABI, simd_device_tag>;     \
    return gen(KOKKOS_LAMBDA(simd_size_t i) {                                            \
      return static_cast<TO>(from_native_ops::extract(v, i));             \
    });                                                                        \
  }

#define KOKKOS_SIMD_IMPL_NATIVE_GATHER_FROM_FN(RET_TYPE, PREFIX, ...) \
  static RET_TYPE PREFIX##_gather_from(R&& in, const IndicesType& indices, [[maybe_unused]] simd_flags<Flags...> flag = {}) { \
    return __VA_ARGS__; \
  }

#define KOKKOS_SIMD_IMPL_NATIVE_MASKED_GATHER_FROM_FN(RET_TYPE, PREFIX, ...) \
  static RET_TYPE PREFIX##_gather_from(R&& in, IndicesType const& indices, MaskType const& mmask, [[maybe_unused]] simd_flags<Flags...> flag = {}) { \
    return __VA_ARGS__; \
  }

#define KOKKOS_SIMD_IMPL_NATIVE_LOAD_HOST(RET_TYPE, FN, SRC_TYPE, ...) \
  template <typename... Flags> \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION \
  KOKKOS_SIMD_IMPL_NATIVE_LOAD(RET_TYPE, FN, SRC_TYPE, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_NATIVE_LOAD_DEVICE(RET_TYPE, FN, SRC_TYPE, ...) \
  template <typename... Flags> \
  KOKKOS_FORCEINLINE_FUNCTION \
  KOKKOS_SIMD_IMPL_NATIVE_LOAD(RET_TYPE, FN, [[maybe_unused]] SRC_TYPE, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_NATIVE_MASKED_LOAD_HOST(RET_TYPE, FN, SRC_TYPE, MASK_TYPE, ...) \
  template <typename... Flags> \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION \
  KOKKOS_SIMD_IMPL_NATIVE_MASKED_LOAD(RET_TYPE, FN, SRC_TYPE, MASK_TYPE, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_NATIVE_MASKED_LOAD_DEVICE(RET_TYPE, FN, SRC_TYPE, MASK_TYPE, ...) \
  template <typename... Flags> \
  KOKKOS_FORCEINLINE_FUNCTION \
  KOKKOS_SIMD_IMPL_NATIVE_MASKED_LOAD(RET_TYPE, FN, [[maybe_unused]] SRC_TYPE, [[maybe_unused]] MASK_TYPE, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_NATIVE_STORE_HOST(FN, DST_TYPE, SRC_TYPE, ...) \
  template <typename... Flags> \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION \
  KOKKOS_SIMD_IMPL_NATIVE_STORE(FN, DST_TYPE, SRC_TYPE, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_NATIVE_STORE_DEVICE(FN, DST_TYPE, SRC_TYPE, ...) \
  template <typename... Flags> \
  KOKKOS_FORCEINLINE_FUNCTION \
  KOKKOS_SIMD_IMPL_NATIVE_STORE(FN, [[maybe_unused]] DST_TYPE, SRC_TYPE, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_NATIVE_MASKED_STORE_HOST(FN, DST_TYPE, SRC_TYPE, MASK_TYPE, ...) \
  template <typename... Flags> \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION \
  KOKKOS_SIMD_IMPL_NATIVE_MASKED_STORE(FN, DST_TYPE, SRC_TYPE, MASK_TYPE, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_NATIVE_MASKED_STORE_DEVICE(FN, DST_TYPE, SRC_TYPE, MASK_TYPE, ...) \
  template <typename... Flags> \
  KOKKOS_FORCEINLINE_FUNCTION \
  KOKKOS_SIMD_IMPL_NATIVE_MASKED_STORE(FN, DST_TYPE, SRC_TYPE, MASK_TYPE, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_NATIVE_EXTRACT_FN_HOST(RET_TYPE, FN, ...) \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION \
  KOKKOS_SIMD_IMPL_NATIVE_EXTRACT_FN(RET_TYPE, FN, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_NATIVE_EXTRACT_FN_DEVICE(RET_TYPE, FN, ...) \
  KOKKOS_FORCEINLINE_FUNCTION \
  KOKKOS_SIMD_IMPL_NATIVE_EXTRACT_FN(RET_TYPE, FN, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_NATIVE_UNARY_OP_HOST(RET_TYPE, FN, ...) \
  KOKKOS_SIMD_IMPL_NATIVE_FN_HOST(RET_TYPE, FN, vector_type, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_NATIVE_UNARY_OP_DEVICE(RET_TYPE, FN, ...) \
  KOKKOS_SIMD_IMPL_NATIVE_FN_DEVICE(RET_TYPE, FN, [[maybe_unused]] vector_type, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_NATIVE_UNARY_MATH_OP_HOST(RET_TYPE, FN, ...) \
  KOKKOS_SIMD_IMPL_NATIVE_UNARY_OP_HOST(RET_TYPE, FN, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_NATIVE_UNARY_MATH_OP_DEVICE(RET_TYPE, FN, ...) \
  KOKKOS_SIMD_IMPL_NATIVE_UNARY_OP_DEVICE(RET_TYPE, FN, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_NATIVE_SHIFT_SCALAR_HOST(RET_TYPE, FN, ...) \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION \
  KOKKOS_SIMD_IMPL_NATIVE_FN_SHIFT_SCALAR(RET_TYPE, FN, vector_type, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_NATIVE_SHIFT_SCALAR_DEVICE(RET_TYPE, FN, ...) \
  KOKKOS_FORCEINLINE_FUNCTION \
  KOKKOS_SIMD_IMPL_NATIVE_FN_SHIFT_SCALAR(RET_TYPE, FN, [[maybe_unused]] vector_type, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_HOST(RET_TYPE, FN, ...) \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION \
  KOKKOS_SIMD_IMPL_NATIVE_FN_2ARGS(RET_TYPE, FN, vector_type, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_DEVICE(RET_TYPE, FN, ...) \
  KOKKOS_FORCEINLINE_FUNCTION \
  KOKKOS_SIMD_IMPL_NATIVE_FN_2ARGS(RET_TYPE, FN, [[maybe_unused]] vector_type, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_NATIVE_BINARY_MATH_OP_HOST(RET_TYPE, FN, ...) \
  KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_HOST(RET_TYPE, FN, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_NATIVE_BINARY_MATH_OP_DEVICE(RET_TYPE, FN, ...) \
  KOKKOS_SIMD_IMPL_NATIVE_BINARY_OP_DEVICE(RET_TYPE, FN, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_NATIVE_TERNARY_MATH_OP_HOST(RET_TYPE, FN, ...) \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION \
  KOKKOS_SIMD_IMPL_NATIVE_FN_3ARGS(RET_TYPE, FN, vector_type, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_NATIVE_TERNARY_MATH_OP_DEVICE(RET_TYPE, FN, ...) \
  KOKKOS_FORCEINLINE_FUNCTION \
  KOKKOS_SIMD_IMPL_NATIVE_FN_3ARGS(RET_TYPE, FN, [[maybe_unused]] vector_type, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_NATIVE_GATHER_FROM_FN_HOST(RET_TYPE, PREFIX, ...) \
  template <Impl::Ranges::contiguous_range R, typename IndicesType, typename... Flags>                       \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION \
  KOKKOS_SIMD_IMPL_NATIVE_GATHER_FROM_FN(RET_TYPE, PREFIX, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_NATIVE_GATHER_FROM_FN_DEVICE(RET_TYPE, PREFIX, ...) \
  template <Impl::Ranges::contiguous_range R, typename IndicesType, typename... Flags>                      \
  KOKKOS_FORCEINLINE_FUNCTION \
  KOKKOS_SIMD_IMPL_NATIVE_GATHER_FROM_FN(RET_TYPE, PREFIX, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_NATIVE_MASKED_GATHER_FROM_FN_HOST(RET_TYPE, PREFIX, ...) \
  template <Impl::Ranges::contiguous_range R, typename IndicesType, typename MaskType, typename... Flags>                      \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION \
  KOKKOS_SIMD_IMPL_NATIVE_MASKED_GATHER_FROM_FN(RET_TYPE, PREFIX, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_NATIVE_MASKED_GATHER_FROM_FN_DEVICE(RET_TYPE, PREFIX, ...) \
  template <Impl::Ranges::contiguous_range R, typename IndicesType, typename MaskType, typename... Flags>    \
  KOKKOS_FORCEINLINE_FUNCTION \
  KOKKOS_SIMD_IMPL_NATIVE_MASKED_GATHER_FROM_FN(RET_TYPE, PREFIX, __VA_ARGS__)

#define KOKKOS_SIMD_IMPL_SIMD_HOST_VECTOR_IMPL(DATA_TYPE, ABI_TYPE, IMPL_TYPE) \
  template <>                                                               \
  struct simd_vector_impl<DATA_TYPE, ABI_TYPE, simd_host_tag> {             \
    using type = IMPL_TYPE;                                                 \
  };

// SIMD_BASE_IMPL
#define KOKKOS_SIMD_BASE_IMPL_DERIVED() \
  KOKKOS_FORCEINLINE_FUNCTION      \
  const Derived& derived() const { return static_cast<const Derived&>(*this); }

#define KOKKOS_SIMD_BASE_IMPL_SUBSCRIPT_OP()                                \
  KOKKOS_FORCEINLINE_FUNCTION                                          \
  constexpr auto operator[](simd_size_t lane) const                    \
    requires requires { Derived::impl_ops::extract(derived(), lane); } \
  {                                                                    \
    return Derived::impl_ops::extract(derived(), lane);                \
  }

#define KOKKOS_SIMD_BASE_IMPL_UNARY_OP(OP, IMPL_FN)                   \
  KOKKOS_FORCEINLINE_FUNCTION                                    \
  constexpr Derived operator OP() const noexcept                 \
    requires requires { Derived::impl_ops::IMPL_FN(derived()); } \
  {                                                              \
    return Derived(Derived::impl_ops::IMPL_FN(derived()));       \
  }

#define KOKKOS_SIMD_BASE_IMPL_BINARY_OP(OP, IMPL_FN)                     \
  KOKKOS_FORCEINLINE_FUNCTION                                       \
  constexpr friend Derived operator OP(Derived const& lhs,          \
                                       Derived const& rhs) noexcept \
    requires requires { Derived::impl_ops::IMPL_FN(lhs, rhs); }     \
  {                                                                 \
    return Derived(Derived::impl_ops::IMPL_FN(lhs, rhs));           \
  }

#define KOKKOS_SIMD_BASE_IMPL_COMPOUND_OP(OP, IMPL_FN)                       \
  KOKKOS_FORCEINLINE_FUNCTION                                           \
  constexpr friend Derived& operator OP(Derived& lhs,                   \
                                        Derived const& rhs) noexcept    \
    requires requires { Derived::impl_ops::IMPL_FN(lhs.m_value, rhs); } \
  {                                                                     \
    Derived::impl_ops::IMPL_FN(lhs.m_value, rhs);                       \
    return lhs;                                                         \
  }

#define KOKKOS_SIMD_BASE_IMPL_MASK_COMPARISON_OP(OP, IMPL_FN) \
  KOKKOS_SIMD_BASE_IMPL_BINARY_OP(OP, IMPL_FN)

#define KOKKOS_SIMD_BASE_IMPL_COMPARISON_OP(OP, IMPL_FN)                           \
  KOKKOS_FORCEINLINE_FUNCTION                                                 \
  constexpr friend auto operator OP(Derived const& lhs,                       \
                                    Derived const& rhs) noexcept              \
    requires requires { Derived::impl_ops::IMPL_FN(lhs, rhs); }               \
  {                                                                           \
    return typename Derived::mask_type(Derived::impl_ops::IMPL_FN(lhs, rhs)); \
  }

#define KOKKOS_SIMD_BASE_IMPL_SHIFT_OP(OP, IMPL_FN, RHS_TYPE)        \
  KOKKOS_FORCEINLINE_FUNCTION                                   \
  constexpr friend Derived operator OP(Derived const& lhs,      \
                                       RHS_TYPE rhs) noexcept   \
    requires requires { Derived::impl_ops::IMPL_FN(lhs, rhs); } \
  {                                                             \
    return Derived(Derived::impl_ops::IMPL_FN(lhs, rhs));       \
  }

#define KOKKOS_SIMD_BASE_IMPL_COMPOUND_SHIFT_OP(OP, IMPL_FN, RHS_TYPE)            \
  KOKKOS_FORCEINLINE_FUNCTION                                                \
  constexpr friend Derived& operator OP(Derived& lhs, RHS_TYPE rhs) noexcept \
    requires requires { Derived::impl_ops::IMPL_FN(lhs.m_value, rhs); }      \
  {                                                                          \
    Derived::impl_ops::IMPL_FN(lhs.m_value, rhs);                            \
    return lhs;                                                              \
  }

// TODO
// will need stuff like KOKKOS_SIMD_DEFINE_BINARY_FN... or MEMORY_PERMUTE... etc for free functions

// gather scatter
// these should eventually be impl native
// BUT can't be removed yet (used in other simd backend as well)

#define KOKKOS_SIMD_DEFINE_GATHER_FROM(PREFIX, DATA_TYPE, ABI)           \
  template <Impl::SimdVecType V, Impl::Ranges::contiguous_range R,             \
            Impl::SimdIntegral I, typename... Flags>                           \
    requires Impl::Ranges::sized_range<R> &&                                   \
             std::same_as<V, basic_simd<DATA_TYPE, ABI>>                 \
  KOKKOS_FORCEINLINE_FUNCTION                                                  \
  constexpr V PREFIX##_gather_from(                                            \
      R&& in, const I& indices,                                                \
      simd_flags<Flags...> flag = simd_flag_default) {                         \
    using impl_ops =                                                           \
        Impl::simd_native_ops<DATA_TYPE, ABI, Impl::simd_backend_t>;     \
    using indices_type = basic_simd<std::int32_t, ABI>;                   \
    auto idx = static_cast<typename indices_type::impl_vector_type>(            \
        indices_type{indices});                                                \
                                                                               \
    return V(impl_ops::PREFIX##_gather_from(in, idx, flag));                   \
  }

#define KOKKOS_SIMD_DEFINE_MASKED_GATHER_FROM(                                 \
    PREFIX, DATA_TYPE, MASK_DATA_TYPE, ABI)                             \
  template <Impl::SimdVecType V, Impl::Ranges::contiguous_range R,             \
            Impl::SimdIntegral I, typename... Flags>                           \
    requires Impl::Ranges::sized_range<R> &&                                   \
             std::same_as<V, basic_simd<DATA_TYPE, ABI>>                 \
  KOKKOS_FORCEINLINE_FUNCTION                                                  \
  constexpr V PREFIX##_gather_from(                                            \
      R&& in, typename I::mask_type const& mask, const I& indices,             \
      simd_flags<Flags...> flag = simd_flag_default) {                         \
    using impl_ops =                                                           \
        Impl::simd_native_ops<DATA_TYPE, ABI, Impl::simd_backend_t>;     \
    using indices_type = basic_simd<std::int32_t, ABI>;                   \
    using mask_type = basic_simd_mask<MASK_DATA_TYPE, ABI>;              \
    auto idx = static_cast<typename indices_type::impl_vector_type>(            \
        indices_type{indices});                                                \
    auto mmask = static_cast<typename mask_type::impl_vector_type>(             \
        mask_type{mask});                                                      \
                                                                               \
    return V(impl_ops::PREFIX##_gather_from(in, idx, mmask, flag));            \
  }

#define KOKKOS_SIMD_DEFINE_SCATTER_TO(PREFIX, DATA_TYPE, ABI)           \
  template <Impl::SimdVecType V, Impl::Ranges::contiguous_range R,          \
            Impl::SimdIntegral I, typename... Flags>                        \
    requires Impl::Ranges::sized_range<R> &&                                \
             std::same_as<V, basic_simd<DATA_TYPE, ABI>>               \
  KOKKOS_FORCEINLINE_FUNCTION constexpr void PREFIX##_scatter_to( \
      const V& v, R&& out, const I& indices,                                \
      [[maybe_unused]] simd_flags<Flags...> flag = simd_flag_default) {     \
      for (Impl::simd_size_t lane = 0; lane < v.size(); ++lane) { \
        out[indices[lane]] = v[lane]; \
      } \
  }

#define KOKKOS_SIMD_DEFINE_MASKED_SCATTER_TO(                                 \
    PREFIX, DATA_TYPE, ABI)                             \
  template <Impl::SimdVecType V, Impl::Ranges::contiguous_range R,          \
            Impl::SimdIntegral I, typename... Flags>                        \
    requires Impl::Ranges::sized_range<R> &&                                \
             std::same_as<V, basic_simd<DATA_TYPE, ABI>>               \
  KOKKOS_FORCEINLINE_FUNCTION constexpr void PREFIX##_scatter_to( \
      const V& v, R&& out, const typename I::mask_type& mask,               \
      const I& indices,                                                     \
      [[maybe_unused]] simd_flags<Flags...> flag = simd_flag_default) {     \
      for (Impl::simd_size_t lane = 0; lane < v.size(); ++lane) { \
        if (mask[lane]) out[indices[lane]] = v[lane]; \
      } \
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
