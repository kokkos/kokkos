// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Half.hpp>

#if defined(__STDCPP_FLOAT16_T__) || defined(__STDCPP_BFLOAT16_T__)
#include <stdfloat>
#endif

// clang-format off

template <class T>
void check_common_types() {
static_assert(std::is_same_v<std::common_type_t<T, T>, T>);
static_assert(std::is_same_v<std::common_type_t<T, float>, float>);
static_assert(std::is_same_v<std::common_type_t<float, T>, float>);
static_assert(std::is_same_v<std::common_type_t<T, double>, double>);
static_assert(std::is_same_v<std::common_type_t<double, T>, double>);
static_assert(std::is_same_v<std::common_type_t<T, long double>, long double>);
static_assert(std::is_same_v<std::common_type_t<long double, T>, long double>);
static_assert(std::is_same_v<std::common_type_t<float, double, T>, double>);
static_assert(std::is_same_v<std::common_type_t<float, T, double>, double>);
static_assert(std::is_same_v<std::common_type_t<T, double, float>, double>);
static_assert(std::is_same_v<std::common_type_t<T, float, float>, float>);
static_assert(std::is_same_v<std::common_type_t<T, float, T>, float>);
static_assert(std::is_same_v<std::common_type_t<T, T, T>, T>);

static_assert(std::is_same_v<std::common_type_t<T, int>, T>);
static_assert(std::is_same_v<std::common_type_t<int, T>, T>);
static_assert(std::is_same_v<std::common_type_t<T, long int>, T>);
static_assert(std::is_same_v<std::common_type_t<long int, T>, T>);
static_assert(std::is_same_v<std::common_type_t<T, long long int>, T>);
static_assert(std::is_same_v<std::common_type_t<long long int, T>, T>);
}
#if !KOKKOS_HALF_T_IS_FLOAT
template void check_common_types<Kokkos::Experimental::half_t>();
#endif
#if defined(__STDCPP_FLOAT16_T__)
template void check_common_types<std::float16_t>();
#endif
#if !KOKKOS_BHALF_T_IS_FLOAT
template void check_common_types<Kokkos::Experimental::bhalf_t>();
#endif
#if defined(__STDCPP_BFLOAT16_T__)
template void check_common_types<std::bfloat16_t>();
#endif

template<class float16, class bfloat16>
inline void check_common_types_both() {
//static_assert(std::is_same_v<std::common_type_t<bfloat16, float16>, float>);
//static_assert(std::is_same_v<std::common_type_t<float16, bfloat16>, float>);
//static_assert(std::is_same_v<std::common_type_t<bfloat16, float16, float>, float>);
//static_assert(std::is_same_v<std::common_type_t<float16, bfloat16, float>, float>);
static_assert(!std::is_convertible_v<float16, bfloat16>);
static_assert(std::is_same_v<std::common_type_t<float, bfloat16, float16>, float>);
static_assert(std::is_same_v<std::common_type_t<float16, float, bfloat16>, float>);
static_assert(std::is_same_v<std::common_type_t<double, bfloat16, float16>, double>);
static_assert(std::is_same_v<std::common_type_t<float16, double, bfloat16>, double>);
static_assert(std::is_same_v<std::common_type_t<long double, bfloat16, float16>, long double>);
static_assert(std::is_same_v<std::common_type_t<float16, long double, bfloat16>, long double>);
}
#if !KOKKOS_HALF_T_IS_FLOAT && !KOKKOS_BHALF_T_IS_FLOAT
template void check_common_types_both<Kokkos::Experimental::half_t, Kokkos::Experimental::bhalf_t>();
#endif
#if defined(__STDCPP_FLOAT16_T__) && defined(__STDCPP_BFLOAT16_T__)
template void check_common_types_both<std::float16_t, std::bfloat16_t>();
#endif

// clang-format on
