// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Half.hpp>

static_assert(std::is_same_v<std::common_type_t<float, double>, double>);

// clang-format off

#if !KOKKOS_BHALF_T_IS_FLOAT
using bfloat16 = Kokkos::Experimental::bhalf_t;
static_assert(std::is_same_v<std::common_type_t<bfloat16, bfloat16>, bfloat16>);
static_assert(std::is_same_v<std::common_type_t<bfloat16, float>, float>);
static_assert(std::is_same_v<std::common_type_t<float, bfloat16>, float>);
static_assert(std::is_same_v<std::common_type_t<bfloat16, double>, double>);
static_assert(std::is_same_v<std::common_type_t<double, bfloat16>, double>);
static_assert(std::is_same_v<std::common_type_t<bfloat16, long double>, long double>);
static_assert(std::is_same_v<std::common_type_t<long double, bfloat16>, long double>);
static_assert(std::is_same_v<std::common_type_t<float, double, bfloat16>, double>);
static_assert(std::is_same_v<std::common_type_t<float, bfloat16, double>, double>);
static_assert(std::is_same_v<std::common_type_t<bfloat16, double, float>, double>);
static_assert(std::is_same_v<std::common_type_t<bfloat16, float, float>, float>);
static_assert(std::is_same_v<std::common_type_t<bfloat16, float, bfloat16>, float>);
static_assert(std::is_same_v<std::common_type_t<bfloat16, bfloat16, bfloat16>, bfloat16>);

static_assert(std::is_same_v<std::common_type_t<bfloat16, int>, bfloat16>);
static_assert(std::is_same_v<std::common_type_t<int, bfloat16>, bfloat16>);
static_assert(std::is_same_v<std::common_type_t<bfloat16, long int>, bfloat16>);
static_assert(std::is_same_v<std::common_type_t<long int, bfloat16>, bfloat16>);
static_assert(std::is_same_v<std::common_type_t<bfloat16, long long int>, bfloat16>);
static_assert(std::is_same_v<std::common_type_t<long long int, bfloat16>, bfloat16>);
#endif

#if !KOKKOS_HALF_T_IS_FLOAT
using float16 = Kokkos::Experimental::half_t;
static_assert(std::is_same_v<std::common_type_t<float16, float16>, float16>);
static_assert(std::is_same_v<std::common_type_t<float16, float>, float>);
static_assert(std::is_same_v<std::common_type_t<float, float16>, float>);
static_assert(std::is_same_v<std::common_type_t<float16, double>, double>);
static_assert(std::is_same_v<std::common_type_t<double, float16>, double>);
static_assert(std::is_same_v<std::common_type_t<float16, long double>, long double>);
static_assert(std::is_same_v<std::common_type_t<long double, float16>, long double>);
static_assert(std::is_same_v<std::common_type_t<float, double, float16>, double>);
static_assert(std::is_same_v<std::common_type_t<float, float16, double>, double>);
static_assert(std::is_same_v<std::common_type_t<float16, double, float>, double>);
static_assert(std::is_same_v<std::common_type_t<float16, float, float>, float>);
static_assert(std::is_same_v<std::common_type_t<float16, float, float16>, float>);
static_assert(std::is_same_v<std::common_type_t<float16, float16, float16>, float16>);

static_assert(std::is_same_v<std::common_type_t<float16, int>, float16>);
static_assert(std::is_same_v<std::common_type_t<int, float16>, float16>);
static_assert(std::is_same_v<std::common_type_t<float16, long int>, float16>);
static_assert(std::is_same_v<std::common_type_t<long int, float16>, float16>);
static_assert(std::is_same_v<std::common_type_t<float16, long long int>, float16>);
static_assert(std::is_same_v<std::common_type_t<long long int, float16>, float16>);
#endif

#if !KOKKOS_HALF_T_IS_FLOAT && !KOKKOS_BHALF_T_IS_FLOAT
static_assert(std::is_same_v<std::common_type_t<bfloat16, float16>, float>);
static_assert(std::is_same_v<std::common_type_t<float16, bfloat16>, float>);
static_assert(std::is_same_v<std::common_type_t<bfloat16, float16, float>, float>);
static_assert(std::is_same_v<std::common_type_t<float16, bfloat16, float>, float>);
static_assert(std::is_same_v<std::common_type_t<float, bfloat16, float16>, float>);
static_assert(std::is_same_v<std::common_type_t<float16, float, bfloat16>, float>);
static_assert(std::is_same_v<std::common_type_t<double, bfloat16, float16>, double>);
static_assert(std::is_same_v<std::common_type_t<float16, double, bfloat16>, double>);
static_assert(std::is_same_v<std::common_type_t<long double, bfloat16, float16>, long double>);
static_assert(std::is_same_v<std::common_type_t<float16, long double, bfloat16>, long double>);
#endif

// clang-format on
