// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Half.hpp>

static_assert(std::is_same_v<std::common_type_t<float, double>, double>);

// clang-format off

#ifdef KOKKOS_IMPL_BHALF_TYPE_DEFINED
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
#endif

#ifdef KOKKOS_IMPL_HALF_TYPE_DEFINED
using float16 = Kokkos::Experimental::bhalf_t;
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
#endif

#if defined(KOKKOS_IMPL_BHALF_TYPE_DEFINED) && \
    defined(KOKKOS_IMPL_HALF_TYPE_DEFINED)
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
