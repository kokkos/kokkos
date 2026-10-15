/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_OPENACC_TRAITS_HPP_
#define DESUL_ATOMICS_OPENACC_TRAITS_HPP_

#include <type_traits>

namespace desul {
namespace Impl {

#ifdef __NVCOMPILER

template <class T>
inline constexpr bool is_openacc_integral_type_v =
    std::is_same_v<T, int> || std::is_same_v<T, unsigned int> ||
    std::is_same_v<T, unsigned long long>;

template <class T>
inline constexpr bool is_openacc_arithmetic_type_v = std::is_same_v<T, float> ||
#ifndef DESUL_CUDA_ARCH_IS_PRE_PASCAL
                                                     std::is_same_v<T, double> ||
#endif
                                                     is_openacc_integral_type_v<T>;

#else

template <class T>
inline constexpr bool is_openacc_integral_type_v = std::is_integral_v<T>;

template <class T>
inline constexpr bool is_openacc_arithmetic_type_v = std::is_arithmetic_v<T>;

#endif

}  // namespace Impl
}  // namespace desul

#endif