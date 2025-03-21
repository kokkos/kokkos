//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef KOKKOS_SIMD_PERF_TEST_HOST_HPP
#define KOKKOS_SIMD_PERF_TEST_HOST_HPP

#include <benchmark/benchmark.h>
#include <Kokkos_SIMD.hpp>

#include "Common.hpp"
#include "PerfTest_Operators.hpp"

#define KOKKOS_IMPL_SIMD_PERFTEST_HOST_UNARY_BENCH(prefix, name, op) \
  benchmark::RegisterBenchmark(                                      \
      benchmark_name<Abi, DataType>("host " #prefix, #name).data(),  \
      host_bench_unary_op<Abi, op, DataType>)
#define KOKKOS_IMPL_SIMD_PERFTEST_HOST_BINARY_BENCH(prefix, name, op) \
  benchmark::RegisterBenchmark(                                       \
      benchmark_name<Abi, DataType>("host " #prefix, #name).data(),   \
      host_bench_binary_op<Abi, op, DataType>)
#define KOKKOS_IMPL_SIMD_PERFTEST_HOST_TERNARY_BENCH(prefix, name, op) \
  benchmark::RegisterBenchmark(                                        \
      benchmark_name<Abi, DataType>("host " #prefix, #name).data(),    \
      host_bench_ternary_op<Abi, op, DataType>)
#define KOKKOS_IMPL_SIMD_PERFTEST_HOST_REDUCTION_BENCH(prefix, name, op) \
  benchmark::RegisterBenchmark(                                          \
      benchmark_name<Abi, DataType>("host " #prefix, #name).data(),      \
      host_bench_reduction_op<Abi, op, DataType>)

template <typename Abi, typename DataType>
inline void host_register_common_benchmarks() {
  using ExecSpace = Kokkos::DefaultHostExecutionSpace;

  if constexpr (is_type_v<Kokkos::Experimental::basic_simd<DataType, Abi>>) {
    KOKKOS_IMPL_SIMD_PERFTEST_HOST_BINARY_BENCH(common, add, plus);
    KOKKOS_IMPL_SIMD_PERFTEST_HOST_BINARY_BENCH(common, sub, minus);
    KOKKOS_IMPL_SIMD_PERFTEST_HOST_BINARY_BENCH(common, multiply, multiplies);

    if constexpr (std::is_floating_point_v<DataType>) {
      KOKKOS_IMPL_SIMD_PERFTEST_HOST_BINARY_BENCH(common, divide, divides);
      KOKKOS_IMPL_SIMD_PERFTEST_HOST_UNARY_BENCH(common, floor, floors);
      KOKKOS_IMPL_SIMD_PERFTEST_HOST_UNARY_BENCH(common, ceil, ceils);
      KOKKOS_IMPL_SIMD_PERFTEST_HOST_UNARY_BENCH(common, round, rounds);
      KOKKOS_IMPL_SIMD_PERFTEST_HOST_UNARY_BENCH(common, truncate, truncates);
    }

    KOKKOS_IMPL_SIMD_PERFTEST_HOST_UNARY_BENCH(common, abs, absolutes);

    if constexpr (std::is_integral_v<DataType>) {
      KOKKOS_IMPL_SIMD_PERFTEST_HOST_BINARY_BENCH(common, shift_right,
                                                  shift_right);
      KOKKOS_IMPL_SIMD_PERFTEST_HOST_BINARY_BENCH(common, shift_left,
                                                  shift_left);
    }

    KOKKOS_IMPL_SIMD_PERFTEST_HOST_BINARY_BENCH(common, min, minimum);
    KOKKOS_IMPL_SIMD_PERFTEST_HOST_BINARY_BENCH(common, max, maximum);

    KOKKOS_IMPL_SIMD_PERFTEST_HOST_REDUCTION_BENCH(common, reduce, reduce);
    KOKKOS_IMPL_SIMD_PERFTEST_HOST_REDUCTION_BENCH(common, reduce_min,
                                                   reduce_min);
    KOKKOS_IMPL_SIMD_PERFTEST_HOST_REDUCTION_BENCH(common, reduce_max,
                                                   reduce_max);
    KOKKOS_IMPL_SIMD_PERFTEST_HOST_REDUCTION_BENCH(common, masked_reduce,
                                                   masked_reduce);
    KOKKOS_IMPL_SIMD_PERFTEST_HOST_REDUCTION_BENCH(common, masked_reduce_min,
                                                   masked_reduce_min);
    KOKKOS_IMPL_SIMD_PERFTEST_HOST_REDUCTION_BENCH(common, masked_reduce_max,
                                                   masked_reduce_max);
  }
}

template <typename Abi, typename DataType>
inline void host_register_math_benchmarks() {
  using ExecSpace = Kokkos::DefaultHostExecutionSpace;

  if constexpr (std::is_floating_point_v<DataType> &&
                is_type_v<Kokkos::Experimental::basic_simd<DataType, Abi>>) {
    KOKKOS_IMPL_SIMD_PERFTEST_HOST_UNARY_BENCH(math, exp, exp_op);
    KOKKOS_IMPL_SIMD_PERFTEST_HOST_UNARY_BENCH(math, exp2, exp2_op);
    KOKKOS_IMPL_SIMD_PERFTEST_HOST_UNARY_BENCH(math, log, log_op);
    KOKKOS_IMPL_SIMD_PERFTEST_HOST_UNARY_BENCH(math, log10, log10_op);
    KOKKOS_IMPL_SIMD_PERFTEST_HOST_UNARY_BENCH(math, log2, log2_op);
    KOKKOS_IMPL_SIMD_PERFTEST_HOST_UNARY_BENCH(math, sqrt, sqrt_op);
    KOKKOS_IMPL_SIMD_PERFTEST_HOST_UNARY_BENCH(math, cbrt, cbrt_op);
    KOKKOS_IMPL_SIMD_PERFTEST_HOST_UNARY_BENCH(math, sin, sin_op);
    KOKKOS_IMPL_SIMD_PERFTEST_HOST_UNARY_BENCH(math, cos, cos_op);
    KOKKOS_IMPL_SIMD_PERFTEST_HOST_UNARY_BENCH(math, tan, tan_op);
    KOKKOS_IMPL_SIMD_PERFTEST_HOST_UNARY_BENCH(math, asin, asin_op);
    KOKKOS_IMPL_SIMD_PERFTEST_HOST_UNARY_BENCH(math, acos, acos_op);
    KOKKOS_IMPL_SIMD_PERFTEST_HOST_UNARY_BENCH(math, atan, atan_op);
    KOKKOS_IMPL_SIMD_PERFTEST_HOST_UNARY_BENCH(math, sinh, sinh_op);
    KOKKOS_IMPL_SIMD_PERFTEST_HOST_UNARY_BENCH(math, cosh, cosh_op);
    KOKKOS_IMPL_SIMD_PERFTEST_HOST_UNARY_BENCH(math, tanh, tanh_op);
    KOKKOS_IMPL_SIMD_PERFTEST_HOST_UNARY_BENCH(math, asinh, asinh_op);
    KOKKOS_IMPL_SIMD_PERFTEST_HOST_UNARY_BENCH(math, acosh, acosh_op);
    KOKKOS_IMPL_SIMD_PERFTEST_HOST_UNARY_BENCH(math, atanh, atanh_op);
    KOKKOS_IMPL_SIMD_PERFTEST_HOST_UNARY_BENCH(math, erf, erf_op);
    KOKKOS_IMPL_SIMD_PERFTEST_HOST_UNARY_BENCH(math, erfc, erfc_op);
    KOKKOS_IMPL_SIMD_PERFTEST_HOST_UNARY_BENCH(math, tgamma, tgamma_op);
    KOKKOS_IMPL_SIMD_PERFTEST_HOST_UNARY_BENCH(math, lgamma, lgamma_op);

    KOKKOS_IMPL_SIMD_PERFTEST_HOST_BINARY_BENCH(math, pow, pow_op);
    KOKKOS_IMPL_SIMD_PERFTEST_HOST_BINARY_BENCH(math, hypot, hypot_op);
    KOKKOS_IMPL_SIMD_PERFTEST_HOST_BINARY_BENCH(math, atan2, atan2_op);
    KOKKOS_IMPL_SIMD_PERFTEST_HOST_BINARY_BENCH(math, copysign, copysign_op);

    KOKKOS_IMPL_SIMD_PERFTEST_HOST_TERNARY_BENCH(math, fma, fma_op);
    KOKKOS_IMPL_SIMD_PERFTEST_HOST_TERNARY_BENCH(math, ternary_hypot,
                                                 ternary_hypot_op);
  }
}

template <typename Abi, typename... DataTypes>
inline void host_register_benchmarks_all_types(
    Kokkos::Experimental::Impl::data_types<DataTypes...>) {
  (host_register_common_benchmarks<Abi, DataTypes>(), ...);
  (host_register_math_benchmarks<Abi, DataTypes>(), ...);
}

template <typename... Abis>
inline void host_register_benchmarks_all_abis(
    Kokkos::Experimental::Impl::abi_set<Abis...>) {
  using DataTypes = Kokkos::Experimental::Impl::data_type_set;
  (host_register_benchmarks_all_types<Abis>(DataTypes()), ...);
}

inline void register_host_benchmarks() {
  host_register_benchmarks_all_abis(Kokkos::Experimental::Impl::host_abi_set());
}

#endif
