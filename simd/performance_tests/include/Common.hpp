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

#ifndef KOKKOS_SIMD_PERF_TEST_COMMON_HPP
#define KOKKOS_SIMD_PERF_TEST_COMMON_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
#include <Kokkos_Random.hpp>
#include <cstdlib>
#include <benchmark/benchmark.h>

#include <Benchmark_Context.hpp>

template <class T, class ExecSpace>
class Args {
 public:
  using view_type =
      Kokkos::View<T*, ExecSpace, Kokkos::MemoryTraits<Kokkos::Aligned>>;
  using mask_view_type = Kokkos::View<bool*, ExecSpace>;

  view_type arg1;
  view_type arg2;
  view_type arg3;
  mask_view_type mask;

  explicit Args(std::size_t size)
      : arg1("arg1", size),
        arg2("arg2", size),
        arg3("arg3", size),
        mask("mask", size) {
    Kokkos::Random_XorShift64_Pool<ExecSpace> random_pool(58051);

    Kokkos::fill_random(arg1, random_pool, 1, 100);
    Kokkos::fill_random(arg2, random_pool, 1, 15);
    Kokkos::fill_random(arg3, random_pool, 1, 100);
  }
};

template <class Abi>
constexpr const char* abi_name() {
  using namespace Kokkos::Experimental::simd_abi;
  if constexpr (std::is_same_v<Abi, scalar>) {
    return "scalar";
#if defined(KOKKOS_ARCH_AVX2)
  } else if constexpr (std::is_same_v<Abi, avx2_fixed_size<4>>) {
    return "avx2 x4";
  } else if constexpr (std::is_same_v<Abi, avx2_fixed_size<8>>) {
    return "avx2 x8";
#elif defined(KOKKOS_ARCH_AVX512XEON)
  } else if constexpr (std::is_same_v<Abi, avx512_fixed_size<8>>) {
    return "avx512 x8";
  } else if constexpr (std::is_same_v<Abi, avx512_fixed_size<16>>) {
    return "avx512 x16";
#elif defined(KOKKOS_ARCH_ARM_NEON)
  } else if constexpr (std::is_same_v<Abi, neon_fixed_size<2>>) {
    return "neon x2";
  } else if constexpr (std::is_same_v<Abi, neon_fixed_size<4>>) {
    return "neon x4";
#endif
  }
}

template <class DataType>
constexpr const char* datatype_name() {
  if constexpr (std::is_same_v<DataType, std::int32_t>) {
    return "i32";
  } else if constexpr (std::is_same_v<DataType, std::uint32_t>) {
    return "u32";
  } else if constexpr (std::is_same_v<DataType, std::int64_t>) {
    return "i64";
  } else if constexpr (std::is_same_v<DataType, std::uint64_t>) {
    return "u64";
  } else if constexpr (std::is_same_v<DataType, float>) {
    return "float";
  } else if constexpr (std::is_same_v<DataType, double>) {
    return "double";
  }
}

template <typename Abi, typename DataType>
std::string benchmark_name(const char* prefix, const char* name) {
  const std::string sp(" ");
  return prefix + sp + abi_name<Abi>() + sp + datatype_name<DataType>() + sp +
         name;
}

// constexpr std::size_t BENCH_SIZE = 100000;
constexpr std::size_t BENCH_SIZE = 1'600'000;

template <class Abi, class UnaryOp, class T>
void host_bench_unary_op(benchmark::State& state) {
  using ExecSpace             = Kokkos::DefaultHostExecutionSpace;
  using simd_type             = Kokkos::Experimental::basic_simd<T, Abi>;
  constexpr std::size_t width = simd_type::size();

  UnaryOp op;

  Args<T, ExecSpace> args(BENCH_SIZE);

  Kokkos::View<T*, ExecSpace> res("res", BENCH_SIZE);

  for (auto _ : state) {
    simd_type a, x;
    for (std::size_t i = 0; i < BENCH_SIZE; i += width) {
      a.copy_from(args.arg1.data() + i,
                  Kokkos::Experimental::simd_flag_aligned);
#ifdef KOKKOS_SIMD_PERFTEST_HOSTFORCESERIAL
      benchmark::DoNotOptimize(x = op.on_host(a));
#else
      x              = op.on_host(a);
#endif
      x.copy_to(res.data() + i, Kokkos::Experimental::simd_flag_aligned);
    }
  }
}

template <class Abi, class BinaryOp, class T>
void host_bench_binary_op(benchmark::State& state) {
  using ExecSpace             = Kokkos::DefaultHostExecutionSpace;
  using simd_type             = Kokkos::Experimental::basic_simd<T, Abi>;
  constexpr std::size_t width = simd_type::size();

  Args<T, ExecSpace> args(BENCH_SIZE);
  BinaryOp op;

  Kokkos::View<T*, ExecSpace> res("res", BENCH_SIZE);

  for (auto _ : state) {
    simd_type a, b, x;
    for (std::size_t i = 0; i < BENCH_SIZE; i += width) {
      a.copy_from(args.arg1.data() + i,
                  Kokkos::Experimental::simd_flag_aligned);
      b.copy_from(args.arg2.data() + i,
                  Kokkos::Experimental::simd_flag_aligned);
#ifdef KOKKOS_SIMD_PERFTEST_HOSTFORCESERIAL
      benchmark::DoNotOptimize(x = op.on_host(a, b));
#else
      x              = op.on_host(a, b);
#endif
      x.copy_to(res.data() + i, Kokkos::Experimental::simd_flag_aligned);
    }
  }

  // benchmark::DoNotOptimize(res(static_cast<std::size_t>(args.arg1(0))));
}

template <class Abi, class TernaryOp, class T>
void host_bench_ternary_op(benchmark::State& state) {
  using ExecSpace             = Kokkos::DefaultHostExecutionSpace;
  using simd_type             = Kokkos::Experimental::basic_simd<T, Abi>;
  constexpr std::size_t width = simd_type::size();

  Args<T, ExecSpace> args(BENCH_SIZE);
  TernaryOp op;

  Kokkos::View<T*, ExecSpace> res("res", BENCH_SIZE);

  for (auto _ : state) {
    simd_type a, b, c, x;
    for (std::size_t i = 0; i < BENCH_SIZE; i += width) {
      a.copy_from(args.arg1.data() + i,
                  Kokkos::Experimental::simd_flag_aligned);
      b.copy_from(args.arg2.data() + i,
                  Kokkos::Experimental::simd_flag_aligned);
      c.copy_from(args.arg3.data() + i,
                  Kokkos::Experimental::simd_flag_aligned);
#ifdef KOKKOS_SIMD_PERFTEST_HOSTFORCESERIAL
      benchmark::DoNotOptimize(x = op.on_host(a, b, c));
#else
      x              = op.on_host(a, b, c);
#endif
      x.copy_to(res.data() + i, Kokkos::Experimental::simd_flag_aligned);
    }
  }

  benchmark::DoNotOptimize(res(static_cast<std::size_t>(args.arg1(0))));
}

template <class Abi, class ReductionOp, class T>
void host_bench_reduction_op(benchmark::State& state) {
  using ExecSpace             = Kokkos::DefaultHostExecutionSpace;
  using simd_type             = Kokkos::Experimental::basic_simd<T, Abi>;
  using mask_type             = typename simd_type::mask_type;
  constexpr std::size_t width = simd_type::size();

  ReductionOp op;

  Args<T, ExecSpace> args(BENCH_SIZE);

  Kokkos::View<typename simd_type::mask_type*, ExecSpace> masks(
      "masks", BENCH_SIZE / width);
  Kokkos::Random_XorShift64_Pool<ExecSpace> random_pool(58051);

  for (std::size_t i = 0; i < masks.size(); i++) {
    masks(i) = mask_type([=](std::size_t) {
      auto generator = random_pool.get_state();
      const bool val = generator.rand() % 2 == 0;
      random_pool.free_state(generator);
      return val;
    });
  }

  Kokkos::View<typename simd_type::value_type*, ExecSpace> res(
      "res", BENCH_SIZE / width);

  for (auto _ : state) {
    simd_type a;
    for (std::size_t i = 0; i < BENCH_SIZE; i += width) {
      a.copy_from(args.arg1.data() + i,
                  Kokkos::Experimental::simd_flag_aligned);
#ifdef KOKKOS_SIMD_PERFTEST_HOSTFORCESERIAL
      benchmark::DoNotOptimize(res(i / width) =
                                   op.on_host(a, masks(i / width)));
#else
      res(i / width) = op.on_host(a, masks(i / width));
#endif
    }
  }

  benchmark::DoNotOptimize(
      res(static_cast<std::size_t>(args.arg1(0)) % res.size()));
}

template <class Abi, class UnaryOp, class T>
void device_bench_unary_op(benchmark::State& state) {
  using ExecSpace             = Kokkos::DefaultExecutionSpace;
  using simd_type             = Kokkos::Experimental::basic_simd<T, Abi>;
  constexpr std::size_t width = simd_type::size();

  UnaryOp op;

  Args<T, ExecSpace> args(BENCH_SIZE);

  Kokkos::View<T*> res("res", BENCH_SIZE);

  for (auto _ : state) {
    Kokkos::Timer timer;
    Kokkos::parallel_for(
        1, KOKKOS_LAMBDA(std::size_t) {
          simd_type a, x;
          for (std::size_t i = 0; i < BENCH_SIZE; i += width) {
            a.copy_from(args.arg1.data() + i,
                        Kokkos::Experimental::simd_flag_aligned);
            x = op.on_device(a);
            x.copy_to(res.data() + i, Kokkos::Experimental::simd_flag_aligned);
          }
        });
    Kokkos::fence("After simd loop");
    state.SetIterationTime(timer.seconds());
  }
}

template <class Abi, class BinaryOp, class T>
void device_bench_binary_op(benchmark::State& state) {
  using ExecSpace             = Kokkos::DefaultExecutionSpace;
  using simd_type             = Kokkos::Experimental::basic_simd<T, Abi>;
  constexpr std::size_t width = simd_type::size();

  BinaryOp op;

  Args<T, ExecSpace> args(BENCH_SIZE);

  Kokkos::View<T*> res("res", BENCH_SIZE);

  for (auto _ : state) {
    Kokkos::Timer timer;
    Kokkos::parallel_for(
        1, KOKKOS_LAMBDA(std::size_t) {
          simd_type a, b, x;
          for (std::size_t i = 0; i < BENCH_SIZE; i += width) {
            a.copy_from(args.arg1.data() + i,
                        Kokkos::Experimental::simd_flag_aligned);
            b.copy_from(args.arg2.data() + i,
                        Kokkos::Experimental::simd_flag_aligned);
            x = op.on_device(a, b);
            x.copy_to(res.data() + i, Kokkos::Experimental::simd_flag_aligned);
          }
        });
    Kokkos::fence("After simd loop");
    state.SetIterationTime(timer.seconds());
  }
}

template <class Abi, class TernaryOp, class T>
void device_bench_ternary_op(benchmark::State& state) {
  using ExecSpace             = Kokkos::DefaultExecutionSpace;
  using simd_type             = Kokkos::Experimental::basic_simd<T, Abi>;
  constexpr std::size_t width = simd_type::size();

  TernaryOp op;

  Args<T, ExecSpace> args(BENCH_SIZE);

  Kokkos::View<T*> res("res", BENCH_SIZE);

  for (auto _ : state) {
    Kokkos::Timer timer;
    Kokkos::parallel_for(
        1, KOKKOS_LAMBDA(std::size_t) {
          simd_type a, b, c, x;
          for (std::size_t i = 0; i < BENCH_SIZE; i += width) {
            a.copy_from(args.arg1.data() + i,
                        Kokkos::Experimental::simd_flag_aligned);
            b.copy_from(args.arg2.data() + i,
                        Kokkos::Experimental::simd_flag_aligned);
            c.copy_from(args.arg3.data() + i,
                        Kokkos::Experimental::simd_flag_aligned);
            x = op.on_device(a, b, c);
            x.copy_to(res.data() + i, Kokkos::Experimental::simd_flag_aligned);
          }
        });
    Kokkos::fence("After simd loop");
    state.SetIterationTime(timer.seconds());
  }
}

template <class Abi, class ReductionOp, class T>
void device_bench_reduction_op(benchmark::State& state) {
  using ExecSpace             = Kokkos::DefaultExecutionSpace;
  using simd_type             = Kokkos::Experimental::basic_simd<T, Abi>;
  using mask_type             = typename simd_type::mask_type;
  constexpr std::size_t width = simd_type::size();

  ReductionOp op;

  Args<T, ExecSpace> args(BENCH_SIZE);

  Kokkos::View<typename simd_type::mask_type*> masks("masks",
                                                     BENCH_SIZE / width);

  auto policy = Kokkos::RangePolicy(ExecSpace(), 0, BENCH_SIZE / width);
  Kokkos::Random_XorShift64_Pool<ExecSpace> random_pool(58051);

  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(std::size_t i) {
        auto generator = random_pool.get_state();
        masks(i)       = mask_type(generator.rand() % 2 == 0);
        random_pool.free_state(generator);
      });
  Kokkos::fence("After filling the masks");

  Kokkos::View<typename simd_type::value_type*> res("res", BENCH_SIZE / width);

  for (auto _ : state) {
    Kokkos::Timer timer;
    Kokkos::parallel_for(
        1, KOKKOS_LAMBDA(std::size_t) {
          simd_type a;
          for (std::size_t i = 0; i < BENCH_SIZE; i += width) {
            a.copy_from(args.arg1.data() + i,
                        Kokkos::Experimental::simd_flag_aligned);
            res(i / width) = op.on_device(a, masks(i / width));
          }
        });
    Kokkos::fence("After simd loop");
    state.SetIterationTime(timer.seconds());
  }
}

// Simple check to loosely test that T is a complete type.
// Some capabilities are only defined for specific data type and abi pairs
// (i.e. extended vector width); this is used to exclude pairs that are not
// defined from being tested.
template <typename T, typename = void>
constexpr bool is_type_v = false;

template <typename T>
constexpr bool is_type_v<T, decltype(void(sizeof(T)))> = true;

#endif
