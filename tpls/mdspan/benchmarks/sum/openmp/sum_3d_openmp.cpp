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
#include <mdspan/mdspan.hpp>

#include "sum_3d_common.hpp"
#include "fill.hpp"

#include <memory>
#include <random>
#include <omp.h>

//================================================================================

using index_type = int;

template <class T, size_t... Es>
using lmdspan = Kokkos::mdspan<T, Kokkos::extents<index_type, Es...>, Kokkos::layout_left>;
template <class T, size_t... Es>
using rmdspan = Kokkos::mdspan<T, Kokkos::extents<index_type, Es...>, Kokkos::layout_right>;

//================================================================================

template <class MDSpan, class... DynSizes>
void BM_MDSpan_Sum_3D_OpenMP(benchmark::State& state, MDSpan, DynSizes... dyn) {
  using value_type = typename MDSpan::value_type;
  auto buffer = std::make_unique<value_type[]>(
    MDSpan{nullptr, dyn...}.mapping().required_span_size()
  );
  auto s = MDSpan{buffer.get(), dyn...};

  int repeats = s.size() > (100*100*100) ? 50 : 1000;

  mdspan_benchmark::fill_random(s);
  for (auto _ : state) {
    #pragma omp parallel default(none) shared(repeats, s)
    {
      for (int r = 0; r < repeats; ++r) {
        value_type sum = 0;
        for (index_type i = omp_get_thread_num(); i < s.extent(0); i += omp_get_num_threads()) {
          for (index_type j = 0; j < s.extent(1); ++j) {
            for (index_type k = 0; k < s.extent(2); ++k) {
              sum += s(i, j, k);
            }
          }
        }
        benchmark::DoNotOptimize(sum);
        auto h = s.data_handle();
        benchmark::DoNotOptimize(h);
      }
    }
  }
  state.SetBytesProcessed(s.size() * sizeof(value_type) * state.iterations() * repeats);
  state.counters["repeats"] = repeats;
}
MDSPAN_BENCHMARK_ALL_3D(BM_MDSpan_Sum_3D_OpenMP, left_, lmdspan, 20, 20, 20);
MDSPAN_BENCHMARK_ALL_3D(BM_MDSpan_Sum_3D_OpenMP, right_, rmdspan, 20, 20, 20);
MDSPAN_BENCHMARK_ALL_3D(BM_MDSpan_Sum_3D_OpenMP, left_, lmdspan, 200, 200, 200);
MDSPAN_BENCHMARK_ALL_3D(BM_MDSpan_Sum_3D_OpenMP, right_, rmdspan, 200, 200, 200);

//================================================================================

template <class MDSpan, class... DynSizes>
void BM_MDSpan_Sum_3D_loop_OpenMP(benchmark::State& state, MDSpan, DynSizes... dyn) {
  using value_type = typename MDSpan::value_type;
  auto buffer = std::make_unique<value_type[]>(
    MDSpan{nullptr, dyn...}.mapping().required_span_size()
  );
  auto s = MDSpan{buffer.get(), dyn...};

  mdspan_benchmark::fill_random(s);
  auto sums_buffer = std::make_unique<value_type[]>(omp_get_max_threads());
  auto* sum = sums_buffer.get();
  for (auto _ : state) {
    auto sbv = sums_buffer.get();
    auto sh = s.data_handle();
    benchmark::DoNotOptimize(sbv);
    benchmark::DoNotOptimize(sh);
    #pragma omp parallel for default(none) shared(s, sum)
    for (index_type i = 0; i < s.extent(0); ++i) {
      for (index_type j = 0; j < s.extent(1); ++j) {
        for (index_type k = 0; k < s.extent(2); ++k) {
          sum[omp_get_thread_num()] += s(i, j, k);
        }
      }
    }
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(s.size() * sizeof(value_type) * state.iterations());
}
MDSPAN_BENCHMARK_ALL_3D(BM_MDSpan_Sum_3D_loop_OpenMP, right_, rmdspan, 200, 200, 200);
MDSPAN_BENCHMARK_ALL_3D(BM_MDSpan_Sum_3D_loop_OpenMP, left_, lmdspan, 200, 200, 200);

//================================================================================

template <class T, class SizeX, class SizeY, class SizeZ>
void BM_Raw_Sum_3D_OpenMP(benchmark::State& state, T, SizeX x, SizeY y, SizeZ z) {
  auto buffer = std::make_unique<T[]>(x * y * z);
  {
    // just for setup...
    auto wrapped = Kokkos::mdspan<T, Kokkos::dextents<index_type, 1>>{buffer.get(), x*y*z};
    mdspan_benchmark::fill_random(wrapped);
  }

  int repeats = x*y*z > (100*100*100) ? 50 : 1000;

  T* data = buffer.get();
  for (auto _ : state) {
    #pragma omp parallel default(none) shared(data,x,y,z,repeats)
    {
      for (int r = 0; r < repeats; ++r) {
        T sum = 0;
        benchmark::DoNotOptimize(sum);
        benchmark::DoNotOptimize(data);
        for (index_type i = omp_get_thread_num(); i < x; i += omp_get_num_threads()) {
          for (index_type j = 0; j < y; ++j) {
            for (index_type k = 0; k < z; ++k) {
              sum += data[k + j*z + i*z*y];
            }
          }
        }
        benchmark::ClobberMemory();
      }
    }
  }
  state.SetBytesProcessed(x * y * z * sizeof(T) * state.iterations() * repeats);
  state.counters["repeats"] = repeats;
}
BENCHMARK_CAPTURE(
  BM_Raw_Sum_3D_OpenMP, size_20_20_20, int(), 20, 20, 20
);
BENCHMARK_CAPTURE(
  BM_Raw_Sum_3D_OpenMP, size_200_200_200, int(), 200, 200, 200
);


//================================================================================

BENCHMARK_MAIN();
