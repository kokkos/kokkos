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
#include "fill.hpp"

#include <mdspan/mdspan.hpp>

#include <benchmark/benchmark.h>

#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <chrono>

//================================================================================

static constexpr int global_delta = 1;

using index_type = int;

template <class T, size_t... Es>
using lmdspan = Kokkos::mdspan<T, Kokkos::extents<index_type, Es...>, Kokkos::layout_left>;
template <class T, size_t... Es>
using rmdspan = Kokkos::mdspan<T, Kokkos::extents<index_type, Es...>, Kokkos::layout_right>;

//================================================================================

template <class MDSpan, class... DynSizes>
void BM_MDSpan_Stencil_3D(benchmark::State& state, MDSpan, DynSizes... dyn) {

  using value_type = typename MDSpan::value_type;
  auto buffer_size = MDSpan{nullptr, dyn...}.mapping().required_span_size();

  auto buffer_s = std::make_unique<value_type[]>(buffer_size);
  auto s = MDSpan{buffer_s.get(), dyn...};
  mdspan_benchmark::fill_random(s);

  auto buffer_o = std::make_unique<value_type[]>(buffer_size);
  auto o = MDSpan{buffer_o.get(), dyn...};
  mdspan_benchmark::fill_random(o);

  int d = global_delta;

  using index_type = typename MDSpan::index_type;
  for (auto _ : state) {
    benchmark::DoNotOptimize(o);
    for(index_type i = d; i < s.extent(0)-d; i ++) {
      for(index_type j = d; j < s.extent(1)-d; j ++) {
        for(index_type k = d; k < s.extent(2)-d; k ++) {
          value_type sum_local = 0;
          for(index_type di = i-d; di < i+d+1; di++) {
          for(index_type dj = j-d; dj < j+d+1; dj++) {
          for(index_type dk = k-d; dk < k+d+1; dk++) {
            sum_local += s(di, dj, dk);
          }}}
          o(i,j,k) = sum_local;
        }
      }
    }
    benchmark::ClobberMemory();
  }
  size_t num_inner_elements = (s.extent(0)-d) * (s.extent(1)-d) * (s.extent(2)-d);
  size_t stencil_num = (2*d+1) * (2*d+1) * (2*d+1);
  state.SetBytesProcessed( num_inner_elements * stencil_num * sizeof(value_type) * state.iterations());
}
MDSPAN_BENCHMARK_ALL_3D(BM_MDSpan_Stencil_3D, right_, rmdspan, 80, 80, 80);
MDSPAN_BENCHMARK_ALL_3D(BM_MDSpan_Stencil_3D, left_, lmdspan, 80, 80, 80);
MDSPAN_BENCHMARK_ALL_3D(BM_MDSpan_Stencil_3D, right_, rmdspan, 400, 400, 400);
MDSPAN_BENCHMARK_ALL_3D(BM_MDSpan_Stencil_3D, left_, lmdspan, 400, 400, 400);

//================================================================================

template <class T, class SizeX, class SizeY, class SizeZ>
void BM_Raw_Stencil_3D_right(benchmark::State& state, T, SizeX x, SizeY y, SizeZ z) {

  using MDSpan = Kokkos::mdspan<T, Kokkos::dextents<index_type, 3>>;
  using value_type = typename MDSpan::value_type;
  auto buffer_size = MDSpan{nullptr, x,y,z}.mapping().required_span_size();

  T* s_ptr = nullptr;
  auto buffer_s = std::make_unique<value_type[]>(buffer_size);
  {
    auto s = MDSpan{buffer_s.get(), x, y, z};
    mdspan_benchmark::fill_random(s);
    s_ptr = s.data_handle();
  }

  T* o_ptr = nullptr;
  auto buffer_o = std::make_unique<value_type[]>(buffer_size);
  {
    auto o = MDSpan{buffer_o.get(), x, y, z};
    mdspan_benchmark::fill_random(o);
    o_ptr = o.data_handle();
  }

  int d = global_delta;

  for (auto _ : state) {
    benchmark::DoNotOptimize(o_ptr);
    for(size_t i = d; i < x-d; i ++) {
      for(size_t j = d; j < y-d; j ++) {
        for(size_t k = d; k < z-d; k ++) {
          value_type sum_local = 0;
          for(size_t di = i-d; di < i+d+1; di++) {
          for(size_t dj = j-d; dj < j+d+1; dj++) {
          for(size_t dk = k-d; dk < k+d+1; dk++) {
            sum_local += s_ptr[dk + dj*z + di*z*y];
          }}}
          o_ptr[k + j*z + i*z*y] = sum_local;
        }
      }
    }
    benchmark::ClobberMemory();
  }
  size_t num_inner_elements = (x-d) * (y-d) * (z-d);
  size_t stencil_num = (2*d+1) * (2*d+1) * (2*d+1);
  state.SetBytesProcessed( num_inner_elements * stencil_num * sizeof(value_type) * state.iterations());
}
BENCHMARK_CAPTURE(BM_Raw_Stencil_3D_right, size_80_80_80, int(), size_t(80), size_t(80), size_t(80));
BENCHMARK_CAPTURE(BM_Raw_Stencil_3D_right, size_400_400_400, int(), size_t(400), size_t(400), size_t(400));

//================================================================================

BENCHMARK_MAIN();
