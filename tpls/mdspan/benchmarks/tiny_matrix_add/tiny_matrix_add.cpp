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

#include <memory>
#include <stdexcept>

#include "fill.hpp"

//================================================================================

using index_type = int;

template <class T, size_t... Es>
using lmdspan = Kokkos::mdspan<T, Kokkos::extents<index_type, Es...>, Kokkos::layout_left>;
template <class T, size_t... Es>
using rmdspan = Kokkos::mdspan<T, Kokkos::extents<index_type, Es...>, Kokkos::layout_right>;

//================================================================================

template <class MDSpan, class... DynSizes>
void BM_MDSpan_TinyMatrixSum_right(benchmark::State& state, MDSpan, DynSizes... dyn) {

  using value_type = typename MDSpan::value_type;
  auto buffer_size = MDSpan{nullptr, dyn...}.mapping().required_span_size();

  auto buffer_s = std::make_unique<value_type[]>(buffer_size);
  auto s = MDSpan{buffer_s.get(), dyn...};
  mdspan_benchmark::fill_random(s);

  auto buffer_o = std::make_unique<value_type[]>(buffer_size);
  auto o = MDSpan{buffer_o.get(), dyn...};
  mdspan_benchmark::fill_random(o);

  for (auto _ : state) {
    benchmark::DoNotOptimize(o);
    auto odata = o.data_handle();
    benchmark::DoNotOptimize(odata);
    benchmark::DoNotOptimize(s);
    auto sdata = s.data_handle();
    benchmark::DoNotOptimize(sdata);
    for(index_type i = 0; i < s.extent(0); i ++) {
      for(index_type j = 0; j < s.extent(1); j ++) {
        for(index_type k = 0; k < s.extent(2); k ++) {
          o(i,j,k) += s(i,j,k);
        }
      }
    }
    benchmark::ClobberMemory();
  }
  size_t num_elements = (s.extent(0) * s.extent(1) * s.extent(2));
  state.SetBytesProcessed( num_elements * 3 * sizeof(value_type) * state.iterations() );
}
MDSPAN_BENCHMARK_ALL_3D(BM_MDSpan_TinyMatrixSum_right, right_, lmdspan, 1000000, 3, 3);
MDSPAN_BENCHMARK_ALL_3D(BM_MDSpan_TinyMatrixSum_right, left_, lmdspan, 1000000, 3, 3);

//================================================================================

template <class T, class SizeX, class SizeY, class SizeZ>
void BM_Raw_Static_TinyMatrixSum_right(benchmark::State& state, T, SizeX x, SizeY y, SizeZ z) {

  using MDSpan = Kokkos::mdspan<T, Kokkos::dextents<index_type, 3>>;
  using value_type = typename MDSpan::value_type;
  auto buffer_size = MDSpan{nullptr, x,y,z}.mapping().required_span_size();

  auto buffer_s = std::make_unique<value_type[]>(buffer_size);
  auto s = MDSpan{buffer_s.get(), x,y,z};
  mdspan_benchmark::fill_random(s);
  T* s_ptr = s.data_handle();

  auto buffer_o = std::make_unique<value_type[]>(buffer_size);
  auto o = MDSpan{buffer_o.get(), x,y,z};
  mdspan_benchmark::fill_random(o);
  T* o_ptr = o.data_handle();

  for (auto _ : state) {
    benchmark::DoNotOptimize(o_ptr);
    benchmark::DoNotOptimize(s_ptr);
    for(index_type i = 0; i < 1000000; i ++) {
      for(index_type j = 0; j < 3; j ++) {
        for(index_type k = 0; k < 3; k ++) {
          o_ptr[k + j*3 + i*3*3] += s_ptr[k + j*3 + i*3*3];
        }
      }
    }
    benchmark::ClobberMemory();
  }
  size_t num_inner_elements = x * y * z;
  state.SetBytesProcessed( num_inner_elements * 3  * sizeof(value_type) * state.iterations());
}
BENCHMARK_CAPTURE(BM_Raw_Static_TinyMatrixSum_right, size_1000000_3_3, int(), 1000000, 3, 3);

//================================================================================

template <class T, class SizeX, class SizeY, class SizeZ>
void BM_Raw_TinyMatrixSum_right(benchmark::State& state, T, SizeX x, SizeY y, SizeZ z) {

  benchmark::DoNotOptimize(x);
  benchmark::DoNotOptimize(y);
  benchmark::DoNotOptimize(z);

  using MDSpan = Kokkos::mdspan<T, Kokkos::dextents<index_type, 3>>;
  using value_type = typename MDSpan::value_type;

  auto buffer_s = std::make_unique<value_type[]>(x * y * z);
  {
    auto s = MDSpan{buffer_s.get(), x, y, z};
    mdspan_benchmark::fill_random(s);
  }
  T* s_ptr = buffer_s.get();

  auto buffer_o = std::make_unique<value_type[]>(x * y * z);
  {
    auto o = MDSpan{buffer_o.get(), x, y, z};
    mdspan_benchmark::fill_random(o);
  }
  T* o_ptr = buffer_o.get();

  for (auto _ : state) {
    benchmark::DoNotOptimize(o_ptr);
    benchmark::DoNotOptimize(s_ptr);
    for(SizeX i = 0; i < x; i ++) {
      for(SizeY j = 0; j < y; j ++) {
        for(SizeZ k = 0; k < z; k ++) {
          o_ptr[k + j*z + i*z*y] += s_ptr[k + j*z + i*z*y];
        }
      }
    }
    benchmark::ClobberMemory();
  }
  size_t num_inner_elements = x * y * z;
  state.SetBytesProcessed( num_inner_elements * y  * sizeof(value_type) * state.iterations());
}
BENCHMARK_CAPTURE(BM_Raw_TinyMatrixSum_right, size_1000000_3_3, int(), size_t(1000000), size_t(3), size_t(3));

//================================================================================

BENCHMARK_MAIN();
