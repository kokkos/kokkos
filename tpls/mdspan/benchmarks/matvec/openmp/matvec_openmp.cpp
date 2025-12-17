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
#include <random>
#include <sstream>
#include <stdexcept>

#include "fill.hpp"

//================================================================================

static constexpr int global_repeat = 1;

//================================================================================

using index_type = int;
template <class T, size_t... Es>
using lmdspan = Kokkos::mdspan<T, Kokkos::extents<index_type, Es...>, Kokkos::layout_left>;
template <class T, size_t... Es>
using rmdspan = Kokkos::mdspan<T, Kokkos::extents<index_type, Es...>, Kokkos::layout_right>;

void throw_runtime_exception(const std::string &msg) {
  std::ostringstream o;
  o << msg;
  throw std::runtime_error(o.str());
}

template<class MDSpan>
void OpenMP_first_touch_2D(MDSpan s) {
  #pragma omp parallel for
  for(index_type i = 0; i < s.extent(0); i ++) {
    for(index_type j = 0; j < s.extent(1); j ++) {
      s(i,j) = 0;
    }
  }
}

template<class MDSpan>
void OpenMP_first_touch_1D(MDSpan s) {
  #pragma omp parallel for
  for(index_type i = 0; i < s.extent(0); i ++) {
    s(i) = 0;
  }
}

//================================================================================

template <class MDSpanMatrix, class... DynSizes>
void BM_MDSpan_OpenMP_MatVec(benchmark::State& state, MDSpanMatrix, DynSizes... dyn) {

  using value_type = typename MDSpanMatrix::value_type;
  using MDSpanVector = lmdspan<value_type,Kokkos::dynamic_extent>;

  auto buffer_size_A = MDSpanMatrix{nullptr, dyn...}.mapping().required_span_size();
  auto buffer_A = std::make_unique<value_type[]>(buffer_size_A);
  auto A = MDSpanMatrix{buffer_A.get(), dyn...};
  OpenMP_first_touch_2D(A);
  mdspan_benchmark::fill_random(A);

  auto buffer_size_x = MDSpanVector{nullptr, A.extent(1)}.mapping().required_span_size();
  auto buffer_x = std::make_unique<value_type[]>(buffer_size_x);
  auto x = MDSpanVector{buffer_x.get(), A.extent(1)};
  OpenMP_first_touch_1D(x);
  mdspan_benchmark::fill_random(x);

  auto buffer_size_y = MDSpanVector{nullptr, A.extent(0)}.mapping().required_span_size();
  auto buffer_y = std::make_unique<value_type[]>(buffer_size_y);
  auto y = MDSpanVector{buffer_y.get(), A.extent(0)};
  OpenMP_first_touch_1D(y);
  mdspan_benchmark::fill_random(y);

  #pragma omp parallel for
  for(index_type i = 0; i < A.extent(0); i ++) {
    value_type y_i = 0;
    for(index_type j = 0; j < A.extent(1); j ++) {
      y_i += A(i,j) * x(j);
    }
    y(i) = y_i;
  }

  int R = 10;
  for (auto _ : state) {
    auto ah = A.data_handle();
    auto yh = y.data_handle();
    auto xh = x.data_handle();
    benchmark::DoNotOptimize(ah);
    benchmark::DoNotOptimize(yh);
    benchmark::DoNotOptimize(xh);
    for(int r=0; r<R; r++) {
    #pragma omp parallel for
    for(index_type i = 0; i < A.extent(0); i ++) {
      value_type y_i = 0;
      for(index_type j = 0; j < A.extent(1); j ++) {
        y_i += A(i,j) * x(j);
      }
      y(i) += y_i;
    }
    }
    benchmark::ClobberMemory();
  }
  size_t num_elements = 2 * A.extent(0) * A.extent(1) + 2 * A.extent(0);
  state.SetBytesProcessed( R * num_elements * sizeof(value_type) * state.iterations() * global_repeat);
  state.counters["repeats"] = global_repeat;
}

BENCHMARK_CAPTURE(BM_MDSpan_OpenMP_MatVec, left, lmdspan<double,Kokkos::dynamic_extent,Kokkos::dynamic_extent>(), 100000, 5000);
BENCHMARK_CAPTURE(BM_MDSpan_OpenMP_MatVec, right, rmdspan<double,Kokkos::dynamic_extent,Kokkos::dynamic_extent>(), 100000, 5000);


template <class MDSpanMatrix, class... DynSizes>
void BM_MDSpan_OpenMP_MatVec_Raw_Left(benchmark::State& state, MDSpanMatrix, DynSizes... dyn) {

  using value_type = typename MDSpanMatrix::value_type;
  using MDSpanVector = lmdspan<value_type,Kokkos::dynamic_extent>;

  auto buffer_size_A = MDSpanMatrix{nullptr, dyn...}.mapping().required_span_size();
  auto buffer_A = std::make_unique<value_type[]>(buffer_size_A);
  auto A = MDSpanMatrix{buffer_A.get(), dyn...};
  OpenMP_first_touch_2D(A);
  mdspan_benchmark::fill_random(A);

  auto buffer_size_x = MDSpanVector{nullptr, A.extent(1)}.mapping().required_span_size();
  auto buffer_x = std::make_unique<value_type[]>(buffer_size_x);
  auto x = MDSpanVector{buffer_x.get(), A.extent(1)};
  OpenMP_first_touch_1D(x);
  mdspan_benchmark::fill_random(x);

  auto buffer_size_y = MDSpanVector{nullptr, A.extent(0)}.mapping().required_span_size();
  auto buffer_y = std::make_unique<value_type[]>(buffer_size_y);
  auto y = MDSpanVector{buffer_y.get(), A.extent(0)};
  OpenMP_first_touch_1D(y);
  mdspan_benchmark::fill_random(y);

  index_type N = A.extent(0);
  index_type M = A.extent(1);

  value_type* p_A = A.data_handle();
  value_type* p_x = x.data_handle();
  value_type* p_y = y.data_handle();

  #pragma omp parallel for
  for(index_type i = 0; i < N; i ++) {
    value_type y_i = 0;
    for(index_type j = 0; j < M; j ++) {
      y_i += p_A[i + j * N] * x[j];
    }
    y[i] = y_i;
  }

  int R = 10;
  for (auto _ : state) {
    auto ah = A.data_handle();
    auto yh = y.data_handle();
    auto xh = x.data_handle();
    benchmark::DoNotOptimize(ah);
    benchmark::DoNotOptimize(yh);
    benchmark::DoNotOptimize(xh);
    for(int r=0; r<R; r++) {
    #pragma omp parallel for
    for(index_type i = 0; i < A.extent(0); i ++) {
      value_type y_i = 0;
      for(index_type j = 0; j < A.extent(1); j ++) {
        y_i += p_A[i + j * N] * p_x[j];
      }
      p_y[i] += y_i;
    }
    }
    benchmark::ClobberMemory();
  }
  size_t num_elements = 2 * A.extent(0) * A.extent(1) + 2 * A.extent(0);
  state.SetBytesProcessed( R * num_elements * sizeof(value_type) * state.iterations() * global_repeat);
  state.counters["repeats"] = global_repeat;
}

BENCHMARK_CAPTURE(BM_MDSpan_OpenMP_MatVec_Raw_Left, left, lmdspan<double,Kokkos::dynamic_extent,Kokkos::dynamic_extent>(), 100000, 5000);

template <class MDSpanMatrix, class... DynSizes>
void BM_MDSpan_OpenMP_MatVec_Raw_Right(benchmark::State& state, MDSpanMatrix, DynSizes... dyn) {

  using value_type = typename MDSpanMatrix::value_type;
  using MDSpanVector = lmdspan<value_type,Kokkos::dynamic_extent>;

  auto buffer_size_A = MDSpanMatrix{nullptr, dyn...}.mapping().required_span_size();
  auto buffer_A = std::make_unique<value_type[]>(buffer_size_A);
  auto A = MDSpanMatrix{buffer_A.get(), dyn...};
  OpenMP_first_touch_2D(A);
  mdspan_benchmark::fill_random(A);

  auto buffer_size_x = MDSpanVector{nullptr, A.extent(1)}.mapping().required_span_size();
  auto buffer_x = std::make_unique<value_type[]>(buffer_size_x);
  auto x = MDSpanVector{buffer_x.get(), A.extent(1)};
  OpenMP_first_touch_1D(x);
  mdspan_benchmark::fill_random(x);

  auto buffer_size_y = MDSpanVector{nullptr, A.extent(0)}.mapping().required_span_size();
  auto buffer_y = std::make_unique<value_type[]>(buffer_size_y);
  auto y = MDSpanVector{buffer_y.get(), A.extent(0)};
  OpenMP_first_touch_1D(y);
  mdspan_benchmark::fill_random(y);

  index_type N = A.extent(0);
  index_type M = A.extent(1);

  value_type* p_A = A.data_handle();
  value_type* p_x = x.data_handle();
  value_type* p_y = y.data_handle();

  #pragma omp parallel for
  for(index_type i = 0; i < N; i ++) {
    value_type y_i = 0;
    for(index_type j = 0; j < M; j ++) {
      y_i += p_A[i * M + j] * x[j];
    }
    y[i] = y_i;
  }

  int R = 10;
  for (auto _ : state) {
    auto ah = A.data_handle();
    auto yh = y.data_handle();
    auto xh = x.data_handle();
    benchmark::DoNotOptimize(ah);
    benchmark::DoNotOptimize(yh);
    benchmark::DoNotOptimize(xh);
    for(int r=0; r<R; r++) {
    #pragma omp parallel for
    for(index_type i = 0; i < A.extent(0); i ++) {
      value_type y_i = 0;
      for(index_type j = 0; j < A.extent(1); j ++) {
        y_i += p_A[i * M + j] * p_x[j];
      }
      p_y[i] += y_i;
    }
    }
    benchmark::ClobberMemory();
  }
  size_t num_elements = 2 * A.extent(0) * A.extent(1) + 2 * A.extent(0);
  state.SetBytesProcessed( R * num_elements * sizeof(value_type) * state.iterations() * global_repeat);
  state.counters["repeats"] = global_repeat;
}

BENCHMARK_CAPTURE(BM_MDSpan_OpenMP_MatVec_Raw_Right, right, rmdspan<double,Kokkos::dynamic_extent,Kokkos::dynamic_extent>(), 100000, 5000);

//================================================================================

BENCHMARK_MAIN();
