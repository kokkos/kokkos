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

#include <omp.h>

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
void OpenMP_first_touch_3D(MDSpan s) {
  #pragma omp parallel for
  for(index_type i = 0; i < s.extent(0); i ++) {
    for(index_type j = 0; j < s.extent(1); j ++) {
      for(index_type k = 0; k < s.extent(2); k ++) {
        s(i,j,k) = 0;
      }
    }
  }
}

//================================================================================

template <class MDSpan, class... DynSizes>
void BM_MDSpan_OpenMP_noloop_TinyMatrixSum(benchmark::State& state, MDSpan, DynSizes... dyn) {

  using value_type = typename MDSpan::value_type;
  auto buffer_size = MDSpan{nullptr, dyn...}.mapping().required_span_size();

  auto buffer_s = std::make_unique<value_type[]>(buffer_size);
  auto s = MDSpan{buffer_s.get(), dyn...};
  OpenMP_first_touch_3D(s);
  mdspan_benchmark::fill_random(s);

  auto buffer_o = std::make_unique<value_type[]>(buffer_size);
  auto o = MDSpan{buffer_o.get(), dyn...};
  OpenMP_first_touch_3D(o);
  mdspan_benchmark::fill_random(o);


  for (auto _ : state) {
    auto oh = o.data_handle();
    auto sh = s.data_handle();
    benchmark::DoNotOptimize(oh);
    benchmark::DoNotOptimize(sh);
#pragma omp parallel
    {
      auto chunk_size = s.extent(0) / omp_get_num_threads();
      auto chunk_start = chunk_size * omp_get_thread_num();
      auto extra = s.extent(0) % chunk_size;
      if (omp_get_thread_num() < extra) {
        chunk_size += 1;
        chunk_start += omp_get_thread_num();
      } else {
        chunk_start += extra;
      }
      auto chunk_end = chunk_start + chunk_size;
      for (index_type i = chunk_start; i < chunk_end; ++i) {
        for (int r = 0; r < global_repeat; r++) {
          for (index_type j = 0; j < s.extent(1); j++) {
            for (index_type k = 0; k < s.extent(2); k++) {
              o(i, j, k) += s(i, j, k);
            }
          }
        }
      }
    }
    benchmark::ClobberMemory();
  }
  size_t num_elements = (s.extent(0) * s.extent(1) * s.extent(2));
  state.SetBytesProcessed( num_elements * 3 * sizeof(value_type) * state.iterations() * global_repeat);
  state.counters["repeats"] = global_repeat;
}
MDSPAN_BENCHMARK_ALL_3D(BM_MDSpan_OpenMP_noloop_TinyMatrixSum, right_, rmdspan, 1000000, 3, 3);
MDSPAN_BENCHMARK_ALL_3D(BM_MDSpan_OpenMP_noloop_TinyMatrixSum, left_, lmdspan, 1000000, 3, 3);

//================================================================================

template <class MDSpan, class... DynSizes>
void BM_MDSpan_OpenMP_TinyMatrixSum(benchmark::State& state, MDSpan, DynSizes... dyn) {

  using value_type = typename MDSpan::value_type;
  using index_type = typename MDSpan::index_type;
  auto buffer_size = MDSpan{nullptr, dyn...}.mapping().required_span_size();

  auto buffer_s = std::make_unique<value_type[]>(buffer_size);
  auto s = MDSpan{buffer_s.get(), dyn...};
  OpenMP_first_touch_3D(s);
  mdspan_benchmark::fill_random(s);

  auto buffer_o = std::make_unique<value_type[]>(buffer_size);
  auto o = MDSpan{buffer_o.get(), dyn...};
  OpenMP_first_touch_3D(o);
  mdspan_benchmark::fill_random(o);

  #pragma omp parallel for
  for(index_type i = 0; i < s.extent(0); i ++) {
    for(int r = 0; r<global_repeat; r++) {
      for(index_type j = 0; j < s.extent(1); j ++) {
        for(index_type k = 0; k < s.extent(2); k ++) {
          o(i,j,k) += s(i,j,k);
        }
      }
    }
  }

  for (auto _ : state) {
    auto oh = o.data_handle();
    auto sh = s.data_handle();
    benchmark::DoNotOptimize(oh);
    benchmark::DoNotOptimize(sh);
    #pragma omp parallel for
    for(index_type i = 0; i < s.extent(0); i ++) {
      for(int r = 0; r<global_repeat; r++) {
        for(index_type j = 0; j < s.extent(1); j ++) {
          for(index_type k = 0; k < s.extent(2); k ++) {
            o(i,j,k) += s(i,j,k);
          }
        }
      }
    }
    benchmark::ClobberMemory();
  }
  size_t num_elements = (s.extent(0) * s.extent(1) * s.extent(2));
  state.SetBytesProcessed( num_elements * 3 * sizeof(value_type) * state.iterations() * global_repeat);
  state.counters["repeats"] = global_repeat;
}
MDSPAN_BENCHMARK_ALL_3D_REAL_TIME(BM_MDSpan_OpenMP_TinyMatrixSum, right_, rmdspan, 1000000, 3, 3);
MDSPAN_BENCHMARK_ALL_3D_REAL_TIME(BM_MDSpan_OpenMP_TinyMatrixSum, left_, lmdspan, 1000000, 3, 3);

//================================================================================

template <class T, class SizeX, class SizeY, class SizeZ>
void BM_Raw_Static_OpenMP_TinyMatrixSum_right(benchmark::State& state, T, SizeX x, SizeY y, SizeZ z) {

  using MDSpan = Kokkos::mdspan<T, Kokkos::dextents<index_type, 3>>;
  using value_type = typename MDSpan::value_type;
  auto buffer_size = MDSpan{nullptr, x,y,z}.mapping().required_span_size();

  auto buffer_s = std::make_unique<value_type[]>(buffer_size);
  auto s = MDSpan{buffer_s.get(), x,y,z};
  OpenMP_first_touch_3D(s);
  mdspan_benchmark::fill_random(s);
  T* s_ptr = s.data_handle();

  auto buffer_o = std::make_unique<value_type[]>(buffer_size);
  auto o = MDSpan{buffer_o.get(), x,y,z};
  OpenMP_first_touch_3D(o);
  mdspan_benchmark::fill_random(o);
  T* o_ptr = o.data_handle();

  #pragma omp parallel for
  for(SizeX i = 0; i < x; i ++) {
      for(int r = 0; r<global_repeat; r++) {
        for(size_t j = 0; j < 3; j ++) {
          for(size_t k = 0; k < 3; k ++) {
            o_ptr[k + j*3 + i*3*3] += s_ptr[k + j*3 + i*3*3];
          }
        }
      }
  }

  for (auto _ : state) {
    benchmark::DoNotOptimize(o_ptr);
    benchmark::DoNotOptimize(s_ptr);
    #pragma omp parallel for
    for(size_t i = 0; i < 1000000; i ++) {
      for(int r = 0; r<global_repeat; r++) {
        for(size_t j = 0; j < 3; j ++) {
          for(size_t k = 0; k < 3; k ++) {
            o_ptr[k + j*3 + i*3*3] += s_ptr[k + j*3 + i*3*3];
          }
        }
      }
    }
    benchmark::ClobberMemory();
  }
  size_t num_inner_elements = x * y * z;
  state.SetBytesProcessed( num_inner_elements * 3 * global_repeat * sizeof(value_type) * state.iterations());
  state.counters["repeats"] = global_repeat;
}
BENCHMARK_CAPTURE(BM_Raw_Static_OpenMP_TinyMatrixSum_right, size_1000000_3_3, int(), 1000000, 3, 3)->UseRealTime();

template <class T, class SizeX, class SizeY, class SizeZ>
void BM_Raw_OpenMP_TinyMatrixSum_right(benchmark::State& state, T, SizeX x, SizeY y, SizeZ z) {

  using MDSpan = Kokkos::mdspan<T, Kokkos::dextents<index_type, 3>>;
  using value_type = typename MDSpan::value_type;
  auto buffer_size = MDSpan{nullptr, x,y,z}.mapping().required_span_size();

  auto buffer_s = std::make_unique<value_type[]>(buffer_size);
  auto s = MDSpan{buffer_s.get(), x,y,z};
  OpenMP_first_touch_3D(s);
  mdspan_benchmark::fill_random(s);
  T* s_ptr = s.data_handle();

  auto buffer_o = std::make_unique<value_type[]>(buffer_size);
  auto o = MDSpan{buffer_o.get(), x,y,z};
  OpenMP_first_touch_3D(o);
  mdspan_benchmark::fill_random(o);
  T* o_ptr = o.data_handle();

#pragma omp parallel for
  for(SizeX i = 0; i < x; i ++) {
    for(int r = 0; r<global_repeat; r++) {
      for(SizeY j = 0; j < y; j ++) {
        for(SizeZ k = 0; k < z; k ++) {
          o_ptr[k + j*y + i*y*z] += s_ptr[k + j*y + i*y*z];
        }
      }
    }
  }

  for (auto _ : state) {
    benchmark::DoNotOptimize(o_ptr);
    benchmark::DoNotOptimize(s_ptr);
#pragma omp parallel for
    for(SizeX i = 0; i < x; i ++) {
      for(int r = 0; r<global_repeat; r++) {
        for(SizeY j = 0; j < y; j ++) {
          for(SizeZ k = 0; k < z; k ++) {
            o_ptr[k + j*y + i*y*z] += s_ptr[k + j*y + i*y*z];
          }
        }
      }
    }
    benchmark::ClobberMemory();
  }
  size_t num_inner_elements = x * y * z;
  state.SetBytesProcessed( num_inner_elements * y * global_repeat * sizeof(value_type) * state.iterations());
  state.counters["repeats"] = global_repeat;
}
BENCHMARK_CAPTURE(BM_Raw_OpenMP_TinyMatrixSum_right, size_1000000_3_3, int(), 1000000, 3, 3)->UseRealTime();

//================================================================================

template <class T, class SizeX, class SizeY, class SizeZ>
void BM_Raw_Static_OpenMP_TinyMatrixSum_left(benchmark::State& state, T, SizeX x, SizeY y, SizeZ z) {

  using MDSpan = Kokkos::mdspan<T, Kokkos::extents<index_type, Kokkos::dynamic_extent, 3, 3>>;
  using value_type = typename MDSpan::value_type;
  auto buffer_size = MDSpan{nullptr, x}.mapping().required_span_size();

  auto buffer_s = std::make_unique<value_type[]>(buffer_size);
  auto s = MDSpan{buffer_s.get(), x};
  OpenMP_first_touch_3D(s);
  mdspan_benchmark::fill_random(s);
  T* s_ptr = s.data_handle();

  auto buffer_o = std::make_unique<value_type[]>(buffer_size);
  auto o = MDSpan{buffer_o.get(), x};
  OpenMP_first_touch_3D(o);
  mdspan_benchmark::fill_random(o);
  T* o_ptr = o.data_handle();

  #pragma omp parallel for
  for(SizeX i = 0; i < x; i ++) {
      for(int r = 0; r<global_repeat; r++) {
        for(size_t j = 0; j < 3; j ++) {
          for(size_t k = 0; k < 3; k ++) {
            o_ptr[k*x*3 + j*x + i] += s_ptr[k*x*3 + j*x + i];
          }
        }
      }
  }

  for (auto _ : state) {
    benchmark::DoNotOptimize(o_ptr);
    benchmark::DoNotOptimize(s_ptr);
    #pragma omp parallel for
    for(size_t i = 0; i < 1000000; i ++) {
      for(int r = 0; r<global_repeat; r++) {
        for(size_t j = 0; j < 3; j ++) {
          for(size_t k = 0; k < 3; k ++) {
            o_ptr[k*x*3 + j*x + i] += s_ptr[k*x*3 + j*x + i];
          }
        }
      }
    }
    benchmark::ClobberMemory();
  }
  size_t num_inner_elements = x * y * z;
  state.SetBytesProcessed( num_inner_elements * 3 * global_repeat * sizeof(value_type) * state.iterations());
  state.counters["repeats"] = global_repeat;
}
BENCHMARK_CAPTURE(BM_Raw_Static_OpenMP_TinyMatrixSum_left, size_1000000_3_3, int(), 1000000, 3, 3)->UseRealTime();

//================================================================================

template <class T, class SizeX, class SizeY, class SizeZ>
void BM_Raw_OpenMP_TinyMatrixSum_left(benchmark::State& state, T, SizeX x, SizeY y, SizeZ z) {

  using MDSpan = Kokkos::mdspan<T, Kokkos::extents<index_type, Kokkos::dynamic_extent, 3, 3>>;
  using value_type = typename MDSpan::value_type;
  auto buffer_size = MDSpan{nullptr, x}.mapping().required_span_size();

  auto buffer_s = std::make_unique<value_type[]>(buffer_size);
  auto s = MDSpan{buffer_s.get(), x};
  OpenMP_first_touch_3D(s);
  mdspan_benchmark::fill_random(s);
  T* s_ptr = s.data_handle();

  auto buffer_o = std::make_unique<value_type[]>(buffer_size);
  auto o = MDSpan{buffer_o.get(), x};
  OpenMP_first_touch_3D(o);
  mdspan_benchmark::fill_random(o);
  T* o_ptr = o.data_handle();

  #pragma omp parallel for
  for(SizeX i = 0; i < x; i ++) {
    for(int r = 0; r<global_repeat; r++) {
      for(SizeY j = 0; j < y; j ++) {
        for(SizeZ k = 0; k < z; k ++) {
          o_ptr[k*x*y + j*x + i] += s_ptr[k*x*y + j*x + i];
        }
      }
    }
  }

  for (auto _ : state) {
    benchmark::DoNotOptimize(o_ptr);
    benchmark::DoNotOptimize(s_ptr);
    #pragma omp parallel for
    for(SizeX i = 0; i < x; i ++) {
      for(int r = 0; r<global_repeat; r++) {
        for(SizeY j = 0; j < y; j ++) {
          for(SizeZ k = 0; k < z; k ++) {
            o_ptr[k*x*y + j*x + i] += s_ptr[k*x*y + j*x + i];
          }
        }
      }
    }
    benchmark::ClobberMemory();
  }
  size_t num_inner_elements = x * y * z;
  state.SetBytesProcessed( num_inner_elements * 3 * global_repeat * sizeof(value_type) * state.iterations());
  state.counters["repeats"] = global_repeat;
}
BENCHMARK_CAPTURE(BM_Raw_OpenMP_TinyMatrixSum_left, size_1000000_3_3, int(), 1000000, 3, 3)->UseRealTime();

template <class MDSpan>
typename MDSpan::value_type*** make_3d_ptr_array(MDSpan s) {
  static_assert(std::is_same<typename MDSpan::layout_type,Kokkos::layout_right>::value,"Creating MD Ptr only works from mdspan with layout_right");
  using value_type = typename MDSpan::value_type;
  using index_type = typename MDSpan::index_type;

  value_type*** ptr= new value_type**[s.extent(0)];
  for(index_type i = 0; i<s.extent(0); i++) {
    ptr[i] = new value_type*[s.extent(1)];
    for(index_type j = 0; j<s.extent(1); j++)
      ptr[i][j]=&s(i,j,0);
  }
  return ptr;
}

template <class T>
void free_3d_ptr_array(T*** ptr, size_t extent_0) {
  for(size_t i=0; i<extent_0; i++)
    delete [] ptr[i];
  delete [] ptr;
}

template <class T, class SizeX, class SizeY, class SizeZ>
void BM_RawMDPtr_OpenMP_TinyMatrixSum_right(benchmark::State& state, T, SizeX x, SizeY y, SizeZ z) {

  using MDSpan = Kokkos::mdspan<T, Kokkos::extents<index_type, Kokkos::dynamic_extent, 3, 3>>;
  using value_type = typename MDSpan::value_type;
  auto buffer_size = MDSpan{nullptr, x}.mapping().required_span_size();

  auto buffer_s = std::make_unique<value_type[]>(buffer_size);
  auto s = MDSpan{buffer_s.get(), x};
  OpenMP_first_touch_3D(s);
  mdspan_benchmark::fill_random(s);
  T*** s_ptr = make_3d_ptr_array(s);

  auto buffer_o = std::make_unique<value_type[]>(buffer_size);
  auto o = MDSpan{buffer_o.get(), x};
  OpenMP_first_touch_3D(o);
  mdspan_benchmark::fill_random(o);
  T*** o_ptr = make_3d_ptr_array(o);

  #pragma omp parallel for
  for(SizeX i = 0; i < x; i ++) {
    for(int r = 0; r<global_repeat; r++) {
      for(size_t j = 0; j < 3; j ++) {
        for(size_t k = 0; k < 3; k ++) {
          o_ptr[i][j][k] += s_ptr[i][j][k];
        }
      }
    }
  }

  for (auto _ : state) {
    benchmark::DoNotOptimize(o_ptr);
    benchmark::DoNotOptimize(s_ptr);
    #pragma omp parallel for
    for(SizeX i = 0; i < x; i ++) {
      for(int r = 0; r<global_repeat; r++) {
        for(size_t j = 0; j < 3; j ++) {
          for(size_t k = 0; k < 3; k ++) {
            o_ptr[i][j][k] += s_ptr[i][j][k];
          }
        }
      }
    }
    benchmark::ClobberMemory();
  }
  size_t num_inner_elements = x * y * z;
  state.SetBytesProcessed( num_inner_elements * 3 * global_repeat * sizeof(value_type) * state.iterations());
  state.counters["repeats"] = global_repeat;
  free_3d_ptr_array(s_ptr,s.extent(0));
  free_3d_ptr_array(o_ptr,o.extent(0));
}
BENCHMARK_CAPTURE(BM_RawMDPtr_OpenMP_TinyMatrixSum_right, size_1000000_3_3, int(), 1000000, 3, 3)->UseRealTime();


//================================================================================

BENCHMARK_MAIN();
