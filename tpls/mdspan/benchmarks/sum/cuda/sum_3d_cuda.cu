/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <mdspan/mdspan.hpp>

#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>

#include "sum_3d_common.hpp"
#include "fill.hpp"

//================================================================================

static constexpr int warpsPerBlock = 4;

//================================================================================

template <class T, size_t... Es>
using lmdspan = Kokkos::mdspan<T, Kokkos::extents<int, Es...>, Kokkos::layout_left>;
template <class T, size_t... Es>
using rmdspan = Kokkos::mdspan<T, Kokkos::extents<int, Es...>, Kokkos::layout_right>;

//================================================================================

template <class Tp>
MDSPAN_FORCE_INLINE_FUNCTION inline
void DoNotOptimize(Tp const& value) {
  // Can't have m constraints on device
  asm volatile("" : : "r"(value) : "memory");
}

template <class Tp>
MDSPAN_FORCE_INLINE_FUNCTION inline
void DoNotOptimize(Tp& value) {
  // Can't have m constraints on device
  asm volatile("" : "+r"(value) : : "memory");
}

//================================================================================

void throw_runtime_exception(const std::string &msg) {
  std::ostringstream o;
  o << msg;
  throw std::runtime_error(o.str());
}

void cuda_internal_error_throw(cudaError e, const char* name,
  const char* file = NULL, const int line = 0) {
  std::ostringstream out;
  out << name << " error( " << cudaGetErrorName(e)
      << "): " << cudaGetErrorString(e);
  if (file) {
    out << " " << file << ":" << line;
  }
  throw_runtime_exception(out.str());
}

inline void cuda_internal_safe_call(cudaError e, const char* name,
       const char* file = NULL,
       const int line   = 0) {
  if (cudaSuccess != e) {
    cuda_internal_error_throw(e, name, file, line);
  }
}

#define CUDA_SAFE_CALL(call) \
  cuda_internal_safe_call(call, #call, __FILE__, __LINE__)

//================================================================================

dim3 get_bench_grid() {
  cudaDeviceProp cudaProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&cudaProp, 0));
  return dim3(cudaProp.multiProcessorCount, 1, 1);
}

dim3 get_bench_thread_block() {
  cudaDeviceProp cudaProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&cudaProp, 1));
  return dim3(1, cudaProp.warpSize, warpsPerBlock);
}

template <class F, class... Args>
__global__
void do_run_kernel(F f, Args... args) {
  f(args...);
}

template <class F, class... Args>
float run_kernel_timed(F&& f, Args&&... args) {
  cudaEvent_t start, stop;
  CUDA_SAFE_CALL(cudaEventCreate(&start));
  CUDA_SAFE_CALL(cudaEventCreate(&stop));

  CUDA_SAFE_CALL(cudaEventRecord(start));
  do_run_kernel<<<get_bench_grid(), get_bench_thread_block()>>>(
    (F&&)f, ((Args&&) args)...
  );
  CUDA_SAFE_CALL(cudaEventRecord(stop));
  CUDA_SAFE_CALL(cudaEventSynchronize(stop));
  float milliseconds = 0;
  CUDA_SAFE_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
  return milliseconds;
}

template <class MDSpan, class... DynSizes>
MDSpan fill_device_mdspan(MDSpan, DynSizes... dyn) {

  using value_type = typename MDSpan::value_type;
  auto buffer_size = MDSpan{nullptr, dyn...}.mapping().required_span_size();
  auto host_buffer = std::make_unique<value_type[]>(
    MDSpan{nullptr, dyn...}.mapping().required_span_size()
  );
  auto host_mdspan = MDSpan{host_buffer.get(), dyn...};
  mdspan_benchmark::fill_random(host_mdspan);

  value_type* device_buffer = nullptr;
  CUDA_SAFE_CALL(cudaMalloc(&device_buffer, buffer_size * sizeof(value_type)));
  CUDA_SAFE_CALL(cudaMemcpy(
    device_buffer, host_buffer.get(), buffer_size * sizeof(value_type), cudaMemcpyHostToDevice
  ));
  return MDSpan{device_buffer, dyn...};
}

//================================================================================

template <class MDSpan, class... DynSizes>
void BM_MDSpan_Cuda_Sum_3D(benchmark::State& state, MDSpan, DynSizes... dyn) {

  using value_type = typename MDSpan::value_type;
  auto s = fill_device_mdspan(MDSpan{}, dyn...);

  int repeats = s.size() > (100*100*100) ? 50 : 1000;

  for (auto _ : state) {
    auto timed = run_kernel_timed(
      [=] __device__ {
        for(int r = 0; r < repeats; ++r) {
          value_type sum_local = 0;
          for(size_t i = blockIdx.x; i < s.extent(0); i += gridDim.x) {
            for(size_t j = threadIdx.z; j < s.extent(1); j += blockDim.z) {
              for(size_t k = threadIdx.y; k < s.extent(2); k += blockDim.y) {
                sum_local += s(i, j, k);
              }
            }
          }
          DoNotOptimize(*(volatile value_type*)(&s(0,0,0)) = sum_local);
          asm volatile ("": : :"memory");
        }
      }
    );
    // units of cuda timer is milliseconds, units of iteration timer is seconds
    state.SetIterationTime(timed * 1e-3);
  }
  state.SetBytesProcessed(s.size() * sizeof(value_type) * state.iterations() * repeats);
  state.counters["repeats"] = repeats;

  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  CUDA_SAFE_CALL(cudaFree(s.data_handle()));
}
MDSPAN_BENCHMARK_ALL_3D_MANUAL(BM_MDSpan_Cuda_Sum_3D, right_, rmdspan, 80, 80, 80);
MDSPAN_BENCHMARK_ALL_3D_MANUAL(BM_MDSpan_Cuda_Sum_3D, left_, lmdspan, 80, 80, 80);
MDSPAN_BENCHMARK_ALL_3D_MANUAL(BM_MDSpan_Cuda_Sum_3D, right_, rmdspan, 400, 400, 400);
MDSPAN_BENCHMARK_ALL_3D_MANUAL(BM_MDSpan_Cuda_Sum_3D, left_, lmdspan, 400, 400, 400);

//================================================================================

template <class T, class SizeX, class SizeY, class SizeZ>
void BM_Raw_Cuda_Sum_3D_right(benchmark::State& state, T, SizeX x, SizeY y, SizeZ z) {

  using value_type = T;
  value_type* data = nullptr;
  {
    // just for setup...
    auto wrapped = Kokkos::mdspan<T, Kokkos::dextents<int, 1>>{};
    auto s = fill_device_mdspan(wrapped, x*y*z);
    data = s.data_handle();
  }

  int repeats = x*y*z > (100*100*100) ? 50 : 1000;

  for (auto _ : state) {
    auto timed = run_kernel_timed(
      [=] __device__ {
        for(int r = 0; r < repeats; ++r) {
          value_type sum_local = 0;
          for(size_t i = blockIdx.x; i < x; i += gridDim.x) {
            for(size_t j = threadIdx.z; j < y; j += blockDim.z) {
              for(size_t k = threadIdx.y; k < z; k += blockDim.y) {
                sum_local += data[k + j*z + i*z*y];
              }
            }
          }
          DoNotOptimize(*(volatile value_type*)(&data[0]) = sum_local);
          asm volatile ("": : :"memory");
        }
      }
    );
    // units of cuda timer is milliseconds, units of iteration timer is seconds
    state.SetIterationTime(timed * 1e-3);
  }
  state.SetBytesProcessed(x * y * z * sizeof(value_type) * state.iterations() * repeats);
  state.counters["repeats"] = repeats;

  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  CUDA_SAFE_CALL(cudaFree(data));
}
BENCHMARK_CAPTURE(BM_Raw_Cuda_Sum_3D_right, size_80_80_80, int(), 80, 80, 80);
BENCHMARK_CAPTURE(BM_Raw_Cuda_Sum_3D_right, size_400_400_400, int(), 400, 400, 400);

//================================================================================

template <class T, class SizeX, class SizeY, class SizeZ>
void BM_Raw_Cuda_Sum_3D_left(benchmark::State& state, T, SizeX x, SizeY y, SizeZ z) {

  using value_type = T;
  value_type* data = nullptr;
  {
    // just for setup...
    auto wrapped = Kokkos::mdspan<T, Kokkos::dextents<int, 1>>{};
    auto s = fill_device_mdspan(wrapped, x*y*z);
    data = s.data_handle();
  }

  int repeats = x*y*z > (100*100*100) ? 50 : 1000;

  for (auto _ : state) {
    auto timed = run_kernel_timed(
    [=] __device__ {
      for(int r = 0; r < repeats; ++r) {
        value_type sum_local = 0;
        for(size_t i = blockIdx.x; i < x; i += gridDim.x) {
          for(size_t j = threadIdx.z; j < y; j += blockDim.z) {
            for(size_t k = threadIdx.y; k < z; k += blockDim.y) {
              sum_local += data[k*x*y + j*x + i];
            }
          }
        }
        DoNotOptimize(*(volatile value_type*)(&data[0]) = sum_local);
        asm volatile ("": : :"memory");
      }
    }
    );
    // units of cuda timer is milliseconds, units of iteration timer is seconds
    state.SetIterationTime(timed * 1e-3);
  }
  state.SetBytesProcessed(x * y * z * sizeof(value_type) * state.iterations() * repeats);
  state.counters["repeats"] = repeats;

  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  CUDA_SAFE_CALL(cudaFree(data));
}
BENCHMARK_CAPTURE(BM_Raw_Cuda_Sum_3D_left, size_80_80_80, int(), 80, 80, 80);
BENCHMARK_CAPTURE(BM_Raw_Cuda_Sum_3D_left, size_400_400_400, int(), 400, 400, 400);

//================================================================================

BENCHMARK_MAIN();
