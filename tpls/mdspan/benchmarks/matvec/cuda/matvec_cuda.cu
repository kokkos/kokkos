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

#include "fill.hpp"

#include <mdspan/mdspan.hpp>

#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <iostream>

//================================================================================

static constexpr int global_delta = 1;
static constexpr int global_repeat = 16;

//================================================================================

using size_type = int;

template <class T, size_t... Es>
using lmdspan = Kokkos::mdspan<T, Kokkos::extents<size_type, Es...>, Kokkos::layout_left>;
template <class T, size_t... Es>
using rmdspan = Kokkos::mdspan<T, Kokkos::extents<size_type, Es...>, Kokkos::layout_right>;

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


template <class F, class... Args>
__global__
void do_run_kernel(F f, Args... args) {
  f(args...);
}

template <class F, class... Args>
float run_kernel_timed(size_t N, size_t M, F&& f, Args&&... args) {
  cudaEvent_t start, stop;
  CUDA_SAFE_CALL(cudaEventCreate(&start));
  CUDA_SAFE_CALL(cudaEventCreate(&stop));

  CUDA_SAFE_CALL(cudaEventRecord(start));
  do_run_kernel<<<(N+255)/256,256>>>(
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

template <class MDSpanMatrix, class... DynSizes>
void BM_MDSpan_CUDA_MatVec(benchmark::State& state, MDSpanMatrix, DynSizes... dyn) {

  using value_type = typename MDSpanMatrix::value_type;
  using MDSpanVector = lmdspan<value_type,Kokkos::dynamic_extent>;

  auto A = fill_device_mdspan(MDSpanMatrix{}, dyn...);
  auto x = fill_device_mdspan(MDSpanVector{}, A.extent(1));
  auto y = fill_device_mdspan(MDSpanVector{}, A.extent(0));

  auto lambda =
      [=] __device__ {
         const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         if(i>=A.extent(0)) return;
         value_type y_i = 0;
         for(size_t j = 0; j < A.extent(1); j ++) {
           y_i += A(i,j) * x(j);
         }
         y(i) = y_i;
      };
  run_kernel_timed(A.extent(0),A.extent(1),lambda);

  for (auto _ : state) {
    auto timed = run_kernel_timed(A.extent(0),A.extent(1),lambda);
    // units of cuda timer is milliseconds, units of iteration timer is seconds
    state.SetIterationTime(timed * 1e-3);
  }
  size_t num_elements = 2 * A.extent(0) * A.extent(1) + 2 * A.extent(0);
  state.SetBytesProcessed( num_elements * sizeof(value_type) * state.iterations() );

  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  CUDA_SAFE_CALL(cudaFree(A.data_handle()));
  CUDA_SAFE_CALL(cudaFree(x.data_handle()));
  CUDA_SAFE_CALL(cudaFree(y.data_handle()));
}

BENCHMARK_CAPTURE(BM_MDSpan_CUDA_MatVec, left, lmdspan<double,Kokkos::dynamic_extent,Kokkos::dynamic_extent>(), 100000, 5000);
BENCHMARK_CAPTURE(BM_MDSpan_CUDA_MatVec, right, rmdspan<double,Kokkos::dynamic_extent,Kokkos::dynamic_extent>(), 100000, 5000);


template <class MDSpanMatrix, class... DynSizes>
void BM_MDSpan_CUDA_MatVec_Raw_Right(benchmark::State& state, MDSpanMatrix, DynSizes... dyn) {

  using value_type = typename MDSpanMatrix::value_type;
  using MDSpanVector = lmdspan<value_type,Kokkos::dynamic_extent>;

  auto A = fill_device_mdspan(MDSpanMatrix{}, dyn...);
  auto x = fill_device_mdspan(MDSpanVector{}, A.extent(1));
  auto y = fill_device_mdspan(MDSpanVector{}, A.extent(0));

  size_t N = A.extent(0);
  size_t M = A.extent(1);

  value_type* p_A = A.data_handle();
  value_type* p_x = x.data_handle();
  value_type* p_y = y.data_handle();

  auto lambda =
      [=] __device__ {
         const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         if(i>=N) return;
         value_type y_i = 0;

         for(size_t j = 0; j < M; j ++) {
           y_i += p_A[i*M+j] * p_x[j];
         }
         p_y[i] = y_i;
      };
  run_kernel_timed(N,M,lambda);

  for (auto _ : state) {
    auto timed = run_kernel_timed(N,M,lambda);
    // units of cuda timer is milliseconds, units of iteration timer is seconds
    state.SetIterationTime(timed * 1e-3);
  }
  size_t num_elements = 2 * A.extent(0) * A.extent(1) + 2 * A.extent(0);
  state.SetBytesProcessed( num_elements * sizeof(value_type) * state.iterations() );

  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  CUDA_SAFE_CALL(cudaFree(A.data_handle()));
  CUDA_SAFE_CALL(cudaFree(x.data_handle()));
  CUDA_SAFE_CALL(cudaFree(y.data_handle()));
}

BENCHMARK_CAPTURE(BM_MDSpan_CUDA_MatVec_Raw_Right, right, rmdspan<double,Kokkos::dynamic_extent,Kokkos::dynamic_extent>(), 100000, 5000);


template <class MDSpanMatrix, class... DynSizes>
void BM_MDSpan_CUDA_MatVec_Raw_Left(benchmark::State& state, MDSpanMatrix, DynSizes... dyn) {

  using value_type = typename MDSpanMatrix::value_type;
  using MDSpanVector = lmdspan<value_type,Kokkos::dynamic_extent>;

  auto A = fill_device_mdspan(MDSpanMatrix{}, dyn...);
  auto x = fill_device_mdspan(MDSpanVector{}, A.extent(1));
  auto y = fill_device_mdspan(MDSpanVector{}, A.extent(0));

  size_t N = A.extent(0);
  size_t M = A.extent(1);

  value_type* p_A = A.data_handle();
  value_type* p_x = x.data_handle();
  value_type* p_y = y.data_handle();

  auto lambda =
      [=] __device__ {
         const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         if(i>=N) return;
         value_type y_i = 0;

         for(size_t j = 0; j < M; j ++) {
           y_i += p_A[i+j*N] * p_x[j];
         }
         p_y[i] = y_i;
      };
  run_kernel_timed(N,M,lambda);

  for (auto _ : state) {
    auto timed = run_kernel_timed(N,M,lambda);
    // units of cuda timer is milliseconds, units of iteration timer is seconds
    state.SetIterationTime(timed * 1e-3);
  }
  size_t num_elements = 2 * A.extent(0) * A.extent(1) + 2 * A.extent(0);
  state.SetBytesProcessed( num_elements * sizeof(value_type) * state.iterations());

  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  CUDA_SAFE_CALL(cudaFree(A.data_handle()));
  CUDA_SAFE_CALL(cudaFree(x.data_handle()));
  CUDA_SAFE_CALL(cudaFree(y.data_handle()));
}

BENCHMARK_CAPTURE(BM_MDSpan_CUDA_MatVec_Raw_Left, left, lmdspan<double,Kokkos::dynamic_extent,Kokkos::dynamic_extent>(), 100000, 5000);
BENCHMARK_MAIN();
