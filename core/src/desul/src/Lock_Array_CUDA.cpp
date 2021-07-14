/* 
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#include <desul/atomics/Lock_Array.hpp>
#include <cinttypes>
#include <string>
#include <sstream>

#ifdef DESUL_HAVE_CUDA_ATOMICS
#ifdef __CUDACC_RDC__
namespace desul {
namespace Impl {
__device__ __constant__ int32_t* CUDA_SPACE_ATOMIC_LOCKS_DEVICE = nullptr;
__device__ __constant__ int32_t* CUDA_SPACE_ATOMIC_LOCKS_NODE = nullptr;
}
}  // namespace desul
#endif

namespace desul {

namespace {

__global__ void init_lock_arrays_cuda_kernel() {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < CUDA_SPACE_ATOMIC_MASK + 1) {
    Impl::CUDA_SPACE_ATOMIC_LOCKS_DEVICE[i] = 0;
    Impl::CUDA_SPACE_ATOMIC_LOCKS_NODE[i] = 0;
  }
}

}  // namespace

namespace Impl {


int32_t* CUDA_SPACE_ATOMIC_LOCKS_DEVICE_h = nullptr;
int32_t* CUDA_SPACE_ATOMIC_LOCKS_NODE_h = nullptr;

// Putting this into anonymous namespace so we don't have multiple defined symbols
// When linking in more than one copy of the object file
namespace {

void check_error_and_throw_cuda(cudaError e, const std::string msg) {
  if(e != cudaSuccess) {
    std::ostringstream out;
    out << "Desul::Error: " << msg << " error(" << cudaGetErrorName(e)
                  << "): " << cudaGetErrorString(e);
    throw std::runtime_error(out.str());
  }
}

}

// define functions
template<typename T>
void init_lock_arrays_cuda() {
  if (CUDA_SPACE_ATOMIC_LOCKS_DEVICE_h != nullptr) return;
  auto error_malloc1 = cudaMalloc(&CUDA_SPACE_ATOMIC_LOCKS_DEVICE_h,
                                 sizeof(int32_t) * (CUDA_SPACE_ATOMIC_MASK + 1));
  check_error_and_throw_cuda(error_malloc1, "init_lock_arrays_cuda: cudaMalloc device locks");

  auto error_malloc2 = cudaMallocHost(&CUDA_SPACE_ATOMIC_LOCKS_NODE_h,
                                 sizeof(int32_t) * (CUDA_SPACE_ATOMIC_MASK + 1));
  check_error_and_throw_cuda(error_malloc2, "init_lock_arrays_cuda: cudaMalloc host locks");

  auto error_sync1 = cudaDeviceSynchronize();
  DESUL_IMPL_COPY_CUDA_LOCK_ARRAYS_TO_DEVICE();
  check_error_and_throw_cuda(error_sync1, "init_lock_arrays_cuda: post mallocs");
  init_lock_arrays_cuda_kernel<<<(CUDA_SPACE_ATOMIC_MASK + 1 + 255) / 256, 256>>>();
  auto error_sync2 = cudaDeviceSynchronize();
  check_error_and_throw_cuda(error_sync2, "init_lock_arrays_cuda: post init kernel");
}

template<typename T>
void finalize_lock_arrays_cuda() {
  if (CUDA_SPACE_ATOMIC_LOCKS_DEVICE_h == nullptr) return;
  cudaFree(CUDA_SPACE_ATOMIC_LOCKS_DEVICE_h);
  cudaFreeHost(CUDA_SPACE_ATOMIC_LOCKS_NODE_h);
  CUDA_SPACE_ATOMIC_LOCKS_DEVICE_h = nullptr;
  CUDA_SPACE_ATOMIC_LOCKS_NODE_h = nullptr;
#ifdef __CUDACC_RDC__
  DESUL_IMPL_COPY_CUDA_LOCK_ARRAYS_TO_DEVICE();
#endif
}

// Instantiate functions
template void init_lock_arrays_cuda<int>();
template void finalize_lock_arrays_cuda<int>();

}  // namespace Impl

}  // namespace desul
#endif
