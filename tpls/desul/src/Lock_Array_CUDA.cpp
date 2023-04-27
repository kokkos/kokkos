/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#include <cinttypes>
#include <desul/atomics/Lock_Array.hpp>
#include <map>
#include <sstream>
#include <string>

#ifdef DESUL_ATOMICS_ENABLE_CUDA_SEPARABLE_COMPILATION
namespace desul {
namespace Impl {
__device__ __constant__ int32_t* CUDA_SPACE_ATOMIC_LOCKS_DEVICE = nullptr;
__device__ __constant__ int32_t* CUDA_SPACE_ATOMIC_LOCKS_NODE = nullptr;
}  // namespace Impl
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

std::map<int, int32_t*> CUDA_SPACE_ATOMIC_LOCKS_DEVICE_h = {};
std::map<int, int32_t*> CUDA_SPACE_ATOMIC_LOCKS_NODE_h = {};

// Putting this into anonymous namespace so we don't have multiple defined symbols
// When linking in more than one copy of the object file
namespace {

void check_error_and_throw_cuda(cudaError e, const std::string msg) {
  if (e != cudaSuccess) {
    std::ostringstream out;
    out << "Desul::Error: " << msg << " error(" << cudaGetErrorName(e)
        << "): " << cudaGetErrorString(e);
    throw std::runtime_error(out.str());
  }
}

}  // namespace

// define functions
template <typename T>
void init_lock_arrays_cuda(int device_id) {
  if (CUDA_SPACE_ATOMIC_LOCKS_DEVICE_h[device_id] != nullptr) return;
  auto error_set_device = cudaSetDevice(device_id);
  check_error_and_throw_cuda(error_set_device, "init_lock_arrays_cuda: cudaSetDevice");
  auto error_malloc1 = cudaMalloc(&CUDA_SPACE_ATOMIC_LOCKS_DEVICE_h[device_id],
                                  sizeof(int32_t) * (CUDA_SPACE_ATOMIC_MASK + 1));
  check_error_and_throw_cuda(error_malloc1,
                             "init_lock_arrays_cuda: cudaMalloc device locks");

  auto error_malloc2 = cudaMallocHost(&CUDA_SPACE_ATOMIC_LOCKS_NODE_h[device_id],
                                      sizeof(int32_t) * (CUDA_SPACE_ATOMIC_MASK + 1));
  check_error_and_throw_cuda(error_malloc2,
                             "init_lock_arrays_cuda: cudaMalloc host locks");

  auto error_sync1 = cudaDeviceSynchronize();
  copy_cuda_lock_arrays_to_device(device_id);
  check_error_and_throw_cuda(error_sync1, "init_lock_arrays_cuda: post mallocs");
  init_lock_arrays_cuda_kernel<<<(CUDA_SPACE_ATOMIC_MASK + 1 + 255) / 256, 256>>>();
  auto error_sync2 = cudaDeviceSynchronize();
  check_error_and_throw_cuda(error_sync2, "init_lock_arrays_cuda: post init kernel");
}

template <typename T>
void finalize_lock_arrays_cuda() {
  for(auto &host_device_lock_arrays: CUDA_SPACE_ATOMIC_LOCKS_DEVICE_h)
  {
    if (host_device_lock_arrays.second == nullptr) continue;
    int device_id = host_device_lock_arrays.first;
    cudaFree(CUDA_SPACE_ATOMIC_LOCKS_DEVICE_h[device_id]);
    cudaFreeHost(CUDA_SPACE_ATOMIC_LOCKS_NODE_h[device_id]);
    CUDA_SPACE_ATOMIC_LOCKS_DEVICE_h[device_id] = nullptr;
    CUDA_SPACE_ATOMIC_LOCKS_NODE_h[device_id] = nullptr;
#ifdef DESUL_ATOMICS_ENABLE_CUDA_SEPARABLE_COMPILATION
    copy_cuda_lock_arrays_to_device(device_id);
#endif
  }
}

// Instantiate functions
template void init_lock_arrays_cuda<int>(int cuda_device);
template void finalize_lock_arrays_cuda<int>();

}  // namespace Impl

}  // namespace desul
