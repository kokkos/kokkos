/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#include <cinttypes>
#include <desul/atomics/Lock_Array.hpp>
#include <sstream>
#include <string>
#include <vector>

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

__global__ void init_lock_arrays_cuda_kernel(int32_t* device_locks,
                                             int32_t* node_locks) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < CUDA_SPACE_ATOMIC_MASK + 1) {
    device_locks[i] = 0;
    node_locks[i] = 0;
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
  if (e != cudaSuccess) {
    std::ostringstream out;
    out << "Desul::Error: " << msg << " error(" << cudaGetErrorName(e)
        << "): " << cudaGetErrorString(e);
    throw std::runtime_error(out.str());
  }
}

}  // namespace

#ifndef DESUL_ATOMICS_ENABLE_CUDA_SEPARABLE_COMPILATION
namespace {
// Function-local statics so that registration from other translation units'
// static initializers works regardless of static initialization order.
std::vector<void (*)()>& cuda_lock_arrays_copiers() {
  static std::vector<void (*)()> copiers;
  return copiers;
}
}  // namespace

void register_cuda_lock_arrays_copier(void (*copier)()) {
  cuda_lock_arrays_copiers().push_back(copier);
}
#endif

namespace {
void copy_all_cuda_lock_arrays_to_device() {
#ifdef DESUL_ATOMICS_ENABLE_CUDA_SEPARABLE_COMPILATION
  copy_cuda_lock_arrays_to_device();
#else
  for (auto copier : cuda_lock_arrays_copiers()) copier();
#endif
}
}  // namespace

// define functions
template <typename T>
void init_lock_arrays_cuda() {
  if (CUDA_SPACE_ATOMIC_LOCKS_DEVICE_h != nullptr) return;
  auto error_malloc1 = cudaMalloc(&CUDA_SPACE_ATOMIC_LOCKS_DEVICE_h,
                                  sizeof(int32_t) * (CUDA_SPACE_ATOMIC_MASK + 1));
  check_error_and_throw_cuda(error_malloc1,
                             "init_lock_arrays_cuda: cudaMalloc device locks");

  auto error_malloc2 = cudaMallocHost(&CUDA_SPACE_ATOMIC_LOCKS_NODE_h,
                                      sizeof(int32_t) * (CUDA_SPACE_ATOMIC_MASK + 1));
  check_error_and_throw_cuda(error_malloc2,
                             "init_lock_arrays_cuda: cudaMalloc host locks");

  copy_all_cuda_lock_arrays_to_device();
  init_lock_arrays_cuda_kernel<<<(CUDA_SPACE_ATOMIC_MASK + 1 + 255) / 256, 256>>>(
      CUDA_SPACE_ATOMIC_LOCKS_DEVICE_h, CUDA_SPACE_ATOMIC_LOCKS_NODE_h);

  auto error_sync = cudaDeviceSynchronize();
  check_error_and_throw_cuda(error_sync, "init_lock_arrays_cuda: post init kernel");
}

template <typename T>
void finalize_lock_arrays_cuda() {
  if (CUDA_SPACE_ATOMIC_LOCKS_DEVICE_h == nullptr) return;
  cudaFree(CUDA_SPACE_ATOMIC_LOCKS_DEVICE_h);
  cudaFreeHost(CUDA_SPACE_ATOMIC_LOCKS_NODE_h);
  CUDA_SPACE_ATOMIC_LOCKS_DEVICE_h = nullptr;
  CUDA_SPACE_ATOMIC_LOCKS_NODE_h = nullptr;
}

// Instantiate functions
template void init_lock_arrays_cuda<int>();
template void finalize_lock_arrays_cuda<int>();

}  // namespace Impl

}  // namespace desul
