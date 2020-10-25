/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#include <cinttypes>
#include <desul/atomics/Lock_Array.hpp>

#ifdef DESUL_HAVE_HIP_ATOMICS
#ifdef DESUL_HIP_RDC
namespace desul {
namespace Impl {
__device__ __constant__ int32_t* HIP_SPACE_ATOMIC_LOCKS_DEVICE = nullptr;
__device__ __constant__ int32_t* HIP_SPACE_ATOMIC_LOCKS_NODE = nullptr;
}  // namespace Impl
}  // namespace desul
#endif

namespace desul {

namespace {

__global__ void init_lock_arrays_hip_kernel() {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < HIP_SPACE_ATOMIC_MASK + 1) {
    Impl::HIP_SPACE_ATOMIC_LOCKS_DEVICE[i] = 0;
    Impl::HIP_SPACE_ATOMIC_LOCKS_NODE[i] = 0;
  }
}

}  // namespace

namespace Impl {

int32_t* HIP_SPACE_ATOMIC_LOCKS_DEVICE_h = nullptr;
int32_t* HIP_SPACE_ATOMIC_LOCKS_NODE_h = nullptr;

void init_lock_arrays_hip() {
  if (HIP_SPACE_ATOMIC_LOCKS_DEVICE_h != nullptr) return;
  hipMalloc(&HIP_SPACE_ATOMIC_LOCKS_DEVICE_h,
            sizeof(int32_t) * (HIP_SPACE_ATOMIC_MASK + 1));
  hipHostMalloc(&HIP_SPACE_ATOMIC_LOCKS_NODE_h,
                sizeof(int32_t) * (HIP_SPACE_ATOMIC_MASK + 1));
  hipDeviceSynchronize();
  DESUL_IMPL_COPY_HIP_LOCK_ARRAYS_TO_DEVICE();
  init_lock_arrays_hip_kernel<<<(HIP_SPACE_ATOMIC_MASK + 1 + 255) / 256, 256>>>();
  hipDeviceSynchronize();
}

void finalize_lock_arrays_hip() {
  if (HIP_SPACE_ATOMIC_LOCKS_DEVICE_h == nullptr) return;
  hipFree(HIP_SPACE_ATOMIC_LOCKS_DEVICE_h);
  hipHostFree(HIP_SPACE_ATOMIC_LOCKS_NODE_h);
  HIP_SPACE_ATOMIC_LOCKS_DEVICE_h = nullptr;
  HIP_SPACE_ATOMIC_LOCKS_NODE_h = nullptr;
#ifdef DESUL_HIP_RDC
  DESUL_IMPL_COPY_HIP_LOCK_ARRAYS_TO_DEVICE();
#endif
}

}  // namespace Impl

}  // namespace desul
#endif

