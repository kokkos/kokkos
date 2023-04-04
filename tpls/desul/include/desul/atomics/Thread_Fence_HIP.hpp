/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_THREAD_FENCE_HIP_HPP_
#define DESUL_ATOMICS_THREAD_FENCE_HIP_HPP_

#include <desul/atomics/Adapt_HIP.hpp>

namespace desul {
namespace Impl {

template <class MemoryOrder>
__device__ void device_atomic_thread_fence(MemoryOrder, MemoryOrderCore) {
  __builtin_amdgcn_fence(HIPMemoryOrder<MemoryOrder>::value, "workgroup");
}

template <class MemoryOrder>
__device__ void device_atomic_thread_fence(MemoryOrder, MemoryOrderDevice) {
  __builtin_amdgcn_fence(HIPMemoryOrder<MemoryOrder>::value, "agent");
}

// FIXME scope larger than strictly necessary
template <class MemoryOrder>
__device__ void device_atomic_thread_fence(MemoryOrder, MemoryOrderNode) {
  __builtin_amdgcn_fence(HIPMemoryOrder<MemoryOrder>::value, "");
}

template <class MemoryOrder>
__device__ void device_atomic_thread_fence(MemoryOrder, MemoryOrderSystem) {
  __builtin_amdgcn_fence(HIPMemoryOrder<MemoryOrder>::value, "");
}

}  // namespace Impl
}  // namespace desul

#endif
