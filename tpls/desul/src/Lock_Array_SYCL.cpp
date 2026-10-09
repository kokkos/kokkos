/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

// FIXME_SYCL Use SYCL_EXT_ONEAPI_DEVICE_GLOBAL when available instead
#ifdef DESUL_SYCL_DEVICE_GLOBAL_SUPPORTED

#include <cinttypes>
#include <desul/atomics/Lock_Array_SYCL.hpp>
#include <vector>

namespace desul::Impl {

#ifdef DESUL_ATOMICS_ENABLE_SYCL_SEPARABLE_COMPILATION
SYCL_EXTERNAL
sycl_device_global<int32_t*> SYCL_SPACE_ATOMIC_LOCKS_DEVICE;
SYCL_EXTERNAL
sycl_device_global<int32_t*> SYCL_SPACE_ATOMIC_LOCKS_NODE;
#endif

int32_t* SYCL_SPACE_ATOMIC_LOCKS_DEVICE_h = nullptr;
int32_t* SYCL_SPACE_ATOMIC_LOCKS_NODE_h = nullptr;

#ifndef DESUL_ATOMICS_ENABLE_SYCL_SEPARABLE_COMPILATION
namespace {
// Function-local statics so that registration from other translation units'
// static initializers works regardless of static initialization order.
std::vector<void (*)(sycl::queue)>& sycl_lock_arrays_copiers() {
  static std::vector<void (*)(sycl::queue)> copiers;
  return copiers;
}
}  // namespace

void register_sycl_lock_arrays_copier(void (*copier)(sycl::queue)) {
  sycl_lock_arrays_copiers().push_back(copier);
}
#endif

namespace {
void copy_all_sycl_lock_arrays_to_device(sycl::queue q) {
#ifdef DESUL_ATOMICS_ENABLE_SYCL_SEPARABLE_COMPILATION
  copy_sycl_lock_arrays_to_device(q);
#else
  for (auto copier : sycl_lock_arrays_copiers()) copier(q);
#endif
}
}  // namespace

template <>
void init_lock_arrays_sycl<int>(sycl::queue q) {
  if (SYCL_SPACE_ATOMIC_LOCKS_DEVICE_h != nullptr) return;

  SYCL_SPACE_ATOMIC_LOCKS_DEVICE_h =
      sycl::malloc_device<int32_t>(SYCL_SPACE_ATOMIC_MASK + 1, q);
  SYCL_SPACE_ATOMIC_LOCKS_NODE_h =
      sycl::malloc_host<int32_t>(SYCL_SPACE_ATOMIC_MASK + 1, q);

  copy_all_sycl_lock_arrays_to_device(q);

  q.memset(SYCL_SPACE_ATOMIC_LOCKS_DEVICE_h,
           0,
           sizeof(int32_t) * (SYCL_SPACE_ATOMIC_MASK + 1));
  q.memset(SYCL_SPACE_ATOMIC_LOCKS_NODE_h,
           0,
           sizeof(int32_t) * (SYCL_SPACE_ATOMIC_MASK + 1));

  q.wait_and_throw();
}

template <>
void finalize_lock_arrays_sycl<int>(sycl::queue q) {
  if (SYCL_SPACE_ATOMIC_LOCKS_DEVICE_h == nullptr) return;

  sycl::free(SYCL_SPACE_ATOMIC_LOCKS_DEVICE_h, q);
  sycl::free(SYCL_SPACE_ATOMIC_LOCKS_NODE_h, q);
  SYCL_SPACE_ATOMIC_LOCKS_DEVICE_h = nullptr;
  SYCL_SPACE_ATOMIC_LOCKS_NODE_h = nullptr;
}

}  // namespace desul::Impl
#endif
