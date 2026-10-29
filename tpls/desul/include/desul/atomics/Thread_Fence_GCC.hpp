/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_THREAD_FENCE_GCC_HPP_
#define DESUL_ATOMICS_THREAD_FENCE_GCC_HPP_

#include <desul/atomics/Adapt_GCC.hpp>

namespace desul {
namespace Impl {

DESUL_IMPL_ACC_ROUTINE_DIRECTIVE
template <class MemoryOrder, class MemoryScope>
void host_atomic_thread_fence(MemoryOrder, MemoryScope) {
#if defined(DESUL_HAVE_OPENACC_ATOMICS) && defined(__NVCOMPILER)
  // FIXME_OPENACC NVHPC cannot lower __atomic_thread_fence in device code.  Test at run
  // time so that the device compiler folds the call away.
  if (acc_on_device(acc_device_host))
#endif
    __atomic_thread_fence(GCCMemoryOrder<MemoryOrder>::value);
}

}  // namespace Impl
}  // namespace desul

#endif
