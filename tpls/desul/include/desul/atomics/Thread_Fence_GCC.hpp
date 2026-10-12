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
  DESUL_IMPL_IF_HOST_BUILTIN
    __atomic_thread_fence(GCCMemoryOrder<MemoryOrder>::value);
}

}  // namespace Impl
}  // namespace desul

#endif
