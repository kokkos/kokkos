/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_THREAD_FENCE_OPENACC_HPP_
#define DESUL_ATOMICS_THREAD_FENCE_OPENACC_HPP_

#include <openacc.h>

#include <desul/atomics/Common.hpp>

namespace desul {
namespace Impl {

#pragma acc routine seq
template <class MemoryOrder, class MemoryScope>
void device_atomic_thread_fence(MemoryOrder, MemoryScope) {
}

}  // namespace Impl
}  // namespace desul

#endif
