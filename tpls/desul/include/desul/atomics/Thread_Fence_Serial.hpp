/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_THREAD_FENCE_SERIAL_HPP_
#define DESUL_ATOMICS_THREAD_FENCE_SERIAL_HPP_

#include <desul/atomics/Common.hpp>

namespace desul {
namespace impl {

template <class MemoryScope>
void host_atomic_thread_fence(MemoryOrderAcquire, MemoryScope) {}

template <class MemoryScope>
void host_atomic_thread_fence(MemoryOrderRelease, MemoryScope) {}

}  // namespace impl
}  // namespace desul

#endif
