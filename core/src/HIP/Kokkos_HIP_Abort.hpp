// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_HIP_ABORT_HPP
#define KOKKOS_HIP_ABORT_HPP

#include <Kokkos_Macros.hpp>
#include <Kokkos_Printf.hpp>

#include <hip/hip_runtime.h>

namespace Kokkos {
namespace Impl {

// The two keywords below are not contradictory. `noinline` is a
// directive to the optimizer.
[[noreturn]] __device__ __attribute__((noinline)) inline void hip_abort(
    char const *msg) {
  const char empty[] = "";
#if defined(_WIN32)
  // ROCm's Windows device runtime ships no glibc __assert_fail. Emit the
  // message via device printf (as the SYCL backend does on _MSC_VER), then
  // abort(). The while(true) guard below preserves [[noreturn]] since abort()
  // is not itself marked [[noreturn]].
  (void)empty;
  Kokkos::printf("Aborting with message %s.\n", msg);
  abort();
#else
  __assert_fail(msg, empty, 0, empty);
#endif
  // This loop is never executed. It's intended to suppress warnings that the
  // function returns, even though it does not. This is necessary because
  // abort() is not marked as [[noreturn]], even though it does not return.
  while (true)
    ;
}

}  // namespace Impl
}  // namespace Kokkos

#endif
