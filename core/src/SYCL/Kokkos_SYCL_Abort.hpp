// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_SYCL_ABORT_HPP
#define KOKKOS_SYCL_ABORT_HPP

#include <Kokkos_Printf.hpp>
#if defined(KOKKOS_ENABLE_SYCL)
#include <sycl/sycl.hpp>
#if defined(__ACPP__)
#include <hipSYCL/sycl/libkernel/sscp/builtins/assert.hpp>
#endif

namespace Kokkos {
namespace Impl {

inline void sycl_abort(char const* msg) {
#if defined(__ACPP__)
  // FIXME_ACPP __assert_fail does not exist on macOS, and Kokkos::printf only
  // prints on the host. In device code, use AdaptiveCpp's assert builtin: its
  // host (OpenMP) device prints the message and aborts, while on Metal the
  // kernel continues ("SYCL doesn't abort", as below).
  Kokkos::printf("Aborting with message %s.\n", msg);
  __acpp_if_target_device(
      __acpp_sscp_assert_fail(msg, __FILE__, __LINE__, "Kokkos::abort"););
// FIXME_WINDOWS __assert_fail isn't defined
#elif defined(NDEBUG) || defined(_MSC_VER)
  Kokkos::printf("Aborting with message %s.\n", msg);
#else
  // Choosing "" here causes problems but a single whitespace character works.
  const char* empty = " ";
  __assert_fail(msg, empty, 0, empty);
#endif
}

}  // namespace Impl
}  // namespace Kokkos

#endif
#endif
