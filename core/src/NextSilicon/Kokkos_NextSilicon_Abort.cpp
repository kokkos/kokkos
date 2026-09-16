// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <csignal>
#include <cstdlib>
#include <iostream>
#include <Kokkos_Assert.hpp>
#include <NextSilicon/Kokkos_NextSilicon_Abort.hpp>
#include <nextapi/intrinsics.h>

namespace Kokkos {
namespace Impl {

void nextsilicon_host_abort(char const* const message) {
  KOKKOS_ASSERT(!__next_is_in_handed_off_code());
  std::cerr << message << '\n';
  std::raise(SIGABRT);
  // std::raise is not [[noreturn]]: it returns if the SIGABRT handler returns
  // normally. Terminate for real rather than falling off the end of a
  // [[noreturn]] function.
  std::abort();
}

}  // namespace Impl
}  // namespace Kokkos
