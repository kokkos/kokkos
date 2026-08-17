// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_NEXTSILICON_ABORT_HPP
#define KOKKOS_NEXTSILICON_ABORT_HPP

#include <nextapi/intrinsics.h>

namespace Kokkos {
namespace Impl {

[[noreturn]] void nextsilicon_host_abort(char const* const message);

[[noreturn, clang::always_inline]] inline void nextsilicon_device_abort(
    char const* const /*message*/) {
  // FIXME_NEXTSILICON Device error message print
  // FIXME_NEXTSILICON Add NextAPI to properly abort application from device
  __builtin_trap();
}

[[noreturn, clang::always_inline]] inline void nextsilicon_abort(
    char const* const message) {
  if (__next_is_in_handed_off_code()) {
    nextsilicon_device_abort(message);
  } else {
    nextsilicon_host_abort(message);
  }
}

}  // namespace Impl
}  // namespace Kokkos

#endif
