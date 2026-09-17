// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_SETENV_HPP
#define KOKKOS_SETENV_HPP

#include <stdlib.h>

namespace Kokkos::Impl {
#ifdef _WIN32
inline int setenv(const char *name, const char *value, int overwrite) {
  int errcode = 0;
  if (!overwrite) {
    size_t envsize = 0;
    errcode        = getenv_s(&envsize, NULL, 0, name);
    if (errcode || envsize) return errcode;
  }
  return _putenv_s(name, value);
}

inline int unsetenv(const char *name) { return _putenv_s(name, ""); }

#else
using ::setenv;
using ::unsetenv;
#endif
}  // namespace Kokkos::Impl

#endif
