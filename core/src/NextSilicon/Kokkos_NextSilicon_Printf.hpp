// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_NEXTSILICON_PRINTF_HPP
#define KOKKOS_NEXTSILICON_PRINTF_HPP

#include <Kokkos_Macros.hpp>

namespace Kokkos {
namespace Impl {

template <typename... Args>
void nextsilicon_printf(const char* /*format*/, Args... /*args*/) {
  // FIXME_NEXTSILICON: CS-515 tracks printing from device
}

}  // namespace Impl
}  // namespace Kokkos

#endif
