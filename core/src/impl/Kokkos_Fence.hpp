// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_FENCE_HPP
#define KOKKOS_FENCE_HPP

#include <string>

namespace Kokkos {
void fence(const std::string& name /*= "Kokkos::fence: Unnamed Global Fence"*/);
}

#endif
