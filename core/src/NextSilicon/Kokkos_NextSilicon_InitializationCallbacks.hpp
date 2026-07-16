// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_NEXTSILICON_INITIALIZATION_CALLBACKS_HPP
#define KOKKOS_NEXTSILICON_INITIALIZATION_CALLBACKS_HPP

#include <functional>
#include <string>

namespace Kokkos::Impl {

void register_nextsilicon_initialization_callback(
    std::string label, std::function<void()> callback);

void run_nextsilicon_initialization_callbacks();

}  // namespace Kokkos::Impl

#endif  // KOKKOS_NEXTSILICON_INITIALIZATION_CALLBACKS_HPP
