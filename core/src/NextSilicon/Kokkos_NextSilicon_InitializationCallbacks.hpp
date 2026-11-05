// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

// Copyright Contributors to the Kokkos project
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef KOKKOS_NEXTSILICON_INITIALIZATION_CALLBACKS_HPP
#define KOKKOS_NEXTSILICON_INITIALIZATION_CALLBACKS_HPP

#include <functional>

namespace Kokkos::Impl {

// Callbacks must not register additional initialization callbacks.
void register_nextsilicon_initialization_callback(
    std::function<void()> callback);

void run_nextsilicon_initialization_callbacks();

}  // namespace Kokkos::Impl

#endif  // KOKKOS_NEXTSILICON_INITIALIZATION_CALLBACKS_HPP
