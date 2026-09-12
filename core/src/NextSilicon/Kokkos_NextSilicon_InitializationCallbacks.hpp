// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_NEXTSILICON_INITIALIZATION_CALLBACKS_HPP
#define KOKKOS_NEXTSILICON_INITIALIZATION_CALLBACKS_HPP

#include <cstdint>
#include <functional>
#include <optional>

namespace Kokkos::Impl {

using NextSiliconInitializationCallbackHandle = std::int64_t;

inline constexpr NextSiliconInitializationCallbackHandle
    invalid_nextsilicon_initialization_callback_handle = -1;

// Callbacks must not register additional initialization callbacks.
NextSiliconInitializationCallbackHandle
register_nextsilicon_initialization_callback(std::function<void()> callback);

// Retrieve the callback associated with handle and clear its entry. Returns
// nullopt if the callback was already retrieved/run or if handle is invalid.
std::optional<std::function<void()>>
retrieve_nextsilicon_initialization_callback(
    NextSiliconInitializationCallbackHandle handle);

void run_nextsilicon_initialization_callbacks();

}  // namespace Kokkos::Impl

#endif  // KOKKOS_NEXTSILICON_INITIALIZATION_CALLBACKS_HPP
