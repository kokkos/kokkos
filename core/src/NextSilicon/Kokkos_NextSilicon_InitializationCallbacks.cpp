// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <NextSilicon/Kokkos_NextSilicon_InitializationCallbacks.hpp>

#include <Kokkos_Abort.hpp>

#include <cstddef>
#include <limits>
#include <optional>
#include <utility>  // std::move
#include <vector>

namespace Kokkos::Impl {

namespace {
using CallbackEntry = std::optional<std::function<void()>>;

std::vector<CallbackEntry>& pending_callbacks() {
  static std::vector<CallbackEntry> pending;
  return pending;
}
}  // namespace

NextSiliconInitializationCallbackHandle
register_nextsilicon_initialization_callback(std::function<void()> callback) {
  auto& pending = pending_callbacks();
  if (pending.size() >
      static_cast<std::size_t>(
          std::numeric_limits<NextSiliconInitializationCallbackHandle>::max()))
    Kokkos::abort(
        "nextsilicon: initialization callback handle overflow. Please report "
        "this.");
  NextSiliconInitializationCallbackHandle handle =
      static_cast<NextSiliconInitializationCallbackHandle>(pending.size());
  pending.push_back(std::move(callback));
  return handle;
}

std::optional<std::function<void()>>
retrieve_nextsilicon_initialization_callback(
    NextSiliconInitializationCallbackHandle handle) {
  auto& pending = pending_callbacks();
  if (handle < 0 || static_cast<std::size_t>(handle) >= pending.size()) {
    return std::nullopt;
  }
  auto callback = std::move(pending[static_cast<std::size_t>(handle)]);
  pending[static_cast<std::size_t>(handle)] = std::nullopt;
  return callback;
}

void run_nextsilicon_initialization_callbacks() {
  auto& pending = pending_callbacks();
  for (auto& callback : pending) {
    if (callback) {
      auto callback_to_run = std::move(callback);
      callback             = std::nullopt;
      (*callback_to_run)();
    }
  }
}

}  // namespace Kokkos::Impl
