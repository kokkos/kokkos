// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <NextSilicon/Kokkos_NextSilicon_InitializationCallbacks.hpp>

#include <Kokkos_Abort.hpp>

#include <optional>
#include <utility>  // std::in_place
#include <vector>

namespace Kokkos::Impl {

namespace {
std::optional<std::vector<std::function<void()>>> pending{std::in_place};
}  // namespace

void register_nextsilicon_initialization_callback(
    std::function<void()> callback) {
  if (!pending)
    Kokkos::abort(
        "nextsilicon: initialization callbacks improperly initialized (1). "
        "Please report this.");
  pending->push_back(std::move(callback));
}

void run_nextsilicon_initialization_callbacks() {
  if (!pending)
    Kokkos::abort(
        "nextsilicon: initialization callbacks improperly initialized (2). "
        "Please report this.");
  for (auto& callback : *pending) {
    callback();
  }
  pending->clear();
}

}  // namespace Kokkos::Impl
