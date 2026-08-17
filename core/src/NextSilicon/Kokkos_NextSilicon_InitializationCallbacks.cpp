// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <NextSilicon/Kokkos_NextSilicon_InitializationCallbacks.hpp>

#include <mutex>
#include <utility>
#include <vector>

namespace Kokkos::Impl {

namespace {

struct NextSiliconInitializationCallbacks {
  std::mutex mutex;
  std::vector<std::function<void()>> pending;
};

NextSiliconInitializationCallbacks& nextsilicon_initialization_callbacks() {
  static NextSiliconInitializationCallbacks callbacks;
  return callbacks;
}

}  // namespace

void register_nextsilicon_initialization_callback(
    std::function<void()> callback) {
  auto& callbacks = nextsilicon_initialization_callbacks();
  std::lock_guard<std::mutex> lock(callbacks.mutex);
  callbacks.pending.push_back(std::move(callback));
}

void run_nextsilicon_initialization_callbacks() {
  auto& callbacks = nextsilicon_initialization_callbacks();
  std::lock_guard<std::mutex> lock(callbacks.mutex);
  for (auto& callback : callbacks.pending) {
    callback();
  }
  callbacks.pending.clear();
}

}  // namespace Kokkos::Impl
