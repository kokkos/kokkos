// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <NextSilicon/Kokkos_NextSilicon_InitializationCallbacks.hpp>

#include <mutex>
#include <utility>
#include <vector>

namespace Kokkos::Impl {

struct NextSiliconInitializationCallbackEntry {
  std::string label;
  std::function<void()> callback;
};

namespace {

struct NextSiliconInitializationCallbacks {
  std::mutex mutex;
  std::vector<NextSiliconInitializationCallbackEntry> pending;
  bool initialized = false;
};

NextSiliconInitializationCallbacks& nextsilicon_initialization_callbacks() {
  static NextSiliconInitializationCallbacks callbacks;
  return callbacks;
}

}  // namespace

void register_nextsilicon_initialization_callback(
    std::string label, std::function<void()> callback) {
  auto& callbacks = nextsilicon_initialization_callbacks();
  {
    std::lock_guard<std::mutex> lock(callbacks.mutex);
    if (!callbacks.initialized) {
      callbacks.pending.push_back({std::move(label), std::move(callback)});
      return;
    }
  }
  callback();
}

void run_nextsilicon_initialization_callbacks() {
  auto& callbacks = nextsilicon_initialization_callbacks();
  while (true) {
    std::vector<NextSiliconInitializationCallbackEntry> pending;
    {
      std::lock_guard<std::mutex> lock(callbacks.mutex);
      if (callbacks.pending.empty()) {
        callbacks.initialized = true;
        return;
      }
      // Drop the mutex before running callbacks in case callback registers a
      // callback. Next iteration will pick it up.
      pending.swap(callbacks.pending);
    }
    for (auto& callback : pending) {
      callback.callback();
    }
  }
}

}  // namespace Kokkos::Impl
