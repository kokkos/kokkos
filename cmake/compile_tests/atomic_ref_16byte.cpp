// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

// Checks whether a 16-byte std::atomic_ref compare-exchange links without
// additional libraries. GCC and Clang may lower it to a libatomic call.

#include <atomic>

#ifdef __cpp_lib_atomic_ref
struct alignas(16) Pair {
  long long first;
  long long second;
};

Pair storage[2];

int main(int argc, char**) {
  Pair& dest = storage[argc % 2];
  Pair expected{0, 0};
  std::atomic_ref<Pair>(dest).compare_exchange_strong(
      expected, Pair{argc, argc}, std::memory_order_relaxed);
  return static_cast<int>(expected.first);
}
#else
int main() { return 0; }
#endif
