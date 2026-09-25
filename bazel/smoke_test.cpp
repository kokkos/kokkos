// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  constexpr int count = 100;
  Kokkos::View<int*> values("values", count);
  Kokkos::parallel_for(
      "fill", count, KOKKOS_LAMBDA(int i) { values(i) = i + 1; });
  int sum = 0;
  Kokkos::parallel_reduce(
      "sum", count, KOKKOS_LAMBDA(int i, int& result) { result += values(i); },
      sum);
  if (sum != count * (count + 1) / 2) {
    std::cerr << "Incorrect parallel reduction: " << sum << '\n';
    return 1;
  }
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, values);
  for (int i = 0; i < count; ++i) {
    if (host(i) != i + 1) return 1;
  }
  return 0;
}
