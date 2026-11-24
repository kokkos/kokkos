// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.simd;
#else
#include <Kokkos_SIMD.hpp>
#endif
#include <cstdint>

// Exercise imported generators without pulling in intrinsic declarations
// through a test framework or other unrelated headers.
template <typename T, int N>
bool check_generator() {
  using simd_type = Kokkos::Experimental::basic_simd<
      T, Kokkos::Experimental::simd_abi::avx2_fixed_size<N>>;
  simd_type values([](auto i) { return T(3 * i + 1); });
  for (int i = 0; i < N; ++i) {
    if (values[i] != T(3 * i + 1)) return false;
  }
  return true;
}

int main() {
  return check_generator<double, 4>() && check_generator<float, 4>() &&
                 check_generator<float, 8>() &&
                 check_generator<std::int32_t, 4>() &&
                 check_generator<std::int32_t, 8>() &&
                 check_generator<std::int64_t, 4>() &&
                 check_generator<std::uint64_t, 4>()
             ? 0
             : 1;
}
