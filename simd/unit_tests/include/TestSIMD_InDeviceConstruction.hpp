// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_TEST_SIMD_IN_DEVICE_CONSTRUCTION_HPP
#define KOKKOS_TEST_SIMD_IN_DEVICE_CONSTRUCTION_HPP

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.simd;
#else
#include <Kokkos_SIMD.hpp>
#endif
#include <SIMDTesting_Utilities.hpp>

class test_simd_in_host_device_construction_functor {
 public:
  using data_type = std::int32_t;

  KOKKOS_INLINE_FUNCTION void operator()(int) const {
    using native_simd_type = Kokkos::Experimental::simd<data_type>;
    native_simd_type s1(data_type{});                                 // Okay
    native_simd_type s2(KOKKOS_LAMBDA(std::size_t i) { return i; });  // Okay

    if constexpr (!std::is_same_v<Kokkos::DefaultExecutionSpace,
                                  Kokkos::Serial>) {
      // native_simd_type currently sets to 'scalar' abi in a device
      // enabled build
      auto s3 = s1 + s2;          // Okay
      auto s4 = Kokkos::abs(s3);  // Okay

      // These fail to compile in non-scalar host simd + device enabled builds
      // simd_[mask]_type contain a vector (intrinsic simd vector)
      using host_simd_type = Kokkos::Experimental::basic_simd<
          data_type, Kokkos::Experimental::simd_abi::Impl::host_fixed_native<data_type>>;
      using host_simd_mask_type = typename host_simd_type::mask_type;

      host_simd_mask_type mask;
      host_simd_type simd1;
      host_simd_type simd2(data_type{});
      host_simd_type simd3(KOKKOS_LAMBDA(std::size_t i) { return i; });

      auto r1 = simd2 + simd3;
      auto r2 = Kokkos::abs(r1);
    }
  }
};

// Sample test
TEST(simd, in_host_device_construction) {
  Kokkos::parallel_for(1, test_simd_in_host_device_construction_functor());
}

#endif
