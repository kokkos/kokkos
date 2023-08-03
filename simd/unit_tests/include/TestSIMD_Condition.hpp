//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef KOKKOS_TEST_SIMD_CONDITION_HPP
#define KOKKOS_TEST_SIMD_CONDITION_HPP

#include <Kokkos_SIMD.hpp>
#include <SIMDTesting_Utilities.hpp>

template <typename Abi>
inline void host_check_condition() {
  auto a = Kokkos::Experimental::condition(
      Kokkos::Experimental::simd<std::int32_t, Abi>(1) > 0,
      Kokkos::Experimental::simd<std::uint64_t, Abi>(16),
      Kokkos::Experimental::simd<std::uint64_t, Abi>(20));
  EXPECT_TRUE(all_of(a == decltype(a)(16)));
}

template <typename... Abis>
inline void host_check_condition_all_abis(
    Kokkos::Experimental::Impl::abi_set<Abis...>) {
  (host_check_condition<Abis>(), ...);
}

template <typename Abi>
KOKKOS_INLINE_FUNCTION void device_check_condition() {
  kokkos_checker checker;
  auto a = Kokkos::Experimental::condition(
      Kokkos::Experimental::simd<std::int32_t, Abi>(1) > 0,
      Kokkos::Experimental::simd<std::uint64_t, Abi>(16),
      Kokkos::Experimental::simd<std::uint64_t, Abi>(20));
  checker.truth(all_of(a == decltype(a)(16)));
}

template <typename... Abis>
KOKKOS_INLINE_FUNCTION void device_check_condition_all_abis(
    Kokkos::Experimental::Impl::abi_set<Abis...>) {
  (device_check_condition<Abis>(), ...);
}

class simd_device_condition_functor {
 public:
  KOKKOS_INLINE_FUNCTION void operator()(int) const {
    device_check_condition_all_abis(
        Kokkos::Experimental::Impl::device_abi_set());
  }
};

TEST(simd, host_condition) {
  host_check_condition_all_abis(Kokkos::Experimental::Impl::host_abi_set());
}

TEST(simd, device_condition) {
  Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::IndexType<int>>(0, 1),
                       simd_device_condition_functor());
}

#endif
