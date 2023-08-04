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

#ifndef KOKKOS_TEST_SIMD_MASK_OPS_HPP
#define KOKKOS_TEST_SIMD_MASK_OPS_HPP

#include <Kokkos_SIMD.hpp>
#include <SIMDTesting_Utilities.hpp>

template <typename Abi>
inline void host_check_mask_ops() {
  using mask_type = Kokkos::Experimental::simd_mask<double, Abi>;
  EXPECT_FALSE(none_of(mask_type(true)));
  EXPECT_TRUE(none_of(mask_type(false)));
  EXPECT_TRUE(all_of(mask_type(true)));
  EXPECT_FALSE(all_of(mask_type(false)));
}

template <typename... Abis>
inline void host_check_mask_ops_all_abis(
    Kokkos::Experimental::Impl::abi_set<Abis...>) {
  (host_check_mask_ops<Abis>(), ...);
}

template <typename Abi>
KOKKOS_INLINE_FUNCTION void device_check_mask_ops() {
  using mask_type = Kokkos::Experimental::simd_mask<double, Abi>;
  kokkos_checker checker;
  checker.truth(!none_of(mask_type(true)));
  checker.truth(none_of(mask_type(false)));
  checker.truth(all_of(mask_type(true)));
  checker.truth(!all_of(mask_type(false)));
}

template <typename... Abis>
KOKKOS_INLINE_FUNCTION void device_check_mask_ops_all_abis(
    Kokkos::Experimental::Impl::abi_set<Abis...>) {
  (device_check_mask_ops<Abis>(), ...);
}

class simd_device_mask_ops_functor {
 public:
  KOKKOS_INLINE_FUNCTION void operator()(int) const {
    device_check_mask_ops_all_abis(
        Kokkos::Experimental::Impl::device_abi_set());
  }
};

TEST(simd, host_mask_ops) {
  host_check_mask_ops_all_abis(Kokkos::Experimental::Impl::host_abi_set());
}

TEST(simd, device_mask_ops) {
  Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::IndexType<int>>(0, 1),
                       simd_device_mask_ops_functor());
}

#endif
