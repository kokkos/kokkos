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

#include <Kokkos_Core.hpp>
#include <TestSYCL_Category.hpp>

namespace {

	        #if defined(__INTEL_LLVM_COMPILER) && __INTEL_LLVM_COMPILER >= 20250000
#define KOKKOS_SYCL_CHECK_SUBGROUP_SIZE(SG_SIZE) \
  auto sg = sycl::ext::oneapi::this_work_item::get_sub_group(); \
                        if(sg.get_max_local_range() != SG_SIZE) \
                Kokkos::abort("Expected subgroup size " ## SG_SIZE);
#else
#define KOKKOS_SYCL_CHECK_SUBGROUP_SIZE(SG_SIZE) \
  auto sg = sycl::ext::oneapi::experimental::this_sub_group(); \
                        if(sg.get_max_local_range() != SG_SIZE) \
                Kokkos::abort("Expected subgroup size SG_SIZE");
#endif

TEST(sycl, subgroup_size) {
#ifndef	SYCL_EXT_ONEAPI_KERNEL_PROPERTIES
  GTEST_SKIP() << " test requires SYCL_EXT_ONEAPI_KERNEL_PROPERTIES to be defined";
#endif
#ifndef KOKKOS_ARCH_INTEL_PVC
  GTEST_SKIP() << " test is designed for PVC architecture";
#endif

	Kokkos::RangePolicy<Kokkos::SYCL,Kokkos::SubGroupSize<16>> range_policy_16(0,10);
        Kokkos::RangePolicy<Kokkos::SYCL,Kokkos::SubGroupSize<32>> range_policy_32(0,10);
        Kokkos::MDRangePolicy<Kokkos::SYCL,Kokkos::SubGroupSize<16>, Kokkos::Rank<2>> mdrange_policy_16({0,0}, {10,10});
        Kokkos::MDRangePolicy<Kokkos::SYCL,Kokkos::SubGroupSize<32>, Kokkos::Rank<2>> mdrange_policy_32({0,0}, {10,10});
        Kokkos::TeamPolicy<Kokkos::SYCL,Kokkos::SubGroupSize<16>> team_policy_16(1, Kokkos::AUTO);
        Kokkos::TeamPolicy<Kokkos::SYCL,Kokkos::SubGroupSize<32>> team_policy_32(1, Kokkos::AUTO);

	Kokkos::parallel_for(range_policy_16, KOKKOS_LAMBDA(int) { KOKKOS_SYCL_CHECK_SUBGROUP_SIZE(16) });
	        Kokkos::parallel_for(range_policy_32, KOKKOS_LAMBDA(int) {
#if defined(__INTEL_LLVM_COMPILER) && __INTEL_LLVM_COMPILER >= 20250000
  auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
#else
  auto sg = sycl::ext::oneapi::experimental::this_sub_group();
#endif
                        if(sg.get_max_local_range() != 32)
                Kokkos::abort("Expected subgroup size 32 for RangePolicy parallel_for");
                        });

double dummy;
		        Kokkos::parallel_reduce(range_policy_16, KOKKOS_LAMBDA(int, double) {
#if defined(__INTEL_LLVM_COMPILER) && __INTEL_LLVM_COMPILER >= 20250000
  auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
#else
  auto sg = sycl::ext::oneapi::experimental::this_sub_group();
#endif
                        if(sg.get_max_local_range() != 16)
                Kokkos::abort("Expected subgroup size 16 for RangePolicy parallel_for");
                        }, dummy);
                Kokkos::parallel_reduce(range_policy_32, KOKKOS_LAMBDA(int, double) {
#if defined(__INTEL_LLVM_COMPILER) && __INTEL_LLVM_COMPILER >= 20250000
  auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
#else
  auto sg = sycl::ext::oneapi::experimental::this_sub_group();
#endif
                        if(sg.get_max_local_range() != 32)
                Kokkos::abort("Expected subgroup size 32 for RangePolicy parallel_for");
                        }, dummy);
		        Kokkos::parallel_scan(range_policy_16, KOKKOS_LAMBDA(int, double, bool) {
#if defined(__INTEL_LLVM_COMPILER) && __INTEL_LLVM_COMPILER >= 20250000
  auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
#else
  auto sg = sycl::ext::oneapi::experimental::this_sub_group();
#endif
                        if(sg.get_max_local_range() != 16)
                Kokkos::abort("Expected subgroup size 16 for RangePolicy parallel_for");
                        }, dummy);
                Kokkos::parallel_scan(range_policy_32, KOKKOS_LAMBDA(int, double, bool) {
#if defined(__INTEL_LLVM_COMPILER) && __INTEL_LLVM_COMPILER >= 20250000
  auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
#else
  auto sg = sycl::ext::oneapi::experimental::this_sub_group();
#endif
                        if(sg.get_max_local_range() != 32)
                Kokkos::abort("Expected subgroup size 32 for RangePolicy parallel_for");
                        }, dummy);
}

}
