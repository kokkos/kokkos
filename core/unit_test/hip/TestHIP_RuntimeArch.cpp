// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif
#include <TestHIP_Category.hpp>

#include <HIP/Kokkos_HIP_Instance.hpp>

namespace Test {

TEST(hip, runtime_arch_traits) {
  auto const wavefront_size = Kokkos::HIP::hip_device_prop().warpSize;
  ASSERT_TRUE(wavefront_size == 32 || wavefront_size == 64);
  ASSERT_EQ(Kokkos::Impl::HIPTraits::host_warp_size(), wavefront_size);
}

}  // namespace Test
