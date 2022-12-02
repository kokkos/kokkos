/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <iostream>

namespace Test {
using CachePreference = Kokkos::Impl::CachePreference;
TEST(TEST_CATEGORY, cuda_cache_config) {
  CachePreference cache_config;
  cudaFuncAttributes attributes;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  Kokkos::TeamPolicy<TEST_EXECSPACE> policy_t;
  dim3 grid{14, 1, 1};
  dim3 block{64, 10, 1};

  int shmem = 0;

  {
    attributes.numRegs         = 32;
    attributes.sharedSizeBytes = 16;
#if defined(KOKKOS_ARCH_VOLTA70)
    shmem = 1000;
#elif defined(KOKKOS_ARCH_AMPERE80)
    shmem = 10000;
#else
    printf(
        "The unit test is only defined for Volta70 and Ampere80 "
        "architectures.\n");
#endif

    modify_launch_configuration_if_desired_occupancy_is_specified(
        grid, policy_t, prop, attributes, block, shmem, cache_config);

    ASSERT_EQ(cache_config, CachePreference::PreferL1);
  }
  {
    attributes.numRegs         = 64;
    attributes.sharedSizeBytes = 32;
#if defined(KOKKOS_ARCH_VOLTA70)
    shmem = 45000;
#elif defined(KOKKOS_ARCH_AMPERE80)
    shmem = 55000;
#else
    printf(
        "The unit test is only defined for Volta70 and Ampere80 "
        "architectures.\n");
#endif

    modify_launch_configuration_if_desired_occupancy_is_specified(
        grid, policy_t, prop, attributes, block, shmem, cache_config);

    ASSERT_EQ(cache_config, CachePreference::PreferEqual);
  }
  {
    cudaFuncAttributes attributes;
    attributes.numRegs         = 32;
    attributes.sharedSizeBytes = 16;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    shmem = 63000;

    modify_launch_configuration_if_desired_occupancy_is_specified(
        grid, policy_t, prop, attributes, block, shmem, cache_config);

    ASSERT_EQ(cache_config, CachePreference::PreferShared);
  }
}

}  // namespace Test
