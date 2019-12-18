/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
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

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <sstream>
#include <type_traits>
#include <gtest/gtest.h>

/// @Kokkos_Feature_Level_Required:2

namespace Test {

// DeepCopy unit tests
// We allocate data in the host. Copy it to the device and copy back the data
// from device back to host We check if the original and copied data are the
// same to evaluate the deep copies.

template <class MemSpaceD, class MemSpaceH>
struct TestIncrMemorySpace_deepcopy {
  using dataType         = double;
  const int num_elements = 10;
  const double value     = 0.5;

  int compare_equal_host(dataType *HostData_send, dataType *HostData_recv) {
    int error = 0;
    for (int i = 0; i < num_elements; ++i) {
      if (HostData_send[i] != HostData_recv[i]) error++;
    }

    return error;
  }

  void testit_DtoH() {
    // Allocate memory on Device space
    dataType *DeviceData = (dataType *)Kokkos::kokkos_malloc<MemSpaceD>(
        "DeviceData", num_elements * sizeof(dataType));
    ASSERT_FALSE(DeviceData == nullptr);

    // Allocate memory on Host space
    dataType *HostData_send = (dataType *)Kokkos::kokkos_malloc<MemSpaceH>(
        "HostData", num_elements * sizeof(dataType));
    ASSERT_FALSE(HostData_send == nullptr);

    // Allocate memory on Host space
    dataType *HostData_recv = (dataType *)Kokkos::kokkos_malloc<MemSpaceH>(
        "HostData", num_elements * sizeof(dataType));
    ASSERT_FALSE(HostData_recv == nullptr);

    for (int i = 0; i < num_elements; ++i) {
      HostData_send[i] = i * value;
      HostData_recv[i] = 0.0;
    }

    // Copy first from Host_send to Device
    Kokkos::Impl::DeepCopy<MemSpaceH, MemSpaceD>(
        DeviceData, HostData_send, num_elements * sizeof(dataType));

    // Copy back from Device to Host
    Kokkos::Impl::DeepCopy<MemSpaceD, MemSpaceH>(
        HostData_recv, DeviceData, num_elements * sizeof(dataType));

    // Check if all data has been copied to and from host to device;
    int sumError = compare_equal_host(HostData_send, HostData_recv);
    ASSERT_EQ(sumError, 0);

    // Free the allocated memory
    Kokkos::kokkos_free<MemSpaceD>(DeviceData);
    Kokkos::kokkos_free<MemSpaceH>(HostData_send);
    Kokkos::kokkos_free<MemSpaceH>(HostData_recv);
  }
};

TEST(TEST_CATEGORY, incr_02c_memspace_deepcopy_DtoH) {
  typedef typename TEST_EXECSPACE::memory_space memory_space;
  typedef typename TEST_EXECSPACE::memory_space host_space;
  TestIncrMemorySpace_deepcopy<memory_space, host_space> test;
  test.testit_DtoH();
}

}  // namespace Test
