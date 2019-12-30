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
#include <gtest/gtest.h>

/// @Kokkos_Feature_Level_Required:3

namespace Test {

using DataType     = double;
const double value = 0.5;

// Unit Test for Reduction

struct Functor {
  DataType *_data;

  Functor(DataType *data) : _data(data) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, double &UpdateSum) const {
    _data[i] = i * value;
    UpdateSum += _data[i];
  }
};

template <class ExecSpace>
struct TestReduction {
  int num_elements = 10;
  DataType *deviceData, *hostData;

  // memory_space for the memory allocation
  // memory_space for the memory allocation
  typedef typename TEST_EXECSPACE::memory_space MemSpaceD;
  typedef Kokkos::HostSpace MemSpaceH;

  // compare and equal
  int compare_equal(double sum) {
    int sum_local = 0;
    for (int i = 0; i < num_elements; ++i) sum_local += i;

    return (sum - (sum_local * value));
  }

  void reduction() {
    double sum = 0.0;
#if defined(KOKKOS_ENABLE_CUDA)
    typedef Kokkos::RangePolicy<ExecSpace, Kokkos::Schedule<Kokkos::Static> >
        range_policy;
#else
    typedef Kokkos::RangePolicy<ExecSpace, Kokkos::Schedule<Kokkos::Dynamic> >
        range_policy;
#endif

    // Allocate Memory for both device and host memory spaces
    deviceData = (DataType *)Kokkos::kokkos_malloc<MemSpaceD>(
        "dataD", num_elements * sizeof(DataType));
    hostData = (DataType *)Kokkos::kokkos_malloc<MemSpaceH>(
        "dataH", num_elements * sizeof(DataType));

    // parallel_reduce call
    Functor func(deviceData);
    Kokkos::parallel_reduce("Reduction", range_policy(0, num_elements), func,
                            sum);

    // Copy the data back to Host memory space
    Kokkos::Impl::DeepCopy<MemSpaceD, MemSpaceH>(
        hostData, deviceData, num_elements * sizeof(DataType));

    // Check if all data has been update correctly
    int sumError = compare_equal(sum);
    ASSERT_EQ(sumError, 0);

    // Free the allocated memory
    Kokkos::kokkos_free<MemSpaceD>(deviceData);
    Kokkos::kokkos_free<MemSpaceH>(hostData);
  }
};

TEST(TEST_CATEGORY, incr_03_reduction) {
  TestReduction<TEST_EXECSPACE> test;
  test.reduction();
}

}  // namespace Test
