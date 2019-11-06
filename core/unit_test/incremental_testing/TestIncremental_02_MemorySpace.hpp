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

namespace Test {

// Test kokkos_malloc, kokkos_free && DeepCopy

template <class MemSpace>
struct TestIncrMemorySpace_malloc {
  const int num_elements = 10;

  void testit_malloc() {
    int *data = (int*) Kokkos::kokkos_malloc<MemSpace>("data", num_elements*sizeof(int));
    ASSERT_FALSE(data == nullptr);
    Kokkos::kokkos_free<MemSpace>(data);
  }

  void testit_free() {

    const int N = 100000;
    const int M = 100000;
    for(int i = 0; i < N; ++i)
    {
      double *data = (double*) Kokkos::kokkos_malloc<MemSpace>("data", M*sizeof(double));
      ASSERT_FALSE(data == nullptr);
      Kokkos::kokkos_free<MemSpace>(data);
    }
  }
};

template <class MemSpaceA, class MemSpaceB>
struct TestIncrMemorySpace_deepcopy {

  const int num_elements = 10;
  typedef Kokkos::View<double*, Kokkos::LayoutRight, MemSpaceA> viewDevice;
  typedef Kokkos::View<double*, Kokkos::LayoutLeft, MemSpaceB> viewHost;

  //fill the host side with 0.5 and device side with 0.0
  void fill_host(viewDevice a_device, viewHost a_host)
  {
    Kokkos::parallel_for("fill a_device", num_elements, KOKKOS_LAMBDA(int i)
    {
      a_device(i) = 0.0;
    });

    auto init = [=] (a_host) {
      for(int i = 0; i < num_elements; ++i)
        a_host(i) = 0.5;
    };
//    parallel_for("init host", num_elements; KOKKOS_LAMBDA(int i))
//    {
//      a_host(i) = 0.5;
//    });

  }

  void fill_device(viewDevice a_device, viewHost a_host)
  {
    Kokkos::parallel_for("fill a_device", num_elements, KOKKOS_LAMBDA(int i)
    {
      a_device(i) = 0.5;
    });

    Kokkos::parallel_for("fill a_host", num_elements, KOKKOS_LAMBDA(int i)
    {
      a_host(i) = 0.0;
    });
  }

  int compare_equal_device(viewDevice a_device)
  {
    int error = 0;
    Kokkos::parallel_reduce("compare_equal", num_elements, KOKKOS_LAMBDA(const int i, int& errorUpdate)
    {
      if(a_device(i) != 0.5) errorUpdate++;
    }, error);
  }

  int compare_equal_host(viewHost a_host)
  {
    int error = 0;
    Kokkos::parallel_reduce("compare_equal", num_elements, KOKKOS_LAMBDA(const int i, int& errorUpdate)
    {
      if(a_host(i) != 0.5) errorUpdate++;
    }, error);
  }

  void testit_HtoD() {
    viewDevice a_device("a_device",num_elements);
    viewHost a_host("a_host",num_elements);

    fill_host(a_device, a_host);

    //Deep copy from host to device
    Kokkos::deep_copy(a_device, a_host);

    // Check if all the numbers on the device are equal;
    int sumError = compare_equal_device(a_device);

    ASSERT_EQ(sumError, 0);
  }

  void testit_DtoH() {
    viewDevice a_device("a_device",num_elements);
    viewHost a_host("a_host",num_elements);

    fill_device(a_device, a_host);

    //Deep copy from device to host
    Kokkos::deep_copy(a_host, a_device);

    // Check if all the numbers on the device are equal;
    int sumError = compare_equal_host(a_host);

    ASSERT_EQ(sumError, 0);
  }
};

TEST_F(TEST_CATEGORY, incr_02_memspace_malloc) {
  TestIncrMemorySpace_malloc<TEST_EXECSPACE> test;
  test.testit_malloc();
}

TEST_F(TEST_CATEGORY, incr_02_memspace_free) {
  TestIncrMemorySpace_malloc<TEST_EXECSPACE> test;
  test.testit_free();
}

TEST_F(TEST_CATEGORY, incr_02_memspace_deepcopy_HtoD) {
  TestIncrMemorySpace_deepcopy<TEST_EXECSPACE, TEST_EXECSPACE> test;
  test.testit_HtoD();
}

TEST_F(TEST_CATEGORY, incr_02_memspace_deepcopy_DtoH) {
  TestIncrMemorySpace_deepcopy<TEST_EXECSPACE, TEST_EXECSPACE> test;
  test.testit_DtoH();
}
}  // namespace Test
