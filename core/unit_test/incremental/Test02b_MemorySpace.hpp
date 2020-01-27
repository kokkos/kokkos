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

/// @Kokkos_Feature_Level_Required:2
// Unit test for Kokkos free.
// We constantly allocate and free the memory.
// If the kokkos_free does not free the allocated memory,
// we will exceed the available space.

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

namespace Test {

using value_type = double;

// Allocate M number of value_type elements N number of times.
const int N = 100000;
const int M = 100000;

template <class ExecSpace>
struct TestIncrMemorySpace_free {
  using memory_space = typename ExecSpace::memory_space;

  void test_free() {
    for (int i = 0; i < N; ++i) {
      auto *data = static_cast<value_type *>(
          Kokkos::kokkos_malloc<memory_space>("data", M * sizeof(value_type)));

      ASSERT_NE(data, nullptr);

      Kokkos::kokkos_free<memory_space>(data);
    }
  }
};

TEST(TEST_CATEGORY, incr_02b_memspace_free) {
  TestIncrMemorySpace_free<TEST_EXECSPACE> test;
  test.test_free();
}

}  // namespace Test
