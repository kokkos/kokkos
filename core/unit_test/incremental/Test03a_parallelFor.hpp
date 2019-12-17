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

/// @Kokkos_Feature_Level_Required:3

namespace Test {

  using dataType         = double;

// parallel-for unit test
// create an array of double dataType and add a constatnt to all elements of the array in a parallel-for

template <class ExecSpace>
struct array {
  const int _N = 10;
  const double value     = 0.5;
  dataType *_data;
  typedef typename TEST_EXECSPACE::memory_space memory_space;

  array(dataType *data)
    :_data(data)
  {
  }

KOKKOS_INLINE_FUNCTION
  void operator() (const int i) const
  {
    _data[i] = i*value;
  }

};

template <class ExecSpace>
struct TestParallel_For
{
  int _N = 10;
  const dataType value = 0.5;
  dataType *_data;

  //memory_space for the memory allocation
  typedef typename TEST_EXECSPACE::memory_space memory_space;

  int compare_equal(dataType* data) {
    int error = 0;
    for (int i = 0; i < _N; ++i) {
      if ( data[i] != i*value) error++;
    }
    return error;
  }

  void testit()
  {
    _data = (dataType *)Kokkos::kokkos_malloc<memory_space>("data",_N*sizeof(dataType));
    Kokkos::parallel_for("parallel_for", _N,array<ExecSpace>(_data));

    // Check if all data has been update correctly
    int sumError = compare_equal(_data);
    ASSERT_EQ(sumError, 0);
    Kokkos::kokkos_free<memory_space>(_data);
  }
};

TEST(TEST_CATEGORY, incr_03a_parallelFor) {
  TestParallel_For<TEST_EXECSPACE> test;
  test.testit();
}

}  // namespace Test
