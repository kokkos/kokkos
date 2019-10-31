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

/*
 * Unit Test for Kokkos::parallel_for and Kokkos::parallel_reduce
 * */
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <sstream>

namespace Test {

#define num_elements 5

template <class ExecSpace>
struct add_x
{
  typedef Kokkos::View<int*, Kokkos::DefaultExecutionSpace::array_layout, Kokkos::DefaultExecutionSpace::memory_space> view_type;

  private:
  view_type arr_inner;

  public :
  const int x = 5;

  KOKKOS_INLINE_FUNCTION
  void fill_array()
  {
    arr_inner = view_type("arr_inner",num_elements);
    Kokkos::parallel_for("init arr_inner", num_elements, KOKKOS_LAMBDA(int i){
      arr_inner(i) = i+1;
    });
  }

  KOKKOS_INLINE_FUNCTION
  void update_serial()
  {
    Kokkos::parallel_for("init arr_inner", num_elements, KOKKOS_LAMBDA(int i){
      arr_inner(i) += x;
    });
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i)
  {
      arr_inner(i) += x;
  }

  KOKKOS_INLINE_FUNCTION
  int compare_equal()
  {
    int error = 0;
    Kokkos::parallel_reduce("compare_equal", num_elements, KOKKOS_LAMBDA(const int i, int& errorUpdate)
    {
      if(arr_inner(i) != (i+x+1)) errorUpdate++;
    }, error);
  }
};

template <class ExecSpace>
struct TestIncrFunctor {

  void testit()
  {
    //Create an object of add_x structure
    add_x<ExecSpace> arr;

    //fill the array in the add_x structures with initial values
    arr.fill_array();

    //update arr
    arr.update();

    int sumError = arr.compare_equal();

    ASSERT_EQ(sumError, 0);
  }
};

//Test to check whether parallel for has added x=5 to arr_inner
TEST_F(TEST_CATEGORY, incr_03_parallel_for) {
  TestIncrFunctor<TEST_EXECSPACE> test;
  test.testit();
}

}  // namespace Test
