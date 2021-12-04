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

namespace Test {

#define LIVE(EXPR, ARGS, RANK) EXPR
#define DIE(EXPR, ARGS, EXPECTED)                                          \
  ASSERT_DEATH(                                                            \
      EXPR,                                                                \
      "Constructor for Kokkos View 'v_" #ARGS                              \
      "' has mismatched number of arguments. Number of arguments = " #ARGS \
      " but dynamic rank = " #EXPECTED)

TEST(TEST_CATEGORY_DEATH, view_construction_with_wrong_params) {
#define PARAM_0
#define PARAM_1 1
#define PARAM_2 1, 1
#define PARAM_3 1, 1, 1
#define PARAM_4 1, 1, 1, 1
#define PARAM_5 1, 1, 1, 1, 1
#define PARAM_6 1, 1, 1, 1, 1, 1
#define PARAM_7 1, 1, 1, 1, 1, 1, 1

#define PARAM_0_RANK 0
#define PARAM_1_RANK 1
#define PARAM_2_RANK 2
#define PARAM_3_RANK 3
#define PARAM_4_RANK 4
#define PARAM_5_RANK 5
#define PARAM_6_RANK 6
#define PARAM_7_RANK 7

  using DType   = int;
  using DType_0 = DType;
  using DType_1 = DType *;
  using DType_2 = DType **;
  using DType_3 = DType ***;
  using DType_4 = DType ****;
  using DType_5 = DType *****;
  using DType_6 = DType ******;
  using DType_7 = DType *******;

  {
    // test View parameters for View dim = 0
    using type = DType_0;
    LIVE({ Kokkos::View<type> v_0("v_0" PARAM_0); }, 0, 0);
    DIE({ Kokkos::View<type> v_1("v_1", PARAM_1); }, 1, 0);
    DIE({ Kokkos::View<type> v_2("v_2", PARAM_2); }, 2, 0);
    DIE({ Kokkos::View<type> v_3("v_3", PARAM_3); }, 3, 0);
    DIE({ Kokkos::View<type> v_4("v_4", PARAM_4); }, 4, 0);
    DIE({ Kokkos::View<type> v_5("v_5", PARAM_5); }, 5, 0);
    DIE({ Kokkos::View<type> v_6("v_6", PARAM_6); }, 6, 0);
    DIE({ Kokkos::View<type> v_7("v_7", PARAM_7); }, 7, 0);
  }

  {
    // test View parameters for View dim = 1
    using type = DType_1;
    DIE({ Kokkos::View<type> v_0("v_0" PARAM_0); }, 0, 1);
    LIVE({ Kokkos::View<type> v_1("v_1", PARAM_1); }, 1, 1);
    DIE({ Kokkos::View<type> v_2("v_2", PARAM_2); }, 2, 1);
    DIE({ Kokkos::View<type> v_3("v_3", PARAM_3); }, 3, 1);
    DIE({ Kokkos::View<type> v_4("v_4", PARAM_4); }, 4, 1);
    DIE({ Kokkos::View<type> v_5("v_5", PARAM_5); }, 5, 1);
    DIE({ Kokkos::View<type> v_6("v_6", PARAM_6); }, 6, 1);
    DIE({ Kokkos::View<type> v_7("v_7", PARAM_7); }, 7, 1);
  }

  {
    // test View parameters for View dim = 2
    using type = DType_2;
    DIE({ Kokkos::View<type> v_0("v_0" PARAM_0); }, 0, 2);
    DIE({ Kokkos::View<type> v_1("v_1", PARAM_1); }, 1, 2);
    LIVE({ Kokkos::View<type> v_2("v_2", PARAM_2); }, 2, 2);
    DIE({ Kokkos::View<type> v_3("v_3", PARAM_3); }, 3, 2);
    DIE({ Kokkos::View<type> v_4("v_4", PARAM_4); }, 4, 2);
    DIE({ Kokkos::View<type> v_5("v_5", PARAM_5); }, 5, 2);
    DIE({ Kokkos::View<type> v_6("v_6", PARAM_6); }, 6, 2);
    DIE({ Kokkos::View<type> v_7("v_7", PARAM_7); }, 7, 2);
  }

  {
    // test View parameters for View dim = 3
    using type = DType_3;
    DIE({ Kokkos::View<type> v_0("v_0" PARAM_0); }, 0, 3);
    DIE({ Kokkos::View<type> v_1("v_1", PARAM_1); }, 1, 3);
    DIE({ Kokkos::View<type> v_2("v_2", PARAM_2); }, 2, 3);
    LIVE({ Kokkos::View<type> v_3("v_3", PARAM_3); }, 3, 3);
    DIE({ Kokkos::View<type> v_4("v_4", PARAM_4); }, 4, 3);
    DIE({ Kokkos::View<type> v_5("v_5", PARAM_5); }, 5, 3);
    DIE({ Kokkos::View<type> v_6("v_6", PARAM_6); }, 6, 3);
    DIE({ Kokkos::View<type> v_7("v_7", PARAM_7); }, 7, 3);
  }

  {
    // test View parameters for View dim = 4
    using type = DType_4;
    DIE({ Kokkos::View<type> v_0("v_0" PARAM_0); }, 0, 4);
    DIE({ Kokkos::View<type> v_1("v_1", PARAM_1); }, 1, 4);
    DIE({ Kokkos::View<type> v_2("v_2", PARAM_2); }, 2, 4);
    DIE({ Kokkos::View<type> v_3("v_3", PARAM_3); }, 3, 4);
    LIVE({ Kokkos::View<type> v_4("v_4", PARAM_4); }, 4, 4);
    DIE({ Kokkos::View<type> v_5("v_5", PARAM_5); }, 5, 4);
    DIE({ Kokkos::View<type> v_6("v_6", PARAM_6); }, 6, 4);
    DIE({ Kokkos::View<type> v_7("v_7", PARAM_7); }, 7, 4);
  }

  {
    // test View parameters for View dim = 5
    using type = DType_5;
    DIE({ Kokkos::View<type> v_0("v_0" PARAM_0); }, 0, 5);
    DIE({ Kokkos::View<type> v_1("v_1", PARAM_1); }, 1, 5);
    DIE({ Kokkos::View<type> v_2("v_2", PARAM_2); }, 2, 5);
    DIE({ Kokkos::View<type> v_3("v_3", PARAM_3); }, 3, 5);
    DIE({ Kokkos::View<type> v_4("v_4", PARAM_4); }, 4, 5);
    LIVE({ Kokkos::View<type> v_5("v_5", PARAM_5); }, 5, 5);
    DIE({ Kokkos::View<type> v_6("v_6", PARAM_6); }, 6, 5);
    DIE({ Kokkos::View<type> v_7("v_7", PARAM_7); }, 7, 5);
  }

  {
    // test View parameters for View dim = 6
    using type = DType_6;
    DIE({ Kokkos::View<type> v_0("v_0" PARAM_0); }, 0, 6);
    DIE({ Kokkos::View<type> v_1("v_1", PARAM_1); }, 1, 6);
    DIE({ Kokkos::View<type> v_2("v_2", PARAM_2); }, 2, 6);
    DIE({ Kokkos::View<type> v_3("v_3", PARAM_3); }, 3, 6);
    DIE({ Kokkos::View<type> v_4("v_4", PARAM_4); }, 4, 6);
    DIE({ Kokkos::View<type> v_5("v_5", PARAM_5); }, 5, 6);
    LIVE({ Kokkos::View<type> v_6("v_6", PARAM_6); }, 6, 6);
    DIE({ Kokkos::View<type> v_7("v_7", PARAM_7); }, 7, 6);
  }

  {
    // test View parameters for View dim = 7
    using type = DType_7;
    DIE({ Kokkos::View<type> v_0("v_0" PARAM_0); }, 0, 7);
    DIE({ Kokkos::View<type> v_1("v_1", PARAM_1); }, 1, 7);
    DIE({ Kokkos::View<type> v_2("v_2", PARAM_2); }, 2, 7);
    DIE({ Kokkos::View<type> v_3("v_3", PARAM_3); }, 3, 7);
    DIE({ Kokkos::View<type> v_4("v_4", PARAM_4); }, 4, 7);
    DIE({ Kokkos::View<type> v_5("v_5", PARAM_5); }, 5, 7);
    DIE({ Kokkos::View<type> v_6("v_6", PARAM_6); }, 6, 7);
    LIVE({ Kokkos::View<type> v_7("v_7", PARAM_7); }, 7, 7);
  }
}

#undef LIVE
#undef DIE
#undef VIEW

}  // namespace Test
