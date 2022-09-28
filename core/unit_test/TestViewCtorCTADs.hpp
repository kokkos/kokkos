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

#include <Kokkos_Core.hpp>

namespace Test {

struct TestViewCtorCTADs {
  using DataType                     = double;
  static constexpr DataType* m       = nullptr;
  static constexpr const DataType* c = nullptr;

  static void test_viewctorcopymove() {
    Kokkos::View<DataType> vt;
    Kokkos::View cd(vt);
    ASSERT_TRUE((std::is_same_v<decltype(cd), decltype(vt)>));

    Kokkos::View md(std::move(vt));
    ASSERT_TRUE((std::is_same_v<decltype(md), decltype(vt)>));
  }

  static void test_viewctor0params() {
    Kokkos::View<DataType> mt0(m);
    Kokkos::View md0(m);
    ASSERT_TRUE((std::is_same_v<decltype(md0), decltype(mt0)>));

    Kokkos::View<const DataType> ct0(c);
    Kokkos::View cd0(c);
    ASSERT_TRUE((std::is_same_v<decltype(cd0), decltype(ct0)>));
  }

  static void test_viewctor1params() {
    Kokkos::View<DataType*> mt1(m, 1);
    Kokkos::View md1(m, 1);
    ASSERT_TRUE((std::is_same_v<decltype(md1), decltype(mt1)>));

    Kokkos::View<const DataType*> ct1(c, 1);
    Kokkos::View cd1(c, 1);
    ASSERT_TRUE((std::is_same_v<decltype(cd1), decltype(ct1)>));
  }

  static void test_viewctor2params() {
    Kokkos::View<DataType**> mt2(m, 1, 1);
    Kokkos::View md2(m, 1, 1);
    ASSERT_TRUE((std::is_same_v<decltype(md2), decltype(mt2)>));

    Kokkos::View<const DataType**> ct2(c, 1, 1);
    Kokkos::View cd2(c, 1, 1);
    ASSERT_TRUE((std::is_same_v<decltype(cd2), decltype(ct2)>));
  }

  static void test_viewctor3params() {
    Kokkos::View<DataType***> mt3(m, 1, 1, 1);
    Kokkos::View md3(m, 1, 1, 1);
    ASSERT_TRUE((std::is_same_v<decltype(md3), decltype(mt3)>));

    Kokkos::View<const DataType***> ct3(c, 1, 1, 1);
    Kokkos::View cd3(c, 1, 1, 1);
    ASSERT_TRUE((std::is_same_v<decltype(cd3), decltype(ct3)>));
  }

  static void test_viewctor4params() {
    Kokkos::View<DataType****> mt4(m, 1, 1, 1, 1);
    Kokkos::View md4(m, 1, 1, 1, 1);
    ASSERT_TRUE((std::is_same_v<decltype(md4), decltype(mt4)>));

    Kokkos::View<const DataType****> ct4(c, 1, 1, 1, 1);
    Kokkos::View cd4(c, 1, 1, 1, 1);
    ASSERT_TRUE((std::is_same_v<decltype(cd4), decltype(ct4)>));
  }

  static void test_viewctor5params() {
    Kokkos::View<DataType*****> mt5(m, 1, 1, 1, 1, 1);
    Kokkos::View md5(m, 1, 1, 1, 1, 1);
    ASSERT_TRUE((std::is_same_v<decltype(md5), decltype(mt5)>));

    Kokkos::View<const DataType*****> ct5(c, 1, 1, 1, 1, 1);
    Kokkos::View cd5(c, 1, 1, 1, 1, 1);
    ASSERT_TRUE((std::is_same_v<decltype(cd5), decltype(ct5)>));
  }

  static void test_viewctor6params() {
    Kokkos::View<DataType******> mt6(m, 1, 1, 1, 1, 1, 1);
    Kokkos::View md6(m, 1, 1, 1, 1, 1, 1);
    ASSERT_TRUE((std::is_same_v<decltype(md6), decltype(mt6)>));

    Kokkos::View<const DataType******> ct6(c, 1, 1, 1, 1, 1, 1);
    Kokkos::View cd6(c, 1, 1, 1, 1, 1, 1);
    ASSERT_TRUE((std::is_same_v<decltype(cd6), decltype(ct6)>));
  }

  static void test_viewctor7params() {
    Kokkos::View<DataType*******> mt7(m, 1, 1, 1, 1, 1, 1, 1);
    Kokkos::View md7(m, 1, 1, 1, 1, 1, 1, 1);
    ASSERT_TRUE((std::is_same_v<decltype(md7), decltype(mt7)>));

    Kokkos::View<const DataType*******> ct7(c, 1, 1, 1, 1, 1, 1, 1);
    Kokkos::View cd7(c, 1, 1, 1, 1, 1, 1, 1);
    ASSERT_TRUE((std::is_same_v<decltype(cd7), decltype(ct7)>));
  }

  static void test_viewctor8params() {
    Kokkos::View<DataType********> mt8(m, 1, 1, 1, 1, 1, 1, 1, 1);
    Kokkos::View md8(m, 1, 1, 1, 1, 1, 1, 1, 1);
    ASSERT_TRUE((std::is_same_v<decltype(md8), decltype(mt8)>));

    Kokkos::View<const DataType********> ct8(c, 1, 1, 1, 1, 1, 1, 1, 1);
    Kokkos::View cd8(c, 1, 1, 1, 1, 1, 1, 1, 1);
    ASSERT_TRUE((std::is_same_v<decltype(cd8), decltype(ct8)>));
  }
};

TEST(TEST_CATEGORY, view_ctor_ctads) {
  TestViewCtorCTADs::test_viewctorcopymove();
  TestViewCtorCTADs::test_viewctor0params();
  TestViewCtorCTADs::test_viewctor1params();
  TestViewCtorCTADs::test_viewctor2params();
  TestViewCtorCTADs::test_viewctor3params();
  TestViewCtorCTADs::test_viewctor4params();
  TestViewCtorCTADs::test_viewctor5params();
  TestViewCtorCTADs::test_viewctor6params();
  TestViewCtorCTADs::test_viewctor7params();
  TestViewCtorCTADs::test_viewctor8params();
}

}  // namespace Test
