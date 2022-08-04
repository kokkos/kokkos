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

template <typename ExecSpace = TEST_EXECSPACE>
struct TestReducerCTADs {
  using scalar        = double*;
  using memspace      = typename ExecSpace::memory_space;
  using view_type     = Kokkos::View<scalar, memspace>;

  static void test_sum() {
    view_type view;

    Kokkos::Sum<scalar, memspace> rt(view);
    Kokkos::Sum rd(view);
    ASSERT_TRUE((std::is_same_v<decltype(rd), decltype(rt)>));

    Kokkos::Sum rdc(rt);
    ASSERT_TRUE((std::is_same_v<decltype(rdc), decltype(rt)>));

    Kokkos::Sum rdm(std::move(rt));
    ASSERT_TRUE((std::is_same_v<decltype(rdm), decltype(rt)>));
  }

  static void test_prod() {
    view_type view;

    Kokkos::Prod<scalar, memspace> rt(view);
    Kokkos::Prod rd(view);
    ASSERT_TRUE((std::is_same_v<decltype(rd), decltype(rt)>));

    Kokkos::Prod rdc(rt);
    ASSERT_TRUE((std::is_same_v<decltype(rdc), decltype(rt)>));

    Kokkos::Prod rdm(std::move(rt));
    ASSERT_TRUE((std::is_same_v<decltype(rdm), decltype(rt)>));
  }
  static void test_min() {
    view_type view;

    Kokkos::Min<scalar, memspace> rt(view);
    Kokkos::Min rd(view);
    ASSERT_TRUE((std::is_same_v<decltype(rd), decltype(rt)>));

    Kokkos::Min rdc(rt);
    ASSERT_TRUE((std::is_same_v<decltype(rdc), decltype(rt)>));

    Kokkos::Min rdm(std::move(rt));
    ASSERT_TRUE((std::is_same_v<decltype(rdm), decltype(rt)>));
  }
  static void test_max() {
    view_type view;

    Kokkos::Max<scalar, memspace> rt(view);
    Kokkos::Max rd(view);
    ASSERT_TRUE((std::is_same_v<decltype(rd), decltype(rt)>));

    Kokkos::Max rdc(rt);
    ASSERT_TRUE((std::is_same_v<decltype(rdc), decltype(rt)>));

    Kokkos::Max rdm(std::move(rt));
    ASSERT_TRUE((std::is_same_v<decltype(rdm), decltype(rt)>));
  }
  static void test_land() {
    view_type view;

    Kokkos::LAnd<scalar, memspace> rt(view);
    Kokkos::LAnd rd(view);
    ASSERT_TRUE((std::is_same_v<decltype(rd), decltype(rt)>));

    Kokkos::LAnd rdc(rt);
    ASSERT_TRUE((std::is_same_v<decltype(rdc), decltype(rt)>));

    Kokkos::LAnd rdm(std::move(rt));
    ASSERT_TRUE((std::is_same_v<decltype(rdm), decltype(rt)>));
  }
  static void test_lor() {
    view_type view;

    Kokkos::LOr<scalar, memspace> rt(view);
    Kokkos::LOr rd(view);
    ASSERT_TRUE((std::is_same_v<decltype(rd), decltype(rt)>));

    Kokkos::LOr rdc(rt);
    ASSERT_TRUE((std::is_same_v<decltype(rdc), decltype(rt)>));

    Kokkos::LOr rdm(std::move(rt));
    ASSERT_TRUE((std::is_same_v<decltype(rdm), decltype(rt)>));
  }
  static void test_band() {
    view_type view;

    Kokkos::BAnd<scalar, memspace> rt(view);
    Kokkos::BAnd rd(view);
    ASSERT_TRUE((std::is_same_v<decltype(rd), decltype(rt)>));

    Kokkos::BAnd rdc(rt);
    ASSERT_TRUE((std::is_same_v<decltype(rdc), decltype(rt)>));

    Kokkos::BAnd rdm(std::move(rt));
    ASSERT_TRUE((std::is_same_v<decltype(rdm), decltype(rt)>));
  }
  static void test_bor() {
    view_type view;

    Kokkos::BOr<scalar, memspace> rt(view);
    Kokkos::BOr rd(view);
    ASSERT_TRUE((std::is_same_v<decltype(rd), decltype(rt)>));

    Kokkos::BOr rdc(rt);
    ASSERT_TRUE((std::is_same_v<decltype(rdc), decltype(rt)>));

    Kokkos::BOr rdm(std::move(rt));
    ASSERT_TRUE((std::is_same_v<decltype(rdm), decltype(rt)>));
  }
  //TODO
  static void test_minloc() {
  }
};

TEST(TEST_CATEGORY, reducer_ctads) {
  TestReducerCTADs<TEST_EXECSPACE>::test_sum();
  TestReducerCTADs<TEST_EXECSPACE>::test_prod();
  TestReducerCTADs<TEST_EXECSPACE>::test_min();
  TestReducerCTADs<TEST_EXECSPACE>::test_max();
  TestReducerCTADs<TEST_EXECSPACE>::test_land();
  TestReducerCTADs<TEST_EXECSPACE>::test_lor();
  TestReducerCTADs<TEST_EXECSPACE>::test_band();
  TestReducerCTADs<TEST_EXECSPACE>::test_bor();
  TestReducerCTADs<TEST_EXECSPACE>::test_minloc();
}

}  // namespace Test
