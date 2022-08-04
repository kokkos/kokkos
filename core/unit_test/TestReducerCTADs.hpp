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
  using execspace = ExecSpace;

  using scalar_type = double;
  using index_type  = int;
  using memspace    = typename execspace::memory_space;

  static void test_sum() {
    Kokkos::View<scalar_type, memspace> view;

    Kokkos::Sum<scalar_type, memspace> rt(view);
    Kokkos::Sum rd(view);
    ASSERT_TRUE((std::is_same_v<decltype(rd), decltype(rt)>));

    Kokkos::Sum rdc(rt);
    ASSERT_TRUE((std::is_same_v<decltype(rdc), decltype(rt)>));

    Kokkos::Sum rdm(std::move(rt));
    ASSERT_TRUE((std::is_same_v<decltype(rdm), decltype(rt)>));
  }

  static void test_prod() {
    Kokkos::View<scalar_type, memspace> view;

    Kokkos::Prod<scalar_type, memspace> rt(view);
    Kokkos::Prod rd(view);
    ASSERT_TRUE((std::is_same_v<decltype(rd), decltype(rt)>));

    Kokkos::Prod rdc(rt);
    ASSERT_TRUE((std::is_same_v<decltype(rdc), decltype(rt)>));

    Kokkos::Prod rdm(std::move(rt));
    ASSERT_TRUE((std::is_same_v<decltype(rdm), decltype(rt)>));
  }

  static void test_min() {
    Kokkos::View<scalar_type, memspace> view;

    Kokkos::Min<scalar_type, memspace> rt(view);
    Kokkos::Min rd(view);
    ASSERT_TRUE((std::is_same_v<decltype(rd), decltype(rt)>));

    Kokkos::Min rdc(rt);
    ASSERT_TRUE((std::is_same_v<decltype(rdc), decltype(rt)>));

    Kokkos::Min rdm(std::move(rt));
    ASSERT_TRUE((std::is_same_v<decltype(rdm), decltype(rt)>));
  }

  static void test_max() {
    Kokkos::View<scalar_type, memspace> view;

    Kokkos::Max<scalar_type, memspace> rt(view);
    Kokkos::Max rd(view);
    ASSERT_TRUE((std::is_same_v<decltype(rd), decltype(rt)>));

    Kokkos::Max rdc(rt);
    ASSERT_TRUE((std::is_same_v<decltype(rdc), decltype(rt)>));

    Kokkos::Max rdm(std::move(rt));
    ASSERT_TRUE((std::is_same_v<decltype(rdm), decltype(rt)>));
  }

  static void test_land() {
    Kokkos::View<scalar_type, memspace> view;

    Kokkos::LAnd<scalar_type, memspace> rt(view);
    Kokkos::LAnd rd(view);
    ASSERT_TRUE((std::is_same_v<decltype(rd), decltype(rt)>));

    Kokkos::LAnd rdc(rt);
    ASSERT_TRUE((std::is_same_v<decltype(rdc), decltype(rt)>));

    Kokkos::LAnd rdm(std::move(rt));
    ASSERT_TRUE((std::is_same_v<decltype(rdm), decltype(rt)>));
  }

  static void test_lor() {
    Kokkos::View<scalar_type, memspace> view;

    Kokkos::LOr<scalar_type, memspace> rt(view);
    Kokkos::LOr rd(view);
    ASSERT_TRUE((std::is_same_v<decltype(rd), decltype(rt)>));

    Kokkos::LOr rdc(rt);
    ASSERT_TRUE((std::is_same_v<decltype(rdc), decltype(rt)>));

    Kokkos::LOr rdm(std::move(rt));
    ASSERT_TRUE((std::is_same_v<decltype(rdm), decltype(rt)>));
  }

  static void test_band() {
    Kokkos::View<scalar_type, memspace> view;

    Kokkos::BAnd<scalar_type, memspace> rt(view);
    Kokkos::BAnd rd(view);
    ASSERT_TRUE((std::is_same_v<decltype(rd), decltype(rt)>));

    Kokkos::BAnd rdc(rt);
    ASSERT_TRUE((std::is_same_v<decltype(rdc), decltype(rt)>));

    Kokkos::BAnd rdm(std::move(rt));
    ASSERT_TRUE((std::is_same_v<decltype(rdm), decltype(rt)>));
  }

  static void test_bor() {
    Kokkos::View<scalar_type, memspace> view;

    Kokkos::BOr<scalar_type, memspace> rt(view);
    Kokkos::BOr rd(view);
    ASSERT_TRUE((std::is_same_v<decltype(rd), decltype(rt)>));

    Kokkos::BOr rdc(rt);
    ASSERT_TRUE((std::is_same_v<decltype(rdc), decltype(rt)>));

    Kokkos::BOr rdm(std::move(rt));
    ASSERT_TRUE((std::is_same_v<decltype(rdm), decltype(rt)>));
  }

  static void test_minloc() {
    Kokkos::View<Kokkos::ValLocScalar<scalar_type, index_type>, memspace> view;

    Kokkos::MinLoc<scalar_type, index_type, memspace> rt(view);
    Kokkos::MinLoc rd(view);
    ASSERT_TRUE((std::is_same_v<decltype(rd), decltype(rt)>));

    Kokkos::MinLoc rdc(rt);
    ASSERT_TRUE((std::is_same_v<decltype(rdc), decltype(rt)>));

    Kokkos::MinLoc rdm(std::move(rt));
    ASSERT_TRUE((std::is_same_v<decltype(rdm), decltype(rt)>));
  }

  static void test_maxloc() {
    Kokkos::View<Kokkos::ValLocScalar<scalar_type, index_type>, memspace> view;

    Kokkos::MaxLoc<scalar_type, index_type, memspace> rt(view);
    Kokkos::MaxLoc rd(view);
    ASSERT_TRUE((std::is_same_v<decltype(rd), decltype(rt)>));

    Kokkos::MaxLoc rdc(rt);
    ASSERT_TRUE((std::is_same_v<decltype(rdc), decltype(rt)>));

    Kokkos::MaxLoc rdm(std::move(rt));
    ASSERT_TRUE((std::is_same_v<decltype(rdm), decltype(rt)>));
  }

  static void test_minmax() {
    Kokkos::View<Kokkos::MinMaxScalar<scalar_type>, memspace> view;

    Kokkos::MinMax<scalar_type, memspace> rt(view);
    Kokkos::MinMax rd(view);
    ASSERT_TRUE((std::is_same_v<decltype(rd), decltype(rt)>));

    Kokkos::MinMax rdc(rt);
    ASSERT_TRUE((std::is_same_v<decltype(rdc), decltype(rt)>));

    Kokkos::MinMax rdm(std::move(rt));
    ASSERT_TRUE((std::is_same_v<decltype(rdm), decltype(rt)>));
  }

  static void test_minmaxloc() {
    Kokkos::View<Kokkos::MinMaxLocScalar<scalar_type, index_type>, memspace>
        view;

    Kokkos::MinMaxLoc<scalar_type, index_type, memspace> rt(view);
    Kokkos::MinMaxLoc rd(view);
    ASSERT_TRUE((std::is_same_v<decltype(rd), decltype(rt)>));

    Kokkos::MinMaxLoc rdc(rt);
    ASSERT_TRUE((std::is_same_v<decltype(rdc), decltype(rt)>));

    Kokkos::MinMaxLoc rdm(std::move(rt));
    ASSERT_TRUE((std::is_same_v<decltype(rdm), decltype(rt)>));
  }

  static void test_maxfirstloc() {
    Kokkos::View<Kokkos::ValLocScalar<scalar_type, index_type>, memspace> view;

    Kokkos::MaxFirstLoc<scalar_type, index_type, memspace> rt(view);
    Kokkos::MaxFirstLoc rd(view);
    ASSERT_TRUE((std::is_same_v<decltype(rd), decltype(rt)>));

    Kokkos::MaxFirstLoc rdc(rt);
    ASSERT_TRUE((std::is_same_v<decltype(rdc), decltype(rt)>));

    Kokkos::MaxFirstLoc rdm(std::move(rt));
    ASSERT_TRUE((std::is_same_v<decltype(rdm), decltype(rt)>));
  }

  static void test_maxfirstloccustomcomparator() {
    Kokkos::View<Kokkos::ValLocScalar<scalar_type, index_type>, memspace> view;

    auto comparator       = [](scalar_type, scalar_type) { return true; };
    using comparator_type = decltype(comparator);

    Kokkos::MaxFirstLocCustomComparator<scalar_type, index_type,
                                        comparator_type, memspace>
        rt(view, comparator);
    Kokkos::MaxFirstLocCustomComparator rd(view, comparator);
    ASSERT_TRUE((std::is_same_v<decltype(rd), decltype(rt)>));

    Kokkos::MaxFirstLocCustomComparator rdc(rt);
    ASSERT_TRUE((std::is_same_v<decltype(rdc), decltype(rt)>));

    Kokkos::MaxFirstLocCustomComparator rdm(std::move(rt));
    ASSERT_TRUE((std::is_same_v<decltype(rdm), decltype(rt)>));
  }

  static void test_minfirstloc() {
    Kokkos::View<Kokkos::ValLocScalar<scalar_type, index_type>, memspace> view;

    Kokkos::MinFirstLoc<scalar_type, index_type, memspace> rt(view);
    Kokkos::MinFirstLoc rd(view);
    ASSERT_TRUE((std::is_same_v<decltype(rd), decltype(rt)>));

    Kokkos::MinFirstLoc rdc(rt);
    ASSERT_TRUE((std::is_same_v<decltype(rdc), decltype(rt)>));

    Kokkos::MinFirstLoc rdm(std::move(rt));
    ASSERT_TRUE((std::is_same_v<decltype(rdm), decltype(rt)>));
  }

  static void test_minfirstloccustomcomparator() {
    Kokkos::View<Kokkos::ValLocScalar<scalar_type, index_type>, memspace> view;

    auto comparator       = [](scalar_type, scalar_type) { return true; };
    using comparator_type = decltype(comparator);

    Kokkos::MinFirstLocCustomComparator<scalar_type, index_type,
                                        comparator_type, memspace>
        rt(view, comparator);
    Kokkos::MinFirstLocCustomComparator rd(view, comparator);
    ASSERT_TRUE((std::is_same_v<decltype(rd), decltype(rt)>));

    Kokkos::MinFirstLocCustomComparator rdc(rt);
    ASSERT_TRUE((std::is_same_v<decltype(rdc), decltype(rt)>));

    Kokkos::MinFirstLocCustomComparator rdm(std::move(rt));
    ASSERT_TRUE((std::is_same_v<decltype(rdm), decltype(rt)>));
  }

  static void test_minmaxfirstlastloc() {
    Kokkos::View<Kokkos::MinMaxLocScalar<scalar_type, index_type>, memspace>
        view;

    Kokkos::MinMaxFirstLastLoc<scalar_type, index_type, memspace> rt(view);
    Kokkos::MinMaxFirstLastLoc rd(view);
    ASSERT_TRUE((std::is_same_v<decltype(rd), decltype(rt)>));

    Kokkos::MinMaxFirstLastLoc rdc(rt);
    ASSERT_TRUE((std::is_same_v<decltype(rdc), decltype(rt)>));

    Kokkos::MinMaxFirstLastLoc rdm(std::move(rt));
    ASSERT_TRUE((std::is_same_v<decltype(rdm), decltype(rt)>));
  }

  static void test_minmaxfirstlastloccustomcomparator() {
    Kokkos::View<Kokkos::MinMaxLocScalar<scalar_type, index_type>, memspace>
        view;

    auto comparator       = [](scalar_type, scalar_type) { return true; };
    using comparator_type = decltype(comparator);

    Kokkos::MinMaxFirstLastLocCustomComparator<scalar_type, index_type,
                                               comparator_type, memspace>
        rt(view, comparator);
    Kokkos::MinMaxFirstLastLocCustomComparator rd(view, comparator);
    ASSERT_TRUE((std::is_same_v<decltype(rd), decltype(rt)>));

    Kokkos::MinMaxFirstLastLocCustomComparator rdc(rt);
    ASSERT_TRUE((std::is_same_v<decltype(rdc), decltype(rt)>));

    Kokkos::MinMaxFirstLastLocCustomComparator rdm(std::move(rt));
    ASSERT_TRUE((std::is_same_v<decltype(rdm), decltype(rt)>));
  }

  static void test_firstloc() {
    Kokkos::View<Kokkos::FirstLocScalar<index_type>, memspace> view;

    Kokkos::FirstLoc<index_type, memspace> rt(view);
    Kokkos::FirstLoc rd(view);
    ASSERT_TRUE((std::is_same_v<decltype(rd), decltype(rt)>));

    Kokkos::FirstLoc rdc(rt);
    ASSERT_TRUE((std::is_same_v<decltype(rdc), decltype(rt)>));

    Kokkos::FirstLoc rdm(std::move(rt));
    ASSERT_TRUE((std::is_same_v<decltype(rdm), decltype(rt)>));
  }

  static void test_lastloc() {
    Kokkos::View<Kokkos::LastLocScalar<index_type>, memspace> view;

    Kokkos::LastLoc<index_type, memspace> rt(view);
    Kokkos::LastLoc rd(view);
    ASSERT_TRUE((std::is_same_v<decltype(rd), decltype(rt)>));

    Kokkos::LastLoc rdc(rt);
    ASSERT_TRUE((std::is_same_v<decltype(rdc), decltype(rt)>));

    Kokkos::LastLoc rdm(std::move(rt));
    ASSERT_TRUE((std::is_same_v<decltype(rdm), decltype(rt)>));
  }

  static void test_stdispartitioned() {
    Kokkos::View<Kokkos::StdIsPartScalar<index_type>, memspace> view;

    Kokkos::StdIsPartitioned<index_type, memspace> rt(view);
    Kokkos::StdIsPartitioned rd(view);
    ASSERT_TRUE((std::is_same_v<decltype(rd), decltype(rt)>));

    Kokkos::StdIsPartitioned rdc(rt);
    ASSERT_TRUE((std::is_same_v<decltype(rdc), decltype(rt)>));

    Kokkos::StdIsPartitioned rdm(std::move(rt));
    ASSERT_TRUE((std::is_same_v<decltype(rdm), decltype(rt)>));
  }

  static void test_stdpartitionpoint() {
    Kokkos::View<Kokkos::StdPartPointScalar<index_type>, memspace> view;

    Kokkos::StdPartitionPoint<index_type, memspace> rt(view);
    Kokkos::StdPartitionPoint rd(view);
    ASSERT_TRUE((std::is_same_v<decltype(rd), decltype(rt)>));

    Kokkos::StdPartitionPoint rdc(rt);
    ASSERT_TRUE((std::is_same_v<decltype(rdc), decltype(rt)>));

    Kokkos::StdPartitionPoint rdm(std::move(rt));
    ASSERT_TRUE((std::is_same_v<decltype(rdm), decltype(rt)>));
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
  TestReducerCTADs<TEST_EXECSPACE>::test_maxloc();
  TestReducerCTADs<TEST_EXECSPACE>::test_minmax();
  TestReducerCTADs<TEST_EXECSPACE>::test_minmaxloc();
  TestReducerCTADs<TEST_EXECSPACE>::test_maxfirstloc();
  TestReducerCTADs<TEST_EXECSPACE>::test_maxfirstloccustomcomparator();
  TestReducerCTADs<TEST_EXECSPACE>::test_minfirstloc();
  TestReducerCTADs<TEST_EXECSPACE>::test_minfirstloccustomcomparator();
  TestReducerCTADs<TEST_EXECSPACE>::test_minmaxfirstlastloc();
  TestReducerCTADs<TEST_EXECSPACE>::test_minmaxfirstlastloccustomcomparator();
  TestReducerCTADs<TEST_EXECSPACE>::test_firstloc();
  TestReducerCTADs<TEST_EXECSPACE>::test_lastloc();
  TestReducerCTADs<TEST_EXECSPACE>::test_stdispartitioned();
  TestReducerCTADs<TEST_EXECSPACE>::test_stdpartitionpoint();
}

}  // namespace Test
