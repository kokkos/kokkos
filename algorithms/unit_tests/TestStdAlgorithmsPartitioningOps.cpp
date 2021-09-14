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

#include <TestStdAlgorithmsCommon.hpp>
#include <std_algorithms/Kokkos_PartitioningOperations.hpp>

namespace KE = Kokkos::Experimental;

namespace Test {
namespace stdalgos {

struct std_algorithms_partitioning_test : public std_algorithms_test {
  enum FixtureViews {
    Mixed,
    NegativeFirst,
    AllNegative,
    AllPositive,
    NegativeLast,
    SingleNegative,
    Count
  };

  void fillFixtureViews(FixtureViews caseNumber) {
    static_view_t tmpView("tmpView");
    auto tmp_view_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), tmpView);

    switch (caseNumber) {
      case FixtureViews::Mixed:
        tmp_view_h(0) = -1;
        tmp_view_h(1) = -2;
        tmp_view_h(2) = 3;
        tmp_view_h(3) = -4;
        tmp_view_h(4) = 5;
        tmp_view_h(5) = -6;
        tmp_view_h(6) = 7;
        tmp_view_h(7) = -8;
        tmp_view_h(8) = 9;
        tmp_view_h(9) = 10;
        break;

      case FixtureViews::NegativeFirst:
        tmp_view_h(0) = -2;
        tmp_view_h(1) = -4;
        tmp_view_h(2) = -6;
        tmp_view_h(3) = -80;
        tmp_view_h(4) = 5;
        tmp_view_h(5) = 7;
        tmp_view_h(6) = 115;
        tmp_view_h(7) = 3;
        tmp_view_h(8) = 9;
        tmp_view_h(9) = 11;
        break;

      case FixtureViews::AllNegative:
        tmp_view_h(0) = -2;
        tmp_view_h(1) = -4;
        tmp_view_h(2) = -6;
        tmp_view_h(3) = -8;
        tmp_view_h(4) = -4;
        tmp_view_h(5) = -12;
        tmp_view_h(6) = -14;
        tmp_view_h(7) = -2;
        tmp_view_h(8) = -6;
        tmp_view_h(9) = -8;
        break;

      case FixtureViews::AllPositive:
        tmp_view_h(0) = 11;
        tmp_view_h(1) = 3;
        tmp_view_h(2) = 17;
        tmp_view_h(3) = 9;
        tmp_view_h(4) = 3;
        tmp_view_h(5) = 11;
        tmp_view_h(6) = 13;
        tmp_view_h(7) = 1;
        tmp_view_h(8) = 9;
        tmp_view_h(9) = 43;
        break;

      case FixtureViews::NegativeLast:
        tmp_view_h(0) = 1;
        tmp_view_h(1) = 11;
        tmp_view_h(2) = 1;
        tmp_view_h(3) = 33;
        tmp_view_h(4) = 3;
        tmp_view_h(5) = 3;
        tmp_view_h(6) = -3;
        tmp_view_h(7) = -5;
        tmp_view_h(8) = -5;
        tmp_view_h(9) = -10;
        break;

      case FixtureViews::SingleNegative:
        tmp_view_h(0) = -200;
        tmp_view_h(1) = 1;
        tmp_view_h(2) = 1;
        tmp_view_h(3) = 3;
        tmp_view_h(4) = 3;
        tmp_view_h(5) = 211;
        tmp_view_h(6) = 3;
        tmp_view_h(7) = 5;
        tmp_view_h(8) = 5;
        tmp_view_h(9) = 11;
        break;

      default: break;
    }

    Kokkos::deep_copy(tmpView, tmp_view_h);
    copyInputViewToFixtureViews(tmpView);
  }

  bool goldSolutionIsPartitioned(FixtureViews caseNumber) const {
    switch (caseNumber) {
      case Mixed: return false;
      case NegativeFirst: return true;
      case AllNegative: return true;
      case AllPositive: return false;
      case NegativeLast: return false;
      case SingleNegative: return true;
      default: return false;
    }
  }

  int goldSolutionPartitionedPoint(FixtureViews caseNumber) const {
    switch (caseNumber) {
      case Mixed: return 2;
      case NegativeFirst: return 4;
      case AllNegative: return 10;
      case AllPositive: return 0;
      case NegativeLast: return 0;
      case SingleNegative: return 1;
      default: return -1;
    }
  }
};

TEST_F(std_algorithms_partitioning_test, is_partitioned_trivial) {
  IsNegativeFunctor<value_type> p;
  const auto result1 = KE::is_partitioned(exespace(), KE::cbegin(m_static_view),
                                          KE::cbegin(m_static_view), p);
  EXPECT_EQ(true, result1);

  const auto result2 = KE::is_partitioned(
      exespace(), KE::cbegin(m_dynamic_view), KE::cbegin(m_dynamic_view), p);
  EXPECT_EQ(true, result2);

  const auto result3 = KE::is_partitioned(
      exespace(), KE::cbegin(m_strided_view), KE::cbegin(m_strided_view), p);
  EXPECT_EQ(true, result3);
}

TEST_F(std_algorithms_partitioning_test, is_partitioned_accepting_iterators) {
  const IsNegativeFunctor<value_type> p;

  for (int id = 0; id < FixtureViews::Count; ++id) {
    fillFixtureViews(static_cast<FixtureViews>(id));
    const bool goldBool =
        goldSolutionIsPartitioned(static_cast<FixtureViews>(id));
    const auto result1 = KE::is_partitioned(
        exespace(), KE::cbegin(m_static_view), KE::cend(m_static_view), p);
    EXPECT_EQ(goldBool, result1);

    const auto result2 = KE::is_partitioned(
        exespace(), KE::cbegin(m_dynamic_view), KE::cend(m_dynamic_view), p);
    EXPECT_EQ(goldBool, result2);

    const auto result3 = KE::is_partitioned(
        exespace(), KE::cbegin(m_strided_view), KE::cend(m_strided_view), p);
    EXPECT_EQ(goldBool, result3);
  }
}

TEST_F(std_algorithms_partitioning_test, is_partitioned_accepting_view) {
  const IsNegativeFunctor<value_type> p;

  for (int id = 0; id < FixtureViews::Count; ++id) {
    fillFixtureViews(static_cast<FixtureViews>(id));
    const bool goldBool =
        goldSolutionIsPartitioned(static_cast<FixtureViews>(id));
    const auto result1 = KE::is_partitioned(exespace(), m_static_view, p);
    EXPECT_EQ(goldBool, result1);

    const auto result2 = KE::is_partitioned(exespace(), m_dynamic_view, p);
    EXPECT_EQ(goldBool, result2);

    const auto result3 = KE::is_partitioned(exespace(), m_strided_view, p);
    EXPECT_EQ(goldBool, result3);
  }
}

TEST_F(std_algorithms_partitioning_test, partition_point) {
  const IsNegativeFunctor<value_type> p;

  for (int id = 0; id < FixtureViews::Count; ++id) {
    fillFixtureViews(static_cast<FixtureViews>(id));
    const auto goldIndex =
        goldSolutionPartitionedPoint(static_cast<FixtureViews>(id));
    auto first1        = KE::cbegin(m_static_view);
    auto last1         = KE::cend(m_static_view);
    const auto result1 = KE::partition_point(exespace(), first1, last1, p);
    EXPECT_EQ(goldIndex, result1 - first1);

    auto first2        = KE::cbegin(m_dynamic_view);
    auto last2         = KE::cend(m_dynamic_view);
    const auto result2 = KE::partition_point(exespace(), first2, last2, p);
    EXPECT_EQ(goldIndex, result2 - first2);

    auto first3        = KE::cbegin(m_strided_view);
    auto last3         = KE::cend(m_strided_view);
    const auto result3 = KE::partition_point(exespace(), first3, last3, p);
    EXPECT_EQ(goldIndex, result3 - first3);
  }
}

}  // namespace stdalgos
}  // namespace Test
