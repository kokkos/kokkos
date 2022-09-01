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
#include <algorithm>

namespace Test {
namespace stdalgos {
namespace TeamPartitionPoint {

namespace KE = Kokkos::Experimental;

template <class ValueType>
struct UnifDist;

template <>
struct UnifDist<int> {
  using dist_type = std::uniform_int_distribution<int>;
  std::mt19937 m_gen;
  dist_type m_dist;

  UnifDist(int a, int b, std::size_t seedIn) : m_dist(a, b) {
    m_gen.seed(seedIn);
  }

  int operator()() { return m_dist(m_gen); }
};

template <class ValueType>
struct GreaterThanValueFunctor {
  ValueType m_val;

  KOKKOS_INLINE_FUNCTION
  GreaterThanValueFunctor(ValueType val) : m_val(val) {}

  KOKKOS_INLINE_FUNCTION
  bool operator()(ValueType val) const { return (val > m_val); }
};

template <class ViewType, class DistancesViewType, class ValueType>
struct TestFunctorA {
  ViewType m_view;
  DistancesViewType m_distancesView;
  ValueType m_threshold;
  int m_apiPick;

  TestFunctorA(const ViewType view,
               const DistancesViewType distancesView,
               ValueType threshold,
               int apiPick)
      : m_view(view),
        m_distancesView(distancesView),
        m_threshold(threshold),
        m_apiPick(apiPick) {}

  template <class MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();
    auto myRowView =
        Kokkos::subview(m_view, myRowIndex, Kokkos::ALL());

    GreaterThanValueFunctor predicate(m_threshold);
    if (m_apiPick == 0) {
      const auto it = KE::partition_point(
          member, KE::cbegin(myRowView), KE::cend(myRowView),
          predicate);

      Kokkos::single(Kokkos::PerTeam(member), [=]() {
        m_distancesView(myRowIndex) =
            KE::distance(KE::cbegin(myRowView), it);
      });
    }

    else if (m_apiPick == 1) {
      const auto it = KE::partition_point(member, myRowView, predicate);

      Kokkos::single(Kokkos::PerTeam(member), [=]() {
        m_distancesView(myRowIndex) =
            KE::distance(KE::begin(myRowView), it);
      });
    }
  }
};

template <class LayoutTag, class ValueType>
void test_A(std::size_t numTeams, std::size_t numCols, int apiId,
            const std::string& sIn) {
  /* description:
     use a rank-2 view randomly filled with values in a range (a,b)
     and run a team-level (one team per row) partition_point with
     predicate = IsGreaterThanValue
     where threshold is set to a number larger than b above
   */
  const auto threshold           = static_cast<ValueType>(1103);
  const auto valueForSureGreater = static_cast<ValueType>(2103);
  const auto valueForSureSmaller = static_cast<ValueType>(111);

  // -----------------------------------------------
  // prepare data
  // -----------------------------------------------
  // construct in memory space associated with default exespace
  auto dataView =
      create_view<ValueType>(LayoutTag{}, numTeams, numCols, "dataView");

  // dataView might not deep copyable (e.g. strided layout) so to
  // randomize it, we make a new view that is for sure deep copyable,
  // modify it on the host, deep copy to device and then launch
  // a kernel to copy to dataView
  auto dataView_dc =
      create_deep_copyable_compatible_view_with_same_extent(dataView);
  auto dataView_dc_h = create_mirror_view(Kokkos::HostSpace(), dataView_dc);

  if (sIn == "trivialEmpty") {
    // do nothing
  }

  else if (sIn == "allTrue") {
    // randomly fill with values greater than threshold
    // so that all elements in each row satisfy the predicate
    // so this counts as being partitioned
    Kokkos::Random_XorShift64_Pool<Kokkos::DefaultHostExecutionSpace> pool(
        452377);
    Kokkos::fill_random(dataView_dc_h, pool, ValueType(2001),
                        ValueType(2501));
  }

  else if (sIn == "allFalse") {
    // randomly fill the view with values smaller than threshold
    // and even in this case each row counts as partitioned
    Kokkos::Random_XorShift64_Pool<Kokkos::DefaultHostExecutionSpace> pool(
        452377);
    Kokkos::fill_random(dataView_dc_h, pool, ValueType(0), ValueType(101));
  }

  else if (sIn == "random") {
    // randomly select a location and make all values before that
    // larger than threshol and all values after to be smaller than threshold
    // so that this picked location does partition the range
    UnifDist<int> indexProducer(0, numCols - 1, 3432779);
    for (std::size_t i = 0; i < dataView_dc_h.extent(0); ++i) {
      const std::size_t a = indexProducer();
      for (std::size_t j = 0; j < a; ++j) {
        dataView_dc_h(i, j) = valueForSureGreater;
      }
      for (std::size_t j = a; j < numCols; ++j) {
        dataView_dc_h(i, j) = valueForSureSmaller;
      }
    }
  }

  // copy to dataView_dc and then to dataView
  Kokkos::deep_copy(dataView_dc, dataView_dc_h);
  // use CTAD
  CopyFunctorRank2 F1(dataView_dc, dataView);
  Kokkos::parallel_for("copy", dataView.extent(0) * dataView.extent(1), F1);

  // -----------------------------------------------
  // launch kokkos kernel
  // -----------------------------------------------
  using space_t = Kokkos::DefaultExecutionSpace;
  Kokkos::TeamPolicy<space_t> policy(numTeams, Kokkos::AUTO());

  // to verify that things work, each team stores the result
  // and then we check that these match what we expect
  Kokkos::View<std::size_t*> distancesView("distances", numTeams);

  // use CTAD for functor
  TestFunctorA fnc(dataView, distancesView, threshold, apiId);
  Kokkos::parallel_for(policy, fnc);

  // -----------------------------------------------
  // check
  // -----------------------------------------------
  auto distancesView_h = create_host_space_copy(distancesView);
  auto dataViewAfterOp_h = create_host_space_copy(dataView);
  GreaterThanValueFunctor predicate(threshold);

  for (std::size_t i = 0; i < dataView_dc_h.extent(0); ++i) {
    auto myRow = Kokkos::subview(dataView_dc_h, i, Kokkos::ALL());
    const auto stdResult = std::partition_point(
        KE::cbegin(myRow), KE::cend(myRow), predicate);

    // our result must match std
    const std::size_t stdDistance = KE::distance(KE::cbegin(myRow), stdResult);
    EXPECT_EQ(stdDistance, distancesView_h(i));
  }

  expect_equal_host_views(dataView_dc_h, dataViewAfterOp_h);
}

template <class LayoutTag, class ValueType>
void run_all_scenarios(const std::string& name, const std::vector<int>& cols) {
  for (int numTeams : teamSizesToTest) {
    for (const auto& numCols : cols) {
      for (int apiId : {0, 1}) {
        test_A<LayoutTag, ValueType>(numTeams, numCols, apiId, name);
      }
    }
  }
}

TEST(std_algorithms_partition_point_team_test, empty) {
  const std::string name      = "trivialEmpty";
  const std::vector<int> cols = {0};
  run_all_scenarios<DynamicTag, double>(name, cols);
  run_all_scenarios<StridedTwoRowsTag, double>(name, cols);
  run_all_scenarios<StridedThreeRowsTag, int>(name, cols);
}

TEST(std_algorithms_partition_point_team_test, all_true) {
  const std::string name      = "allTrue";
  const std::vector<int> cols = {13, 101, 1444, 5153};
  run_all_scenarios<DynamicTag, double>(name, cols);
  run_all_scenarios<StridedTwoRowsTag, double>(name, cols);
  run_all_scenarios<StridedThreeRowsTag, int>(name, cols);
}

TEST(std_algorithms_partition_point_team_test, all_false) {
  const std::string name      = "allFalse";
  const std::vector<int> cols = {13, 101, 1444, 5153};
  run_all_scenarios<DynamicTag, double>(name, cols);
  run_all_scenarios<StridedTwoRowsTag, double>(name, cols);
  run_all_scenarios<StridedThreeRowsTag, int>(name, cols);
}

TEST(std_algorithms_partition_point_team_test, random) {
  const std::string name      = "random";
  const std::vector<int> cols = {13, 101, 1444, 5153};
  run_all_scenarios<DynamicTag, double>(name, cols);
  run_all_scenarios<StridedTwoRowsTag, double>(name, cols);
  run_all_scenarios<StridedThreeRowsTag, int>(name, cols);
}

}  // namespace TeamPartitionPoint
}  // namespace stdalgos
}  // namespace Test
