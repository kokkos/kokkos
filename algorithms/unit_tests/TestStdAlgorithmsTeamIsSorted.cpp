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

namespace Test {
namespace stdalgos {
namespace TeamIsSorted {

namespace KE = Kokkos::Experimental;

template <class ValueType>
struct GreaterThanValueFunctor {
  ValueType m_val;

  KOKKOS_INLINE_FUNCTION
  GreaterThanValueFunctor(ValueType val) : m_val(val) {}

  KOKKOS_INLINE_FUNCTION
  bool operator()(ValueType val) const { return (val > m_val); }
};

template <class ViewType, class ReturnViewType>
struct TestFunctorA {
  ViewType m_view;
  ReturnViewType m_returnsView;
  int m_apiPick;

  TestFunctorA(const ViewType view,
	       const ReturnViewType returnsView,
               int apiPick)
      : m_view(view),
        m_returnsView(returnsView),
        m_apiPick(apiPick) {}

  template <class MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const MemberType& member) const
  {
    const auto myRowIndex = member.league_rank();
    auto myRowView        = Kokkos::subview(m_view, myRowIndex, Kokkos::ALL());

    //GreaterThanValueFunctor predicate(m_threshold);
    if (m_apiPick == 0) {
      const bool result = KE::is_sorted(member, KE::cbegin(myRowView), KE::cend(myRowView));

      Kokkos::single(Kokkos::PerTeam(member),
                     [=]() { m_returnsView(myRowIndex) = result; });

    }
    // else if (m_apiPick == 1) {
    //   auto myCount = KE::count_if(member, myRowView, predicate);
    //   Kokkos::single(Kokkos::PerTeam(member),
    //                  [=]() { m_returnsView(myRowIndex) = myCount; });
    // }
  }
};

template <class LayoutTag, class ValueType>
void test_A(std::size_t numTeams, std::size_t numCols,
	    int apiId,
	    bool makeDataSortedOnPurpose)
{
  /* description:
     use a rank-2 view randomly filled with values,
     and run a team-level is_sorted
   */

  // -----------------------------------------------
  // prepare data
  // -----------------------------------------------
  // construct in memory space associated with default exespace
  auto dataView = create_view<ValueType>(LayoutTag{}, numTeams, numCols, "dataView");

  // dataView might not deep copyable (e.g. strided layout) so to
  // randomize it, we make a new view that is for sure deep copyable,
  // modify it on the host, deep copy to device and then launch
  // a kernel to copy to dataView
  auto dataView_dc =
      create_deep_copyable_compatible_view_with_same_extent(dataView);
  auto dataView_dc_h = create_mirror_view(Kokkos::HostSpace(), dataView_dc);

  if (makeDataSortedOnPurpose){
    for (std::size_t i=0; i<dataView_dc_h.extent(0); ++i){
      for (std::size_t j=0; j<dataView_dc_h.extent(1); ++j){
	dataView_dc_h(i,j) = ValueType(j);
      }
    }
  }
  else{
    // randomly fill the view
    using rand_pool = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultHostExecutionSpace>;
    rand_pool pool(45234977);
    Kokkos::fill_random(dataView_dc_h, pool, ValueType{5}, ValueType{1545});
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
  Kokkos::View<bool*> returnView("returnView", numTeams);

  // use CTAD for functor
  TestFunctorA fnc(dataView, returnView, apiId);
  Kokkos::parallel_for(policy, fnc);

  // -----------------------------------------------
  // check
  // -----------------------------------------------
  auto returnView_h = create_host_space_copy(returnView);
  for (std::size_t i = 0; i < dataView_dc_h.extent(0); ++i)
  {
    auto myRow = Kokkos::subview(dataView_dc_h, i, Kokkos::ALL());
    if (apiId <= 1) {
      auto stdResult = std::is_sorted(KE::cbegin(myRow), KE::cend(myRow));
      EXPECT_TRUE(stdResult == returnView_h(i));

      // note that we have to be careful because when we have only
      // 0, 1 columns, then the data is sorted by definition
      // and when we have 2 columns it is very likely it is sorted
      // so only do the following check for large enough cols count
      if (numCols <= 1){
	EXPECT_TRUE(stdResult == true);
      }
      else if (numCols > 10){
	EXPECT_TRUE(stdResult == makeDataSortedOnPurpose);
      }
    }
  }

  // dataView should remain unchanged
  auto dataViewAfterOp_h = create_host_space_copy(dataView);
  expect_equal_host_views(dataView_dc_h, dataViewAfterOp_h);
}

template <class LayoutTag, class ValueType>
void run_all_scenarios(bool makeDataSortedOnPurpose)
{
  for (int numTeams : teamSizesToTest) {
    for (const auto& numCols : {0, 1, 2, 13, 101, 1444, 8153}) {
      for (int apiId : {0}) {
	test_A<LayoutTag, ValueType>(numTeams, numCols, apiId, makeDataSortedOnPurpose);
      }
    }
  }
}

TEST(std_algorithms_is_sorted_team_test, test_data_almost_certainly_not_sorted) {
  run_all_scenarios<DynamicTag, double>(false);
  run_all_scenarios<StridedTwoRowsTag, double>(false);
  run_all_scenarios<StridedThreeRowsTag, unsigned>(false);
}

TEST(std_algorithms_is_sorted_team_test, test_data_certainly_sorted) {
  run_all_scenarios<DynamicTag, double>(true);
  run_all_scenarios<StridedTwoRowsTag, double>(true);
  run_all_scenarios<StridedThreeRowsTag, unsigned>(true);
}

}  // namespace TeamIsSorted
}  // namespace stdalgos
}  // namespace Test
