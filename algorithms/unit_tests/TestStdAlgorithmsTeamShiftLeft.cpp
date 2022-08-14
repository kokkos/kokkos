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
namespace TeamShiftLeft {

namespace KE = Kokkos::Experimental;

template <class ViewType, class DistancesViewType>
struct TestFunctorA {
  ViewType m_view;
  DistancesViewType m_distancesView;
  std::size_t m_shift;
  int m_apiPick;

  TestFunctorA(const ViewType view,
               const DistancesViewType distancesView,
	       std::size_t shift,
               int apiPick)
      : m_view(view),
        m_distancesView(distancesView),
        m_shift(shift),
        m_apiPick(apiPick) {}

  template <class MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const MemberType& member) const
  {
    const auto myRowIndex = member.league_rank();
    auto myRowView = Kokkos::subview(m_view, myRowIndex, Kokkos::ALL());

    if (m_apiPick == 0)
    {
      auto it = KE::shift_left(member,
			       KE::begin(myRowView),
			       KE::end(myRowView),
			       m_shift);

      Kokkos::single(Kokkos::PerTeam(member), [=]() {
        m_distancesView(myRowIndex) = KE::distance(KE::begin(myRowView), it);
      });
    }
    else if (m_apiPick == 1)
    {
      auto it = KE::shift_left(member, myRowView, m_shift);

      Kokkos::single(Kokkos::PerTeam(member), [=]() {
        m_distancesView(myRowIndex) = KE::distance(KE::begin(myRowView), it);
      });
    }
  }
};

// shift_left is only supported starting from C++20
template <class ForwardIterator>
ForwardIterator my_std_shift_left(ForwardIterator first,
				  ForwardIterator last,
				  typename std::iterator_traits<ForwardIterator>::difference_type n)
{
  // copied from
  // https://github.com/llvm/llvm-project/blob/main/libcxx/include/__algorithm/shift_left.h

  if (n == 0) {
    return last;
  }

  ForwardIterator m = first;
  for (; n > 0; --n) {
    if (m == last) {
      return first;
    }
    ++m;
  }
  return std::move(m, last, first);
}

template <class LayoutTag, class ValueType>
void test_A(std::size_t numTeams, std::size_t numCols,
	    std::size_t shift,
            int apiId)
{
  /* description:
     randomly fill a source view and copy a copyCount set of values
     for each row into a destination view. The operation is done via
     a team parfor with one row per team.
   */

  // -----------------------------------------------
  // prepare data
  // -----------------------------------------------
  // create a view in the memory space associated with default exespace
  // with as many rows as the number of teams and fill it with random
  // values from an arbitrary range
  auto [dataView, dataViewBeforeOp_h] = create_view_and_fill_randomly(
      LayoutTag{}, numTeams, numCols,
      Kokkos::pair{ValueType(11), ValueType(523)}, "dataView");

  // -----------------------------------------------
  // launch kokkos kernel
  // -----------------------------------------------
  using space_t = Kokkos::DefaultExecutionSpace;
  Kokkos::TeamPolicy<space_t> policy(numTeams, Kokkos::AUTO());

  // copy_n returns an iterator so to verify that it is correct
  // each team stores the distance of the returned iterator from the
  // beginning of the interval that team operates on and then we check
  // that these distances match the expectation
  Kokkos::View<std::size_t*> distancesView("distancesView", numTeams);

  // use CTAD for functor
  TestFunctorA fnc(dataView, distancesView, shift, apiId);
  Kokkos::parallel_for(policy, fnc);

  // -----------------------------------------------
  // run std algo
  // -----------------------------------------------
  // here I can use dataViewBeforeOp_h to run std algo on
  // since that contains a valid copy of the data
  auto distancesView_h   = create_host_space_copy(distancesView);
  for (std::size_t i = 0; i < dataViewBeforeOp_h.extent(0); ++i){
    auto myRow = Kokkos::subview(dataViewBeforeOp_h, i, Kokkos::ALL());
    auto it = my_std_shift_left(KE::begin(myRow), KE::end(myRow), shift);
    const std::size_t stdDistance = KE::distance(KE::begin(myRow), it);
    EXPECT_EQ(stdDistance, distancesView_h(i));
  }

  auto dataViewAfterOp_h = create_host_space_copy(dataView);
  expect_equal_host_views(dataViewBeforeOp_h, dataViewAfterOp_h);
}

template <class Tag, class ValueType>
void run_all_scenarios() {
  // prepare a map where, for a given set of num cols
  // we provide a list of shifts to use for testing
  // key = num of columns
  // value = list of shifts
  // Note that the cornerCase number is here since the shiftLeft algo
  // should work even when the shift given is way larger than the range.
  constexpr std::size_t cornerCase = 100000;
  const std::map<int, std::vector<std::size_t>> scenarios = {
    {0, {0, cornerCase}},
      {2, {0, 1, 2, cornerCase}},
      {6, {0, 1, 2, 5, cornerCase}},
      {13, {0, 1, 2, 8, 11, cornerCase}},
      {56, {0, 1, 2, 8, 11, 33, 56, cornerCase}},
      {123, {0, 1, 11, 33, 56, 89, 112, cornerCase}},
      {3145, {0, 1, 11, 33, 56, 89, 112, 5677, cornerCase}}};

  for (int num_teams : teamSizesToTest) {
    for (const auto& scenario : scenarios) {
      const std::size_t numCols = scenario.first;
      for (int copyCount : scenario.second) {
        for (int apiId : {0, 1}) {
          test_A<Tag, ValueType>(num_teams, numCols, copyCount, apiId);
        }
      }
    }
  }
}

TEST(std_algorithms_shift_left_team_test, test) {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, unsigned>();
}

}  // namespace TeamCopy_n
}  // namespace stdalgos
}  // namespace Test
