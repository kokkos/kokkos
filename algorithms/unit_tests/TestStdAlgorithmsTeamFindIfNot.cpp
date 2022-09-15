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
namespace TeamFindIfNot {

namespace KE = Kokkos::Experimental;

template <class ValueType>
struct GreaterEqualFunctor {
  ValueType m_val;

  KOKKOS_INLINE_FUNCTION
  GreaterEqualFunctor(ValueType val) : m_val(val) {}

  KOKKOS_INLINE_FUNCTION
  bool operator()(ValueType val) const { return (val >= m_val); }
};

template <class DataViewType, class GreaterThanValuesViewType,
          class DistancesViewType>
struct TestFunctorA {
  DataViewType m_dataView;
  GreaterThanValuesViewType m_greaterThanValuesView;
  DistancesViewType m_distancesView;
  int m_apiPick;

  TestFunctorA(const DataViewType dataView,
               const GreaterThanValuesViewType greaterThanValuesView,
               DistancesViewType distancesView, int apiPick)
      : m_dataView(dataView),
        m_greaterThanValuesView(greaterThanValuesView),
        m_distancesView(distancesView),
        m_apiPick(apiPick) {}

  template <class MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();
    auto myRowViewFrom = Kokkos::subview(m_dataView, myRowIndex, Kokkos::ALL());
    const auto val     = m_greaterThanValuesView(myRowIndex);
    GreaterEqualFunctor unaryOp{val};

    switch (m_apiPick) {
      case 0: {
        auto it = KE::find_if_not(member, KE::cbegin(myRowViewFrom),
                                  KE::cend(myRowViewFrom), unaryOp);

        Kokkos::single(Kokkos::PerTeam(member), [=]() {
          m_distancesView(myRowIndex) =
              KE::distance(KE::cbegin(myRowViewFrom), it);
        });

        break;
      }

      case 1: {
        auto it = KE::find_if_not(member, myRowViewFrom, unaryOp);

        Kokkos::single(Kokkos::PerTeam(member), [=]() {
          m_distancesView(myRowIndex) =
              KE::distance(KE::begin(myRowViewFrom), it);
        });

        break;
      }
    }
  }
};

template <class LayoutTag, class ValueType>
void test_A(const bool predicatesReturnTrue, std::size_t numTeams,
            std::size_t numCols, int apiId) {
  /* description:
     use a rank-2 view randomly filled with values,
     and run a team-level find_if_not
   */

  // -----------------------------------------------
  // prepare data
  // -----------------------------------------------
  // create a view in the memory space associated with default exespace
  // with as many rows as the number of teams and fill it with random
  // values from an arbitrary range.
  const ValueType lowerBound{5}, upperBound{523};
  auto [dataView, dataViewBeforeOp_h] = create_random_view_and_host_clone(
      LayoutTag{}, numTeams, numCols, Kokkos::pair{lowerBound, upperBound},
      "dataView");

  // -----------------------------------------------
  // launch kokkos kernel
  // -----------------------------------------------
  using space_t = Kokkos::DefaultExecutionSpace;
  Kokkos::TeamPolicy<space_t> policy(numTeams, Kokkos::AUTO());

  // find_if_not returns an iterator so to verify that it is correct
  // each team stores the distance of the returned iterator from the
  // beginning of the interval that team operates on and then we check
  // that these distances match the std result
  Kokkos::View<std::size_t*> distancesView("distancesView", numTeams);

  // If predicatesReturnTrue == true, we want to ensure that for each dataView's
  // row find_if_not always returns end iterator. To do that,
  // GreaterEqualFunctor predicate created for each row must return true for
  // every value in that row, so it needs to compare each value with value
  // smaller than lowerBound.
  //
  // If searchedValuesExist == false we want to ensure the opposite -
  // GreaterEqualFunctor needs to return false for every value of each
  // dataView's row, so the predicate is constructed with value randomly picked
  // from range [upperBound, upperBound*2).
  Kokkos::View<ValueType*> greaterEqualValuesView("greaterEqualValuesView",
                                                  numTeams);
  auto greaterEqualValuesView_h =
      create_mirror_view(Kokkos::HostSpace(), greaterEqualValuesView);

  using rand_pool =
      Kokkos::Random_XorShift64_Pool<Kokkos::DefaultHostExecutionSpace>;
  rand_pool pool(lowerBound * upperBound);

  if (predicatesReturnTrue) {
    Kokkos::fill_random(greaterEqualValuesView_h, pool, 0, lowerBound);
  } else {
    Kokkos::fill_random(greaterEqualValuesView_h, pool, upperBound,
                        upperBound * 2);
  }

  Kokkos::deep_copy(greaterEqualValuesView, greaterEqualValuesView_h);

  // use CTAD for functor
  TestFunctorA fnc(dataView, greaterEqualValuesView, distancesView, apiId);
  Kokkos::parallel_for(policy, fnc);

  // -----------------------------------------------
  // run cpp-std kernel and check
  // -----------------------------------------------
  auto distancesView_h = create_host_space_copy(distancesView);
  for (std::size_t i = 0; i < dataView.extent(0); ++i) {
    auto rowFrom = Kokkos::subview(dataViewBeforeOp_h, i, Kokkos::ALL());
    const auto rowFromBegin = KE::begin(rowFrom);
    const auto rowFromEnd   = KE::end(rowFrom);
    const auto val          = greaterEqualValuesView_h(i);
    const GreaterEqualFunctor unaryOp{val};

    auto it = std::find_if_not(rowFromBegin, rowFromEnd, unaryOp);

    const std::size_t stdDistance      = KE::distance(rowFromBegin, it);
    const std::size_t beginEndDistance = KE::distance(rowFromBegin, rowFromEnd);
    const auto gotDist                 = distancesView_h(i);

    if (predicatesReturnTrue) {
      EXPECT_EQ(stdDistance, beginEndDistance);
    } else {
      EXPECT_LT(stdDistance, beginEndDistance);
    }

    EXPECT_EQ(stdDistance, distancesView_h(i));
  }
}

template <class LayoutTag, class ValueType>
void run_all_scenarios(const bool predicatesReturnTrue) {
  for (int numTeams : teamSizesToTest) {
    for (const auto& numCols : {1, 2, 13, 101, 1444, 8153}) {
      for (int apiId : {0, 1}) {
        test_A<LayoutTag, ValueType>(predicatesReturnTrue, numTeams, numCols,
                                     apiId);
      }
    }
  }
}

TEST(std_algorithms_find_if_not_team_test, predicates_return_true) {
  constexpr bool predicatesReturnTrue = true;

  run_all_scenarios<DynamicTag, double>(predicatesReturnTrue);
  run_all_scenarios<StridedTwoRowsTag, int>(predicatesReturnTrue);
  run_all_scenarios<StridedThreeRowsTag, unsigned>(predicatesReturnTrue);
}

TEST(std_algorithms_find_if_not_team_test, predicates_return_false) {
  constexpr bool predicatesReturnTrue = false;

  run_all_scenarios<DynamicTag, double>(predicatesReturnTrue);
  run_all_scenarios<StridedTwoRowsTag, int>(predicatesReturnTrue);
  run_all_scenarios<StridedThreeRowsTag, unsigned>(predicatesReturnTrue);
}

}  // namespace TeamFindIfNot
}  // namespace stdalgos
}  // namespace Test
