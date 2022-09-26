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
// Questions? Contact Christian R. Trott crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <TestStdAlgorithmsCommon.hpp>

#if not defined KOKKOS_ENABLE_OPENMPTARGET

namespace Test {
namespace stdalgos {
namespace TeamReduce {

namespace KE = Kokkos::Experimental;

template <class ValueType>
struct PlusFunctor {
  KOKKOS_INLINE_FUNCTION
  ValueType operator()(const ValueType& lhs, const ValueType& rhs) const {
    return lhs + rhs;
  }
};

template <class DataViewType, class ReductionInitValuesViewType,
          class ReduceResultsViewType, class BinaryPredType>
struct TestFunctorA {
  DataViewType m_dataView;
  ReductionInitValuesViewType m_reductionInitValuesView;
  ReduceResultsViewType m_reduceResultsView;
  int m_apiPick;
  BinaryPredType m_binaryPred;

  TestFunctorA(const DataViewType dataView,
               const ReductionInitValuesViewType reductionInitValuesView,
               const ReduceResultsViewType reduceResultsView, int apiPick,
               BinaryPredType binaryPred)
      : m_dataView(dataView),
        m_reductionInitValuesView(reductionInitValuesView),
        m_reduceResultsView(reduceResultsView),
        m_apiPick(apiPick),
        m_binaryPred(binaryPred) {}

  template <class MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();

    auto myRowViewFrom = Kokkos::subview(m_dataView, myRowIndex, Kokkos::ALL());

    const auto rowFromBegin     = KE::cbegin(myRowViewFrom);
    const auto rowFromEnd       = KE::cend(myRowViewFrom);
    const auto initReductionVal = m_reductionInitValuesView(myRowIndex);

    switch (m_apiPick) {
      case 0: {
        const auto result = KE::reduce(member, rowFromBegin, rowFromEnd);
        Kokkos::single(Kokkos::PerTeam(member),
                       [=]() { m_reduceResultsView(myRowIndex) = result; });
        break;
      }

      case 1: {
        const auto result = KE::reduce(member, myRowViewFrom);
        Kokkos::single(Kokkos::PerTeam(member),
                       [=]() { m_reduceResultsView(myRowIndex) = result; });
        break;
      }

      case 2: {
        const auto result =
            KE::reduce(member, rowFromBegin, rowFromEnd, initReductionVal);
        Kokkos::single(Kokkos::PerTeam(member),
                       [=]() { m_reduceResultsView(myRowIndex) = result; });
        break;
      }

      case 3: {
        const auto result = KE::reduce(member, myRowViewFrom, initReductionVal);
        Kokkos::single(Kokkos::PerTeam(member),
                       [=]() { m_reduceResultsView(myRowIndex) = result; });
        break;
      }

      case 4: {
        const auto result = KE::reduce(member, rowFromBegin, rowFromEnd,
                                       initReductionVal, m_binaryPred);
        Kokkos::single(Kokkos::PerTeam(member),
                       [=]() { m_reduceResultsView(myRowIndex) = result; });
        break;
      }

      case 5: {
        const auto result =
            KE::reduce(member, myRowViewFrom, initReductionVal, m_binaryPred);
        Kokkos::single(Kokkos::PerTeam(member),
                       [=]() { m_reduceResultsView(myRowIndex) = result; });
        break;
      }
    }
  }
};

template <class LayoutTag, class ValueType>
void test_A(std::size_t numTeams, std::size_t numCols, int apiId) {
  /* description:
     use a rank-2 view randomly filled with values,
     and run a team-level reduce
   */

  // -----------------------------------------------
  // prepare data
  // -----------------------------------------------
  // create a view in the memory space associated with default exespace
  // with as many rows as the number of teams and fill it with random
  // values from an arbitrary range.
  constexpr ValueType lowerBound = 5;
  constexpr ValueType upperBound = 523;
  const auto bounds              = make_bounds(lowerBound, upperBound);

  auto [dataView, dataViewBeforeOp_h] = create_random_view_and_host_clone(
      LayoutTag{}, numTeams, numCols, bounds, "dataView");

  // Create view of reduce init values to be used by test cases
  Kokkos::View<ValueType*, Kokkos::DefaultHostExecutionSpace>
      reductionInitValuesView_h("reductionInitValuesView_h", numTeams);
  using rand_pool =
      Kokkos::Random_XorShift64_Pool<Kokkos::DefaultHostExecutionSpace>;
  rand_pool pool(lowerBound * upperBound);
  Kokkos::fill_random(reductionInitValuesView_h, pool, lowerBound, upperBound);

  // -----------------------------------------------
  // launch kokkos kernel
  // -----------------------------------------------
  using space_t = Kokkos::DefaultExecutionSpace;
  Kokkos::TeamPolicy<space_t> policy(numTeams, Kokkos::AUTO());

  // to verify that things work, each team stores the result of its reduce
  // call, and then we check that these match what we expect
  Kokkos::View<ValueType*> reduceResultsView("reduceResultsView", numTeams);

  PlusFunctor<ValueType> binaryPred;

  // use CTAD for functor
  auto reductionInitValuesView =
      Kokkos::create_mirror_view_and_copy(space_t(), reductionInitValuesView_h);
  TestFunctorA fnc(dataView, reductionInitValuesView, reduceResultsView, apiId,
                   binaryPred);
  Kokkos::parallel_for(policy, fnc);

  // -----------------------------------------------
  // run cpp-std kernel and check
  // -----------------------------------------------

  auto reduceResultsView_h = create_host_space_copy(reduceResultsView);

  for (std::size_t i = 0; i < dataView.extent(0); ++i) {
    auto rowFrom = Kokkos::subview(dataViewBeforeOp_h, i, Kokkos::ALL());

    const auto rowFromBegin = KE::cbegin(rowFrom);
    const auto rowFromEnd   = KE::cend(rowFrom);
    const auto initVal      = reductionInitValuesView_h(i);

    switch (apiId) {
      case 0:
      case 1: {
        const ValueType result = testing_reduce(rowFromBegin, rowFromEnd);
        if constexpr (std::is_floating_point_v<ValueType>) {
          EXPECT_FLOAT_EQ(result, reduceResultsView_h(i));
        } else {
          EXPECT_EQ(result, reduceResultsView_h(i));
        }

        break;
      }

      case 2:
      case 3: {
        const ValueType result =
            testing_reduce(rowFromBegin, rowFromEnd, initVal);
        if constexpr (std::is_floating_point_v<ValueType>) {
          EXPECT_FLOAT_EQ(result, reduceResultsView_h(i));
        } else {
          EXPECT_EQ(result, reduceResultsView_h(i));
        }

        break;
      }

      case 4:
      case 5: {
        const ValueType result =
            testing_reduce(rowFromBegin, rowFromEnd, initVal, binaryPred);
        if constexpr (std::is_floating_point_v<ValueType>) {
          EXPECT_FLOAT_EQ(result, reduceResultsView_h(i));
        } else {
          EXPECT_EQ(result, reduceResultsView_h(i));
        }

        break;
      }
    }
  }
}

template <class LayoutTag, class ValueType>
void run_all_scenarios() {
  for (int numTeams : teamSizesToTest) {
    for (const auto& numCols : {0, 1, 2, 13, 101, 1444, 8153}) {
      for (int apiId : {0, 1, 2, 3, 4, 5}) {
        test_A<LayoutTag, ValueType>(numTeams, numCols, apiId);
      }
    }
  }
}

TEST(std_algorithms_reduce_team_test, test) {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, unsigned>();
}

}  // namespace TeamReduce
}  // namespace stdalgos
}  // namespace Test

#endif
