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

namespace Test {
namespace stdalgos {
namespace TeamTransformReduce {

namespace KE = Kokkos::Experimental;

template <class ValueType>
struct PlusFunctor {
  KOKKOS_INLINE_FUNCTION
  ValueType operator()(const ValueType& lhs, const ValueType& rhs) const {
    return lhs + rhs;
  }
};

template <class ValueType>
struct MultipliesFunctor {
  KOKKOS_INLINE_FUNCTION
  ValueType operator()(const ValueType& lhs, const ValueType& rhs) const {
    return lhs * rhs;
  }
};

template <class ValueType>
struct PlusOneFunctor {
  KOKKOS_INLINE_FUNCTION
  ValueType operator()(const ValueType& val) const { return val + 1; };
};

template <class FirstDataViewType, class SecondDataViewType,
          class InitValuesViewType, class ResultsViewType,
          class BinaryJoinerType, class BinaryTransformType,
          class UnaryTransformType>
struct TestFunctorA {
  FirstDataViewType m_firstDataView;
  SecondDataViewType m_secondDataView;
  InitValuesViewType m_initValuesView;
  ResultsViewType m_resultsView;
  BinaryJoinerType m_binaryJoiner;
  BinaryTransformType m_binaryTransform;
  UnaryTransformType m_unaryTransform;
  int m_apiPick;

  TestFunctorA(const FirstDataViewType firstDataView,
               const SecondDataViewType secondDataview,
               const InitValuesViewType initValuesView,
               const ResultsViewType resultsView, BinaryJoinerType binaryJoiner,
               BinaryTransformType binaryTransform,
               UnaryTransformType unaryTransform, int apiPick)
      : m_firstDataView(firstDataView),
        m_secondDataView(secondDataview),
        m_initValuesView(initValuesView),
        m_resultsView(resultsView),
        m_binaryJoiner(binaryJoiner),
        m_binaryTransform(binaryTransform),
        m_unaryTransform(unaryTransform),
        m_apiPick(apiPick) {}

  template <class MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const int rowIndex = member.league_rank();

    auto firstDataRow =
        Kokkos::subview(m_firstDataView, rowIndex, Kokkos::ALL());
    auto firstDataRowBegin = KE::cbegin(firstDataRow);
    auto firstDataRowEnd   = KE::cend(firstDataRow);

    auto secondDataRow =
        Kokkos::subview(m_secondDataView, rowIndex, Kokkos::ALL());
    auto secondDataRowBegin = KE::cbegin(secondDataRow);

    const auto initVal = m_initValuesView(rowIndex);

    switch (m_apiPick) {
      case 0: {
        const auto result =
            KE::transform_reduce(member, firstDataRowBegin, firstDataRowEnd,
                                 secondDataRowBegin, initVal);
        Kokkos::single(Kokkos::PerTeam(member),
                       [=]() { m_resultsView(rowIndex) = result; });
        break;
      }

      case 1: {
        const auto result =
            KE::transform_reduce(member, firstDataRow, secondDataRow, initVal);
        Kokkos::single(Kokkos::PerTeam(member),
                       [=]() { m_resultsView(rowIndex) = result; });
        break;
      }

      case 2: {
        const auto result = KE::transform_reduce(
            member, firstDataRowBegin, firstDataRowEnd, secondDataRowBegin,
            initVal, m_binaryJoiner, m_binaryTransform);
        Kokkos::single(Kokkos::PerTeam(member),
                       [=]() { m_resultsView(rowIndex) = result; });
        break;
      }

      case 3: {
        const auto result =
            KE::transform_reduce(member, firstDataRow, secondDataRow, initVal,
                                 m_binaryJoiner, m_binaryTransform);
        Kokkos::single(Kokkos::PerTeam(member),
                       [=]() { m_resultsView(rowIndex) = result; });
        break;
      }

      case 4: {
        const auto result =
            KE::transform_reduce(member, firstDataRowBegin, firstDataRowEnd,
                                 initVal, m_binaryJoiner, m_unaryTransform);
        Kokkos::single(Kokkos::PerTeam(member),
                       [=]() { m_resultsView(rowIndex) = result; });
        break;
      }

      case 5: {
        const auto result = KE::transform_reduce(
            member, firstDataRow, initVal, m_binaryJoiner, m_unaryTransform);
        Kokkos::single(Kokkos::PerTeam(member),
                       [=]() { m_resultsView(rowIndex) = result; });
        break;
      }
    }
  }
};

template <class LayoutTag, class ValueType>
void test_A(std::size_t numTeams, std::size_t numCols, int apiId) {
  /* description:
     use a rank-2 view randomly filled with values,
     and run a team-level transform_reduce
   */

  // -----------------------------------------------
  // prepare data
  // -----------------------------------------------
  // create a view in the memory space associated with default exespace
  // with as many rows as the number of teams and fill it with random
  // values from an arbitrary range.
  constexpr auto lowerBound = ValueType{5};
  constexpr auto upperBound = ValueType{523};
  Kokkos::pair bounds{lowerBound, upperBound};
  auto [firstDataView, firstDataViewBeforeOp_h] =
      create_random_view_and_host_clone(LayoutTag{}, numTeams, numCols, bounds,
                                        "firstDataView");
  auto [secondDataView, secondDataViewBeforeOp_h] =
      create_random_view_and_host_clone(LayoutTag{}, numTeams, numCols, bounds,
                                        "secondDataView");

  // Create view of init values to be used by test cases
  Kokkos::View<ValueType*, Kokkos::DefaultHostExecutionSpace> initValuesView_h(
      "initValuesView_h", numTeams);
  using rand_pool =
      Kokkos::Random_XorShift64_Pool<Kokkos::DefaultHostExecutionSpace>;
  rand_pool pool(lowerBound * upperBound);
  Kokkos::fill_random(initValuesView_h, pool, lowerBound, upperBound);

  // -----------------------------------------------
  // launch kokkos kernel
  // -----------------------------------------------
  using space_t = Kokkos::DefaultExecutionSpace;
  Kokkos::TeamPolicy<space_t> policy(numTeams, Kokkos::AUTO());

  // to verify that things work, each team stores the result of its
  // transform_reduce call, and then we check that these match what we expect
  Kokkos::View<ValueType*> resultsView("resultsView", numTeams);

  PlusFunctor<ValueType> binaryJoiner;
  MultipliesFunctor<ValueType> binaryTransform;
  PlusOneFunctor<ValueType> unaryTransform;

  // use CTAD for functor
  auto initValuesView =
      Kokkos::create_mirror_view_and_copy(space_t(), initValuesView_h);
  TestFunctorA fnc(firstDataView, secondDataView, initValuesView, resultsView,
                   binaryJoiner, binaryTransform, unaryTransform, apiId);
  Kokkos::parallel_for(policy, fnc);

  // -----------------------------------------------
  // run cpp-std kernel and check
  // -----------------------------------------------

  auto resultsView_h = create_host_space_copy(resultsView);

  for (std::size_t i = 0; i < firstDataView.extent(0); ++i) {
    auto firstDataRow =
        Kokkos::subview(firstDataViewBeforeOp_h, i, Kokkos::ALL());

    const auto firstDataRowBegin = KE::cbegin(firstDataRow);
    const auto firstDataRowEnd   = KE::cend(firstDataRow);

    auto secondDataRow =
        Kokkos::subview(secondDataViewBeforeOp_h, i, Kokkos::ALL());

    const auto secondDataRowBegin = KE::cbegin(secondDataRow);

    const auto initVal = initValuesView_h(i);

    switch (apiId) {
      case 0:
      case 1: {
        const auto result_h = resultsView_h(i);
        const auto result   = std::transform_reduce(
            firstDataRowBegin, firstDataRowEnd, secondDataRowBegin, initVal);

        if constexpr (std::is_floating_point_v<ValueType>) {
          EXPECT_FLOAT_EQ(result, resultsView_h(i));
        } else {
          EXPECT_EQ(result, resultsView_h(i));
        }

        break;
      }

      case 2:
      case 3: {
        const ValueType result = std::transform_reduce(
            firstDataRowBegin, firstDataRowEnd, secondDataRowBegin, initVal,
            binaryJoiner, binaryTransform);

        if constexpr (std::is_floating_point_v<ValueType>) {
          EXPECT_FLOAT_EQ(result, resultsView_h(i));
        } else {
          EXPECT_EQ(result, resultsView_h(i));
        }

        break;
      }

      case 4:
      case 5: {
        const ValueType result =
            std::transform_reduce(firstDataRowBegin, firstDataRowEnd, initVal,
                                  binaryJoiner, unaryTransform);

        if constexpr (std::is_floating_point_v<ValueType>) {
          EXPECT_FLOAT_EQ(result, resultsView_h(i));
        } else {
          EXPECT_EQ(result, resultsView_h(i));
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

TEST(std_algorithms_transform_reduce_team_test, test) {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, unsigned>();
}

}  // namespace TeamTransformReduce
}  // namespace stdalgos
}  // namespace Test
