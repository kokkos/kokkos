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
namespace TeamTransformExclusiveScan {

namespace KE = Kokkos::Experimental;

template <class ValueType>
struct PlusFunctor {
  KOKKOS_INLINE_FUNCTION
  ValueType operator()(const ValueType& lhs, const ValueType& rhs) const {
    return lhs + rhs;
  }
};

template <class ValueType>
struct MultipliesByTwoFunctor {
  KOKKOS_INLINE_FUNCTION
  ValueType operator()(const ValueType& value) const { return value * 2; }
};

template <class SourceViewType, class DestViewType, class DistancesViewType,
          class InitValuesViewType, class BinaryOpType, class UnaryOpType>
struct TestFunctorA {
  SourceViewType m_sourceView;
  DestViewType m_destView;
  DistancesViewType m_distancesView;
  InitValuesViewType m_initValuesView;
  BinaryOpType m_binaryOp;
  UnaryOpType m_unaryOp;
  int m_apiPick;

  TestFunctorA(const SourceViewType sourceView, const DestViewType destView,
               const DistancesViewType distancesView,
               const InitValuesViewType initValuesView, BinaryOpType binaryOp,
               UnaryOpType unaryOp, int apiPick)
      : m_sourceView(sourceView),
        m_destView(destView),
        m_distancesView(distancesView),
        m_initValuesView(initValuesView),
        m_binaryOp(binaryOp),
        m_unaryOp(unaryOp),
        m_apiPick(apiPick) {}

  template <class MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto rowIndex = member.league_rank();

    auto rowViewSrc  = Kokkos::subview(m_sourceView, rowIndex, Kokkos::ALL());
    auto rowViewDest = Kokkos::subview(m_destView, rowIndex, Kokkos::ALL());

    const auto initVal = m_initValuesView(rowIndex);

    switch (m_apiPick) {
      case 0: {
        auto it = KE::transform_exclusive_scan(
            member, KE::cbegin(rowViewSrc), KE::cend(rowViewSrc),
            KE::begin(rowViewDest), initVal, m_binaryOp, m_unaryOp);
        Kokkos::single(Kokkos::PerTeam(member), [=] {
          m_distancesView(rowIndex) = KE::distance(KE::begin(rowViewDest), it);
        });

        break;
      }

      case 1: {
        auto it = KE::transform_exclusive_scan(member, rowViewSrc, rowViewDest,
                                               initVal, m_binaryOp, m_unaryOp);
        Kokkos::single(Kokkos::PerTeam(member), [=] {
          m_distancesView(rowIndex) = KE::distance(KE::begin(rowViewDest), it);
        });

        break;
      }
    }
  }
};

template <class LayoutTag, class ValueType>
void test_A(std::size_t numTeams, std::size_t numCols, int apiId) {
  /* description:
     use a rank-2 view randomly filled with values,
     and run a team-level transform_exclusive_scan
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

  auto [sourceView, sourceViewBeforeOp_h] = create_random_view_and_host_clone(
      LayoutTag{}, numTeams, numCols, bounds, "sourceView");

  // -----------------------------------------------
  // launch kokkos kernel
  // -----------------------------------------------
  using space_t = Kokkos::DefaultExecutionSpace;
  Kokkos::TeamPolicy<space_t> policy(numTeams, Kokkos::AUTO());

  // create the destination view
  Kokkos::View<ValueType**> destView("destView", numTeams, numCols);

  // tranform_exclusive_scan returns an iterator so to verify that it is correct
  // each team stores the distance of the returned iterator from the beginning
  // of the interval that team operates on and then we check that these
  // distances match the std result
  Kokkos::View<std::size_t*> distancesView("distancesView", numTeams);

  PlusFunctor<ValueType> binaryOp;
  MultipliesByTwoFunctor<ValueType> unaryOp;

  // Create view of reduce init values to be used by test cases
  Kokkos::View<ValueType*, Kokkos::DefaultHostExecutionSpace> initValuesView_h(
      "initValuesView_h", numTeams);
  using rand_pool =
      Kokkos::Random_XorShift64_Pool<Kokkos::DefaultHostExecutionSpace>;
  rand_pool pool(lowerBound * upperBound);
  Kokkos::fill_random(initValuesView_h, pool, lowerBound, upperBound);

  // use CTAD for functor
  auto initValuesView =
      Kokkos::create_mirror_view_and_copy(space_t(), initValuesView_h);
  TestFunctorA fnc(sourceView, destView, distancesView, initValuesView,
                   binaryOp, unaryOp, apiId);
  Kokkos::parallel_for(policy, fnc);

  // -----------------------------------------------
  // run cpp-std kernel and check
  // -----------------------------------------------
  auto distancesView_h = create_host_space_copy(distancesView);
  Kokkos::View<ValueType**, Kokkos::HostSpace> stdDestView("stdDestView",
                                                           numTeams, numCols);

  for (std::size_t i = 0; i < sourceView.extent(0); ++i) {
    auto rowFrom = Kokkos::subview(sourceViewBeforeOp_h, i, Kokkos::ALL());
    auto rowDest = Kokkos::subview(stdDestView, i, Kokkos::ALL());

    const auto initValue = initValuesView_h(i);

#if defined(__GNUC__) && __GNUC__ == 8
#define transform_exclusive_scan testing_transform_exclusive_scan
#else
#define transform_exclusive_scan std::transform_exclusive_scan
#endif

    switch (apiId) {
      case 0:
      case 1: {
        auto it = transform_exclusive_scan(
            KE::cbegin(rowFrom), KE::cend(rowFrom), KE::begin(rowDest),
            initValue, binaryOp, unaryOp);
        const std::size_t stdDistance = KE::distance(KE::begin(rowDest), it);
        EXPECT_EQ(stdDistance, distancesView_h(i));
        break;
      }
    }

#undef transform_exclusive_scan
  }

  auto dataViewAfterOp_h = create_host_space_copy(destView);
  expect_equal_host_views(stdDestView, dataViewAfterOp_h);
}

template <class LayoutTag, class ValueType>
void run_all_scenarios() {
  for (int numTeams : teamSizesToTest) {
    for (const auto& numCols : {0, 1, 2, 13, 101, 1444, 8153}) {
      for (int apiId : {0, 1}) {
        test_A<LayoutTag, ValueType>(numTeams, numCols, apiId);
      }
    }
  }
}

TEST(std_algorithms_transform_exclusive_scan_team_test, test) {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, unsigned>();
}

}  // namespace TeamTransformExclusiveScan
}  // namespace stdalgos
}  // namespace Test

#endif
