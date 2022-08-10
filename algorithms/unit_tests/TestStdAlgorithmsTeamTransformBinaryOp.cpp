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
namespace TeamTransformBinaryOp {

namespace KE = Kokkos::Experimental;

template <class ValueType>
struct AddValuesBinaryOp {
  KOKKOS_INLINE_FUNCTION
  ValueType operator()(const ValueType& a, const ValueType& b) const {
    return a + b;
  }
};

template <class SourceView1Type, class SourceView2Type, class DestViewType,
          class DistancesViewType>
struct TestFunctorA {
  SourceView1Type m_sourceView1;
  SourceView2Type m_sourceView2;
  DestViewType m_destView;
  DistancesViewType m_distancesView;
  int m_apiPick;

  TestFunctorA(const SourceView1Type fromView1, const SourceView2Type fromView2,
               const DestViewType destView,
               const DistancesViewType distancesView, int apiPick)
      : m_sourceView1(fromView1),
        m_sourceView2(fromView2),
        m_destView(destView),
        m_distancesView(distancesView),
        m_apiPick(apiPick) {}

  template <class MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();
    auto myRowView1From =
        Kokkos::subview(m_sourceView1, myRowIndex, Kokkos::ALL());
    auto myRowView2From =
        Kokkos::subview(m_sourceView2, myRowIndex, Kokkos::ALL());
    auto myRowViewDest = Kokkos::subview(m_destView, myRowIndex, Kokkos::ALL());

    using value_type = typename SourceView1Type::value_type;
    if (m_apiPick == 0) {
      auto it = KE::transform(
          member, KE::cbegin(myRowView1From), KE::cend(myRowView1From),
          KE::cbegin(myRowView2From), KE::begin(myRowViewDest),
          AddValuesBinaryOp<value_type>());

      Kokkos::single(Kokkos::PerTeam(member), [=]() {
        m_distancesView(myRowIndex) =
            KE::distance(KE::begin(myRowViewDest), it);
      });
    } else if (m_apiPick == 1) {
      auto it = KE::transform(member, myRowView1From, myRowView2From,
                              myRowViewDest, AddValuesBinaryOp<value_type>());

      Kokkos::single(Kokkos::PerTeam(member), [=]() {
        m_distancesView(myRowIndex) =
            KE::distance(KE::begin(myRowViewDest), it);
      });
    }
  }
};

// template <class ViewType>
// auto fill_view_randomly(ViewType view) {
//   // view might not deep copyable (e.g. strided layout) so to fill it
//   // we make a new view that is for sure deep copyable, modify it on the host
//   // deep copy to device and then launch copy kernel to view
//   auto view_dc   =
//   create_deep_copyable_compatible_view_with_same_extent(view); auto view_dc_h
//   = create_mirror_view(Kokkos::HostSpace(), view_dc);

//   // randomly fill the view
//   Kokkos::Random_XorShift64_Pool<Kokkos::DefaultHostExecutionSpace>
//   pool(12371); Kokkos::fill_random(view_dc_h, pool, 0, 523);

//   // copy to view_dc and then to view
//   Kokkos::deep_copy(view_dc, view_dc_h);
//   // use CTAD
//   CopyFunctorRank2 F1(view_dc, view);
//   Kokkos::parallel_for("copy", view.extent(0) * view.extent(1), F1);
//   return view_dc_h;
// }

template <class LayoutTag, class ValueType>
void test_A(std::size_t numTeams, std::size_t numCols, int apiId) {
  /* description:
     team level transform with each team handling a row of
     two rank-2 source views and applying a binary op that
     add each pair of element from those two views
   */

  // -----------------------------------------------
  // prepare data
  // -----------------------------------------------
  // create a view in the memory space associated with default exespace
  // with as many rows as the number of teams and fill it with random
  // values from an arbitrary range
  auto [sourceView1, sourceView1BeforeOp_h] = create_view_and_fill_randomly(
      LayoutTag{}, numTeams, numCols, std::pair{ValueType(0), ValueType(523)},
      "sourceView1", 317539 /*random seed*/);
  auto [sourceView2, sourceView2BeforeOp_h] = create_view_and_fill_randomly(
      LayoutTag{}, numTeams, numCols, std::pair{ValueType(0), ValueType(523)},
      "sourceView2", 957313 /*random seed*/);

  // -----------------------------------------------
  // launch kokkos kernel
  // -----------------------------------------------
  using space_t = Kokkos::DefaultExecutionSpace;
  Kokkos::TeamPolicy<space_t> policy(numTeams, Kokkos::AUTO());
  // create the destination view
  Kokkos::View<ValueType**> destView("destView", numTeams, numCols);
  // make a host copy of the dest view that we can check below
  // to be all zeros
  auto destViewBeforeOp_h = create_host_space_copy(destView);

  // copy returns an iterator so to verify that it is correct
  // each team stores the distance of the returned iterator from the
  // beginning of the interval that team operates on and then we check
  // that these distances match the expectation
  Kokkos::View<std::size_t*> distancesView("distancesView", numTeams);

  // use CTAD for functor
  TestFunctorA fnc(sourceView1, sourceView2, destView, distancesView, apiId);
  Kokkos::parallel_for(policy, fnc);

  // -----------------------------------------------
  // check
  // -----------------------------------------------
  auto distancesView_h   = create_host_space_copy(distancesView);
  auto destViewAfterOp_h = create_host_space_copy(destView);
  for (std::size_t i = 0; i < destViewAfterOp_h.extent(0); ++i) {
    for (std::size_t j = 0; j < destViewAfterOp_h.extent(1); ++j) {
      // elements in dest view should be the sum of source elements
      EXPECT_EQ(destViewAfterOp_h(i, j),
                sourceView1BeforeOp_h(i, j) + sourceView2BeforeOp_h(i, j));
    }

    // each team should return an iterator whose distance from the
    // beginning of the row equals the num of columns since
    // each team transforms all elements in each row
    EXPECT_TRUE(distancesView_h(i) == numCols);
  }
}

template <class LayoutTag, class ValueType>
void run_all_scenarios() {
  for (int numTeams : teamSizesToTest) {
    for (const auto& numCols : {0, 1, 2, 13, 101, 1444, 11113}) {
      for (int apiId : {0, 1}) {
        test_A<LayoutTag, ValueType>(numTeams, numCols, apiId);
      }
    }
  }
}

TEST(std_algorithms_transform_team_test, test_binary_op) {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, unsigned>();
}

}  // namespace TeamTransformBinaryOp
}  // namespace stdalgos
}  // namespace Test
