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
#include <Kokkos_Random.hpp>
#include <algorithm>

namespace Test {
namespace stdalgos {
namespace TeamRotateCopy {

namespace KE = Kokkos::Experimental;

// // impl is here for std because it is only avail from c++>=17
// template <class InputIterator, class OutputIterator, class BinaryPredicate>
// auto my_rotate(InputIterator first, InputIterator last,
//                     OutputIterator result, BinaryPredicate pred) {
//   if (first != last) {
//     typename OutputIterator::value_type t(*first);
//     *result = t;
//     ++result;
//     while (++first != last) {
//       if (!pred(t, *first)) {
//         t       = *first;
//         *result = t;
//         ++result;
//       }
//     }
//   }
//   return result;
// }

// template <class InputIterator, class OutputIterator>
// auto my_rotate(InputIterator first, InputIterator last,
//                     OutputIterator result) {
//   using value_type = typename OutputIterator::value_type;
//   using func_t     = IsEqualFunctor<value_type>;
//   return my_rotate(first, last, result, func_t());
// }

template <class ViewTypeFrom, class ViewTypeDest, class ViewItDist, class MemberType>
struct FunctorA {
  ViewTypeFrom m_view_from;
  ViewTypeDest m_view_dest;
  ViewItDist m_view_dist;
  int m_pivotShift;
  int m_api_pick;

  FunctorA(const ViewTypeFrom viewFrom,
	   const ViewTypeDest viewDest,
	   const ViewItDist view_dist,
	   int pivotShift,
	   int apiPick)
      : m_view_from(viewFrom),
	m_view_dest(viewDest),
	m_view_dist(view_dist),
	m_pivotShift(pivotShift),
        m_api_pick(apiPick) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const MemberType& member) const
  {
    const auto myRowIndex = member.league_rank();
    auto myRowViewFrom = Kokkos::subview(m_view_from, myRowIndex, Kokkos::ALL());
    auto myRowViewDest = Kokkos::subview(m_view_dest, myRowIndex, Kokkos::ALL());

    if (m_api_pick == 0) {
      auto pivot = KE::cbegin(myRowViewFrom) + m_pivotShift;
      auto it = KE::rotate_copy(member,
				KE::cbegin(myRowViewFrom),
				pivot,
				KE::cend(myRowViewFrom),
				KE::begin(myRowViewDest));
      m_view_dist(myRowIndex) = KE::distance(KE::begin(myRowViewDest), it);
    }
    else{
      auto it = KE::rotate_copy(member, myRowViewFrom, m_pivotShift, myRowViewDest);
      m_view_dist(myRowIndex) = KE::distance(KE::begin(myRowViewDest), it);
    }
  }
};

template <class Tag, class ValueType>
void test_A(std::size_t num_teams,
	    std::size_t num_cols,
	    std::size_t pivotShift,
	    int apiId)
{
  /* description: */

  //
  // fill v
  //
  auto v = create_view<ValueType>(Tag{}, num_teams, num_cols, "v");
  auto v_dc   = create_deep_copyable_compatible_view_with_same_extent(v);
  auto v_dc_h = create_mirror_view(Kokkos::HostSpace(), v_dc);
  Kokkos::Random_XorShift64_Pool<Kokkos::DefaultHostExecutionSpace> pool(12371);
  Kokkos::fill_random(v_dc_h, pool, 0, 523);
  // copy to v_dc and then to v
  Kokkos::deep_copy(v_dc, v_dc_h);
  CopyFunctorRank2<decltype(v_dc), decltype(v)> F1(v_dc, v);
  Kokkos::parallel_for("copy", v.extent(0) * v.extent(1), F1);

  //
  // make a copy of v on host and use it to run host algorithm
  //
  auto v2_h = create_host_space_copy(v);
  Kokkos::View<ValueType**, Kokkos::HostSpace> v_gold_h("vgold",
							v2_h.extent(0),
                                                        v2_h.extent(1));
  for (std::size_t i = 0; i < v_dc_h.extent(0); ++i)
  {
    auto rowFrom      = Kokkos::subview(v2_h, i, Kokkos::ALL());
    auto rowDest      = Kokkos::subview(v_gold_h, i, Kokkos::ALL());
    auto pivot = KE::cbegin(rowFrom) + pivotShift;
    auto it           = std::rotate_copy(KE::cbegin(rowFrom),
					 pivot, KE::cend(rowFrom),
					 KE::begin(rowDest));
    const int dist = KE::distance(KE::begin(rowDest), it);
    EXPECT_TRUE(dist == (int) (num_cols));
  }

  // launch kernel
  using space_t          = Kokkos::DefaultExecutionSpace;
  using policy_type      = Kokkos::TeamPolicy<space_t>;
  using team_member_type = typename policy_type::member_type;
  policy_type policy(num_teams, Kokkos::AUTO());

  auto v3 = create_view<ValueType>(Tag{}, num_teams, num_cols, "v3");
  auto distances = create_view<int>(DynamicTag{}, num_teams, "view_it_dist");
  using functor_type = FunctorA<decltype(v), decltype(v3), decltype(distances), team_member_type>;
  functor_type fnc(v, v3, distances, pivotShift, apiId);
  Kokkos::parallel_for(policy, fnc);

  // check
  auto distances_h = create_host_space_copy(distances);
  for (std::size_t i = 0; i < v.extent(0); ++i) {
    EXPECT_TRUE(distances_h(i) == (int) (num_cols));
  }

  auto v3_h = create_host_space_copy(v3);
  for (std::size_t i = 0; i < v3_h.extent(0); ++i) {
    for (std::size_t j = 0; j < v3_h.extent(1); ++j) {
      EXPECT_TRUE(v3_h(i, j) == v_gold_h(i, j));
    }
  }
}

template <class ValueType>
struct UnifDist;

template <>
struct UnifDist<int> {
  using dist_type = std::uniform_int_distribution<int>;
  std::mt19937 m_gen;
  dist_type m_dist;

  UnifDist(int b, std::size_t seedIn) : m_dist(0, b) { m_gen.seed(seedIn); }

  int operator()() { return m_dist(m_gen); }
};

template <class Tag, class ValueType>
void run_all_scenarios()
{

  for (int num_teams : team_sizes_to_test) {
    for (const auto& numCols : {0, 1, 2, 13, 101, 1444, 11153}) {
      UnifDist<int> pivotsProducer(numCols, 3123377);
      // 7 is an arbitrary number of pivots to test
      for (int k=0; k<7; ++k){
	const auto pivotIndex = pivotsProducer();
	for (int apiId : {0,1}) {
	  test_A<Tag, ValueType>(num_teams, numCols, pivotIndex, apiId);
	}
      }
    }
  }
}

TEST(std_algorithms_rotate_copy_team_test, test) {
  run_all_scenarios<DynamicTag, int>();
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedTwoRowsTag, double>();
  run_all_scenarios<StridedThreeRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, double>();
}

}  // namespace TeamRotateCopy
}  // namespace stdalgos
}  // namespace Test
