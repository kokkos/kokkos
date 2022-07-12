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

namespace Test {
namespace stdalgos {
namespace TeamUniqueCopy {

namespace KE = Kokkos::Experimental;

template <class ValueType>
struct UnifDist;

template <>
struct UnifDist<int> {
  using dist_type = std::uniform_int_distribution<int>;
  std::mt19937 m_gen;
  dist_type m_dist;

  UnifDist() : m_dist(2, 9) { m_gen.seed(1034343); }

  int operator()() { return m_dist(m_gen); }
};

template <class ViewTypeFrom, class ViewTypeDest, class ViewDist,
          class MemberType>
struct FunctorA {
  ViewTypeFrom m_view_from;
  ViewTypeDest m_view_dest;
  ViewDist m_view_d;
  int m_api_pick;

  FunctorA(const ViewTypeFrom viewFrom, const ViewTypeDest viewDest,
           const ViewDist viewD, int apiPick)
      : m_view_from(viewFrom),
        m_view_dest(viewDest),
        m_view_d(viewD),
        m_api_pick(apiPick) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();
    auto myRowViewFrom =
        Kokkos::subview(m_view_from, myRowIndex, Kokkos::ALL());
    auto myRowViewDest =
        Kokkos::subview(m_view_dest, myRowIndex, Kokkos::ALL());

    if (m_api_pick == 0) {
      auto it =
          KE::unique_copy(member, KE::begin(myRowViewFrom),
                          KE::end(myRowViewFrom), KE::begin(myRowViewDest));
      m_view_d(myRowIndex) = KE::distance(KE::begin(myRowViewDest), it);
    } else if (m_api_pick == 1) {
      auto it = KE::unique_copy(member, myRowViewFrom, myRowViewDest);
      m_view_d(myRowIndex) = KE::distance(KE::begin(myRowViewDest), it);
    } else if (m_api_pick == 2) {
      using comparator_t =
          CustomEqualityComparator<typename ViewTypeFrom::value_type>;
      auto it              = KE::unique_copy(member, KE::begin(myRowViewFrom),
                                KE::end(myRowViewFrom),
                                KE::begin(myRowViewDest), comparator_t());
      m_view_d(myRowIndex) = KE::distance(KE::begin(myRowViewDest), it);
    } else if (m_api_pick == 3) {
      using comparator_t =
          CustomEqualityComparator<typename ViewTypeFrom::value_type>;
      auto it =
          KE::unique_copy(member, myRowViewFrom, myRowViewDest, comparator_t());
      m_view_d(myRowIndex) = KE::distance(KE::begin(myRowViewDest), it);
    }
  }
};

// impl is here for std because it is only avail from c++>=17
template <class InputIterator, class OutputIterator, class BinaryPredicate>
auto my_unique_copy(InputIterator first, InputIterator last,
                    OutputIterator result, BinaryPredicate pred) {
  if (first != last) {
    typename OutputIterator::value_type t(*first);
    *result = t;
    ++result;
    while (++first != last) {
      if (!pred(t, *first)) {
        t       = *first;
        *result = t;
        ++result;
      }
    }
  }
  return result;
}

template <class InputIterator, class OutputIterator>
auto my_unique_copy(InputIterator first, InputIterator last,
                    OutputIterator result) {
  using value_type = typename OutputIterator::value_type;
  using func_t     = IsEqualFunctor<value_type>;
  return my_unique_copy(first, last, result, func_t());
}

template <class Tag, class ValueType>
void test_A(std::size_t num_teams, std::size_t num_cols, int apiId) {
  /* description: */

  //
  // fill v
  //
  auto v = create_view<ValueType>(Tag{}, num_teams, num_cols, "v");
  // v might not deep copyable so to modify it on the host
  auto v_dc   = create_deep_copyable_compatible_view_with_same_extent(v);
  auto v_dc_h = create_mirror_view(Kokkos::HostSpace(), v_dc);
  UnifDist<ValueType> randObj;
  for (std::size_t i = 0; i < v_dc_h.extent(0); ++i) {
    for (std::size_t j = 0; j < v_dc_h.extent(1); ++j) {
      v_dc_h(i, j) = randObj();
    }
  }
  // copy to v_dc and then to v
  Kokkos::deep_copy(v_dc, v_dc_h);
  CopyFunctorRank2<decltype(v_dc), decltype(v)> F1(v_dc, v);
  Kokkos::parallel_for("copy", v.extent(0) * v.extent(1), F1);

  //
  // make a copy of v on host and use it to run host algorithm
  //
  Kokkos::View<int*, Kokkos::HostSpace> gold_distances("view_it_dist",
                                                       num_teams);
  auto v2_h = create_host_space_copy(v);
  Kokkos::View<ValueType**, Kokkos::HostSpace> v_gold_h("vgold", v2_h.extent(0),
                                                        v2_h.extent(1));
  for (std::size_t i = 0; i < v_dc_h.extent(0); ++i) {
    auto rowFrom      = Kokkos::subview(v2_h, i, Kokkos::ALL());
    auto rowDest      = Kokkos::subview(v_gold_h, i, Kokkos::ALL());
    auto it           = my_unique_copy(KE::cbegin(rowFrom), KE::cend(rowFrom),
                             KE::begin(rowDest));
    gold_distances(i) = KE::distance(KE::begin(rowDest), it);
  }

  // launch kernel
  using space_t          = Kokkos::DefaultExecutionSpace;
  using policy_type      = Kokkos::TeamPolicy<space_t>;
  using team_member_type = typename policy_type::member_type;
  policy_type policy(num_teams, Kokkos::AUTO());

  auto distances = create_view<int>(DynamicTag{}, num_teams, "view_it_dist");
  auto v3        = create_view<ValueType>(Tag{}, num_teams, num_cols, "v3");
  using functor_type = FunctorA<decltype(v), decltype(v3), decltype(distances),
                                team_member_type>;
  functor_type fnc(v, v3, distances, apiId);
  Kokkos::parallel_for(policy, fnc);

  (void)apiId;
  // check
  auto distances_h = create_host_space_copy(distances);
  for (std::size_t i = 0; i < v.extent(0); ++i) {
    EXPECT_TRUE(distances_h(i) == gold_distances(i));
  }

  auto v3_h = create_host_space_copy(v3);
  for (std::size_t i = 0; i < v.extent(0); ++i) {
    for (std::size_t j = 0; j < v.extent(1); ++j) {
      EXPECT_TRUE(v3_h(i, j) == v_gold_h(i, j));
    }
  }
}

template <class Tag, class ValueType>
void run_all_scenarios() {
  for (int num_teams : team_sizes_to_test) {
    for (const auto& numCols : {0, 1, 2, 13, 101, 1444, 11153}) {
      for (int apiId : {0, 1, 2, 3}) {
        test_A<Tag, ValueType>(num_teams, numCols, apiId);
      }
    }
  }
}

TEST(std_algorithms_unique_copy_team_test, test) {
  run_all_scenarios<DynamicTag, int>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, int>();
}

}  // namespace TeamUniqueCopy
}  // namespace stdalgos
}  // namespace Test
