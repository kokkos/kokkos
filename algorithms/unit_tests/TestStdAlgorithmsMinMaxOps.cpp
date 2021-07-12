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
#include <std_algorithms/Kokkos_MinMaxOperations.hpp>

namespace Test {

struct std_algorithms_min_max : public ::testing::Test
{
  static constexpr std::size_t m_numElements = 10;

  using value_type = int;

  using static_view_t = Kokkos::View<value_type[m_numElements]>;
  static_view_t m_static_view{"std-algo-test-1D-contiguous-view-static"};

  using dyn_view_t = Kokkos::View<value_type*>;
  dyn_view_t m_dynamic_view{"std-algo-test-1D-contiguous-view-dynamic", m_numElements};

  using strided_view_t = Kokkos::View<value_type*, Kokkos::LayoutStride>;
  Kokkos::LayoutStride layout{m_numElements, 2};
  strided_view_t m_strided_view{"std-algo-test-1D-strided-view", layout};

  template<class ViewFromType>
  void copyViewToFixturesViews(ViewFromType viewFrom)
  {
    stdalgos::CopyFunctor<ViewFromType, static_view_t> F1(viewFrom, m_static_view);
    Kokkos::parallel_for("_std_algo_copy1", m_numElements, F1);

    stdalgos::CopyFunctor<ViewFromType, dyn_view_t> F2(viewFrom, m_dynamic_view);
    Kokkos::parallel_for("_std_algo_copy2", m_numElements, F2);

    stdalgos::CopyFunctor<ViewFromType, strided_view_t> F3(viewFrom, m_strided_view);
    Kokkos::parallel_for("_std_algo_copy2", m_numElements, F3);
  }

  void fillFixtureViews(int caseNumber)
  {
    using tmp_view_t = Kokkos::View<value_type*>;
    tmp_view_t tmpView("tmpView", m_numElements);
    auto tmp_view_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), tmpView);
    if (caseNumber == 1){
      tmp_view_h(0) = 0;
      tmp_view_h(1) = 0;
      tmp_view_h(2) = 0;
      tmp_view_h(3) = 2;
      tmp_view_h(4) = 2;
      tmp_view_h(5) = 1;
      tmp_view_h(6) = 1;
      tmp_view_h(7) = 1;
      tmp_view_h(8) = 1;
      tmp_view_h(9) = 0;
    }

    else if (caseNumber == 2){
      tmp_view_h(0) = 1;
      tmp_view_h(1) = 2;
      tmp_view_h(2) = 3;
      tmp_view_h(3) = 4;
      tmp_view_h(4) = 5;
      tmp_view_h(5) = 6;
      tmp_view_h(6) = 7;
      tmp_view_h(7) = 8;
      tmp_view_h(8) = 9;
      tmp_view_h(9) = 10;
    }

    else if (caseNumber == 3){
      tmp_view_h(0) = 8;
      tmp_view_h(1) = 8;
      tmp_view_h(2) = -1;
      tmp_view_h(3) = -1;
      tmp_view_h(4) = 5;
      tmp_view_h(5) = 5;
      tmp_view_h(6) = 5;
      tmp_view_h(7) = 8;
      tmp_view_h(8) = 2;
      tmp_view_h(9) = 1;
    }

    else if (caseNumber == 4){
      tmp_view_h(0) = 2;
      tmp_view_h(1) = 2;
      tmp_view_h(2) = 2;
      tmp_view_h(3) = 2;
      tmp_view_h(4) = 2;
      tmp_view_h(5) = 2;
      tmp_view_h(6) = 2;
      tmp_view_h(7) = 2;
      tmp_view_h(8) = 2;
      tmp_view_h(9) = 2;
    }

    else{
    }

    Kokkos::deep_copy(tmpView, tmp_view_h);
    copyViewToFixturesViews(tmpView);
  }

  std::pair<int, value_type> goldIndexValuePair(int caseNumber)
  {
    if(caseNumber==1){ return {3,2}; }
    else if(caseNumber==2){ return {9,10}; }
    else if(caseNumber==3){ return {0,8};  }
    else if(caseNumber==4){ return {0,2};  }
    else{
      return {};
    }
  }
};

template<class goldPairType, class ItType, class TestedViewType>
void std_algo_min_max_verify(const goldPairType & goldPair,
			     ItType result,
			     TestedViewType testedView)
{
  namespace KE = Kokkos::Experimental;
  EXPECT_EQ(result-KE::cbegin(testedView), std::get<0>(goldPair));

  using gold_view_t = Kokkos::View<int>;
  gold_view_t goldView("gold");
  using it_type = decltype(result);
  stdalgos::CopyFromIteratorFunctor<it_type, gold_view_t> cf(result, goldView);
  Kokkos::parallel_for("_std_algo_copy", 1, cf);

  auto gold_v_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), goldView);
  EXPECT_EQ(gold_v_h(), std::get<1>(goldPair));
}

///
/// TRIVIAL DATA
///
TEST_F(std_algorithms_min_max, max_element_static_view_empty) {
  namespace KE = Kokkos::Experimental;

  const auto testedView = m_static_view;
  for (int id=1; id<=4; ++id)
  {
    // if we pass empty range, should return last
    auto result = KE::max_element(KE::cbegin(testedView), KE::cbegin(testedView));
    EXPECT_TRUE(result==KE::cbegin(testedView));
  }
}

TEST_F(std_algorithms_min_max, max_element_dynamic_view_empty) {
  namespace KE = Kokkos::Experimental;

  const auto testedView = m_dynamic_view;
  for (int id=1; id<=4; ++id)
  {
    // if we pass empty range, should return last
    auto result = KE::max_element(KE::cbegin(testedView), KE::cbegin(testedView));
    EXPECT_TRUE(result==KE::cbegin(testedView));
  }
}

TEST_F(std_algorithms_min_max, max_element_strided_view_empty) {
  namespace KE = Kokkos::Experimental;

  const auto testedView = m_strided_view;
  for (int id=1; id<=4; ++id)
  {
    // if we pass empty range, should return last
    auto result = KE::max_element(KE::cbegin(testedView), KE::cbegin(testedView));
    EXPECT_TRUE(result==KE::cbegin(testedView));
  }
}

///
/// NON-TRIVIAL DATA
///
TEST_F(std_algorithms_min_max, max_element_static_view_api_accept_iterators) {
  namespace KE = Kokkos::Experimental;

  const auto testedView = m_static_view;
  for (int id=1; id<=4; ++id)
  {
    fillFixtureViews(id);
    const auto goldPair = goldIndexValuePair(id);
    auto result = KE::max_element(KE::cbegin(testedView), KE::cend(testedView));
    std_algo_min_max_verify(goldPair, result, testedView);
  }
}

TEST_F(std_algorithms_min_max, max_element_static_view_api_accept_view) {
  namespace KE = Kokkos::Experimental;

  const auto testedView = m_static_view;
  for (int id=1; id<=4; ++id)
  {
    fillFixtureViews(id);
    const auto goldPair = goldIndexValuePair(id);
    auto result = KE::max_element(m_static_view);
    std_algo_min_max_verify(goldPair, result, testedView);
  }
}

TEST_F(std_algorithms_min_max, max_element_dynamic_view_api_accept_iterators) {
  namespace KE = Kokkos::Experimental;

  const auto testedView = m_dynamic_view;
  for (int id=1; id<=4; ++id)
  {
    fillFixtureViews(id);
    const auto goldPair = goldIndexValuePair(id);
    auto result = KE::max_element(KE::cbegin(testedView), KE::cend(testedView));
    std_algo_min_max_verify(goldPair, result, testedView);
  }
}

TEST_F(std_algorithms_min_max, max_element_dynamic_view_api_accept_view) {
  namespace KE = Kokkos::Experimental;

  const auto testedView = m_dynamic_view;
  for (int id=1; id<=4; ++id)
  {
    fillFixtureViews(id);
    const auto goldPair = goldIndexValuePair(id);
    auto result = KE::max_element(testedView);
    std_algo_min_max_verify(goldPair, result, testedView);
  }
}

TEST_F(std_algorithms_min_max, max_element_strided_view_api_accept_itertors) {
  namespace KE = Kokkos::Experimental;

  const auto testedView = m_strided_view;
  for (int id=1; id<=4; ++id)
  {
    fillFixtureViews(id);
    const auto goldPair = goldIndexValuePair(id);
    auto result = KE::max_element(KE::cbegin(testedView), KE::cend(testedView));
    std_algo_min_max_verify(goldPair, result, testedView);
  }
}

TEST_F(std_algorithms_min_max, max_element_strided_view_api_accept_view) {
  namespace KE = Kokkos::Experimental;

  const auto testedView = m_strided_view;
  for (int id=1; id<=4; ++id)
  {
    fillFixtureViews(id);
    const auto goldPair = goldIndexValuePair(id);
    auto result = KE::max_element(testedView);
    std_algo_min_max_verify(goldPair, result, testedView);
  }
}




}  // namespace Test
