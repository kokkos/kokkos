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

struct std_algorithms_min_max : public ::testing::Test {
  using value_type = int;

  const int m_numberOfFillingCases = 4;

  // views that we need to test
  using static_view_t = Kokkos::View<value_type[10]>;
  static_view_t m_static_view{"std-algo-min-max-ops-test-1D-contiguous-view-static"};

  using dyn_view_t = Kokkos::View<value_type*>;
  dyn_view_t m_dynamic_view{"std-algo-min-max-ops-test-1D-contiguous-view-dynamic", 10};

  using strided_view_t = Kokkos::View<value_type*, Kokkos::LayoutStride>;
  Kokkos::LayoutStride layout{10, 2};
  strided_view_t m_strided_view{"std-algo-min-max-ops-test-1D-strided-view", layout};

  // utility methods
  template <class ViewFromType>
  void copyInputViewToFixtureViews(ViewFromType viewFrom) {
    stdalgos::CopyFunctor<ViewFromType, static_view_t> F1(viewFrom,
                                                          m_static_view);
    Kokkos::parallel_for("_std_algo_copy1", viewFrom.extent(0), F1);

    stdalgos::CopyFunctor<ViewFromType, dyn_view_t> F2(viewFrom,
                                                       m_dynamic_view);
    Kokkos::parallel_for("_std_algo_copy2", viewFrom.extent(0), F2);

    stdalgos::CopyFunctor<ViewFromType, strided_view_t> F3(viewFrom,
                                                           m_strided_view);
    Kokkos::parallel_for("_std_algo_copy3", viewFrom.extent(0), F3);
  }

  void fillFixtureViews(int caseNumber) {
    using tmp_view_t = Kokkos::View<value_type*>;
    tmp_view_t tmpView("tmpView", 10);
    auto tmp_view_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), tmpView);
    if (caseNumber == 1) {
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

    else if (caseNumber == 2) {
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

    else if (caseNumber == 3) {
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

    else if (caseNumber == 4) {
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

    else {
    }

    Kokkos::deep_copy(tmpView, tmp_view_h);
    copyInputViewToFixtureViews(tmpView);
  }

  std::pair<int, value_type> goldIndexValuePair(int caseNumber) {
    if (caseNumber == 1) {
      return {3, 2};
    } else if (caseNumber == 2) {
      return {9, 10};
    } else if (caseNumber == 3) {
      return {0, 8};
    } else if (caseNumber == 4) {
      return {0, 2};
    } else {
      return {};
    }
  }
};

template<class ValueType>
struct StdAlgoMinMaxOpsTestCustomLessThanComparator
{
  KOKKOS_INLINE_FUNCTION
  bool operator()(const ValueType & a, const ValueType & b) const
  {
    return a < b;
  }

  StdAlgoMinMaxOpsTestCustomLessThanComparator(){}
};

template <class GoldItindexValuePairType, class ItType, class TestedViewType>
void std_algo_min_max_test_verify(const GoldItindexValuePairType & goldPair,
			     const ItType result,
                             TestedViewType testedView)
{
  namespace KE = Kokkos::Experimental;

  // check that iterator is correct
  EXPECT_EQ(result - KE::cbegin(testedView), std::get<0>(goldPair));

  using gold_view_t = Kokkos::View<int>;
  gold_view_t goldView("gold");
  using it_type = decltype(result);
  stdalgos::CopyFromIteratorFunctor<it_type, gold_view_t> cf(result, goldView);
  Kokkos::parallel_for("_std_algo_copy", 1, cf);

  auto gold_v_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), goldView);
  EXPECT_EQ(gold_v_h(), std::get<1>(goldPair));
}


/// --------------------------
/// MACROS to simplify things
/// --------------------------
#define MAX_ELEMENT_TRIVIAL_DATA_TEST(VIEWTOTEST)			\
  namespace KE = Kokkos::Experimental;					\
  const auto myTestedView = VIEWTOTEST;					\
   									\
  /* if we pass empty range, should return last */			\
  auto result = KE::max_element(KE::cbegin(myTestedView), KE::cbegin(myTestedView)); \
  EXPECT_TRUE(result == KE::cbegin(myTestedView));			\
									\
  /* if we pass empty range, should return last */			\
  auto it0 = KE::cbegin(myTestedView) + 3;				\
  auto it1 = it0;							\
  auto result2 = KE::max_element(it0, it1);				\
  EXPECT_TRUE(result2 == it1);						\

#define MAX_ELEMENT_NON_TRIVIAL_DATA_TEST(VIEWTOTEST, USEVIEW)	\
  namespace KE = Kokkos::Experimental;				\
  const auto myTestedView = VIEWTOTEST;				\
  for (int id = 1; id <= m_numberOfFillingCases; ++id) {		\
    fillFixtureViews(id);						\
    const auto goldPair = goldIndexValuePair(id);			\
    if (USEVIEW){							\
      const auto result = KE::max_element(myTestedView);		\
      std_algo_min_max_test_verify(goldPair, result, myTestedView);	\
      const auto result2 = KE::max_element("MYCUSTOMLABEL1", myTestedView); \
      std_algo_min_max_test_verify(goldPair, result2, myTestedView);	\
    }									\
    else{								\
      const auto result = KE::max_element(KE::cbegin(myTestedView), KE::cend(myTestedView)); \
      std_algo_min_max_test_verify(goldPair, result, myTestedView);	\
      const auto result2 = KE::max_element("MYCUSTOMLABEL2", KE::cbegin(myTestedView), KE::cend(myTestedView)); \
      std_algo_min_max_test_verify(goldPair, result2, myTestedView);	\
    }									\
  }									\

#define MAX_ELEMENT_NON_TRIVIAL_DATA_CUSTOM_COMP_TEST(VIEWTOTEST, USEVIEW) \
  namespace KE = Kokkos::Experimental;				\
  const auto myTestedView = VIEWTOTEST;				\
  for (int id = 1; id <= m_numberOfFillingCases; ++id) {		\
    fillFixtureViews(id);						\
    const auto goldPair = goldIndexValuePair(id);			\
    StdAlgoMinMaxOpsTestCustomLessThanComparator<value_type> comp;	\
    if (USEVIEW){							\
      const auto result = KE::max_element(myTestedView, comp);		\
      std_algo_min_max_test_verify(goldPair, result, myTestedView);	\
      const auto result2 = KE::max_element("MYCUSTOMLABEL3", myTestedView, comp); \
      std_algo_min_max_test_verify(goldPair, result2, myTestedView);	\
    }									\
    else{								\
      const auto result = KE::max_element(KE::cbegin(myTestedView), KE::cend(myTestedView), comp); \
      std_algo_min_max_test_verify(goldPair, result, myTestedView);	\
      const auto result2 = KE::max_element("MYCUSTOMLABEL4", KE::cbegin(myTestedView), KE::cend(myTestedView), comp); \
      std_algo_min_max_test_verify(goldPair, result2, myTestedView);	\
    }									\
  }									\

///
/// TRIVIAL case
///
TEST_F(std_algorithms_min_max, max_element_static_view_empty) {
  MAX_ELEMENT_TRIVIAL_DATA_TEST(m_static_view);
}

TEST_F(std_algorithms_min_max, max_element_dynamic_view_empty) {
  MAX_ELEMENT_TRIVIAL_DATA_TEST(m_dynamic_view);
}

TEST_F(std_algorithms_min_max, max_element_strided_view_empty) {
  MAX_ELEMENT_TRIVIAL_DATA_TEST(m_strided_view);
}

///
/// NON-TRIVIAL DATA for STATIC VIEW
///
TEST_F(std_algorithms_min_max, max_element_static_view_api_accept_iterators) {
  MAX_ELEMENT_NON_TRIVIAL_DATA_TEST(m_static_view, false);
}

TEST_F(std_algorithms_min_max, max_element_static_view_api_accept_iterators_custom_comp) {
  MAX_ELEMENT_NON_TRIVIAL_DATA_CUSTOM_COMP_TEST(m_static_view, false);
}

TEST_F(std_algorithms_min_max, max_element_static_view_api_accept_view) {
  MAX_ELEMENT_NON_TRIVIAL_DATA_TEST(m_static_view, true);
}

TEST_F(std_algorithms_min_max, max_element_static_view_api_accept_view_custom_comp) {
  MAX_ELEMENT_NON_TRIVIAL_DATA_CUSTOM_COMP_TEST(m_static_view, true);
}

///
/// NON-TRIVIAL DATA for DYNAMIC VIEW
///
TEST_F(std_algorithms_min_max, max_element_dynamic_view_api_accept_iterators) {
  MAX_ELEMENT_NON_TRIVIAL_DATA_TEST(m_dynamic_view, false);
}

TEST_F(std_algorithms_min_max, max_element_dynamic_view_api_accept_iterators_custom_comp) {
  MAX_ELEMENT_NON_TRIVIAL_DATA_CUSTOM_COMP_TEST(m_dynamic_view, false);
}

TEST_F(std_algorithms_min_max, max_element_dynamic_view_api_accept_view) {
  MAX_ELEMENT_NON_TRIVIAL_DATA_TEST(m_dynamic_view, true);
}

TEST_F(std_algorithms_min_max, max_element_dynamic_view_api_accept_view_custom_comp) {
  MAX_ELEMENT_NON_TRIVIAL_DATA_CUSTOM_COMP_TEST(m_dynamic_view, true);
}

///
/// NON-TRIVIAL DATA for STRIDED VIEW
///
TEST_F(std_algorithms_min_max, max_element_strided_view_api_accept_iterators) {
  MAX_ELEMENT_NON_TRIVIAL_DATA_TEST(m_strided_view, false);
}

TEST_F(std_algorithms_min_max, max_element_strided_view_api_accept_iterators_custom_comp) {
  MAX_ELEMENT_NON_TRIVIAL_DATA_CUSTOM_COMP_TEST(m_strided_view, false);
}

TEST_F(std_algorithms_min_max, max_element_strided_view_api_accept_view) {
  MAX_ELEMENT_NON_TRIVIAL_DATA_TEST(m_strided_view, true);
}

TEST_F(std_algorithms_min_max, max_element_strided_view_api_accept_view_custom_comp) {
  MAX_ELEMENT_NON_TRIVIAL_DATA_CUSTOM_COMP_TEST(m_strided_view, true);
}

}  // namespace Test
