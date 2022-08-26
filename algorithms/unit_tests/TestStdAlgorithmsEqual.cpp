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
#include <algorithm>

namespace Test::stdalgos::Equal {

namespace KE = Kokkos::Experimental;

template <class ViewType>
void test_equal(const ViewType view) {
  auto copy = create_deep_copyable_compatible_clone(view);

  // pass iterators
  EXPECT_TRUE(
      KE::equal(exespace(), KE::begin(view), KE::end(view), KE::begin(copy)));
  // pass views
  EXPECT_TRUE(KE::equal(exespace(), view, copy));

  // modify copy - make the last element different
  const auto extent = view.extent(0);
  if (extent > 0) {
    KE::fill(exespace(), KE::end(copy) - 1, KE::end(copy), 1);

    // pass const iterators
    EXPECT_FALSE(KE::equal(exespace(), KE::cbegin(view), KE::cend(view),
                           KE::cbegin(copy)));
    // pass views
    EXPECT_FALSE(KE::equal("label", exespace(), view, copy));
  }
}

template <class ViewType>
void test_equal_custom_comparator(const ViewType view) {
  using value_t = typename ViewType::value_type;
  const auto p  = CustomEqualityComparator<value_t>();
  auto copy     = create_deep_copyable_compatible_clone(view);

  // pass iterators
  EXPECT_TRUE(KE::equal(exespace(), KE::begin(view), KE::end(view),
                        KE::begin(copy), p));
  // pass views
  EXPECT_TRUE(KE::equal(exespace(), view, copy, p));

  // modify copy - make the last element different
  const auto extent = view.extent(0);
  if (extent > 0) {
    KE::fill(exespace(), KE::end(copy) - 1, KE::end(copy), 1);

    // pass const iterators
    EXPECT_FALSE(KE::equal("label", exespace(), KE::cbegin(view),
                           KE::cend(view), KE::cbegin(copy), p));
    // pass views
    EXPECT_FALSE(KE::equal(exespace(), view, copy, p));
  }
}

template <class ViewType>
void test_equal_4_iterators(const ViewType view) {
  using value_t = typename ViewType::value_type;
  const auto p  = CustomEqualityComparator<value_t>();
  auto copy     = create_deep_copyable_compatible_clone(view);

  // pass iterators
  EXPECT_TRUE(KE::equal(exespace(), KE::begin(view), KE::end(view),
                        KE::begin(copy), KE::end(copy)));
  // pass const and non-const iterators, custom comparator
  EXPECT_TRUE(KE::equal("label", exespace(), KE::cbegin(view), KE::cend(view),
                        KE::begin(copy), KE::end(copy), p));

  const auto extent = view.extent(0);
  if (extent > 0) {
    // use different length ranges, pass const iterators
    EXPECT_FALSE(KE::equal(exespace(), KE::cbegin(view), KE::cend(view),
                           KE::cbegin(copy), KE::cend(copy) - 1));

    // modify copy - make the last element different
    KE::fill(exespace(), KE::end(copy) - 1, KE::end(copy), 1);
    // pass const iterators
    EXPECT_FALSE(KE::equal(exespace(), KE::cbegin(view), KE::cend(view),
                           KE::cbegin(copy), KE::cend(copy)));
  }
}

template <class Tag, class ValueType>
void run_all_scenarios() {
  for (const auto& scenario : default_scenarios) {
    auto view = create_view<ValueType>(Tag{}, scenario.second, "equal");
    test_equal(view);
    test_equal_custom_comparator(view);
    test_equal_4_iterators(view);
  }
}

TEST(std_algorithms_equal_test, test) {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoTag, int>();
  run_all_scenarios<StridedThreeTag, unsigned>();
}

}  // namespace Test::stdalgos::Equal
