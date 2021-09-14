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
#include <TestStdAlgorithmsCommon2.hpp>
#include <iterator>
#include <std_algorithms/Kokkos_BeginEnd.hpp>
#include <std_algorithms/Kokkos_NonModifyingSequenceOperations.hpp>
#include <algorithm>
#include <numeric>

namespace Test {
namespace stdalgos {
namespace Mismatch {

namespace KE = Kokkos::Experimental;

template <class ViewType1, class ViewType2>
void test_mismatch(const ViewType1 view_1, const ViewType2 view_2) {
  auto first_1 = KE::begin(view_1);
  auto last_1  = KE::end(view_1);
  auto first_2 = KE::begin(view_2);
  auto last_2  = KE::end(view_2);

  {
    // default comparator, pass iterators
    auto ret = KE::mismatch(exespace(), first_1, last_1, first_2, last_2);
    auto ret_with_label =
        KE::mismatch("label", exespace(), first_1, last_1, first_2, last_2);

    EXPECT_EQ(ret.first, KE::end(view_1));
    EXPECT_EQ(ret.second, KE::end(view_2));
    EXPECT_EQ(ret, ret_with_label);
  }
  {
    // default comparator, pass views
    auto ret            = KE::mismatch(exespace(), view_1, view_2);
    auto ret_with_label = KE::mismatch("label", exespace(), view_1, view_2);

    EXPECT_EQ(ret.first, KE::cend(view_1));
    EXPECT_EQ(ret.second, KE::cend(view_2));
    EXPECT_EQ(ret, ret_with_label);
  }

  using value_t_1 = typename ViewType1::value_type;
  const auto comp = CustomEqualityComparator<value_t_1>();
  {
    // custom comparator, pass iterators
    auto ret = KE::mismatch(exespace(), first_1, last_1, first_2, last_2, comp);
    auto ret_with_label = KE::mismatch("label", exespace(), first_1, last_1,
                                       first_2, last_2, comp);

    EXPECT_EQ(ret.first, KE::end(view_1));
    EXPECT_EQ(ret.second, KE::end(view_2));
    EXPECT_EQ(ret, ret_with_label);
  }
  {
    // custom comparator, pass views
    auto ret = KE::mismatch(exespace(), view_1, view_2, comp);
    auto ret_with_label =
        KE::mismatch("label", exespace(), view_1, view_2, comp);

    EXPECT_EQ(ret.first, KE::cend(view_1));
    EXPECT_EQ(ret.second, KE::cend(view_2));
    EXPECT_EQ(ret, ret_with_label);
  }
}

template <class Tag, class ValueType>
void run_all_scenarios() {
  for (const auto& scenario : default_scenarios) {
    {
      auto view1 = create_view<ValueType>(Tag{}, scenario.second, "mismatch_1");
      auto view2 = create_view<ValueType>(Tag{}, scenario.second, "mismatch_2");

      test_mismatch(view1, view2);
    }
  }
}

TEST(std_algorithms_mismatch_test, test) {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoTag, int>();
  run_all_scenarios<StridedThreeTag, unsigned>();
}

}  // namespace Mismatch
}  // namespace stdalgos
}  // namespace Test
