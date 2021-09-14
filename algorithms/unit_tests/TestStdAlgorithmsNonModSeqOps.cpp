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
#include <std_algorithms/Kokkos_BeginEnd.hpp>
#include <std_algorithms/Kokkos_NonModifyingSequenceOperations.hpp>

namespace KE = Kokkos::Experimental;

namespace Test {
namespace stdalgos {

struct std_algorithms_non_mod_seq_ops_test : public std_algorithms_test {};

template <class ViewType1, class ViewType2, class... Args>
void mismatch_test_helper(ViewType1 view_operand1, ViewType2 view_operand2,
                          Args&&... args) {
  //
  // same length no mismatch
  // remember that KE::end() return iterator to element after last one
  // begin1	          end1
  //   |                   |
  //   0,0,0,0,0,0,0,0,0,0 *
  // begin2		  end2
  //   |                   |
  //   0,0,0,0,0,0,0,0,0,0 *
  {
    stdalgos::fill_zero(view_operand1, view_operand2);
    auto first1 = KE::begin(view_operand1);
    auto last1  = KE::end(view_operand1);
    auto first2 = KE::begin(view_operand2);
    auto last2  = KE::end(view_operand2);
    auto result = KE::mismatch(exespace(), first1, last1, first2, last2,
                               std::forward<Args>(args)...);
    EXPECT_EQ(result.first, KE::end(view_operand1));
    EXPECT_EQ(result.second, KE::end(view_operand2));
  }

  //
  // same length with mismatch at 6 from first range
  // remember that KE::end() return iterator to element after last one
  // begin1	          end1
  //   |                   |
  //   0,0,0,0,0,0,1,1,1,1 *
  // begin2		  end2
  //   |                   |
  //   0,0,0,0,0,0,0,0,0,0 *
  {
    stdalgos::fill_zero(view_operand1, view_operand2);
    const auto functor = IncrementElementWiseFunctor<value_type>();
    KE::for_each(exespace(), KE::begin(view_operand1) + 6,
                 KE::end(view_operand1), functor);
    auto first1 = KE::begin(view_operand1);
    auto last1  = KE::end(view_operand1);
    auto first2 = KE::begin(view_operand2);
    auto last2  = KE::end(view_operand2);
    auto result = KE::mismatch(exespace(), first1, last1, first2, last2,
                               std::forward<Args>(args)...);
    EXPECT_EQ(result.first, KE::begin(view_operand1) + 6);
    EXPECT_EQ(result.second, KE::begin(view_operand2) + 6);
  }

  // same length with mismatch at 6 from second range
  // remember that KE::end() return iterator to element after last one
  // begin1	          end1
  //   |                   |
  //   0,0,0,0,0,0,0,0,0,0 *
  // begin2		  end2
  //   |                   |
  //   0,0,0,0,0,0,1,1,1,1 *
  {
    stdalgos::fill_zero(view_operand1, view_operand2);
    const auto functor = IncrementElementWiseFunctor<value_type>();
    KE::for_each(exespace(), KE::begin(view_operand2) + 6,
                 KE::end(view_operand2), functor);
    auto first1 = KE::begin(view_operand1);
    auto last1  = KE::end(view_operand1);
    auto first2 = KE::begin(view_operand2);
    auto last2  = KE::end(view_operand2);
    auto result = KE::mismatch(exespace(), first1, last1, first2, last2,
                               std::forward<Args>(args)...);
    EXPECT_EQ(result.first, KE::begin(view_operand1) + 6);
    EXPECT_EQ(result.second, KE::begin(view_operand2) + 6);
  }

  //
  // first range shorter, no mismatch
  // remember that KE::end() return iterator to element after last one
  // begin1	        end1
  //   |                 |
  //   0,0,0,0,0,0,0,0,0,0 *
  // begin2		  end2
  //   |                   |
  //   0,0,0,0,0,0,0,0,0,0 *
  {
    stdalgos::fill_zero(view_operand1, view_operand2);
    auto first1 = KE::begin(view_operand1);
    auto last1  = KE::end(view_operand1) - 1;
    auto first2 = KE::begin(view_operand2);
    auto last2  = KE::end(view_operand2);
    auto result = KE::mismatch(exespace(), first1, last1, first2, last2,
                               std::forward<Args>(args)...);
    ASSERT_EQ(result.first, last1);
    ASSERT_EQ(result.second, first2 + (last1 - first1));
  }

  //
  // first range shorter, mismatch at 3
  // remember that KE::end() return iterator to element after last one
  // begin1	        end1
  //   |                 |
  //   0,0,0,1,1,1,1,1,1,1 *
  // begin2		  end2
  //   |                   |
  //   0,0,0,0,0,0,0,0,0,0 *
  {
    stdalgos::fill_zero(view_operand1, view_operand2);
    const auto functor = IncrementElementWiseFunctor<value_type>();
    KE::for_each(exespace(), KE::begin(view_operand1) + 3,
                 KE::end(view_operand1), functor);
    auto first1 = KE::begin(view_operand1);
    auto last1  = KE::end(view_operand1) - 1;
    auto first2 = KE::begin(view_operand2);
    auto last2  = KE::end(view_operand2);
    auto result = KE::mismatch(exespace(), first1, last1, first2, last2,
                               std::forward<Args>(args)...);
    ASSERT_EQ(result.first, first1 + 3);
    ASSERT_EQ(result.second, first2 + 3);
  }

  //
  // second range shorter, no mismatch
  // begin1		  end1
  //   |                   |
  //   0,0,0,0,0,0,0,0,0,0 *
  // begin2	    end2
  //   |             |
  //   0,0,0,0,0,0,0,0,0,0 *
  //
  {
    stdalgos::fill_zero(view_operand1, view_operand2);
    auto first1 = KE::begin(view_operand1);
    auto last1  = KE::end(view_operand1);
    auto first2 = KE::begin(view_operand2);
    auto last2  = KE::end(view_operand2) - 3;
    auto result = KE::mismatch(exespace(), first1, last1, first2, last2,
                               std::forward<Args>(args)...);
    ASSERT_EQ(result.first, last1 + 7);
    ASSERT_EQ(result.second, last2);
  }

  //
  // second range shorter, mismatch at 2
  // begin1		  end1
  //   |                   |
  //   0,0,0,0,0,0,0,0,0,0 *
  // begin2	    end2
  //   |             |
  //   0,0,1,1,1,1,1,1,1,1 *
  //
  {
    stdalgos::fill_zero(view_operand1, view_operand2);
    const auto functor = IncrementElementWiseFunctor<value_type>();
    KE::for_each(exespace(), KE::begin(view_operand2) + 2,
                 KE::end(view_operand2), functor);
    auto first1 = KE::begin(view_operand1);
    auto last1  = KE::end(view_operand1);
    auto first2 = KE::begin(view_operand2);
    auto last2  = KE::end(view_operand2) - 3;
    auto result = KE::mismatch(exespace(), first1, last1, first2, last2,
                               std::forward<Args>(args)...);
    ASSERT_EQ(result.first, first1 + 2);
    ASSERT_EQ(result.second, first2 + 2);
  }
}

TEST_F(std_algorithms_non_mod_seq_ops_test, mismatch_default_binary_predicate) {
  mismatch_test_helper(m_static_view, m_dynamic_view);
}

TEST_F(std_algorithms_non_mod_seq_ops_test,
       mismatch_user_provided_binary_predicate) {
  using predicate_type = CustomEqualityComparator<value_type>;
  mismatch_test_helper(m_static_view, m_dynamic_view, predicate_type());
}

TEST_F(std_algorithms_non_mod_seq_ops_test, mismatch_view) {
  auto no_mismatch = KE::mismatch(exespace(), m_static_view, m_dynamic_view);
  EXPECT_EQ(KE::cend(m_static_view), no_mismatch.first);
  EXPECT_EQ(KE::cend(m_dynamic_view), no_mismatch.second);

  const auto functor = IncrementElementWiseFunctor<value_type>();
  KE::for_each(exespace(), KE::begin(m_static_view) + 5, KE::end(m_static_view),
               functor);

  auto mismatched = KE::mismatch(exespace(), m_static_view, m_dynamic_view);
  EXPECT_EQ(KE::cbegin(m_static_view) + 5, mismatched.first);
  EXPECT_EQ(KE::cbegin(m_dynamic_view) + 5, mismatched.second);
}

}  // namespace stdalgos
}  // namespace Test
