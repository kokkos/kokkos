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

#ifndef KOKKOS_ALGORITHMS_UNITTESTS_TESTSORT_COMMON_HPP
#define KOKKOS_ALGORITHMS_UNITTESTS_TESTSORT_COMMON_HPP

#include <TestSort.hpp>

namespace Test {
TEST(TEST_CATEGORY, SortUnsigned) {
  Impl::test_sort<TEST_EXECSPACE, unsigned>(171);
}

TEST(TEST_CATEGORY, NestedSort) {
  Impl::test_nested_sort<TEST_EXECSPACE, unsigned>(171, 0U, UINT_MAX);
  Impl::test_nested_sort<TEST_EXECSPACE, float>(42, -1e6f, 1e6f);
  Impl::test_nested_sort<TEST_EXECSPACE, char>(67, CHAR_MIN, CHAR_MAX);
}

TEST(TEST_CATEGORY, NestedSortByKey) {
  // Second/third template arguments are key and value respectively.
  // In sort_by_key_X functions, a key view and a value view are both permuted
  // to make the keys sorted. This means that the value type doesn't need to be
  // ordered, unlike key
  Impl::test_nested_sort_by_key<TEST_EXECSPACE, unsigned, unsigned>(
      161, 0U, UINT_MAX, 0U, UINT_MAX);
  Impl::test_nested_sort_by_key<TEST_EXECSPACE, float, char>(
      267, -1e6f, 1e6f, CHAR_MIN, CHAR_MAX);
  Impl::test_nested_sort_by_key<TEST_EXECSPACE, char, double>(
      11, CHAR_MIN, CHAR_MAX, 2.718, 3.14);
}

}  // namespace Test
#endif
