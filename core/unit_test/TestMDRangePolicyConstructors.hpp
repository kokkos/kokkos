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

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

namespace {

template <class IndexType>
void construct_range_policy_variable_type() {
  (void)Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>>{
      Kokkos::Array<IndexType, 2>{}, Kokkos::Array<IndexType, 2>{}};

  (void)Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>>{
      {{IndexType(0), IndexType(0)}}, {{IndexType(2), IndexType(2)}}};

  (void)Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>>{
      {IndexType(0), IndexType(0)}, {IndexType(2), IndexType(2)}};
}

TEST(TEST_CATEGORY, md_range_policy_construction_from_arrays) {
  {
    // Check that construction from Kokkos::Array of the specified index type
    // works.
    using IndexType = unsigned long long;
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>,
                          Kokkos::IndexType<IndexType>>
        p1(Kokkos::Array<IndexType, 2>{{0, 1}},
           Kokkos::Array<IndexType, 2>{{2, 3}});
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>,
                          Kokkos::IndexType<IndexType>>
        p2(Kokkos::Array<IndexType, 2>{{0, 1}},
           Kokkos::Array<IndexType, 2>{{2, 3}});
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>,
                          Kokkos::IndexType<IndexType>>
        p3(Kokkos::Array<IndexType, 2>{{0, 1}},
           Kokkos::Array<IndexType, 2>{{2, 3}},
           Kokkos::Array<IndexType, 1>{{4}});
  }
  {
    // Check that construction from double-braced initliazer list
    // works.
    using index_type = unsigned long long;
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>> p1({{0, 1}},
                                                              {{2, 3}});
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>,
                          Kokkos::IndexType<index_type>>
        p2({{0, 1}}, {{2, 3}});
  }
  {
    // Check that construction from Kokkos::Array of long compiles for backwards
    // compability.  This was broken in
    // https://github.com/kokkos/kokkos/pull/3527/commits/88ea8eec6567c84739d77bdd25fdbc647fae28bb#r512323639
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>> p1(
        Kokkos::Array<long, 2>{{0, 1}}, Kokkos::Array<long, 2>{{2, 3}});
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>> p2(
        Kokkos::Array<long, 2>{{0, 1}}, Kokkos::Array<long, 2>{{2, 3}});
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>> p3(
        Kokkos::Array<long, 2>{{0, 1}}, Kokkos::Array<long, 2>{{2, 3}},
        Kokkos::Array<long, 1>{{4}});
  }

  // Check that construction from various index types works.
  construct_range_policy_variable_type<char>();
  construct_range_policy_variable_type<int>();
  construct_range_policy_variable_type<unsigned long>();
  construct_range_policy_variable_type<std::int64_t>();
}

#ifndef KOKKOS_COMPILER_NVHPC       // FIXME_NVHPC
#ifndef KOKKOS_ENABLE_OPENMPTARGET  // FIXME_OPENMPTARGET
TEST(TEST_CATEGORY_DEATH, policy_bounds_unsafe_narrowing_conversions) {
  using Policy = Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>,
                                       Kokkos::IndexType<unsigned>>;

  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_DEATH(
      {
        (void)Policy({-1, 0}, {2, 3});
      },
      "unsafe narrowing conversion");
}
#endif
#endif

}  // namespace
