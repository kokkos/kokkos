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

#include <cstdio>
#include <sstream>
#include <iostream>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include "helper_functions/CommonExecPolicyTests.hpp"

namespace {

void test_range_policy_run_time_parameters() {
  using policy_t     = Kokkos::RangePolicy<>;
  using index_t      = policy_t::index_type;
  index_t work_begin = 5;
  index_t work_end   = 15;
  index_t chunk_size = 10;
  {
    policy_t p(work_begin, work_end);
    ASSERT_EQ(p.begin(), work_begin);
    ASSERT_EQ(p.end(), work_end);
  }
  {
    policy_t p(Kokkos::DefaultExecutionSpace(), work_begin, work_end);
    ASSERT_EQ(p.begin(), work_begin);
    ASSERT_EQ(p.end(), work_end);
  }
  {
    policy_t p(work_begin, work_end, Kokkos::ChunkSize(chunk_size));
    ASSERT_EQ(p.begin(), work_begin);
    ASSERT_EQ(p.end(), work_end);
    ASSERT_EQ(p.chunk_size(), chunk_size);
  }
  {
    policy_t p(Kokkos::DefaultExecutionSpace(), work_begin, work_end,
               Kokkos::ChunkSize(chunk_size));
    ASSERT_EQ(p.begin(), work_begin);
    ASSERT_EQ(p.end(), work_end);
    ASSERT_EQ(p.chunk_size(), chunk_size);
  }
  {
    policy_t p;
    ASSERT_EQ(p.begin(), index_t(0));
    ASSERT_EQ(p.end(), index_t(0));
    ASSERT_EQ(p.chunk_size(), index_t(0));

    p = policy_t(work_begin, work_end, Kokkos::ChunkSize(chunk_size));
    ASSERT_EQ(p.begin(), work_begin);
    ASSERT_EQ(p.end(), work_end);
    ASSERT_EQ(p.chunk_size(), chunk_size);
  }
  {
    policy_t p1(work_begin, work_end, Kokkos::ChunkSize(chunk_size));
    policy_t p2(p1);
    ASSERT_EQ(p1.begin(), p2.begin());
    ASSERT_EQ(p1.end(), p2.end());
    ASSERT_EQ(p1.chunk_size(), p2.chunk_size());
  }
}

TEST(TEST_CATEGORY, range_policy_semi_regular) {
  check_semiregular<Kokkos::RangePolicy>();
}

TEST(TEST_CATEGORY, range_policy_compile_time_parameters) {
  test_compile_time_parameters<Kokkos::RangePolicy>();
}

TEST(TEST_CATEGORY, range_policy_run_time_parameters) {
  test_range_policy_run_time_parameters();
}

TEST(TEST_CATEGORY, range_policy_worktag) {
  test_worktag<Kokkos::RangePolicy>();
}

TEST(TEST_CATEGORY, range_policy_occupancy) {
  test_prefer_desired_occupancy<Kokkos::RangePolicy>();
}

}  // namespace
