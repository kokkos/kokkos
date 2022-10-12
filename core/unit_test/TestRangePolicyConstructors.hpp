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

namespace Test {

struct SomeTag {};

template <typename ExecutionSpace>
struct TestRangePolicyConstructors {

public:

  TestRangePolicyConstructors() {
    check_semiregular();
    test_compile_time_parameters();
    test_runtime_parameters();
    test_worktag();
    test_prefer_desired_occupancy();
  }

private:

  void check_semiregular() {
    // semiregular is copyable and default initializable
    // (regular requires equality comparable)
    using policy_t = Kokkos::RangePolicy<ExecutionSpace>;

    static_assert(std::is_default_constructible<policy_t>::value, "");
    static_assert(std::is_copy_constructible<policy_t>::value, "");
    static_assert(std::is_move_constructible<policy_t>::value, "");
    static_assert(std::is_copy_assignable<policy_t>::value, "");
    static_assert(std::is_move_assignable<policy_t>::value, "");
    static_assert(std::is_destructible<policy_t>::value, "");
  }

  void test_compile_time_parameters() {
    {
      using policy_t = Kokkos::RangePolicy<>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      static_assert(std::is_same<execution_space, Kokkos::DefaultExecutionSpace>::value, "");
      static_assert(std::is_same<index_type, typename execution_space::size_type>::value, "");
      static_assert(std::is_same<schedule_type, Kokkos::Schedule<Kokkos::Static>>::value, "");
      static_assert(std::is_void<work_tag>::value, "");
    }

    {
      using policy_t = Kokkos::RangePolicy<ExecutionSpace>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      static_assert(std::is_same<execution_space, ExecutionSpace>::value, "");
      static_assert(std::is_same<index_type, typename execution_space::size_type>::value, "");
      static_assert(std::is_same<schedule_type, Kokkos::Schedule<Kokkos::Static>>::value, "");
      static_assert(std::is_void<work_tag>::value, "");
    }

    {
      using policy_t = Kokkos::RangePolicy<ExecutionSpace,
                                           Kokkos::Schedule<Kokkos::Dynamic>>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      static_assert(std::is_same<execution_space, ExecutionSpace>::value, "");
      static_assert(std::is_same<index_type, typename execution_space::size_type>::value, "");
      static_assert(std::is_same<schedule_type, Kokkos::Schedule<Kokkos::Dynamic>>::value, "");
      static_assert(std::is_void<work_tag>::value, "");
    }

    {
      using policy_t = Kokkos::RangePolicy<ExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>,
                                           Kokkos::IndexType<long>>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      static_assert(std::is_same<execution_space, ExecutionSpace>::value, "");
      static_assert(std::is_same<index_type, long>::value, "");
      static_assert(std::is_same<schedule_type, Kokkos::Schedule<Kokkos::Dynamic>>::value, "");
      static_assert(std::is_void<work_tag>::value, "");
    }

    {
      using policy_t = Kokkos::RangePolicy<Kokkos::IndexType<long>, ExecutionSpace,
                                           Kokkos::Schedule<Kokkos::Dynamic>>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      static_assert(std::is_same<execution_space, ExecutionSpace>::value, "");
      static_assert(std::is_same<index_type, long>::value, "");
      static_assert(std::is_same<schedule_type, Kokkos::Schedule<Kokkos::Dynamic>>::value, "");
      static_assert(std::is_void<work_tag>::value, "");
    }

    {
      using policy_t = Kokkos::RangePolicy<ExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>,
                                           Kokkos::IndexType<long>, SomeTag>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      static_assert(std::is_same<execution_space, ExecutionSpace>::value, "");
      static_assert(std::is_same<index_type, long>::value, "");
      static_assert(std::is_same<schedule_type, Kokkos::Schedule<Kokkos::Dynamic>>::value, "");
      static_assert(std::is_same<work_tag, SomeTag>::value, "");
    }

    {
      using policy_t = Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>, ExecutionSpace,
                                           Kokkos::IndexType<long>, SomeTag>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      static_assert(std::is_same<execution_space, ExecutionSpace>::value, "");
      static_assert(std::is_same<index_type, long>::value, "");
      static_assert(std::is_same<schedule_type, Kokkos::Schedule<Kokkos::Dynamic>>::value, "");
      static_assert(std::is_same<work_tag, SomeTag>::value, "");
    }

    {
      using policy_t = Kokkos::RangePolicy<SomeTag, Kokkos::Schedule<Kokkos::Dynamic>,
                                           Kokkos::IndexType<long>, ExecutionSpace>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      static_assert(std::is_same<execution_space, ExecutionSpace>::value, "");
      static_assert(std::is_same<index_type, long>::value, "");
      static_assert(std::is_same<schedule_type, Kokkos::Schedule<Kokkos::Dynamic>>::value, "");
      static_assert(std::is_same<work_tag, SomeTag>::value, "");
    }

    {
      using policy_t = Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      static_assert(std::is_same<execution_space, Kokkos::DefaultExecutionSpace>::value, "");
      static_assert(std::is_same<index_type, typename execution_space::size_type>::value, "");
      static_assert(std::is_same<schedule_type, Kokkos::Schedule<Kokkos::Dynamic>>::value, "");
      static_assert(std::is_void<work_tag>::value, "");
    }

    {
      using policy_t = Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>,
                                           Kokkos::IndexType<long>>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      static_assert(std::is_same<execution_space, Kokkos::DefaultExecutionSpace>::value, "");
      static_assert(std::is_same<index_type, long>::value, "");
      static_assert(std::is_same<schedule_type, Kokkos::Schedule<Kokkos::Dynamic>>::value, "");
      static_assert(std::is_void<work_tag>::value, "");
    }

    {
      using policy_t = Kokkos::RangePolicy<Kokkos::IndexType<long>,
                                           Kokkos::Schedule<Kokkos::Dynamic>>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      static_assert(std::is_same<execution_space, Kokkos::DefaultExecutionSpace>::value, "");
      static_assert(std::is_same<index_type, long>::value, "");
      static_assert(std::is_same<schedule_type, Kokkos::Schedule<Kokkos::Dynamic>>::value, "");
      static_assert(std::is_void<work_tag>::value, "");
    }

    {
      using policy_t = Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>,
                                           Kokkos::IndexType<long>, SomeTag>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      static_assert(std::is_same<execution_space, Kokkos::DefaultExecutionSpace>::value, "");
      static_assert(std::is_same<index_type, long>::value, "");
      static_assert(std::is_same<schedule_type, Kokkos::Schedule<Kokkos::Dynamic>>::value, "");
      static_assert(std::is_same<work_tag, SomeTag>::value, "");
    }

    {
      using policy_t = Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>,
                                           Kokkos::IndexType<long>, SomeTag>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      static_assert(std::is_same<execution_space, Kokkos::DefaultExecutionSpace>::value, "");
      static_assert(std::is_same<index_type, long>::value, "");
      static_assert(std::is_same<schedule_type, Kokkos::Schedule<Kokkos::Dynamic>>::value, "");
      static_assert(std::is_same<work_tag, SomeTag>::value, "");
    }

    {
      using policy_t = Kokkos::RangePolicy<SomeTag, Kokkos::Schedule<Kokkos::Dynamic>,
                                           Kokkos::IndexType<long>>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      static_assert(std::is_same<execution_space, Kokkos::DefaultExecutionSpace>::value, "");
      static_assert(std::is_same<index_type, long>::value, "");
      static_assert(std::is_same<schedule_type, Kokkos::Schedule<Kokkos::Dynamic>>::value, "");
      static_assert(std::is_same<work_tag, SomeTag>::value, "");
    }
  }

  void test_runtime_parameters() {
    using policy_t     = Kokkos::RangePolicy<>;
    using index_t      = policy_t::index_type;
    index_t work_begin = 5;
    index_t work_end   = 15;
    index_t chunk_size = 10;
    {
      policy_t p(work_begin, work_end);
      ASSERT_EQ(p.begin(), work_begin);
      ASSERT_EQ(p.end(),   work_end);
    }
    {
      policy_t p(ExecutionSpace(), work_begin, work_end);
      ASSERT_EQ(p.begin(), work_begin);
      ASSERT_EQ(p.end(),   work_end);
    }
    {
      policy_t p(work_begin, work_end, Kokkos::ChunkSize(chunk_size));
      ASSERT_EQ(p.begin(),      work_begin);
      ASSERT_EQ(p.end(),        work_end);
      ASSERT_EQ(p.chunk_size(), chunk_size);
    }
    {
      policy_t p(ExecutionSpace(), work_begin, work_end,
                 Kokkos::ChunkSize(chunk_size));
      ASSERT_EQ(p.begin(),      work_begin);
      ASSERT_EQ(p.end(),        work_end);
      ASSERT_EQ(p.chunk_size(), chunk_size);
    }
    {
      policy_t p;
      ASSERT_EQ(p.begin(),      index_t(0));
      ASSERT_EQ(p.end(),        index_t(0));
      ASSERT_EQ(p.chunk_size(), index_t(0));

      p = policy_t(work_begin, work_end, Kokkos::ChunkSize(chunk_size));
      ASSERT_EQ(p.begin(),      work_begin);
      ASSERT_EQ(p.end(),        work_end);
      ASSERT_EQ(p.chunk_size(), chunk_size);
    }
    {
      policy_t p1(work_begin, work_end, Kokkos::ChunkSize(chunk_size));
      policy_t p2(p1);
      ASSERT_EQ(p1.begin(),      p2.begin());
      ASSERT_EQ(p1.end(),        p2.end());
      ASSERT_EQ(p1.chunk_size(), p2.chunk_size());
    }
  }

  void test_worktag() {
    struct WorkTag1 {};
    struct WorkTag2 {};

    using policy_t_no_worktag = Kokkos::RangePolicy<>;
    using policy_t_worktag1 = Kokkos::Impl::WorkTagTrait::policy_with_trait<policy_t_no_worktag, WorkTag1>;
    using policy_t_worktag2 = Kokkos::Impl::WorkTagTrait::policy_with_trait<policy_t_worktag1, WorkTag2>;

    policy_t_no_worktag p0;
    static_assert(std::is_void<typename decltype(p0)::work_tag>::value, "");

    auto p1 = policy_t_worktag1(p0);
    static_assert(std::is_same<typename decltype(p1)::work_tag, WorkTag1>::value, "");

    auto p2 = policy_t_worktag2(p1);
    static_assert(std::is_same<typename decltype(p2)::work_tag, WorkTag2>::value, "");

    // NOTE line directly below does not currently compile
    //using policy_t_void_tag = Kokkos::Impl::WorkTagTrait::policy_with_trait<policy_t_no_worktag, void>;
    //auto p3 = policy_t_void_tag(p2);
    //static_assert(std::is_void<decltype(p3)::work_tag>::value, "");
  }

  void test_prefer_desired_occupancy() {
    using policy_t = Kokkos::RangePolicy<ExecutionSpace>;
    policy_t policy;

    static_assert(!policy_t::experimental_contains_desired_occupancy, "");

    // MaximizeOccupancy -> MaximizeOccupancy
    auto const policy_still_no_occ = Kokkos::Experimental::prefer(
        policy, Kokkos::Experimental::MaximizeOccupancy{});
    static_assert(!decltype(policy_still_no_occ)::experimental_contains_desired_occupancy, "");

    // MaximizeOccupancy -> DesiredOccupancy
    auto const policy_with_occ = Kokkos::Experimental::prefer(
        policy_still_no_occ, Kokkos::Experimental::DesiredOccupancy{33});
    static_assert(decltype(policy_with_occ)::experimental_contains_desired_occupancy, "");
    EXPECT_EQ(policy_with_occ.impl_get_desired_occupancy().value(), 33);

    // DesiredOccupancy -> DesiredOccupancy
    auto const policy_change_occ = Kokkos::Experimental::prefer(
        policy_with_occ, Kokkos::Experimental::DesiredOccupancy{24});
    static_assert(decltype(policy_change_occ)::experimental_contains_desired_occupancy, "");
    EXPECT_EQ(policy_change_occ.impl_get_desired_occupancy().value(), 24);

    // DesiredOccupancy -> DesiredOccupancy w/ hint
    auto policy_with_occ_and_hint = Kokkos::Experimental::require(
        policy_change_occ, Kokkos::Experimental::WorkItemProperty::HintLightWeight);
    EXPECT_EQ(policy_with_occ_and_hint.impl_get_desired_occupancy().value(), 24);

    // DesiredOccupancy -> MaximizeOccupancy
    auto const policy_drop_occ = Kokkos::Experimental::prefer(
        policy_with_occ_and_hint, Kokkos::Experimental::MaximizeOccupancy{});
    static_assert(!decltype(policy_drop_occ)::experimental_contains_desired_occupancy, "");
  }
};

TEST(TEST_CATEGORY, range_policy_construction) {
  TestRangePolicyConstructors<TEST_EXECSPACE>();
}

} // namespace Test
