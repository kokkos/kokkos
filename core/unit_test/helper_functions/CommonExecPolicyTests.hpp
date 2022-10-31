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

#include <Kokkos_Core.hpp>

struct SomeTag {};

template <template <class...> class PolicyType>
void check_semiregular() {
  using Policy = PolicyType<>;

  // semiregular is copyable and default initializable
  // (regular requires equality comparable)
  static_assert(std::is_default_constructible_v<Policy>);
  static_assert(std::is_copy_constructible_v<Policy>);
  static_assert(std::is_move_constructible_v<Policy>);
  static_assert(std::is_copy_assignable_v<Policy>);
  static_assert(std::is_move_assignable_v<Policy>);
  static_assert(std::is_destructible_v<Policy>);
}

template <typename Policy, typename ExpectedExecutionSpace,
          typename ExpectedIndexType, typename ExpectedScheduleType,
          typename ExpectedWorkTag>
void test_compile_time_parameters_for_type() {
  using execution_space = typename Policy::execution_space;
  using index_type      = typename Policy::index_type;
  using schedule_type   = typename Policy::schedule_type;
  using work_tag        = typename Policy::work_tag;

  static_assert(std::is_same_v<execution_space, ExpectedExecutionSpace>);
  static_assert(std::is_same_v<index_type, ExpectedIndexType>);
  static_assert(std::is_same_v<schedule_type, ExpectedScheduleType>);
  static_assert(std::is_same_v<work_tag, ExpectedWorkTag>);
}

template <template <class...> class PolicyType>
void test_compile_time_parameters() {
  {
    using Policy = PolicyType<>;
    test_compile_time_parameters_for_type<
        Policy, Kokkos::DefaultExecutionSpace,
        Kokkos::DefaultExecutionSpace::size_type,
        Kokkos::Schedule<Kokkos::Static>, void>();
  }

  {
    using Policy = PolicyType<TEST_EXECSPACE>;
    test_compile_time_parameters_for_type<
        Policy, TEST_EXECSPACE, TEST_EXECSPACE::size_type,
        Kokkos::Schedule<Kokkos::Static>, void>();
  }

  {
    using Policy =
        Kokkos::TeamPolicy<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Dynamic>>;
    test_compile_time_parameters_for_type<
        Policy, TEST_EXECSPACE, TEST_EXECSPACE::size_type,
        Kokkos::Schedule<Kokkos::Dynamic>, void>();
  }

  {
    using Policy =
        Kokkos::TeamPolicy<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Dynamic>,
                           Kokkos::IndexType<long>>;
    test_compile_time_parameters_for_type<Policy, TEST_EXECSPACE, long,
                                          Kokkos::Schedule<Kokkos::Dynamic>,
                                          void>();
  }

  {
    using Policy = Kokkos::TeamPolicy<Kokkos::IndexType<long>, TEST_EXECSPACE,
                                      Kokkos::Schedule<Kokkos::Dynamic>>;
    test_compile_time_parameters_for_type<Policy, TEST_EXECSPACE, long,
                                          Kokkos::Schedule<Kokkos::Dynamic>,
                                          void>();
  }

  {
    using Policy =
        Kokkos::TeamPolicy<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Dynamic>,
                           Kokkos::IndexType<long>, SomeTag>;
    test_compile_time_parameters_for_type<Policy, TEST_EXECSPACE, long,
                                          Kokkos::Schedule<Kokkos::Dynamic>,
                                          SomeTag>();
  }

  {
    using Policy =
        Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>, TEST_EXECSPACE,
                           Kokkos::IndexType<long>, SomeTag>;
    test_compile_time_parameters_for_type<Policy, TEST_EXECSPACE, long,
                                          Kokkos::Schedule<Kokkos::Dynamic>,
                                          SomeTag>();
  }

  {
    using Policy =
        Kokkos::TeamPolicy<SomeTag, Kokkos::Schedule<Kokkos::Dynamic>,
                           Kokkos::IndexType<long>, TEST_EXECSPACE>;
    test_compile_time_parameters_for_type<Policy, TEST_EXECSPACE, long,
                                          Kokkos::Schedule<Kokkos::Dynamic>,
                                          SomeTag>();
  }

  {
    using Policy = Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>>;
    test_compile_time_parameters_for_type<
        Policy, Kokkos::DefaultExecutionSpace,
        Kokkos::DefaultExecutionSpace::size_type,
        Kokkos::Schedule<Kokkos::Dynamic>, void>();
  }

  {
    using Policy = Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>,
                                      Kokkos::IndexType<long>>;
    test_compile_time_parameters_for_type<
        Policy, Kokkos::DefaultExecutionSpace, long,
        Kokkos::Schedule<Kokkos::Dynamic>, void>();
  }

  {
    using Policy = Kokkos::TeamPolicy<Kokkos::IndexType<long>,
                                      Kokkos::Schedule<Kokkos::Dynamic>>;
    test_compile_time_parameters_for_type<
        Policy, Kokkos::DefaultExecutionSpace, long,
        Kokkos::Schedule<Kokkos::Dynamic>, void>();
  }

  {
    using Policy = Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>,
                                      Kokkos::IndexType<long>, SomeTag>;
    test_compile_time_parameters_for_type<
        Policy, Kokkos::DefaultExecutionSpace, long,
        Kokkos::Schedule<Kokkos::Dynamic>, SomeTag>();
  }

  {
    using Policy =
        Kokkos::TeamPolicy<SomeTag, Kokkos::Schedule<Kokkos::Dynamic>,
                           Kokkos::IndexType<long>>;
    test_compile_time_parameters_for_type<
        Policy, Kokkos::DefaultExecutionSpace, long,
        Kokkos::Schedule<Kokkos::Dynamic>, SomeTag>();
  }
}

template <template <class...> class PolicyType>
void test_worktag() {
  struct WorkTag1 {};
  struct WorkTag2 {};

  using Policy = PolicyType<>;
  using Policy_worktag1 =
      Kokkos::Impl::WorkTagTrait::policy_with_trait<Policy, WorkTag1>;
  using Policy_worktag2 =
      Kokkos::Impl::WorkTagTrait::policy_with_trait<Policy_worktag1, WorkTag2>;

  Policy p0;
  static_assert(std::is_void_v<typename decltype(p0)::work_tag>);

  auto p1 = Policy_worktag1(p0);
  static_assert(std::is_same_v<typename decltype(p1)::work_tag, WorkTag1>);

  auto p2 = Policy_worktag2(p1);
  static_assert(std::is_same_v<typename decltype(p2)::work_tag, WorkTag2>);

  // NOTE line directly below does not currently compile
  // using Policy_void_tag =
  //    Kokkos::Impl::WorkTagTrait::policy_with_trait<Policy,
  //                                                  void>;
  // auto p3 = Policy_void_tag(p2);
  // static_assert(std::is_void_v<decltype(p3)::work_tag>);
}

template <template <class...> class PolicyType>
void test_prefer_desired_occupancy() {
  using Policy = PolicyType<>;
  Policy policy;

  static_assert(!Policy::experimental_contains_desired_occupancy);

  // MaximizeOccupancy -> MaximizeOccupancy
  auto const policy_still_no_occ = Kokkos::Experimental::prefer(
      policy, Kokkos::Experimental::MaximizeOccupancy{});
  static_assert(
      !decltype(policy_still_no_occ)::experimental_contains_desired_occupancy);

  // MaximizeOccupancy -> DesiredOccupancy
  auto const policy_with_occ = Kokkos::Experimental::prefer(
      policy_still_no_occ, Kokkos::Experimental::DesiredOccupancy{33});
  static_assert(
      decltype(policy_with_occ)::experimental_contains_desired_occupancy);
  EXPECT_EQ(policy_with_occ.impl_get_desired_occupancy().value(), 33);

  // DesiredOccupancy -> DesiredOccupancy
  auto const policy_change_occ = Kokkos::Experimental::prefer(
      policy_with_occ, Kokkos::Experimental::DesiredOccupancy{24});
  static_assert(
      decltype(policy_change_occ)::experimental_contains_desired_occupancy);
  EXPECT_EQ(policy_change_occ.impl_get_desired_occupancy().value(), 24);

  // DesiredOccupancy -> DesiredOccupancy w/ hint
  auto policy_with_occ_and_hint = Kokkos::Experimental::require(
      policy_change_occ,
      Kokkos::Experimental::WorkItemProperty::HintLightWeight);
  EXPECT_EQ(policy_with_occ_and_hint.impl_get_desired_occupancy().value(), 24);

  // DesiredOccupancy -> MaximizeOccupancy
  auto const policy_drop_occ = Kokkos::Experimental::prefer(
      policy_with_occ_and_hint, Kokkos::Experimental::MaximizeOccupancy{});
  static_assert(
      !decltype(policy_drop_occ)::experimental_contains_desired_occupancy);
}
