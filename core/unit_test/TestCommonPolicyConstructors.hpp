//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <Kokkos_Core.hpp>

namespace {

// Dummy policy for testing base class.
template <class... Args>
struct DummyPolicy : Kokkos::Impl::PolicyTraits<Args...> {
  using execution_policy = DummyPolicy;
};

// Asserts that a policy constructor is semiregular.
// Semiregular is copyable and default initializable
// (regular requires equality comparable).
template <class Policy>
constexpr bool check_semiregular() {
  static_assert(std::is_default_constructible_v<Policy>);
  static_assert(std::is_copy_constructible_v<Policy>);
  static_assert(std::is_move_constructible_v<Policy>);
  static_assert(std::is_copy_assignable_v<Policy>);
  static_assert(std::is_move_assignable_v<Policy>);
  static_assert(std::is_destructible_v<Policy>);

  return true;
}

static_assert(check_semiregular<DummyPolicy<>>());
static_assert(check_semiregular<Kokkos::RangePolicy<>>());
static_assert(check_semiregular<Kokkos::TeamPolicy<>>());
static_assert(check_semiregular<Kokkos::MDRangePolicy<Kokkos::Rank<2>>>());

// Asserts that worktag conversion works properly.
template <class Policy>
constexpr bool test_worktag() {
  struct WorkTag1 {};
  struct WorkTag2 {};

  // Apply WorkTag1
  using PolicyWithWorkTag1 =
      Kokkos::Impl::WorkTagTrait::policy_with_trait<Policy, WorkTag1>;
  // Swap for WorkTag2
  using PolicyWithWorkTag2 =
      Kokkos::Impl::WorkTagTrait::policy_with_trait<PolicyWithWorkTag1,
                                                    WorkTag2>;

  static_assert(std::is_void_v<typename Policy::work_tag>);
  static_assert(
      std::is_same_v<typename PolicyWithWorkTag1::work_tag, WorkTag1>);
  static_assert(
      std::is_same_v<typename PolicyWithWorkTag2::work_tag, WorkTag2>);

  // Currently not possible to remove the work tag from a policy.
  // Uncomment the line below to see the compile error.
  // using PolicyRemoveWorkTag =
  // Kokkos::Impl::WorkTagTrait::policy_with_trait<PolicyWithWorkTag2, void>;
  // static_assert(std::is_void_v<PolicyRemoveWorkTag::work_tag>);

  return true;
}

static_assert(test_worktag<DummyPolicy<>>());
static_assert(test_worktag<Kokkos::RangePolicy<>>());
static_assert(test_worktag<Kokkos::TeamPolicy<>>());
static_assert(test_worktag<Kokkos::MDRangePolicy<Kokkos::Rank<2>>>());

// Assert that occupancy conversion and hints work properly.
template <class Policy>
void test_prefer_desired_occupancy() {
  Policy policy;

  using Kokkos::Experimental::DesiredOccupancy;
  using Kokkos::Experimental::MaximizeOccupancy;
  using Kokkos::Experimental::WorkItemProperty;

  static_assert(!Policy::experimental_contains_desired_occupancy);

  // MaximizeOccupancy -> MaximizeOccupancy
  auto const policy_still_no_occ =
      Kokkos::Experimental::prefer(policy, MaximizeOccupancy{});
  static_assert(
      !decltype(policy_still_no_occ)::experimental_contains_desired_occupancy);

  // MaximizeOccupancy -> DesiredOccupancy
  auto const policy_with_occ =
      Kokkos::Experimental::prefer(policy_still_no_occ, DesiredOccupancy{33});
  static_assert(
      decltype(policy_with_occ)::experimental_contains_desired_occupancy);
  EXPECT_EQ(policy_with_occ.impl_get_desired_occupancy().value(), 33);

  // DesiredOccupancy -> DesiredOccupancy
  auto const policy_change_occ =
      Kokkos::Experimental::prefer(policy_with_occ, DesiredOccupancy{24});
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
      policy_with_occ_and_hint, MaximizeOccupancy{});
  static_assert(
      !decltype(policy_drop_occ)::experimental_contains_desired_occupancy);

  // The following demonstrates that the size of a policy
  // does not increase if the user decides not to use
  // experimental procedures.
  // We must disable in some case since an EBO failure with
  // VS 16.11.3 and CUDA 11.4.2 will effect the
  // DummyPolicy instantiation.
#if !(defined(_WIN32) && defined(KOKKOS_ENABLE_CUDA))
  // assert that base class has size 1. We are only
  // concerned with calling this if the PolicyType
  // template is DummyPolicy. In all other cases,
  // "!std::is_same_v<Policy, DummyPolicy<Args...>>=true"
  // and the test automatically passes.
  static_assert(!std::is_same_v<Policy, DummyPolicy<>> ||
                sizeof(decltype(policy)) == 1);

  // assert that size of the policy with
  // experimental proceedure is the size
  // of the procedure struct. Again,
  // only tested for DummyPolicy.
  static_assert(!std::is_same_v<Policy, DummyPolicy<>> ||
                sizeof(decltype(policy_with_occ)) == sizeof(DesiredOccupancy));
#endif
}

TEST(TEST_CATEGORY, execution_policy_occupancy_and_hint) {
  test_prefer_desired_occupancy<DummyPolicy<>>();
  test_prefer_desired_occupancy<Kokkos::RangePolicy<>>();
  test_prefer_desired_occupancy<Kokkos::TeamPolicy<>>();
  test_prefer_desired_occupancy<Kokkos::MDRangePolicy<Kokkos::Rank<2>>>();
}

template <typename Policy, typename ExpectedExecutionSpace,
          typename ExpectedIndexType, typename ExpectedScheduleType,
          typename ExpectedWorkTag>
constexpr bool compile_time_test() {
  using execution_space = typename Policy::execution_space;
  using index_type      = typename Policy::index_type;
  using schedule_type   = typename Policy::schedule_type;
  using work_tag        = typename Policy::work_tag;

  static_assert(std::is_same_v<execution_space, ExpectedExecutionSpace>);
  static_assert(std::is_same_v<index_type, ExpectedIndexType>);
  static_assert(std::is_same_v<schedule_type, ExpectedScheduleType>);
  static_assert(std::is_same_v<work_tag, ExpectedWorkTag>);

  return true;
}

// Separate class type from class template args so that different
// combinations of template args can be used, while still including
// any necessary templates args (stored in "Args...").
// Example: MDRangePolicy required an iteration pattern be included.
template <template <class...> class PolicyType, class... Args>
constexpr bool test_compile_time_parameters() {
  struct SomeTag {};

  using TestExecSpace    = TEST_EXECSPACE;
  using DefaultExecSpace = Kokkos::DefaultExecutionSpace;
  using TestIndex        = TestExecSpace::size_type;
  using DefaultIndex     = DefaultExecSpace::size_type;
  using LongIndex        = Kokkos::IndexType<long>;
  using StaticSchedule   = Kokkos::Schedule<Kokkos::Static>;
  using DynamicSchedule  = Kokkos::Schedule<Kokkos::Dynamic>;

  // clang-format off
  compile_time_test<PolicyType<                                                            Args...>, DefaultExecSpace, DefaultIndex, StaticSchedule,  void   >();
  compile_time_test<PolicyType<TestExecSpace,                                              Args...>, TestExecSpace,    TestIndex,    StaticSchedule,  void   >();
  compile_time_test<PolicyType<DynamicSchedule,                                            Args...>, DefaultExecSpace, DefaultIndex, DynamicSchedule, void   >();
  compile_time_test<PolicyType<TestExecSpace,   DynamicSchedule,                           Args...>, TestExecSpace,    TestIndex,    DynamicSchedule, void   >();
  compile_time_test<PolicyType<DynamicSchedule, LongIndex,                                 Args...>, DefaultExecSpace, long,         DynamicSchedule, void   >();
  compile_time_test<PolicyType<LongIndex,       DynamicSchedule,                           Args...>, DefaultExecSpace, long,         DynamicSchedule, void   >();
  compile_time_test<PolicyType<TestExecSpace,   DynamicSchedule, LongIndex,                Args...>, TestExecSpace,    long,         DynamicSchedule, void   >();
  compile_time_test<PolicyType<LongIndex,       TestExecSpace,   DynamicSchedule,          Args...>, TestExecSpace,    long,         DynamicSchedule, void   >();
  compile_time_test<PolicyType<DynamicSchedule, LongIndex,       SomeTag,                  Args...>, DefaultExecSpace, long,         DynamicSchedule, SomeTag>();
  compile_time_test<PolicyType<SomeTag,         DynamicSchedule, LongIndex,                Args...>, DefaultExecSpace, long,         DynamicSchedule, SomeTag>();
  compile_time_test<PolicyType<TestExecSpace,   DynamicSchedule, LongIndex, SomeTag,       Args...>, TestExecSpace,    long,         DynamicSchedule, SomeTag>();
  compile_time_test<PolicyType<DynamicSchedule, TestExecSpace,   LongIndex, SomeTag,       Args...>, TestExecSpace,    long,         DynamicSchedule, SomeTag>();
  compile_time_test<PolicyType<SomeTag,         DynamicSchedule, LongIndex, TestExecSpace, Args...>, TestExecSpace,    long,         DynamicSchedule, SomeTag>();
  // clang-format on

  return true;
}

static_assert(test_compile_time_parameters<Kokkos::RangePolicy>());
static_assert(test_compile_time_parameters<Kokkos::TeamPolicy>());
static_assert(
    test_compile_time_parameters<Kokkos::MDRangePolicy, Kokkos::Rank<2>>());

}  // namespace
