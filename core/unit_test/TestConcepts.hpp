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

namespace TestConcept {

using ExecutionSpace = TEST_EXECSPACE;
using MemorySpace    = typename ExecutionSpace::memory_space;
using DeviceType     = typename ExecutionSpace::device_type;

static_assert(Kokkos::is_execution_space<ExecutionSpace>{}, "");
static_assert(Kokkos::is_execution_space<ExecutionSpace const>{}, "");
static_assert(!Kokkos::is_execution_space<ExecutionSpace &>{}, "");
static_assert(!Kokkos::is_execution_space<ExecutionSpace const &>{}, "");

static_assert(Kokkos::is_memory_space<MemorySpace>{}, "");
static_assert(Kokkos::is_memory_space<MemorySpace const>{}, "");
static_assert(!Kokkos::is_memory_space<MemorySpace &>{}, "");
static_assert(!Kokkos::is_memory_space<MemorySpace const &>{}, "");

static_assert(Kokkos::is_device<DeviceType>{}, "");
static_assert(Kokkos::is_device<DeviceType const>{}, "");
static_assert(!Kokkos::is_device<DeviceType &>{}, "");
static_assert(!Kokkos::is_device<DeviceType const &>{}, "");

static_assert(!Kokkos::is_device<ExecutionSpace>{}, "");
static_assert(!Kokkos::is_device<MemorySpace>{}, "");

static_assert(Kokkos::is_space<ExecutionSpace>{}, "");
static_assert(Kokkos::is_space<MemorySpace>{}, "");
static_assert(Kokkos::is_space<DeviceType>{}, "");
static_assert(Kokkos::is_space<ExecutionSpace const>{}, "");
static_assert(Kokkos::is_space<MemorySpace const>{}, "");
static_assert(Kokkos::is_space<DeviceType const>{}, "");
static_assert(!Kokkos::is_space<ExecutionSpace &>{}, "");
static_assert(!Kokkos::is_space<MemorySpace &>{}, "");
static_assert(!Kokkos::is_space<DeviceType &>{}, "");

static_assert(Kokkos::is_execution_space_v<ExecutionSpace>, "");
static_assert(!Kokkos::is_execution_space_v<ExecutionSpace &>, "");

static_assert(
    std::is_same<float, Kokkos::Impl::remove_cvref_t<float const &>>{}, "");
static_assert(std::is_same<int, Kokkos::Impl::remove_cvref_t<int &>>{}, "");
static_assert(std::is_same<int, Kokkos::Impl::remove_cvref_t<int const>>{}, "");
static_assert(std::is_same<float, Kokkos::Impl::remove_cvref_t<float>>{}, "");

/*-------------------------------------------------
  begin test for team_handle concept

  Read before moving on to the test below:
  The implementation inside Kokkos/core of the team_handle concept follows
  that of execution space, memory space, etc as shown here:

    https://github.com/kokkos/kokkos/blob/61d7db55fceac3318c987a291f77b844fd94c165/core/src/Kokkos_Concepts.hpp#L160

  which has a key aspect: for performance reasons, it does *not* check the
  complete trait. So below we also provide a complete team handle concept trait
  based on this

    https://kokkos.github.io/kokkos-core-wiki/API/core/policies/TeamHandleConcept.html

  which completely checks the trait and we use to validate things.
  This complete trait was originally used as implementation but was moved
  here as discussed in this PR: https://github.com/kokkos/kokkos/pull/5375

  ------------------------------------------------- */
namespace TestIsTeamHandle {

template <typename T>
struct is_team_handle_complete_trait_check {
 private:
  struct TrivialFunctor {
    void operator()(double &) {}
  };
  using test_value_type = double;
  test_value_type lvalueForMethodsNeedingIt_;
  test_value_type *ptrForMethodsNeedingIt_;
  // we use Sum here but any other reducer can be used
  // since we just want something that meets the ReducerConcept
  using reduction_to_test_t = ::Kokkos::Sum<test_value_type>;

  // nested aliases
  template <class U>
  using ExecutionSpaceArchetypeAlias = typename U::execution_space;
  template <class U>
  using ScratchMemorySpaceArchetypeAlias = typename U::scratch_memory_space;

  // "indices" methods
  template <class U>
  using TeamRankArchetypeExpr = decltype(std::declval<U const &>().team_rank());

  template <class U>
  using TeamSizeArchetypeExpr = decltype(std::declval<U const &>().team_size());

  template <class U>
  using LeagueRankArchetypeExpr =
      decltype(std::declval<U const &>().league_rank());

  template <class U>
  using LeagueSizeArchetypeExpr =
      decltype(std::declval<U const &>().league_size());

  // scratch space
  template <class U>
  using TeamShmemArchetypeExpr =
      decltype(std::declval<U const &>().team_shmem());

  template <class U>
  using TeamScratchArchetypeExpr =
      decltype(std::declval<U const &>().team_scratch(int{}));

  template <class U>
  using ThreadScracthArchetypeExpr =
      decltype(std::declval<U const &>().thread_scratch(int{}));

  // collectives
  template <class U>
  using TeamBarrierArchetypeExpr =
      decltype(std::declval<U const &>().team_barrier());

  template <class U>
  using TeamBroadcastArchetypeExpr =
      decltype(std::declval<U const &>().team_broadcast(
          lvalueForMethodsNeedingIt_, int{}));

  template <class U>
  using TeamBroadcastAcceptClosureArchetypeExpr =
      decltype(std::declval<U const &>().team_broadcast(
          TrivialFunctor{}, lvalueForMethodsNeedingIt_, int{}));

  template <class U>
  using TeamReducedArchetypeExpr =
      decltype(std::declval<U const &>().team_reduce(
          std::declval<reduction_to_test_t>()));

  template <class U>
  using TeamScanArchetypeExpr = decltype(std::declval<U const &>().team_scan(
      lvalueForMethodsNeedingIt_, ptrForMethodsNeedingIt_));

 public:
  static constexpr bool value =
      Kokkos::is_detected_v<ExecutionSpaceArchetypeAlias, T> &&
      Kokkos::is_detected_v<ScratchMemorySpaceArchetypeAlias, T> &&
      //
      Kokkos::is_detected_exact_v<int, TeamRankArchetypeExpr, T> &&
      Kokkos::is_detected_exact_v<int, TeamSizeArchetypeExpr, T> &&
      Kokkos::is_detected_exact_v<int, LeagueRankArchetypeExpr, T> &&
      Kokkos::is_detected_exact_v<int, LeagueSizeArchetypeExpr, T> &&
      //
      Kokkos::is_detected_exact_v<
          Kokkos::detected_t<ScratchMemorySpaceArchetypeAlias, T> const &,
          TeamShmemArchetypeExpr, T> &&
      Kokkos::is_detected_exact_v<
          Kokkos::detected_t<ScratchMemorySpaceArchetypeAlias, T> const &,
          TeamScratchArchetypeExpr, T> &&
      Kokkos::is_detected_exact_v<
          Kokkos::detected_t<ScratchMemorySpaceArchetypeAlias, T> const &,
          ThreadScracthArchetypeExpr, T> &&
      //
      Kokkos::is_detected_exact_v<void, TeamBarrierArchetypeExpr, T> &&
      Kokkos::is_detected_exact_v<void, TeamBroadcastArchetypeExpr, T> &&
      Kokkos::is_detected_exact_v<void, TeamBroadcastAcceptClosureArchetypeExpr,
                                  T> &&
      Kokkos::is_detected_exact_v<void, TeamReducedArchetypeExpr, T> &&
      Kokkos::is_detected_exact_v<test_value_type, TeamScanArchetypeExpr, T>;
  constexpr operator bool() const noexcept { return value; }
};

template <class T>
inline constexpr bool is_team_handle_complete_trait_check_v =
    is_team_handle_complete_trait_check<T>::value;

// actual test begins here

/*
  disabling as follows:

  - OpenMPTARGET: due to this
    https://github.com/kokkos/kokkos/blob/2d6cbad7e079eb45ae69ac6a59929d9fcf10409a/core/src/OpenMPTarget/Kokkos_OpenMPTarget_Exec.hpp#L860

    - OpenACC: not supporting teams yet
   */
#if not defined KOKKOS_ENABLE_OPENMPTARGET && not defined KOKKOS_ENABLE_OPENACC
using space_t  = TEST_EXECSPACE;
using policy_t = Kokkos::TeamPolicy<space_t>;
using member_t = typename policy_t::member_type;

// is_team_handle uses the actual core implementation
static_assert(Kokkos::is_team_handle<member_t>::value);
static_assert(Kokkos::is_team_handle_v<member_t>);
static_assert(Kokkos::is_team_handle_v<member_t const>);
static_assert(!Kokkos::is_team_handle_v<member_t &>);
static_assert(!Kokkos::is_team_handle_v<member_t const &>);
static_assert(!Kokkos::is_team_handle_v<member_t *>);
static_assert(!Kokkos::is_team_handle_v<member_t const *>);
static_assert(!Kokkos::is_team_handle_v<member_t *const>);

// is_team_handle_complete_trait_check uses the FULL trait class above
static_assert(is_team_handle_complete_trait_check<member_t>::value);
static_assert(is_team_handle_complete_trait_check_v<member_t>);
static_assert(is_team_handle_complete_trait_check_v<member_t const>);
static_assert(!is_team_handle_complete_trait_check_v<member_t &>);
static_assert(!is_team_handle_complete_trait_check_v<member_t const &>);
static_assert(!is_team_handle_complete_trait_check_v<member_t *>);
static_assert(!is_team_handle_complete_trait_check_v<member_t const *>);
static_assert(!is_team_handle_complete_trait_check_v<member_t *const>);
#endif

}  // end namespace TestIsTeamHandle
/*-------------------------------------------------
  end test for team_handle concept
  -------------------------------------------------*/

}  // namespace TestConcept
