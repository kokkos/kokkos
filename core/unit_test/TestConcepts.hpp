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
struct FakeExeSpace {};
struct FakeScratchMemorySpace {};

#define a1() using execution_space = FakeExeSpace;
#define a2() using scratch_memory_space = FakeScratchMemorySpace;
#define m1() KOKKOS_INLINE_FUNCTION int team_rank() const noexcept;
#define m2() KOKKOS_INLINE_FUNCTION int team_size() const noexcept;
#define m3() KOKKOS_INLINE_FUNCTION int league_rank() const noexcept;
#define m4() KOKKOS_INLINE_FUNCTION int league_size() const noexcept;

#define m5()             \
  KOKKOS_INLINE_FUNCTION \
  const FakeScratchMemorySpace &team_shmem() const;

#define m6()             \
  KOKKOS_INLINE_FUNCTION \
  const FakeScratchMemorySpace &team_scratch(int level) const;

#define m7()             \
  KOKKOS_INLINE_FUNCTION \
  const FakeScratchMemorySpace &thread_scratch(int level) const;

#define m8()             \
  KOKKOS_INLINE_FUNCTION \
  void team_barrier() const noexcept;

#define m9()                                  \
  template <typename T>                       \
  KOKKOS_INLINE_FUNCTION void team_broadcast( \
      T &value, const int source_team_rank) const noexcept;

#define m10()                                 \
  template <class Closure, typename T>        \
  KOKKOS_INLINE_FUNCTION void team_broadcast( \
      Closure const &f, T &value, const int source_team_rank) const noexcept;

#define m11()                                                         \
  template <typename ReducerType>                                     \
  KOKKOS_INLINE_FUNCTION void team_reduce(ReducerType const &reducer) \
      const noexcept;

#define m12()                                                             \
  template <typename T>                                                   \
  KOKKOS_INLINE_FUNCTION T team_scan(T const &value, T *const global = 0) \
      const noexcept;

struct ValidTeamMember {
  using team_handle = ValidTeamMember;
  a1() a2() m1() m2() m3() m4() m5() m6() m7() m8() m9() m10() m11() m12()
};
struct InvalidTeamMember1 {
  /*using team_handle  = InvalidTeamMember1;*/
  /*a1()*/ a2() m1() m2() m3() m4() m5() m6() m7() m8() m9() m10() m11() m12()
};
struct InvalidTeamMember2 {
  /*using team_handle  = InvalidTeamMember2;*/
  a1() /*a2()*/ m1() m2() m3() m4() m5() m6() m7() m8() m9() m10() m11() m12()
};
struct InvalidTeamMember3 {
  /*using team_handle  = InvalidTeamMember3;*/
  a1() a2() /*m1()*/ m2() m3() m4() m5() m6() m7() m8() m9() m10() m11() m12()
};
struct InvalidTeamMember4 {
  /*using team_handle  = InvalidTeamMember4;*/
  a1() a2() m1() /*m2()*/ m3() m4() m5() m6() m7() m8() m9() m10() m11() m12()
};
struct InvalidTeamMember5 {
  /*using team_handle  = InvalidTeamMember5;*/
  a1() a2() m1() m2() /*m3()*/ m4() m5() m6() m7() m8() m9() m10() m11() m12()
};
struct InvalidTeamMember6 {
  /*using team_handle  = InvalidTeamMember6;*/
  a1() a2() m1() m2() m3() /*m4()*/ m5() m6() m7() m8() m9() m10() m11() m12()
};
struct InvalidTeamMember7 {
  /*using team_handle  = InvalidTeamMember7;*/
  a1() a2() m1() m2() m3() m4() /*m5()*/ m6() m7() m8() m9() m10() m11() m12()
};
struct InvalidTeamMember8 {
  /*using team_handle  = InvalidTeamMember8;*/
  a1() a2() m1() m2() m3() m4() m5() /*m6()*/ m7() m8() m9() m10() m11() m12()
};
struct InvalidTeamMember9 {
  /*using team_handle  = InvalidTeamMember9;*/
  a1() a2() m1() m2() m3() m4() m5() m6() /*m7()*/ m8() m9() m10() m11() m12()
};
struct InvalidTeamMember10 {
  /*using team_handle  = InvalidTeamMember10;*/
  a1() a2() m1() m2() m3() m4() m5() m6() m7() /*m8()*/ m9() m10() m11() m12()
};
struct InvalidTeamMember11 {
  /*using team_handle  = InvalidTeamMember11;*/
  a1() a2() m1() m2() m3() m4() m5() m6() m7() m8() /*m9()*/ m10() m11() m12()
};
struct InvalidTeamMember12 {
  /*using team_handle  = InvalidTeamMember12;*/
  a1() a2() m1() m2() m3() m4() m5() m6() m7() m8() m9() /*m10()*/ m11() m12()
};
struct InvalidTeamMember13 {
  /*using team_handle  = InvalidTeamMember13;*/
  a1() a2() m1() m2() m3() m4() m5() m6() m7() m8() m9() m10() /*m11()*/ m12()
};
struct InvalidTeamMember14 {
  /*using team_handle  = InvalidTeamMember14;*/
  a1() a2() m1() m2() m3() m4() m5() m6() m7() m8() m9() m10() m11() /*m12()*/
};
struct InvalidTeamMember15 {
  /*using team_handle  = InvalidTeamMember15;*/
  a1() a2() m1() m2() m3() m4() m5() m6() m7() m8() m9() m10() m11() m12()
};

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

// the following use a custom struct defined above
static_assert(Kokkos::is_team_handle_v<ValidTeamMember>);
static_assert(!Kokkos::is_team_handle_v<InvalidTeamMember1>);
static_assert(!Kokkos::is_team_handle_v<InvalidTeamMember2>);
static_assert(!Kokkos::is_team_handle_v<InvalidTeamMember3>);
static_assert(!Kokkos::is_team_handle_v<InvalidTeamMember4>);
static_assert(!Kokkos::is_team_handle_v<InvalidTeamMember5>);
static_assert(!Kokkos::is_team_handle_v<InvalidTeamMember6>);
static_assert(!Kokkos::is_team_handle_v<InvalidTeamMember7>);
static_assert(!Kokkos::is_team_handle_v<InvalidTeamMember8>);
static_assert(!Kokkos::is_team_handle_v<InvalidTeamMember9>);
static_assert(!Kokkos::is_team_handle_v<InvalidTeamMember10>);
static_assert(!Kokkos::is_team_handle_v<InvalidTeamMember11>);
static_assert(!Kokkos::is_team_handle_v<InvalidTeamMember12>);
static_assert(!Kokkos::is_team_handle_v<InvalidTeamMember13>);
static_assert(!Kokkos::is_team_handle_v<InvalidTeamMember14>);

static_assert(is_team_handle_complete_trait_check_v<ValidTeamMember>);
static_assert(!is_team_handle_complete_trait_check_v<InvalidTeamMember1>);
static_assert(!is_team_handle_complete_trait_check_v<InvalidTeamMember2>);
static_assert(!is_team_handle_complete_trait_check_v<InvalidTeamMember3>);
static_assert(!is_team_handle_complete_trait_check_v<InvalidTeamMember4>);
static_assert(!is_team_handle_complete_trait_check_v<InvalidTeamMember5>);
static_assert(!is_team_handle_complete_trait_check_v<InvalidTeamMember6>);
static_assert(!is_team_handle_complete_trait_check_v<InvalidTeamMember7>);
static_assert(!is_team_handle_complete_trait_check_v<InvalidTeamMember8>);
static_assert(!is_team_handle_complete_trait_check_v<InvalidTeamMember9>);
static_assert(!is_team_handle_complete_trait_check_v<InvalidTeamMember10>);
static_assert(!is_team_handle_complete_trait_check_v<InvalidTeamMember11>);
static_assert(!is_team_handle_complete_trait_check_v<InvalidTeamMember12>);
static_assert(!is_team_handle_complete_trait_check_v<InvalidTeamMember13>);
static_assert(!is_team_handle_complete_trait_check_v<InvalidTeamMember14>);

}  // end namespace TestIsTeamHandle
/*-------------------------------------------------
  end test for team_handle concept
  -------------------------------------------------*/

}  // namespace TestConcept
