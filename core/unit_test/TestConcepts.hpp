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

#include <gtest/gtest.h>
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

namespace TestIsTeamHandle {

struct FakeScratchMemorySpace {};

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
  m1() m2() m3() m4() m5() m6() m7() m8() m9() m10() m11() m12()
};
struct InvalidTeamMember1 {
  /*m1()*/ m2() m3() m4() m5() m6() m7() m8() m9() m10() m11() m12()
};
struct InvalidTeamMember2 {
  m1() /*m2()*/ m3() m4() m5() m6() m7() m8() m9() m10() m11() m12()
};
struct InvalidTeamMember3 {
  m1() m2() /*m3()*/ m4() m5() m6() m7() m8() m9() m10() m11() m12()
};
struct InvalidTeamMember4 {
  m1() m2() m3() /*m4()*/ m5() m6() m7() m8() m9() m10() m11() m12()
};
struct InvalidTeamMember5 {
  m1() m2() m3() m4() /*m5()*/ m6() m7() m8() m9() m10() m11() m12()
};
struct InvalidTeamMember6 {
  m1() m2() m3() m4() m5() /*m6()*/ m7() m8() m9() m10() m11() m12()
};
struct InvalidTeamMember7 {
  m1() m2() m3() m4() m5() m6() /*m7()*/ m8() m9() m10() m11() m12()
};
struct InvalidTeamMember8 {
  m1() m2() m3() m4() m5() m6() m7() /*m8()*/ m9() m10() m11() m12()
};
struct InvalidTeamMember9 {
  m1() m2() m3() m4() m5() m6() m7() m8() /*m9()*/ m10() m11() m12()
};
struct InvalidTeamMember10 {
  m1() m2() m3() m4() m5() m6() m7() m8() m9() /*m10()*/ m11() m12()
};
struct InvalidTeamMember11 {
  m1() m2() m3() m4() m5() m6() m7() m8() m9() m10() /*m11()*/ m12()
};
struct InvalidTeamMember12 {
  m1() m2() m3() m4() m5() m6() m7() m8() m9() m10() m11() /*m12()*/
};

TEST(TEST_CATEGORY, team_handle_concept) {
  using space_t  = TEST_EXECSPACE;
  using policy_t = Kokkos::TeamPolicy<space_t>;
  using member_t = typename policy_t::member_type;

  static_assert(Kokkos::is_team_handle<member_t>::value, "");
  static_assert(Kokkos::is_team_handle_v<member_t>, "");
  static_assert(Kokkos::is_team_handle_v<member_t const>, "");

  static_assert(!Kokkos::is_team_handle_v<member_t &>, "");
  static_assert(!Kokkos::is_team_handle_v<member_t const &>, "");
  static_assert(!Kokkos::is_team_handle_v<member_t *>, "");
  static_assert(!Kokkos::is_team_handle_v<member_t const *>, "");
  static_assert(!Kokkos::is_team_handle_v<member_t *const>, "");

  /*
    disabling as follows:

    - OpenMPTARGET: due to this
    https://github.com/kokkos/kokkos/blob/2d6cbad7e079eb45ae69ac6a59929d9fcf10409a/core/src/OpenMPTarget/Kokkos_OpenMPTarget_Exec.hpp#L860

    - OpenACC: not supporting teams yet
   */
#if not defined KOKKOS_ENABLE_OPENMPTARGET && not defined KOKKOS_ENABLE_OPENACC
  static_assert(Kokkos::is_team_handle_v<ValidTeamMember>, "");
#endif

  static_assert(!Kokkos::is_team_handle_v<InvalidTeamMember1>, "");
  static_assert(!Kokkos::is_team_handle_v<InvalidTeamMember2>, "");
  static_assert(!Kokkos::is_team_handle_v<InvalidTeamMember3>, "");
  static_assert(!Kokkos::is_team_handle_v<InvalidTeamMember4>, "");
  static_assert(!Kokkos::is_team_handle_v<InvalidTeamMember5>, "");
  static_assert(!Kokkos::is_team_handle_v<InvalidTeamMember6>, "");
  static_assert(!Kokkos::is_team_handle_v<InvalidTeamMember7>, "");
  static_assert(!Kokkos::is_team_handle_v<InvalidTeamMember8>, "");
  static_assert(!Kokkos::is_team_handle_v<InvalidTeamMember9>, "");
  static_assert(!Kokkos::is_team_handle_v<InvalidTeamMember10>, "");
  static_assert(!Kokkos::is_team_handle_v<InvalidTeamMember11>, "");
  static_assert(!Kokkos::is_team_handle_v<InvalidTeamMember12>, "");
}
}  // end namespace TestIsTeamHandle

}  // namespace TestConcept
