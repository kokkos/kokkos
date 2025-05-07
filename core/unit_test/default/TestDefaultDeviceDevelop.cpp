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

#include <TestDefaultDeviceType_Category.hpp>

namespace Kokkos {
// Another attempt to insert dummy overload of parallel_for
// Doesn't work: if the range_policy function returns different types
// depending on the host vs device variant getting this failure:
//   terminate called after throwing an instance of 'std::runtime_error'
//     what():  cudaFuncGetAttributes(&attr, func) error(
//     cudaErrorInvalidResourceHandle): invalid resource handle
//     /ascldap/users/crtrott/Kokkos/kokkos/core/src/Cuda/Kokkos_Cuda_KernelLaunch.hpp:140

#if 0
  struct SinkPolicy {
    template<class Arg>
    KOKKOS_FUNCTION constexpr static bool sink(const Arg& arg) {
      return sizeof(Arg) > 0 && (&arg!=nullptr);
    }
    template<class ... Args>
    KOKKOS_FUNCTION explicit SinkPolicy(const Args& ... args) {
      (void) (sink(args) && ... && true);
    }
  };

  template<class ... Args>
  KOKKOS_FUNCTION
  void parallel_for(const SinkPolicy&, const Args& ...) {}
#endif
}  // namespace Kokkos

namespace Kokkos {
template <class Exec, class... Args>
  requires(Kokkos::is_execution_space_v<Exec> &&
           (!std::is_convertible_v<Args, size_t> && ... && true))
KOKKOS_INLINE_FUNCTION auto range_policy(const Exec& exec, size_t N,
                                         const Args&... args) {
  // This constructor doesn't work on device
  KOKKOS_IF_ON_HOST((return Kokkos::RangePolicy<Exec>(exec, 0, N, args...);))
  // Calling device callable RangePolicy ctor to not get "calling host function"
  // warning here If this guy doesn't return exactly the same type as above,
  // getting issues
  KOKKOS_IF_ON_DEVICE(
      (return Kokkos::RangePolicy<Exec>(Impl::InternalTag{}, exec);))

  // KOKKOS_IF_ON_DEVICE(( return SinkPolicy(args...); ))
  //(void) exec; (void) N; (void) (SinkPolicy::sink(args) && ... && true);
}

template <class Exec, class... Args>
  requires(!Kokkos::is_execution_space_v<Exec> &&
           (!std::is_convertible_v<Args, size_t> && ... && true))
KOKKOS_INLINE_FUNCTION auto range_policy(const Exec& exec, size_t N,
                                         const Args&... args) {
  return Kokkos::TeamVectorRange(exec, 0, N);
}
}  // namespace Kokkos

template <class Exec, class X, class Y>
KOKKOS_INLINE_FUNCTION void axpy(const Exec& exec, const X& x, const Y& y) {
  Kokkos::parallel_for(
      Kokkos::range_policy(exec, x.extent(0)),
      KOKKOS_LAMBDA(const int& i) { x(i) += y(i); });
}

void foo() {
  int N = 1000000;
  int M = 100;

  Kokkos::View<float*> VA("VA", N), VB("VB", N);
  Kokkos::View<float**> MA("MA", N / M, M), MB("MB", N / M, M);
  Kokkos::deep_copy(VA, 1);
  Kokkos::deep_copy(VB, 2);
  Kokkos::deep_copy(MA, 1);
  Kokkos::deep_copy(MB, 2);

  // Call axpy from host
  axpy(Kokkos::DefaultExecutionSpace(), VA, VB);

  using team_t = typename Kokkos::TeamPolicy<>::member_type;
  Kokkos::parallel_for(
      "TeamThing", Kokkos::TeamPolicy(N / M, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const team_t& team) {
        // call axpy from device with team handle
        axpy(team, Kokkos::subview(MA, team.league_rank(), Kokkos::ALL()),
             Kokkos::subview(MB, team.league_rank(), Kokkos::ALL()));
      });

  // check
  size_t result = 0;
  Kokkos::parallel_reduce(
      "Check1", VA.extent(0),
      KOKKOS_LAMBDA(int i, size_t& val) { val += VA(i); }, result);
  printf("%lu %lu\n", result, size_t(3) * VA.extent(0));
  ASSERT_EQ(result, size_t(3) * VA.extent(0));
  Kokkos::parallel_reduce(
      "Check2", MA.extent(0),
      KOKKOS_LAMBDA(int i, size_t& val) {
        for (int j = 0; j < MA.extent(1); j++) val += MA(i, j);
      },
      result);
  printf("%lu %lu\n", result, size_t(3) * MA.extent(0) * MA.extent(1));
  ASSERT_EQ(result, size_t(3) * MA.extent(0) * MA.extent(1));
}

namespace Test {

TEST(defaultdevicetype, development_test) { foo(); }

}  // namespace Test
