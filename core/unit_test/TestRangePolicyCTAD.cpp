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
#include "Kokkos_Core_fwd.hpp"

#if !defined(KOKKOS_COMPILER_NVCC)

namespace {

template <class... Args>
using PolicyMaker = decltype(::Kokkos::RangePolicy(std::declval<Args>()...));

template <class Policy, class... Args>
inline constexpr bool IsSamePolicy =
    std::is_same_v<Policy, PolicyMaker<Args...>>;

#define KOKKOS_TEST_RANGE_POLICY(...) static_assert(IsSamePolicy<__VA_ARGS__>)

struct TestRangePolicyCTAD {
  struct ImplicitlyConvertibleToDefaultExecutionSpace {
    operator Kokkos::DefaultExecutionSpace() const {
      return Kokkos::DefaultExecutionSpace();
    }
  };
  static_assert(!Kokkos::is_execution_space_v<
                ImplicitlyConvertibleToDefaultExecutionSpace>);

  using des = Kokkos::DefaultExecutionSpace;
  using nes = ImplicitlyConvertibleToDefaultExecutionSpace;
  using i64 = int64_t;
  using i32 = int32_t;
  using cs  = Kokkos::ChunkSize;

  // RangePolicy()

  // Guard against GGC 8.4 bug
  // error: cannot deduce template arguments for ‘RangePolicy’ from ()
  // error: template argument 2 is invalid
#if !defined(KOKKOS_COMPILER_GNU) || (KOKKOS_COMPILER_GNU > 900)
  KOKKOS_TEST_RANGE_POLICY(Kokkos::RangePolicy<> /*, no argument */);
#endif

  // RangePolicy(index_type, index_type)

  KOKKOS_TEST_RANGE_POLICY(Kokkos::RangePolicy<>, i64, i64);
  KOKKOS_TEST_RANGE_POLICY(Kokkos::RangePolicy<>, i64, i32);
  KOKKOS_TEST_RANGE_POLICY(Kokkos::RangePolicy<>, i32, i64);
  KOKKOS_TEST_RANGE_POLICY(Kokkos::RangePolicy<>, i32, i32);

  // RangePolicy(index_type, index_type, Args...)

  KOKKOS_TEST_RANGE_POLICY(Kokkos::RangePolicy<>, i64, i64, cs);
  KOKKOS_TEST_RANGE_POLICY(Kokkos::RangePolicy<>, i64, i32, cs);
  KOKKOS_TEST_RANGE_POLICY(Kokkos::RangePolicy<>, i32, i64, cs);
  KOKKOS_TEST_RANGE_POLICY(Kokkos::RangePolicy<>, i32, i32, cs);

  // RangePolicy(execution_space, index_type, index_type)

  KOKKOS_TEST_RANGE_POLICY(Kokkos::RangePolicy<>, des, i64, i64);
  KOKKOS_TEST_RANGE_POLICY(Kokkos::RangePolicy<>, des, i32, i32);
  KOKKOS_TEST_RANGE_POLICY(Kokkos::RangePolicy<>, nes, i64, i64);
  KOKKOS_TEST_RANGE_POLICY(Kokkos::RangePolicy<>, nes, i32, i32);

  // RangePolicy(execution_space, index_type, index_type, Args...)

  KOKKOS_TEST_RANGE_POLICY(Kokkos::RangePolicy<>, des, i64, i64, cs);
  KOKKOS_TEST_RANGE_POLICY(Kokkos::RangePolicy<>, des, i32, i32, cs);
  KOKKOS_TEST_RANGE_POLICY(Kokkos::RangePolicy<>, nes, i64, i64, cs);
  KOKKOS_TEST_RANGE_POLICY(Kokkos::RangePolicy<>, nes, i32, i32, cs);
};  // TestRangePolicyCTAD struct

// To eliminate maybe_unused warning on some compilers
[[maybe_unused]] const Kokkos::DefaultExecutionSpace des =
    TestRangePolicyCTAD::ImplicitlyConvertibleToDefaultExecutionSpace();

}  // namespace

#endif
