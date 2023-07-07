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

struct TestRangePolicyCTADs {
  struct SomeExecutionSpace {
    using execution_space = SomeExecutionSpace;
    using size_type       = size_t;
  };
  static_assert(Kokkos::is_execution_space_v<SomeExecutionSpace>);

  struct ImplicitlyConvertibleToDefaultExecutionSpace {
    operator Kokkos::DefaultExecutionSpace() const;
  };
  static_assert(!Kokkos::is_execution_space_v<
                ImplicitlyConvertibleToDefaultExecutionSpace>);

  [[maybe_unused]] static inline Kokkos::DefaultExecutionSpace des;
  [[maybe_unused]] static inline ImplicitlyConvertibleToDefaultExecutionSpace
      notEs;
  [[maybe_unused]] static inline SomeExecutionSpace ses;

  [[maybe_unused]] static inline int64_t i64;
  [[maybe_unused]] static inline int32_t i32;
  [[maybe_unused]] static inline Kokkos::ChunkSize cs{0};

  //  RangePolicy

  // Workaround for gcc 8.4 bug, as
  //    static_assert(std::is_same_v<Kokkos::RangePolicy<>,
  //                                 decltype(Kokkos::RangePolicy())>);
  // gives us:
  //    error: cannot deduce template arguments for ‘RangePolicy’ from ()
  //    error: template argument 2 is invalid
  [[maybe_unused]] static inline Kokkos::RangePolicy rp;
  static_assert(std::is_same_v<Kokkos::RangePolicy<>, decltype(rp)>);

  static_assert(std::is_same_v<Kokkos::RangePolicy<>,
                               decltype(Kokkos::RangePolicy(des, i64, i64))>);
  static_assert(std::is_same_v<Kokkos::RangePolicy<>,
                               decltype(Kokkos::RangePolicy(notEs, i64, i64))>);
  static_assert(std::is_same_v<Kokkos::RangePolicy<SomeExecutionSpace>,
                               decltype(Kokkos::RangePolicy(ses, i64, i64))>);
  static_assert(std::is_same_v<Kokkos::RangePolicy<>,
                               decltype(Kokkos::RangePolicy(des, i32, i32))>);
  static_assert(std::is_same_v<Kokkos::RangePolicy<>,
                               decltype(Kokkos::RangePolicy(notEs, i32, i32))>);
  static_assert(std::is_same_v<Kokkos::RangePolicy<SomeExecutionSpace>,
                               decltype(Kokkos::RangePolicy(ses, i32, i32))>);

  // RangePolicy(index_type, index_type)

  static_assert(std::is_same_v<Kokkos::RangePolicy<>,
                               decltype(Kokkos::RangePolicy(i64, i64))>);
  static_assert(std::is_same_v<Kokkos::RangePolicy<>,
                               decltype(Kokkos::RangePolicy(i32, i32))>);

  // RangePolicy(execution_space, index_type, index_type, Args...)

  static_assert(
      std::is_same_v<Kokkos::RangePolicy<>,
                     decltype(Kokkos::RangePolicy(des, i64, i64, cs))>);
  static_assert(
      std::is_same_v<Kokkos::RangePolicy<>,
                     decltype(Kokkos::RangePolicy(notEs, i64, i64, cs))>);
  static_assert(
      std::is_same_v<Kokkos::RangePolicy<SomeExecutionSpace>,
                     decltype(Kokkos::RangePolicy(ses, i64, i64, cs))>);
  static_assert(
      std::is_same_v<Kokkos::RangePolicy<>,
                     decltype(Kokkos::RangePolicy(des, i32, i32, cs))>);
  static_assert(
      std::is_same_v<Kokkos::RangePolicy<>,
                     decltype(Kokkos::RangePolicy(notEs, i32, i32, cs))>);
  static_assert(
      std::is_same_v<Kokkos::RangePolicy<SomeExecutionSpace>,
                     decltype(Kokkos::RangePolicy(ses, i32, i32, cs))>);

  // RangePolicy(index_type, index_type, Args...)

  static_assert(std::is_same_v<Kokkos::RangePolicy<>,
                               decltype(Kokkos::RangePolicy(i64, i64, cs))>);
  static_assert(std::is_same_v<Kokkos::RangePolicy<>,
                               decltype(Kokkos::RangePolicy(i32, i32, cs))>);
};  // TestRangePolicyCTADs struct

}  // namespace
