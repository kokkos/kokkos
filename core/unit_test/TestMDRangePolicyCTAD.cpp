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

struct TestMDRangePolicyCTADs {
  struct SomeExecutionSpace {
    using execution_space = SomeExecutionSpace;
    using size_type       = size_t;
  };
  static_assert(Kokkos::is_execution_space_v<SomeExecutionSpace>);

  struct ImplicitlyConvertibleToDefaultExecutionSpace {
    [[maybe_unused]] operator Kokkos::DefaultExecutionSpace() const;
  };
  static_assert(!Kokkos::is_execution_space_v<
                ImplicitlyConvertibleToDefaultExecutionSpace>);

  [[maybe_unused]] static inline Kokkos::DefaultExecutionSpace des;
  [[maybe_unused]] static inline ImplicitlyConvertibleToDefaultExecutionSpace
      notEs;
  [[maybe_unused]] static inline SomeExecutionSpace ses;

  [[maybe_unused]] static inline int t[5];
  [[maybe_unused]] static inline int64_t tt[5];
  [[maybe_unused]] static inline Kokkos::Array<int64_t, 3> a;
  [[maybe_unused]] static inline Kokkos::Array<int64_t, 2> aa;

  // MDRangePolicy with C array parameters

  static_assert(
      std::is_same_v<Kokkos::MDRangePolicy<Kokkos::Rank<std::size(t)>>,
                     decltype(Kokkos::MDRangePolicy(t, t))>);
  static_assert(
      std::is_same_v<Kokkos::MDRangePolicy<Kokkos::Rank<std::size(t)>>,
                     decltype(Kokkos::MDRangePolicy(t, t, tt))>);
  static_assert(
      std::is_same_v<Kokkos::MDRangePolicy<Kokkos::Rank<std::size(t)>>,
                     decltype(Kokkos::MDRangePolicy(des, t, tt))>);
  static_assert(
      std::is_same_v<Kokkos::MDRangePolicy<Kokkos::Rank<std::size(t)>>,
                     decltype(Kokkos::MDRangePolicy(notEs, t, t))>);

  static_assert(
      std::is_same_v<
          Kokkos::MDRangePolicy<SomeExecutionSpace, Kokkos::Rank<std::size(t)>>,
          decltype(Kokkos::MDRangePolicy(ses, t, t))>);

  // MDRangePolicy with Kokkos::Array parameters

  static_assert(
      std::is_same_v<Kokkos::MDRangePolicy<Kokkos::Rank<std::size(a)>>,
                     decltype(Kokkos::MDRangePolicy(a, a))>);
  static_assert(
      std::is_same_v<Kokkos::MDRangePolicy<Kokkos::Rank<std::size(a)>>,
                     decltype(Kokkos::MDRangePolicy(a, a, aa))>);
  static_assert(
      std::is_same_v<Kokkos::MDRangePolicy<Kokkos::Rank<std::size(a)>>,
                     decltype(Kokkos::MDRangePolicy(des, a, a))>);
  static_assert(
      std::is_same_v<Kokkos::MDRangePolicy<Kokkos::Rank<std::size(a)>>,
                     decltype(Kokkos::MDRangePolicy(notEs, a, a))>);
  static_assert(
      std::is_same_v<Kokkos::MDRangePolicy<Kokkos::Rank<std::size(a)>>,
                     decltype(Kokkos::MDRangePolicy(des, a, a, aa))>);
  static_assert(
      std::is_same_v<Kokkos::MDRangePolicy<Kokkos::Rank<std::size(a)>>,
                     decltype(Kokkos::MDRangePolicy(notEs, a, a, aa))>);

  static_assert(
      std::is_same_v<
          Kokkos::MDRangePolicy<SomeExecutionSpace, Kokkos::Rank<std::size(a)>>,
          decltype(Kokkos::MDRangePolicy(ses, a, a))>);
  static_assert(
      std::is_same_v<
          Kokkos::MDRangePolicy<SomeExecutionSpace, Kokkos::Rank<std::size(a)>>,
          decltype(Kokkos::MDRangePolicy(ses, a, a, aa))>);
};

}  // namespace
