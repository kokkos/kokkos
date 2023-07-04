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
    operator Kokkos::DefaultExecutionSpace() const;
  };
  static_assert(!Kokkos::is_execution_space_v<
                ImplicitlyConvertibleToDefaultExecutionSpace>);

  static inline Kokkos::DefaultExecutionSpace des;
  static inline ImplicitlyConvertibleToDefaultExecutionSpace notEs;
  static inline SomeExecutionSpace ses;
  static inline TEST_EXECSPACE es;

  static inline int t[5];
  static inline int64_t tt[5];
  static inline Kokkos::Array<int64_t, 3> a;
  static inline Kokkos::Array<int64_t, 2> aa;

  static inline Kokkos::MDRangePolicy<Kokkos::Rank<2>> p2;

  // MDRangePolicy copy/move

  static_assert(std::is_same_v<Kokkos::MDRangePolicy<Kokkos::Rank<2>>,
                               decltype(Kokkos::MDRangePolicy(p2))>);
  static_assert(std::is_same_v<Kokkos::MDRangePolicy<Kokkos::Rank<2>>,
                               decltype(Kokkos::MDRangePolicy(std::move(p2)))>);

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

  using MDRangePolicyTestExecSpaceCArray = std::conditional_t<
      std::is_same_v<TEST_EXECSPACE, Kokkos::DefaultExecutionSpace>,
      Kokkos::MDRangePolicy<Kokkos::Rank<std::size(t)>>,
      Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<std::size(t)>>>;
  static_assert(std::is_same_v<MDRangePolicyTestExecSpaceCArray,
                               decltype(Kokkos::MDRangePolicy(es, t, t))>);

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

  using MDRangePolicyTestExecSpaceKokkosArray = std::conditional_t<
      std::is_same_v<TEST_EXECSPACE, Kokkos::DefaultExecutionSpace>,
      Kokkos::MDRangePolicy<Kokkos::Rank<std::size(a)>>,
      Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<std::size(a)>>>;
  static_assert(std::is_same_v<MDRangePolicyTestExecSpaceKokkosArray,
                               decltype(Kokkos::MDRangePolicy(es, a, a))>);
  static_assert(std::is_same_v<MDRangePolicyTestExecSpaceKokkosArray,
                               decltype(Kokkos::MDRangePolicy(es, a, a, aa))>);
};

}  // namespace
