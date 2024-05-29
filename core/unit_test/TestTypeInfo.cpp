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

#include <Kokkos_TypeInfo.hpp>

namespace {

struct Foo {};
using FooAlias = Foo;
enum Bar { BAR_0, BAR_1, BAR_2 };
union Baz {
  int i;
  float f;
};

// clang-format off
#if defined(__clang__)
static_assert(Kokkos::TypeInfo<Foo>::name()      == "(anonymous namespace)::Foo");
static_assert(Kokkos::TypeInfo<FooAlias>::name() == "(anonymous namespace)::Foo");
static_assert(Kokkos::TypeInfo<Bar>::name()      == "(anonymous namespace)::Bar");
static_assert(Kokkos::TypeInfo<Baz>::name()      == "(anonymous namespace)::Baz");
#elif defined(__GNUC__)
static_assert(Kokkos::TypeInfo<Foo>::name()      == "(anonymous)::Foo");
static_assert(Kokkos::TypeInfo<FooAlias>::name() == "(anonymous)::Foo");
static_assert(Kokkos::TypeInfo<Bar>::name()      == "(anonymous)::Bar");
static_assert(Kokkos::TypeInfo<Baz>::name()      == "(anonymous)::Baz");
#elif defined(_MSC_VER)
static_assert(Kokkos::TypeInfo<Foo>::name()      == "class `anonymous-namespace'::Foo");
static_assert(Kokkos::TypeInfo<FooAlias>::name() == "class `anonymous-namespace'::Foo");
static_assert(Kokkos::TypeInfo<Bar>::name()      == "enum `anonymous-namespace'::Bar");
static_assert(Kokkos::TypeInfo<Baz>::name()      == "union `anonymous-namespace'::Baz");
#else
#error how did I ended up here?
#endif
// clang-format on

}  // namespace
