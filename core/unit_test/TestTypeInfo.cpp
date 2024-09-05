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

#include <type_traits>

namespace {

struct Foo {};
using FooAlias = Foo;
enum Bar { BAR_0, BAR_1, BAR_2 };
union Baz {
  int i;
  float f;
};

[[maybe_unused]] auto func = [](int) {};  // < line 31
//                           ^  column 30

using Lambda = decltype(func);

// clang-format off
#if defined(__clang__)
static_assert(Kokkos::TypeInfo<Foo>::name()      == "(anonymous namespace)::Foo");
static_assert(Kokkos::TypeInfo<FooAlias>::name() == "(anonymous namespace)::Foo");
static_assert(Kokkos::TypeInfo<Bar>::name()      == "(anonymous namespace)::Bar");
static_assert(Kokkos::TypeInfo<Baz>::name()      == "(anonymous namespace)::Baz");
static_assert(Kokkos::TypeInfo<Lambda>::name()   == "(anonymous namespace)::(lambda at "  __FILE__  ":31:30)");
#elif defined(__EDG__)
static_assert(Kokkos::TypeInfo<Foo>::name()      == "<unnamed>::Foo");
static_assert(Kokkos::TypeInfo<FooAlias>::name() == "<unnamed>::Foo");
static_assert(Kokkos::TypeInfo<Bar>::name()      == "<unnamed>::Bar");
static_assert(Kokkos::TypeInfo<Baz>::name()      == "<unnamed>::Baz");
static_assert(Kokkos::TypeInfo<Lambda>::name()   == "lambda [](int)->void");
#elif defined(__GNUC__)
static_assert(Kokkos::TypeInfo<Foo>::name()      == "{anonymous}::Foo");
static_assert(Kokkos::TypeInfo<FooAlias>::name() == "{anonymous}::Foo");
static_assert(Kokkos::TypeInfo<Bar>::name()      == "{anonymous}::Bar");
static_assert(Kokkos::TypeInfo<Baz>::name()      == "{anonymous}::Baz");
static_assert(Kokkos::TypeInfo<Lambda>::name()   == "{anonymous}::<lambda(int)>");

#elif defined(_MSC_VER)
static_assert(Kokkos::TypeInfo<Foo>::name()      == "struct `anonymous-namespace'::Foo");
static_assert(Kokkos::TypeInfo<FooAlias>::name() == "struct `anonymous-namespace'::Foo");
static_assert(Kokkos::TypeInfo<Bar>::name()      == "enum `anonymous-namespace'::Bar");
static_assert(Kokkos::TypeInfo<Baz>::name()      == "union `anonymous-namespace'::Baz");
#ifndef KOKKOS_ENABLE_CXX17
static_assert(Kokkos::TypeInfo<Lambda>::name().starts_with("class `anonymous-namespace'::<lambda_");
// underscore followed by some 32-bit hash that seems sensitive to the content of the current source code file
static_assert(Kokkos::TypeInfo<Lambda>::name().ends_with(">");
#endif
#else
#error how did I ended up here?
#endif
// clang-format on

using T = void;
static_assert(!std::is_default_constructible_v<Kokkos::TypeInfo<T>>);

}  // namespace
