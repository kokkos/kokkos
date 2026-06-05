// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>

#ifndef KOKKOS_FUNCTION
static_assert(false, "KOKKOS_FUNCTION macro is not defined!");
#endif

#ifndef KOKKOS_INLINE_FUNCTION
static_assert(false, "KOKKOS_INLINE_FUNCTION macro is not defined!");
#endif

#ifndef KOKKOS_FORCEINLINE_FUNCTION
static_assert(false, "KOKKOS_FORCEINLINE_FUNCTION macro is not defined!");
#endif

#ifndef KOKKOS_RELOCATABLE_FUNCTION
static_assert(false, "KOKKOS_RELOCATABLE_FUNCTION macro is not defined!");
#endif

#if !defined(KOKKOS_INLINE_FUNCTION_DELETED)
static_assert(false, "KOKKOS_INLINE_FUNCTION_DELETED macro is not defined!");
#endif

#if !defined(KOKKOS_DEFAULTED_FUNCTION)
static_assert(false, "KOKKOS_DEFAULTED_FUNCTION macro is not defined!");
#endif

#ifndef KOKKOS_DEDUCTION_GUIDE
static_assert(false, "KOKKOS_DEDUCTION_GUIDE macro is not defined!");
#endif

#ifndef KOKKOS_LAMBDA
static_assert(false, "KOKKOS_LAMBDA macro is not defined!");
#endif

#ifndef KOKKOS_CLASS_LAMBDA
static_assert(false, "KOKKOS_CLASS_LAMBDA macro is not defined!");
#endif

namespace {

#define KOKKOS_TEST_FUNCTION_ANNOTATION(ANNOTATION)                         \
  struct Struct_##ANNOTATION {                                              \
    int m_val = 0;                                                          \
    static ANNOTATION constexpr int kokkos_function_static() { return 42; } \
    ANNOTATION static constexpr int static_kokkos_function() { return 42; } \
    ANNOTATION constexpr int kokkos_function() const { return m_val; }      \
  };                                                                        \
  ANNOTATION constexpr int Fun_##ANNOTATION() {                             \
    Struct_##ANNOTATION fun{42};                                            \
    return fun.kokkos_function();                                           \
  }                                                                         \
  static_assert(Struct_##ANNOTATION::static_kokkos_function() == 42);       \
  static_assert(Struct_##ANNOTATION::kokkos_function_static() == 42);       \
  static_assert(Fun_##ANNOTATION() == 42)

KOKKOS_TEST_FUNCTION_ANNOTATION(KOKKOS_FUNCTION);
KOKKOS_TEST_FUNCTION_ANNOTATION(KOKKOS_INLINE_FUNCTION);
KOKKOS_TEST_FUNCTION_ANNOTATION(KOKKOS_FORCEINLINE_FUNCTION);

template <class T>
struct Foo /* NOLINT(cppcoreguidelines-special-member-functions) */ {
  KOKKOS_INLINE_FUNCTION_DELETED Foo(Foo const&)            = delete;
  KOKKOS_INLINE_FUNCTION_DELETED Foo& operator-(Foo const&) = delete;
  KOKKOS_DEFAULTED_FUNCTION Foo()                           = default;
  T m_val;
  Foo(T val) : m_val(val) {}
};
template <class T>
KOKKOS_DEDUCTION_GUIDE Foo(T) -> Foo<T>;
[[maybe_unused]] auto FooDeduced = Foo(3.14);

struct Bar {
  int m_val = 3;
  KOKKOS_FUNCTION int fun() const {
    auto dec = KOKKOS_LAMBDA(int x) { return 2 * x; };
    auto lam = KOKKOS_CLASS_LAMBDA() { return m_val; };
    return dec(lam());
  }
};
[[maybe_unused]] auto Fun = Bar{}.fun();

}  // namespace
