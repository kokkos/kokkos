// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_ASSERT_HPP
#define KOKKOS_ASSERT_HPP

#include <Kokkos_Macros.hpp>
// Even for C++20 modules builds, Kokkos_Assert.hpp must be included
// (and can't be part of the kokkos.core module). That means it
// can't include other Kokkos headers to avoid symbol clashes.
#ifndef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
#include <Kokkos_Abort.hpp>
#endif

#if !defined(NDEBUG) || defined(KOKKOS_ENFORCE_CONTRACTS) || \
    defined(KOKKOS_ENABLE_DEBUG)
#define KOKKOS_EXPECTS(predicate, ...)                                         \
  {                                                                            \
    if (!bool(predicate)) {                                                    \
      Kokkos::printf(                                                          \
        "Kokkos contract violation:\n"                                         \
        "  Expected precondition `" #predicate "` evaluated false.\n"          \
        "Error at " KOKKOS_IMPL_TOSTRING(__FILE__) ":" KOKKOS_IMPL_TOSTRING(   \
          __LINE__) " \n"                                                      \
        __VA_OPT__(__VA_ARGS__));                                              \
      ::Kokkos::abort("");                                                     \
    }                                                                          \
  }
#define KOKKOS_ENSURES(predicate, ...)                                         \
  {                                                                            \
    if (!bool(predicate)) {                                                    \
      Kokkos::printf(                                                          \
        "Kokkos contract violation:\n"                                         \
        "  Ensured postcondition `" #predicate "` evaluated false.\n"          \
        "Error at " KOKKOS_IMPL_TOSTRING(__FILE__) ":" KOKKOS_IMPL_TOSTRING(   \
          __LINE__) " \n"                                                      \
        __VA_OPT__(__VA_ARGS__));                                              \
      ::Kokkos::abort("");                                                     \
    }                                                                          \
  }
#define KOKKOS_ASSERT(predicate, ...)                                          \
  {                                                                            \
    if (!bool(predicate)) {                                                    \
      Kokkos::printf(                                                          \
        "Kokkos contract violation:\n"                                         \
        "  Asserted condition `" #predicate "` evaluated false.\n"             \
        "Error at " KOKKOS_IMPL_TOSTRING(__FILE__) ":" KOKKOS_IMPL_TOSTRING(   \
          __LINE__) " \n"                                                      \
        __VA_OPT__(__VA_ARGS__));                                              \
      ::Kokkos::abort("");                                                     \
    }                                                                          \
  }

#else  // not debug mode
#define KOKKOS_EXPECTS(...)
#define KOKKOS_ENSURES(...)
#define KOKKOS_ASSERT(...)
#endif  // end debug mode ifdefs

#endif /* #ifndef KOKKOS_ASSERT_HPP */
