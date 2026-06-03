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

#ifndef KOKKOS_FORCEINLINE_LAMBDA
static_assert(false, "KOKKOS_FORCEINLINE_LAMBDA macro is not defined!");
#endif

#ifndef KOKKOS_CLASS_FORCEINLINE_LAMBDA
static_assert(false, "KOKKOS_CLASS_FORCEINLINE_LAMBDA macro is not defined!");
#endif
