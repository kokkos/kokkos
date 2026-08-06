// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_HOST_EXP_ITERATE_TILE_INT_HPP
#define KOKKOS_HOST_EXP_ITERATE_TILE_INT_HPP

#include <Kokkos_Array.hpp>
#include <Kokkos_Macros.hpp>

#include <limits>

#if defined(KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION) && \
    defined(KOKKOS_ENABLE_PRAGMA_IVDEP) && !defined(__CUDA_ARCH__)
#define KOKKOS_MDRANGE_IVDEP
#endif

#if defined(KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION) && \
    defined(KOKKOS_ENABLE_PRAGMA_IVDEP) && !defined(__CUDA_ARCH__)
#define KOKKOS_MDRANGE_IVDEP_INNERMOST_LOOP
#endif

#ifdef KOKKOS_MDRANGE_IVDEP_INNERMOST_LOOP
#if defined(__clang__)
#define KOKKOS_ENABLE_IVDEP_INNERMOST_LOOP \
  _Pragma("clang loop vectorize(assume_safety)")
#elif defined(KOKKOS_COMPILER_GNU) && (KOKKOS_COMPILER_GNU >= 1150)
#define KOKKOS_ENABLE_IVDEP_INNERMOST_LOOP _Pragma("GCC ivdep")
#elif defined(_MSC_VER)
#define KOKKOS_ENABLE_IVDEP_INNERMOST_LOOP
#else
#define KOKKOS_ENABLE_IVDEP_INNERMOST_LOOP _Pragma("ivdep")
#endif
#else
#define KOKKOS_ENABLE_IVDEP_INNERMOST_LOOP
#endif

namespace Kokkos {
namespace Impl {

// parallel_for, non-tagged

// LayoutRight
// d = 0 to start
#define KOKKOS_IMPL_PRECOMP_LOOP_R_1(func, type, extent_st, extent_en, d, ...) \
  KOKKOS_ENABLE_IVDEP_INNERMOST_LOOP                                           \
  for (type i0 = static_cast<type>(extent_st[d]);                              \
       i0 < static_cast<type>(extent_en[d]); ++i0) {                           \
    KOKKOS_IMPL_APPLY(func, __VA_ARGS__, i0)                                   \
  }

#define KOKKOS_IMPL_PRECOMP_LOOP_R_2(func, type, extent_st, extent_en, d, ...) \
  for (type i1 = static_cast<type>(extent_st[d]);                              \
       i1 < static_cast<type>(extent_en[d]); ++i1) {                           \
    KOKKOS_IMPL_PRECOMP_LOOP_R_1(func, type, extent_st, extent_en, d + 1,      \
                                 __VA_ARGS__, i1)                              \
  }

#define KOKKOS_IMPL_PRECOMP_LOOP_R_3(func, type, extent_st, extent_en, d, ...) \
  for (type i2 = static_cast<type>(extent_st[d]);                              \
       i2 < static_cast<type>(extent_en[d]); ++i2) {                           \
    KOKKOS_IMPL_PRECOMP_LOOP_R_2(func, type, extent_st, extent_en, d + 1,      \
                                 __VA_ARGS__, i2)                              \
  }

#define KOKKOS_IMPL_PRECOMP_LOOP_R_4(func, type, extent_st, extent_en, d, ...) \
  for (type i3 = static_cast<type>(extent_st[d]);                              \
       i3 < static_cast<type>(extent_en[d]); ++i3) {                           \
    KOKKOS_IMPL_PRECOMP_LOOP_R_3(func, type, extent_st, extent_en, d + 1,      \
                                 __VA_ARGS__, i3)                              \
  }

#define KOKKOS_IMPL_PRECOMP_LOOP_R_5(func, type, extent_st, extent_en, d, ...) \
  for (type i4 = static_cast<type>(extent_st[d]);                              \
       i4 < static_cast<type>(extent_en[d]); ++i4) {                           \
    KOKKOS_IMPL_PRECOMP_LOOP_R_4(func, type, extent_st, extent_en, d + 1,      \
                                 __VA_ARGS__, i4)                              \
  }

#define KOKKOS_IMPL_PRECOMP_LOOP_R_6(func, type, extent_st, extent_en, d, ...) \
  for (type i5 = static_cast<type>(extent_st[d]);                              \
       i5 < static_cast<type>(extent_en[d]); ++i5) {                           \
    KOKKOS_IMPL_PRECOMP_LOOP_R_5(func, type, extent_st, extent_en, d + 1,      \
                                 __VA_ARGS__, i5)                              \
  }

#define KOKKOS_IMPL_PRECOMP_LOOP_R_7(func, type, extent_st, extent_en, d, ...) \
  for (type i6 = static_cast<type>(extent_st[d]);                              \
       i6 < static_cast<type>(extent_en[d]); ++i6) {                           \
    KOKKOS_IMPL_PRECOMP_LOOP_R_6(func, type, extent_st, extent_en, d + 1,      \
                                 __VA_ARGS__, i6)                              \
  }

#define KOKKOS_IMPL_PRECOMP_LOOP_R_8(func, type, extent_st, extent_en, d, ...) \
  for (type i7 = static_cast<type>(extent_st[d]);                              \
       i7 < static_cast<type>(extent_en[d]); ++i7) {                           \
    KOKKOS_IMPL_PRECOMP_LOOP_R_7(func, type, extent_st, extent_en, d + 1,      \
                                 __VA_ARGS__, i7)                              \
  }

// LayoutLeft
// d = rank-1 to start
#define KOKKOS_IMPL_PRECOMP_LOOP_L_1(func, type, extent_st, extent_en, d, ...) \
  KOKKOS_ENABLE_IVDEP_INNERMOST_LOOP                                           \
  for (type i0 = static_cast<type>(extent_st[d]);                              \
       i0 < static_cast<type>(extent_en[d]); ++i0) {                           \
    KOKKOS_IMPL_APPLY(func, i0, __VA_ARGS__)                                   \
  }

#define KOKKOS_IMPL_PRECOMP_LOOP_L_2(func, type, extent_st, extent_en, d, ...) \
  for (type i1 = static_cast<type>(extent_st[d]);                              \
       i1 < static_cast<type>(extent_en[d]); ++i1) {                           \
    KOKKOS_IMPL_PRECOMP_LOOP_L_1(func, type, extent_st, extent_en, d - 1, i1,  \
                                 __VA_ARGS__)                                  \
  }

#define KOKKOS_IMPL_PRECOMP_LOOP_L_3(func, type, extent_st, extent_en, d, ...) \
  for (type i2 = static_cast<type>(extent_st[d]);                              \
       i2 < static_cast<type>(extent_en[d]); ++i2) {                           \
    KOKKOS_IMPL_PRECOMP_LOOP_L_2(func, type, extent_st, extent_en, d - 1, i2,  \
                                 __VA_ARGS__)                                  \
  }

#define KOKKOS_IMPL_PRECOMP_LOOP_L_4(func, type, extent_st, extent_en, d, ...) \
  for (type i3 = static_cast<type>(extent_st[d]);                              \
       i3 < static_cast<type>(extent_en[d]); ++i3) {                           \
    KOKKOS_IMPL_PRECOMP_LOOP_L_3(func, type, extent_st, extent_en, d - 1, i3,  \
                                 __VA_ARGS__)                                  \
  }

#define KOKKOS_IMPL_PRECOMP_LOOP_L_5(func, type, extent_st, extent_en, d, ...) \
  for (type i4 = static_cast<type>(extent_st[d]);                              \
       i4 < static_cast<type>(extent_en[d]); ++i4) {                           \
    KOKKOS_IMPL_PRECOMP_LOOP_L_4(func, type, extent_st, extent_en, d - 1, i4,  \
                                 __VA_ARGS__)                                  \
  }

#define KOKKOS_IMPL_PRECOMP_LOOP_L_6(func, type, extent_st, extent_en, d, ...) \
  for (type i5 = static_cast<type>(extent_st[d]);                              \
       i5 < static_cast<type>(extent_en[d]); ++i5) {                           \
    KOKKOS_IMPL_PRECOMP_LOOP_L_5(func, type, extent_st, extent_en, d - 1, i5,  \
                                 __VA_ARGS__)                                  \
  }

#define KOKKOS_IMPL_PRECOMP_LOOP_L_7(func, type, extent_st, extent_en, d, ...) \
  for (type i6 = static_cast<type>(extent_st[d]);                              \
       i6 < static_cast<type>(extent_en[d]); ++i6) {                           \
    KOKKOS_IMPL_PRECOMP_LOOP_L_6(func, type, extent_st, extent_en, d - 1, i6,  \
                                 __VA_ARGS__)                                  \
  }

#define KOKKOS_IMPL_PRECOMP_LOOP_L_8(func, type, extent_st, extent_en, d, ...) \
  for (type i7 = static_cast<type>(extent_st[d]);                              \
       i7 < static_cast<type>(extent_en[d]); ++i7) {                           \
    KOKKOS_IMPL_PRECOMP_LOOP_L_7(func, type, extent_st, extent_en, d - 1, i7,  \
                                 __VA_ARGS__)                                  \
  }

// Left vs Right
// TODO: rank not necessary to pass through, can hardcode the values
#define KOKKOS_IMPL_PRECOMP_LOOP_LAYOUT_1(func, type, is_left, extent_st, \
                                          extent_en, rank)                \
  KOKKOS_ENABLE_IVDEP_INNERMOST_LOOP                                      \
  for (type i0 = static_cast<type>(extent_st[0]);                         \
       i0 < static_cast<type>(extent_en[0]); ++i0) {                      \
    KOKKOS_IMPL_APPLY(func, i0)                                           \
  }

#define KOKKOS_IMPL_PRECOMP_LOOP_LAYOUT_2(func, type, is_left, extent_st,      \
                                          extent_en, rank)                     \
  if (is_left) {                                                               \
    for (type i1 = static_cast<type>(extent_st[rank - 1]);                     \
         i1 < static_cast<type>(extent_en[rank - 1]); ++i1) {                  \
      KOKKOS_IMPL_PRECOMP_LOOP_L_1(func, type, extent_st, extent_en, rank - 2, \
                                   i1)                                         \
    }                                                                          \
  } else {                                                                     \
    for (type i1 = static_cast<type>(extent_st[0]);                            \
         i1 < static_cast<type>(extent_en[0]); ++i1) {                         \
      KOKKOS_IMPL_PRECOMP_LOOP_R_1(func, type, extent_st, extent_en, 1, i1)    \
    }                                                                          \
  }

#define KOKKOS_IMPL_PRECOMP_LOOP_LAYOUT_3(func, type, is_left, extent_st,      \
                                          extent_en, rank)                     \
  if (is_left) {                                                               \
    for (type i2 = static_cast<type>(extent_st[rank - 1]);                     \
         i2 < static_cast<type>(extent_en[rank - 1]); ++i2) {                  \
      KOKKOS_IMPL_PRECOMP_LOOP_L_2(func, type, extent_st, extent_en, rank - 2, \
                                   i2)                                         \
    }                                                                          \
  } else {                                                                     \
    for (type i2 = static_cast<type>(extent_st[0]);                            \
         i2 < static_cast<type>(extent_en[0]); ++i2) {                         \
      KOKKOS_IMPL_PRECOMP_LOOP_R_2(func, type, extent_st, extent_en, 1, i2)    \
    }                                                                          \
  }

#define KOKKOS_IMPL_PRECOMP_LOOP_LAYOUT_4(func, type, is_left, extent_st,      \
                                          extent_en, rank)                     \
  if (is_left) {                                                               \
    for (type i3 = static_cast<type>(extent_st[rank - 1]);                     \
         i3 < static_cast<type>(extent_en[rank - 1]); ++i3) {                  \
      KOKKOS_IMPL_PRECOMP_LOOP_L_3(func, type, extent_st, extent_en, rank - 2, \
                                   i3)                                         \
    }                                                                          \
  } else {                                                                     \
    for (type i3 = static_cast<type>(extent_st[0]);                            \
         i3 < static_cast<type>(extent_en[0]); ++i3) {                         \
      KOKKOS_IMPL_PRECOMP_LOOP_R_3(func, type, extent_st, extent_en, 1, i3)    \
    }                                                                          \
  }

#define KOKKOS_IMPL_PRECOMP_LOOP_LAYOUT_5(func, type, is_left, extent_st,      \
                                          extent_en, rank)                     \
  if (is_left) {                                                               \
    for (type i4 = static_cast<type>(extent_st[rank - 1]);                     \
         i4 < static_cast<type>(extent_en[rank - 1]); ++i4) {                  \
      KOKKOS_IMPL_PRECOMP_LOOP_L_4(func, type, extent_st, extent_en, rank - 2, \
                                   i4)                                         \
    }                                                                          \
  } else {                                                                     \
    for (type i4 = static_cast<type>(extent_st[0]);                            \
         i4 < static_cast<type>(extent_en[0]); ++i4) {                         \
      KOKKOS_IMPL_PRECOMP_LOOP_R_4(func, type, extent_st, extent_en, 1, i4)    \
    }                                                                          \
  }

#define KOKKOS_IMPL_PRECOMP_LOOP_LAYOUT_6(func, type, is_left, extent_st,      \
                                          extent_en, rank)                     \
  if (is_left) {                                                               \
    for (type i5 = static_cast<type>(extent_st[rank - 1]);                     \
         i5 < static_cast<type>(extent_en[rank - 1]); ++i5) {                  \
      KOKKOS_IMPL_PRECOMP_LOOP_L_5(func, type, extent_st, extent_en, rank - 2, \
                                   i5)                                         \
    }                                                                          \
  } else {                                                                     \
    for (type i5 = static_cast<type>(extent_st[0]);                            \
         i5 < static_cast<type>(extent_en[0]); ++i5) {                         \
      KOKKOS_IMPL_PRECOMP_LOOP_R_5(func, type, extent_st, extent_en, 1, i5)    \
    }                                                                          \
  }

#define KOKKOS_IMPL_PRECOMP_LOOP_LAYOUT_7(func, type, is_left, extent_st,      \
                                          extent_en, rank)                     \
  if (is_left) {                                                               \
    for (type i6 = static_cast<type>(extent_st[rank - 1]);                     \
         i6 < static_cast<type>(extent_en[rank - 1]); ++i6) {                  \
      KOKKOS_IMPL_PRECOMP_LOOP_L_6(func, type, extent_st, extent_en, rank - 2, \
                                   i6)                                         \
    }                                                                          \
  } else {                                                                     \
    for (type i6 = static_cast<type>(extent_st[0]);                            \
         i6 < static_cast<type>(extent_en[0]); ++i6) {                         \
      KOKKOS_IMPL_PRECOMP_LOOP_R_6(func, type, extent_st, extent_en, 1, i6)    \
    }                                                                          \
  }

#define KOKKOS_IMPL_PRECOMP_LOOP_LAYOUT_8(func, type, is_left, extent_st,      \
                                          extent_en, rank)                     \
  if (is_left) {                                                               \
    for (type i7 = static_cast<type>(extent_st[rank - 1]);                     \
         i7 < static_cast<type>(extent_en[rank - 1]); ++i7) {                  \
      KOKKOS_IMPL_PRECOMP_LOOP_L_7(func, type, extent_st, extent_en, rank - 2, \
                                   i7)                                         \
    }                                                                          \
  } else {                                                                     \
    for (type i7 = static_cast<type>(extent_st[0]);                            \
         i7 < static_cast<type>(extent_en[0]); ++i7) {                         \
      KOKKOS_IMPL_PRECOMP_LOOP_R_7(func, type, extent_st, extent_en, 1, i7)    \
    }                                                                          \
  }

// tagged macros

// LayoutRight
// d = 0 to start
#define KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_R_1(tag, func, type, extent_st, \
                                            extent_en, d, ...)          \
  KOKKOS_ENABLE_IVDEP_INNERMOST_LOOP                                    \
  for (type i0 = static_cast<type>(extent_st[d]);                       \
       i0 < static_cast<type>(extent_en[d]); ++i0) {                    \
    KOKKOS_IMPL_TAGGED_APPLY(tag, func, __VA_ARGS__, i0)                \
  }

#define KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_R_2(tag, func, type, extent_st,        \
                                            extent_en, d, ...)                 \
  for (type i1 = static_cast<type>(extent_st[d]);                              \
       i1 < static_cast<type>(extent_en[d]); ++i1) {                           \
    KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_R_1(tag, func, type, extent_st, extent_en, \
                                        d + 1, __VA_ARGS__, i1)                \
  }

#define KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_R_3(tag, func, type, extent_st,        \
                                            extent_en, d, ...)                 \
  for (type i2 = static_cast<type>(extent_st[d]);                              \
       i2 < static_cast<type>(extent_en[d]); ++i2) {                           \
    KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_R_2(tag, func, type, extent_st, extent_en, \
                                        d + 1, __VA_ARGS__, i2)                \
  }

#define KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_R_4(tag, func, type, extent_st,        \
                                            extent_en, d, ...)                 \
  for (type i3 = static_cast<type>(extent_st[d]);                              \
       i3 < static_cast<type>(extent_en[d]); ++i3) {                           \
    KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_R_3(tag, func, type, extent_st, extent_en, \
                                        d + 1, __VA_ARGS__, i3)                \
  }

#define KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_R_5(tag, func, type, extent_st,        \
                                            extent_en, d, ...)                 \
  for (type i4 = static_cast<type>(extent_st[d]);                              \
       i4 < static_cast<type>(extent_en[d]); ++i4) {                           \
    KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_R_4(tag, func, type, extent_st, extent_en, \
                                        d + 1, __VA_ARGS__, i4)                \
  }

#define KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_R_6(tag, func, type, extent_st,        \
                                            extent_en, d, ...)                 \
  for (type i5 = static_cast<type>(extent_st[d]);                              \
       i5 < static_cast<type>(extent_en[d]); ++i5) {                           \
    KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_R_5(tag, func, type, extent_st, extent_en, \
                                        d + 1, __VA_ARGS__, i5)                \
  }

#define KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_R_7(tag, func, type, extent_st,        \
                                            extent_en, d, ...)                 \
  for (type i6 = static_cast<type>(extent_st[d]);                              \
       i6 < static_cast<type>(extent_en[d]); ++i6) {                           \
    KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_R_6(tag, func, type, extent_st, extent_en, \
                                        d + 1, __VA_ARGS__, i6)                \
  }

#define KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_R_8(tag, func, type, extent_st,        \
                                            extent_en, d, ...)                 \
  for (type i7 = static_cast<type>(extent_st[d]);                              \
       i7 < static_cast<type>(extent_en[d]); ++i7) {                           \
    KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_R_7(tag, func, type, extent_st, extent_en, \
                                        d + 1, __VA_ARGS__, i7)                \
  }

// LayoutLeft
// d = rank-1 to start
#define KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_L_1(tag, func, type, extent_st, \
                                            extent_en, d, ...)          \
  KOKKOS_ENABLE_IVDEP_INNERMOST_LOOP                                    \
  for (type i0 = static_cast<type>(extent_st[d]);                       \
       i0 < static_cast<type>(extent_en[d]); ++i0) {                    \
    KOKKOS_IMPL_TAGGED_APPLY(tag, func, i0, __VA_ARGS__)                \
  }

#define KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_L_2(tag, func, type, extent_st,        \
                                            extent_en, d, ...)                 \
  for (type i1 = static_cast<type>(extent_st[d]);                              \
       i1 < static_cast<type>(extent_en[d]); ++i1) {                           \
    KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_L_1(tag, func, type, extent_st, extent_en, \
                                        d - 1, i1, __VA_ARGS__)                \
  }

#define KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_L_3(tag, func, type, extent_st,        \
                                            extent_en, d, ...)                 \
  for (type i2 = static_cast<type>(extent_st[d]);                              \
       i2 < static_cast<type>(extent_en[d]); ++i2) {                           \
    KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_L_2(tag, func, type, extent_st, extent_en, \
                                        d - 1, i2, __VA_ARGS__)                \
  }

#define KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_L_4(tag, func, type, extent_st,        \
                                            extent_en, d, ...)                 \
  for (type i3 = static_cast<type>(extent_st[d]);                              \
       i3 < static_cast<type>(extent_en[d]); ++i3) {                           \
    KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_L_3(tag, func, type, extent_st, extent_en, \
                                        d - 1, i3, __VA_ARGS__)                \
  }

#define KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_L_5(tag, func, type, extent_st,        \
                                            extent_en, d, ...)                 \
  for (type i4 = static_cast<type>(extent_st[d]);                              \
       i4 < static_cast<type>(extent_en[d]); ++i4) {                           \
    KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_L_4(tag, func, type, extent_st, extent_en, \
                                        d - 1, i4, __VA_ARGS__)                \
  }

#define KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_L_6(tag, func, type, extent_st,        \
                                            extent_en, d, ...)                 \
  for (type i5 = static_cast<type>(extent_st[d]);                              \
       i5 < static_cast<type>(extent_en[d]); ++i5) {                           \
    KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_L_5(tag, func, type, extent_st, extent_en, \
                                        d - 1, i5, __VA_ARGS__)                \
  }

#define KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_L_7(tag, func, type, extent_st,        \
                                            extent_en, d, ...)                 \
  for (type i6 = static_cast<type>(extent_st[d]);                              \
       i6 < static_cast<type>(extent_en[d]); ++i6) {                           \
    KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_L_6(tag, func, type, extent_st, extent_en, \
                                        d - 1, i6, __VA_ARGS__)                \
  }

#define KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_L_8(tag, func, type, extent_st,        \
                                            extent_en, d, ...)                 \
  for (type i7 = static_cast<type>(extent_st[d]);                              \
       i7 < static_cast<type>(extent_en[d]); ++i7) {                           \
    KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_L_7(tag, func, type, extent_st, extent_en, \
                                        d - 1, i7, __VA_ARGS__)                \
  }

// Left vs Right
// TODO: rank not necessary to pass through, can hardcode the values
#define KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_LAYOUT_1(tag, func, type, is_left,   \
                                                 extent_st, extent_en, rank) \
  KOKKOS_ENABLE_IVDEP_INNERMOST_LOOP                                         \
  for (type i0 = static_cast<type>(extent_st[0]);                            \
       i0 < static_cast<type>(extent_en[0]); ++i0) {                         \
    KOKKOS_IMPL_TAGGED_APPLY(tag, func, i0)                                  \
  }

#define KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_LAYOUT_2(tag, func, type, is_left,   \
                                                 extent_st, extent_en, rank) \
  if (is_left) {                                                             \
    for (type i1 = static_cast<type>(extent_st[rank - 1]);                   \
         i1 < static_cast<type>(extent_en[rank - 1]); ++i1) {                \
      KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_L_1(tag, func, type, extent_st,        \
                                          extent_en, rank - 2, i1)           \
    }                                                                        \
  } else {                                                                   \
    for (type i1 = static_cast<type>(extent_st[0]);                          \
         i1 < static_cast<type>(extent_en[0]); ++i1) {                       \
      KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_R_1(tag, func, type, extent_st,        \
                                          extent_en, 1, i1)                  \
    }                                                                        \
  }

#define KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_LAYOUT_3(tag, func, type, is_left,   \
                                                 extent_st, extent_en, rank) \
  if (is_left) {                                                             \
    for (type i2 = static_cast<type>(extent_st[rank - 1]);                   \
         i2 < static_cast<type>(extent_en[rank - 1]); ++i2) {                \
      KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_L_2(tag, func, type, extent_st,        \
                                          extent_en, rank - 2, i2)           \
    }                                                                        \
  } else {                                                                   \
    for (type i2 = static_cast<type>(extent_st[0]);                          \
         i2 < static_cast<type>(extent_en[0]); ++i2) {                       \
      KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_R_2(tag, func, type, extent_st,        \
                                          extent_en, 1, i2)                  \
    }                                                                        \
  }

#define KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_LAYOUT_4(tag, func, type, is_left,   \
                                                 extent_st, extent_en, rank) \
  if (is_left) {                                                             \
    for (type i3 = static_cast<type>(extent_st[rank - 1]);                   \
         i3 < static_cast<type>(extent_en[rank - 1]); ++i3) {                \
      KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_L_3(tag, func, type, extent_st,        \
                                          extent_en, rank - 2, i3)           \
    }                                                                        \
  } else {                                                                   \
    for (type i3 = static_cast<type>(extent_st[0]);                          \
         i3 < static_cast<type>(extent_en[0]); ++i3) {                       \
      KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_R_3(tag, func, type, extent_st,        \
                                          extent_en, 1, i3)                  \
    }                                                                        \
  }

#define KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_LAYOUT_5(tag, func, type, is_left,   \
                                                 extent_st, extent_en, rank) \
  if (is_left) {                                                             \
    for (type i4 = static_cast<type>(extent_st[rank - 1]);                   \
         i4 < static_cast<type>(extent_en[rank - 1]); ++i4) {                \
      KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_L_4(tag, func, type, extent_st,        \
                                          extent_en, rank - 2, i4)           \
    }                                                                        \
  } else {                                                                   \
    for (type i4 = static_cast<type>(extent_st[0]);                          \
         i4 < static_cast<type>(extent_en[0]); ++i4) {                       \
      KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_R_4(tag, func, type, extent_st,        \
                                          extent_en, 1, i4)                  \
    }                                                                        \
  }

#define KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_LAYOUT_6(tag, func, type, is_left,   \
                                                 extent_st, extent_en, rank) \
  if (is_left) {                                                             \
    for (type i5 = static_cast<type>(extent_st[rank - 1]);                   \
         i5 < static_cast<type>(extent_en[rank - 1]); ++i5) {                \
      KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_L_5(tag, func, type, extent_st,        \
                                          extent_en, rank - 2, i5)           \
    }                                                                        \
  } else {                                                                   \
    for (type i5 = static_cast<type>(extent_st[0]);                          \
         i5 < static_cast<type>(extent_en[0]); ++i5) {                       \
      KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_R_5(tag, func, type, extent_st,        \
                                          extent_en, 1, i5)                  \
    }                                                                        \
  }

#define KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_LAYOUT_7(tag, func, type, is_left,   \
                                                 extent_st, extent_en, rank) \
  if (is_left) {                                                             \
    for (type i6 = static_cast<type>(extent_st[rank - 1]);                   \
         i6 < static_cast<type>(extent_en[rank - 1]); ++i6) {                \
      KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_L_6(tag, func, type, extent_st,        \
                                          extent_en, rank - 2, i6)           \
    }                                                                        \
  } else {                                                                   \
    for (type i6 = static_cast<type>(extent_st[0]);                          \
         i6 < static_cast<type>(extent_en[0]); ++i6) {                       \
      KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_R_6(tag, func, type, extent_st,        \
                                          extent_en, 1, i6)                  \
    }                                                                        \
  }

#define KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_LAYOUT_8(tag, func, type, is_left,   \
                                                 extent_st, extent_en, rank) \
  if (is_left) {                                                             \
    for (type i7 = static_cast<type>(extent_st[rank - 1]);                   \
         i7 < static_cast<type>(extent_en[rank - 1]); ++i7) {                \
      KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_L_7(tag, func, type, extent_st,        \
                                          extent_en, rank - 2, i7)           \
    }                                                                        \
  } else {                                                                   \
    for (type i7 = static_cast<type>(extent_st[0]);                          \
         i7 < static_cast<type>(extent_en[0]); ++i7) {                       \
      KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_R_7(tag, func, type, extent_st,        \
                                          extent_en, 1, i7)                  \
    }                                                                        \
  }

// end tagged macros

// Structs for calling loops when loop extents are pre-computed
template <int Rank, bool IsLeft, typename IType, typename Tagged,
          typename Enable = void>
struct Tile_Loop_Type_PreComp;

template <bool IsLeft, typename IType>
struct Tile_Loop_Type_PreComp<1, IsLeft, IType, void, void> {
  template <typename Func, typename ExtentType>
  static void apply(Func const& func, ExtentType const& extent_st,
                    ExtentType const& extent_en) {
    KOKKOS_IMPL_PRECOMP_LOOP_LAYOUT_1(func, IType, IsLeft, extent_st, extent_en,
                                      1);
  }
};

template <bool IsLeft, typename IType>
struct Tile_Loop_Type_PreComp<2, IsLeft, IType, void, void> {
  template <typename Func, typename ExtentType>
  static void apply(Func const& func, ExtentType const& extent_st,
                    ExtentType const& extent_en) {
    KOKKOS_IMPL_PRECOMP_LOOP_LAYOUT_2(func, IType, IsLeft, extent_st, extent_en,
                                      2);
  }
};

template <bool IsLeft, typename IType>
struct Tile_Loop_Type_PreComp<3, IsLeft, IType, void, void> {
  template <typename Func, typename ExtentType>
  static void apply(Func const& func, ExtentType const& extent_st,
                    ExtentType const& extent_en) {
    KOKKOS_IMPL_PRECOMP_LOOP_LAYOUT_3(func, IType, IsLeft, extent_st, extent_en,
                                      3);
  }
};

template <bool IsLeft, typename IType>
struct Tile_Loop_Type_PreComp<4, IsLeft, IType, void, void> {
  template <typename Func, typename ExtentType>
  static void apply(Func const& func, ExtentType const& extent_st,
                    ExtentType const& extent_en) {
    KOKKOS_IMPL_PRECOMP_LOOP_LAYOUT_4(func, IType, IsLeft, extent_st, extent_en,
                                      4);
  }
};

template <bool IsLeft, typename IType>
struct Tile_Loop_Type_PreComp<5, IsLeft, IType, void, void> {
  template <typename Func, typename ExtentType>
  static void apply(Func const& func, ExtentType const& extent_st,
                    ExtentType const& extent_en) {
    KOKKOS_IMPL_PRECOMP_LOOP_LAYOUT_5(func, IType, IsLeft, extent_st, extent_en,
                                      5);
  }
};

template <bool IsLeft, typename IType>
struct Tile_Loop_Type_PreComp<6, IsLeft, IType, void, void> {
  template <typename Func, typename ExtentType>
  static void apply(Func const& func, ExtentType const& extent_st,
                    ExtentType const& extent_en) {
    KOKKOS_IMPL_PRECOMP_LOOP_LAYOUT_6(func, IType, IsLeft, extent_st, extent_en,
                                      6);
  }
};

template <bool IsLeft, typename IType>
struct Tile_Loop_Type_PreComp<7, IsLeft, IType, void, void> {
  template <typename Func, typename ExtentType>
  static void apply(Func const& func, ExtentType const& extent_st,
                    ExtentType const& extent_en) {
    KOKKOS_IMPL_PRECOMP_LOOP_LAYOUT_7(func, IType, IsLeft, extent_st, extent_en,
                                      7);
  }
};

template <bool IsLeft, typename IType>
struct Tile_Loop_Type_PreComp<8, IsLeft, IType, void, void> {
  template <typename Func, typename ExtentType>
  static void apply(Func const& func, ExtentType const& extent_st,
                    ExtentType const& extent_en) {
    KOKKOS_IMPL_PRECOMP_LOOP_LAYOUT_8(func, IType, IsLeft, extent_st, extent_en,
                                      8);
  }
};

// tagged versions

template <bool IsLeft, typename IType, typename Tagged>
struct Tile_Loop_Type_PreComp<1, IsLeft, IType, Tagged,
                              std::enable_if_t<!std::is_void_v<Tagged>>> {
  template <typename Func, typename ExtentType>
  static void apply(Func const& func, ExtentType const& extent_st,
                    ExtentType const& extent_en) {
    KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_LAYOUT_1(Tagged{}, func, IType, IsLeft,
                                             extent_st, extent_en, 1);
  }
};

template <bool IsLeft, typename IType, typename Tagged>
struct Tile_Loop_Type_PreComp<2, IsLeft, IType, Tagged,
                              std::enable_if_t<!std::is_void_v<Tagged>>> {
  template <typename Func, typename ExtentType>
  static void apply(Func const& func, ExtentType const& extent_st,
                    ExtentType const& extent_en) {
    KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_LAYOUT_2(Tagged{}, func, IType, IsLeft,
                                             extent_st, extent_en, 2);
  }
};

template <bool IsLeft, typename IType, typename Tagged>
struct Tile_Loop_Type_PreComp<3, IsLeft, IType, Tagged,
                              std::enable_if_t<!std::is_void_v<Tagged>>> {
  template <typename Func, typename ExtentType>
  static void apply(Func const& func, ExtentType const& extent_st,
                    ExtentType const& extent_en) {
    KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_LAYOUT_3(Tagged{}, func, IType, IsLeft,
                                             extent_st, extent_en, 3);
  }
};

template <bool IsLeft, typename IType, typename Tagged>
struct Tile_Loop_Type_PreComp<4, IsLeft, IType, Tagged,
                              std::enable_if_t<!std::is_void_v<Tagged>>> {
  template <typename Func, typename ExtentType>
  static void apply(Func const& func, ExtentType const& extent_st,
                    ExtentType const& extent_en) {
    KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_LAYOUT_4(Tagged{}, func, IType, IsLeft,
                                             extent_st, extent_en, 4);
  }
};

template <bool IsLeft, typename IType, typename Tagged>
struct Tile_Loop_Type_PreComp<5, IsLeft, IType, Tagged,
                              std::enable_if_t<!std::is_void_v<Tagged>>> {
  template <typename Func, typename ExtentType>
  static void apply(Func const& func, ExtentType const& extent_st,
                    ExtentType const& extent_en) {
    KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_LAYOUT_5(Tagged{}, func, IType, IsLeft,
                                             extent_st, extent_en, 5);
  }
};

template <bool IsLeft, typename IType, typename Tagged>
struct Tile_Loop_Type_PreComp<6, IsLeft, IType, Tagged,
                              std::enable_if_t<!std::is_void_v<Tagged>>> {
  template <typename Func, typename ExtentType>
  static void apply(Func const& func, ExtentType const& extent_st,
                    ExtentType const& extent_en) {
    KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_LAYOUT_6(Tagged{}, func, IType, IsLeft,
                                             extent_st, extent_en, 6);
  }
};

template <bool IsLeft, typename IType, typename Tagged>
struct Tile_Loop_Type_PreComp<7, IsLeft, IType, Tagged,
                              std::enable_if_t<!std::is_void_v<Tagged>>> {
  template <typename Func, typename ExtentType>
  static void apply(Func const& func, ExtentType const& extent_st,
                    ExtentType const& extent_en) {
    KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_LAYOUT_7(Tagged{}, func, IType, IsLeft,
                                             extent_st, extent_en, 7);
  }
};

template <bool IsLeft, typename IType, typename Tagged>
struct Tile_Loop_Type_PreComp<8, IsLeft, IType, Tagged,
                              std::enable_if_t<!std::is_void_v<Tagged>>> {
  template <typename Func, typename ExtentType>
  static void apply(Func const& func, ExtentType const& extent_st,
                    ExtentType const& extent_en) {
    KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_LAYOUT_8(Tagged{}, func, IType, IsLeft,
                                             extent_st, extent_en, 8);
  }
};
// end Structs for calling loops

// For ParallelFor
template <typename RP, typename Functor, typename Tag, typename ValueType>
  requires std::is_same_v<typename RP::index_type, int>
struct HostIterateTile<RP, Functor, Tag, ValueType,
                       std::enable_if_t<std::is_void_v<ValueType>>> {
  using index_type = typename RP::index_type;
  using point_type = Kokkos::Array<index_type, RP::rank>;

  using value_type = ValueType;

  inline HostIterateTile(RP const& rp, Functor const& func)
      : m_rp(rp), m_func(func) {
    // Choose between iteration space that can be represented by:
    // 1. std::int64_t (maximum range possible)
    // 2. int (more vectorization friendly)
    m_int_enough = true;

    for (int i_rank = 0; i_rank < RP::rank; ++i_rank) {
      if ((m_rp.m_upper[i_rank] > std::numeric_limits<int>::max()) ||
          (m_rp.m_lower[i_rank] < std::numeric_limits<int>::min())) {
        m_int_enough = false;
        break;
      }
    }
  }

  inline bool check_iteration_bounds(point_type& partial_tile,
                                     const point_type& offset) const {
    bool is_full_tile = true;

    for (int i = 0; i < RP::rank; ++i) {
      if ((offset[i] + m_rp.m_tile[i]) <= m_rp.m_upper[i]) {
        partial_tile[i] = m_rp.m_tile[i];
      } else {
        is_full_tile = false;
        partial_tile[i] =
            m_rp.m_upper[i] - offset[i];  // remaining elements in dimension i
      }
    }

    return is_full_tile;
  }  // end check bounds

  template <int Rank>
  struct RankTag {
    using type = RankTag<Rank>;
    enum { value = (int)Rank };
  };

  template <typename IType>
  inline void compute_extents(IType tile_idx, point_type& extents_st,
                              point_type& extents_en) const {
    point_type offset;
    point_type tiledims;

    if constexpr (RP::outer_direction == Iterate::Left) {
      for (int i = 0; i < RP::rank; ++i) {
        offset[i] =
            (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i];
        tile_idx /= m_rp.m_tile_end[i];
      }
    } else {
      for (int i = RP::rank - 1; i >= 0; --i) {
        offset[i] =
            (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i];
        tile_idx /= m_rp.m_tile_end[i];
      }
    }

    // Check if offset+tiledim in bounds - if not, replace tile dims with the
    // partial tile dims
    check_iteration_bounds(tiledims, offset);

    for (int i = 0; i < RP::rank; ++i) {
      extents_st[i] = static_cast<int>(offset[i]);
      extents_en[i] = static_cast<int>(offset[i]) + tiledims[i];
    }
  }

  template <typename IType>
  inline void operator()(IType tile_idx) const {
    if (m_int_enough) {
      Kokkos::Array<int, RP::rank> extents_st, extents_en;
      compute_extents(tile_idx, extents_st, extents_en);
      Tile_Loop_Type_PreComp<RP::rank, (RP::inner_direction == Iterate::Left),
                             index_type, Tag>::apply(m_func, extents_st,
                                                     extents_en);
    } else {
      point_type offset;
      point_type tiledims;

      if constexpr (RP::outer_direction == Iterate::Left) {
        for (int i = 0; i < RP::rank; ++i) {
          offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] +
                      m_rp.m_lower[i];
          tile_idx /= m_rp.m_tile_end[i];
        }
      } else {
        for (int i = RP::rank - 1; i >= 0; --i) {
          offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] +
                      m_rp.m_lower[i];
          tile_idx /= m_rp.m_tile_end[i];
        }
      }

      // Check if offset+tiledim in bounds - if not, replace tile dims with the
      // partial tile dims
      const bool full_tile = check_iteration_bounds(tiledims, offset);

      Tile_Loop_Type<RP::rank, (RP::inner_direction == Iterate::Left),
                     index_type, Tag>::apply(m_func, full_tile, offset,
                                             m_rp.m_tile, tiledims);
    }
  }

  template <typename... Args>
  std::enable_if_t<(sizeof...(Args) == RP::rank && std::is_void_v<Tag>), void>
  apply(Args&&... args) const {
    m_func(args...);
  }

  template <typename... Args>
  std::enable_if_t<(sizeof...(Args) == RP::rank && !std::is_void_v<Tag>), void>
  apply(Args&&... args) const {
    m_func(m_tag, args...);
  }

  RP const m_rp;
  Functor const m_func;
  std::conditional_t<std::is_void_v<Tag>, int, Tag> m_tag{};
  bool m_int_enough;
};

// ------------------------------------------------------------------ //

#undef KOKKOS_IMPL_TAGGED_LOOP_R_1
#undef KOKKOS_IMPL_TAGGED_LOOP_R_2
#undef KOKKOS_IMPL_TAGGED_LOOP_R_3
#undef KOKKOS_IMPL_TAGGED_LOOP_R_4
#undef KOKKOS_IMPL_TAGGED_LOOP_R_5
#undef KOKKOS_IMPL_TAGGED_LOOP_R_6
#undef KOKKOS_IMPL_TAGGED_LOOP_R_7
#undef KOKKOS_IMPL_TAGGED_LOOP_R_8
#undef KOKKOS_IMPL_TAGGED_LOOP_L_1
#undef KOKKOS_IMPL_TAGGED_LOOP_L_2
#undef KOKKOS_IMPL_TAGGED_LOOP_L_3
#undef KOKKOS_IMPL_TAGGED_LOOP_L_4
#undef KOKKOS_IMPL_TAGGED_LOOP_L_5
#undef KOKKOS_IMPL_TAGGED_LOOP_L_6
#undef KOKKOS_IMPL_TAGGED_LOOP_L_7
#undef KOKKOS_IMPL_TAGGED_LOOP_L_8
#undef KOKKOS_IMPL_TAGGED_LOOP_LAYOUT_1
#undef KOKKOS_IMPL_TAGGED_LOOP_LAYOUT_2
#undef KOKKOS_IMPL_TAGGED_LOOP_LAYOUT_3
#undef KOKKOS_IMPL_TAGGED_LOOP_LAYOUT_4
#undef KOKKOS_IMPL_TAGGED_LOOP_LAYOUT_5
#undef KOKKOS_IMPL_TAGGED_LOOP_LAYOUT_6
#undef KOKKOS_IMPL_TAGGED_LOOP_LAYOUT_7
#undef KOKKOS_IMPL_TAGGED_LOOP_LAYOUT_8
#undef KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_R_1
#undef KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_R_2
#undef KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_R_3
#undef KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_R_4
#undef KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_R_5
#undef KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_R_6
#undef KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_R_7
#undef KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_R_8
#undef KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_L_1
#undef KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_L_2
#undef KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_L_3
#undef KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_L_4
#undef KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_L_5
#undef KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_L_6
#undef KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_L_7
#undef KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_L_8
#undef KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_LAYOUT_1
#undef KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_LAYOUT_2
#undef KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_LAYOUT_3
#undef KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_LAYOUT_4
#undef KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_LAYOUT_5
#undef KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_LAYOUT_6
#undef KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_LAYOUT_7
#undef KOKKOS_IMPL_TAGGED_PRECOMP_LOOP_LAYOUT_8

}  // namespace Impl
}  // namespace Kokkos

#endif
