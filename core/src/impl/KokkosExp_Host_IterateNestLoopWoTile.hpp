// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_HOST_EXP_ITERATE_NESTLOOPWOTILE_HPP
#define KOKKOS_HOST_EXP_ITERATE_NESTLOOPWOTILE_HPP

#include <type_traits>

#include <Kokkos_Layout.hpp>
#include <Kokkos_Macros.hpp>

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

// MDRangePolicy iteration via a nested loop without tiles

// Currently, only handles ParallelFor.
// For ParallelReduce ReferenceType will be either a scalar or an array
// the requires clause `requires std::is_void_v<ReferenceType>` to be removed
// when extending for ParallelReduce.
template <typename RP, typename Functor, typename Tag,
          typename ReferenceType = void>
  requires std::is_void_v<ReferenceType>
struct HostIterateNestLoopWoTile {
  // The loop variable has to be int in order to encourage
  // auto-vectorization. References in this comment:
  // https://github.com/kokkos/kokkos/pull/8721#issuecomment-4936232872
  // FIXME: better to obtain this via RP::tile_type::value_type
  using loop_var_type = int;

  inline HostIterateNestLoopWoTile(RP const& rp, Functor const& func)
      : m_rp(rp), m_func(func) {}

  // inner-most loop inside a separate function to encourage vectorization.
  // Proof of vectorization with GCC using -fopt-info-vec-all.

  // ----------------------------------------------------------------------- //
  // \brief Nested loops with recursive template instantiation
  //
  // Nested for loop order depends on the iteration order:
  //  Iterate::Left:
  //    Outermost loop corresponds to right-most index.
  //  Iterate::Right:
  //    Outermost loop corresponds to left-most index.
  // The fastest changing index is always in innermost loop.
  //
  // Indices accumulated in parameter pack Idxs...
  //
  // Functor call order depends on the iteration order:
  //  Iterate::Left:
  //    functor(i_0, i_1, i_2, ..., i_{R-1})
  //  Iterate::Right:
  //    functor(i_{R-1}, ..., i_2, i_1, i_0)
  //
  //  \tparam IterLevel iteration level of the nested loops
  //  \tpraram Idxs... index pack
  template <unsigned IterLevel, typename... Idxs>
  inline void iterate(std::integral_constant<unsigned, IterLevel>,
                      Idxs... idxs) const {
    const loop_var_type start = (RP::outer_direction == Iterate::Left)
                                    ? m_rp.m_lower[RP::rank - 1 - IterLevel]
                                    : m_rp.m_lower[IterLevel];
    const loop_var_type end   = (RP::outer_direction == Iterate::Left)
                                    ? m_rp.m_upper[RP::rank - 1 - IterLevel]
                                    : m_rp.m_upper[IterLevel];

    for (loop_var_type idx = start; idx < end; ++idx) {
      if constexpr (RP::outer_direction == Iterate::Left) {
        iterate(std::integral_constant<unsigned, IterLevel + 1>(), idx,
                idxs...);
      } else {
        iterate(std::integral_constant<unsigned, IterLevel + 1>(), idxs...,
                idx);
      }
    }
  }

  template <typename... Idxs>
  inline void iterate(std::integral_constant<unsigned, RP::rank - 1>,
                      Idxs... idxs) const {
    if constexpr (RP::outer_direction == Iterate::Left) {
      KOKKOS_ENABLE_IVDEP_INNERMOST_LOOP
      for (loop_var_type i = m_rp.m_lower[0]; i < m_rp.m_upper[0]; ++i) {
        if constexpr (std::is_void_v<Tag>) {
          m_func(i, idxs...);
        } else {
          m_func(Tag{}, i, idxs...);
        }
      }
    } else {
      KOKKOS_ENABLE_IVDEP_INNERMOST_LOOP
      for (loop_var_type i = m_rp.m_lower[RP::rank - 1];
           i < m_rp.m_upper[RP::rank - 1]; ++i) {
        if constexpr (std::is_void_v<Tag>) {
          m_func(idxs..., i);
        } else {
          m_func(Tag{}, idxs..., i);
        }
      }
    }
  }

  inline void execute() const {
    iterate(std::integral_constant<unsigned, 0u>());
  }

  const RP m_rp;
  const Functor m_func;
};

}  // namespace Impl
}  // namespace Kokkos

#endif
