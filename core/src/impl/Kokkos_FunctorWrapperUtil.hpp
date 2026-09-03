// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_IMPL_KOKKOS_FUNCTORWRAPPERUTIL_HPP
#define KOKKOS_IMPL_KOKKOS_FUNCTORWRAPPERUTIL_HPP

#include <type_traits>

namespace Kokkos::Impl {

// Helper to allow passing an indexless functor to the parallel_for backend
// through special interface such as Kokkos::GraphNodeThen and Kokkos::Single.
template <typename Functor>
struct IndexlessFunctorWrapper {
  Functor m_functor;

  // One of WorkTagOrIndex or MaybeIndex contains the index, the other can be
  // the worktag. We try to determine which is which and discard the index.
  template <typename WorkTagOrIndex, typename... MaybeIndex>
  KOKKOS_FUNCTION void operator()(WorkTagOrIndex, MaybeIndex...) const {
    static_assert(sizeof...(MaybeIndex) <= 1);
    if constexpr (sizeof...(MaybeIndex) == 0) {
      m_functor();
    } else {
      static_assert(std::is_empty_v<WorkTagOrIndex>);
      m_functor(WorkTagOrIndex{});
    }
  }
};

// Helper to allow passing an indexless functor to the parallel_reduce backend
// through special interface such as Kokkos::GraphNodeThen and Kokkos::Single.
template <class FunctorType, class WorkTag>
struct IndexlessReductionFunctorWrapper {
  FunctorType m_functor;

  // One of WorkTagOrIndex or IndexOrFirstRet contains the index, the other can
  // be the worktag or the first return type. We try to determine which is
  // which and discard the index.
  template <class WorkTagOrIndex, class IndexOrFirstRet, class... ReturnTypes>
  KOKKOS_INLINE_FUNCTION void operator()(WorkTagOrIndex&& wtOrIdx,
                                         IndexOrFirstRet&& idxOrFirstRet,
                                         ReturnTypes&&... rets) const {
    if constexpr (std::is_void_v<WorkTag>) {
      m_functor(std::forward<IndexOrFirstRet>(idxOrFirstRet),
                std::forward<ReturnTypes>(rets)...);
    } else {
      m_functor(std::forward<WorkTagOrIndex>(wtOrIdx),
                std::forward<ReturnTypes>(rets)...);
    }
  }
};

}  //  namespace Kokkos::Impl

#endif  // KOKKOS_IMPL_KOKKOS_FUNCTORWRAPPERUTIL_HPP
