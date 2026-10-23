// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_IMPL_UNBIND_INDEX_ARGUMENT_HPP
#define KOKKOS_IMPL_UNBIND_INDEX_ARGUMENT_HPP

#include <type_traits>

namespace Kokkos::Impl {

// Helper to allow passing an indexless functor to the parallel_for backend
// through special interface such as Kokkos::GraphNodeThen and Kokkos::Single.
template <typename Functor>
struct UnbindIndexArgument {
  Functor m_functor;

  // One of WorkTagOrIndex or MaybeIndex contains the index, the other can be
  // the worktag. We try to determine which is which and discard the index.
  template <typename WorkTagOrIndex, typename... MaybeIndex>
  KOKKOS_FUNCTION void operator()(WorkTagOrIndex wtOrIdx, MaybeIndex...) const {
    static_assert(sizeof...(MaybeIndex) <= 1);
    if constexpr (sizeof...(MaybeIndex) == 0) {
      m_functor();
    } else {
      static_assert(std::is_empty_v<WorkTagOrIndex>);
      m_functor(wtOrIdx);
    }
  }
};

}  //  namespace Kokkos::Impl

#endif  // KOKKOS_IMPL_UNBIND_INDEX_ARGUMENT_HPP
