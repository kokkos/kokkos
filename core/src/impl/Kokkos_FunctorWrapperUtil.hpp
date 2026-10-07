// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_IMPL_FUNCTOR_WRAPPER_UTIL_HPP
#define KOKKOS_IMPL_FUNCTOR_WRAPPER_UTIL_HPP

#include <type_traits>

namespace Kokkos::Impl {

// Helper to allow passing an indexless functor to the parallel_for backend
// through special interface such as Kokkos::GraphNodeThen and Kokkos::Single.
template <class Functor, class WorkTag>
struct IndexlessFunctorWrapper {
  Functor m_functor;

  template <std::integral IdxT>
    requires(std::is_same_v<WorkTag, void>)
  KOKKOS_FUNCTION void operator()(const IdxT&) const {
    m_functor();
  }

  template <class AWorkTag, std::integral IdxT>
    requires(!std::is_same_v<WorkTag, void> &&
             std::is_same_v<AWorkTag, WorkTag>)
  KOKKOS_FUNCTION void operator()(const AWorkTag& tag, const IdxT&) const {
    m_functor(tag);
  }
};

// Helper to allow passing an indexless functor to the parallel_for backend
// but with a value being produced by the Functor such as in Kokkos::single
template <class FunctorType, class ValueView, class WorkTag>
struct IndexlessValueFunctorWrapper {
  FunctorType m_functor;
  ValueView m_value;

  template <std::integral IdxT>
    requires(std::is_same_v<WorkTag, void>)
  KOKKOS_FUNCTION void operator()(const IdxT&) const {
    m_functor(m_value());
  }

  template <class AWorkTag, std::integral IdxT>
    requires(!std::is_same_v<WorkTag, void> &&
             std::is_same_v<AWorkTag, WorkTag>)
  KOKKOS_FUNCTION void operator()(const AWorkTag& tag, const IdxT&) const {
    m_functor(tag, m_value());
  }
};

}  //  namespace Kokkos::Impl

#endif  // KOKKOS_IMPL_FUNCTOR_WRAPPER_UTIL_HPP
