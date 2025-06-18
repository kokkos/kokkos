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

#ifndef KOKKOS_IMPL_KOKKOS_GRAPHNODETHENIMPL_HPP
#define KOKKOS_IMPL_KOKKOS_GRAPHNODETHENIMPL_HPP

#include <Kokkos_ExecPolicy.hpp>
#include <impl/Kokkos_GraphNodeThenPolicy.hpp>

namespace Kokkos::Impl {
// Helper for the 'then', such that the user can indeed pass a callable that
// takes no index.
template <typename Functor, typename WorkTag>
struct ThenWrapper {
  Functor functor;

  template <typename Ignored, typename... Args>
  KOKKOS_FUNCTION void operator()(Ignored, Args...) const {
    static_assert(sizeof...(Args) <= 1);
    if constexpr (sizeof...(Args) == 0) {
      functor();
    } else {
      static_assert(is_tag_class_v<Ignored>);
      functor(Ignored{});
    }
  }
};

template <typename ExecutionSpace, typename Policy, typename Functor>
struct GraphNodeThenImpl
    : public GraphNodeKernelImpl<
          ExecutionSpace,
          Kokkos::RangePolicy<ExecutionSpace, IsGraphKernelTag,
                              Kokkos::LaunchBounds<1>,
                              typename Policy::work_tag>,
          ThenWrapper<Functor, typename Policy::work_tag>, ParallelForTag> {
  using inner_policy_t =
      Kokkos::RangePolicy<ExecutionSpace, IsGraphKernelTag,
                          Kokkos::LaunchBounds<1>, typename Policy::work_tag>;
  using wrapper_t = ThenWrapper<Functor, typename Policy::work_tag>;
  using base_t = GraphNodeKernelImpl<ExecutionSpace, inner_policy_t, wrapper_t,
                                     ParallelForTag>;

  template <typename Label, typename T>
  GraphNodeThenImpl(Label&& label, const ExecutionSpace& exec, Policy,
                    T&& functor)
      : base_t(std::forward<Label>(label), exec,
               wrapper_t{std::forward<T>(functor)},
               inner_policy_t(exec, 0, 1)) {}
};

}  // namespace Kokkos::Impl

#endif  // KOKKOS_IMPL_KOKKOS_GRAPHNODETHENIMPL_HPP
