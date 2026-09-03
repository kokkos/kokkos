// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_IMPL_KOKKOS_GRAPHNODETHENIMPL_HPP
#define KOKKOS_IMPL_KOKKOS_GRAPHNODETHENIMPL_HPP

#include <Kokkos_ExecPolicy.hpp>
#include <impl/Kokkos_FunctorWrapperUtil.hpp>
#include <impl/Kokkos_GraphImpl_fwd.hpp>
#include <impl/Kokkos_GraphNodeThenPolicy.hpp>

namespace Kokkos::Impl {

// MSVC needs the 'policy' template argument to have a distinct name from the
// 'policy' template argument of the base class GraphNodeKernelImpl, see
// https://github.com/kokkos/kokkos/pull/8190#issuecomment-3083682368.
template <typename ExecutionSpace, typename ThenPolicyType, typename Functor>
struct GraphNodeThenImpl
    : public GraphNodeKernelImpl<
          ExecutionSpace,
          Kokkos::RangePolicy<ExecutionSpace, IsGraphKernelTag,
                              Kokkos::LaunchBounds<1>,
                              typename ThenPolicyType::work_tag>,
          IndexlessFunctorWrapper<Functor>, ParallelForTag> {
  using inner_policy_t = Kokkos::RangePolicy<ExecutionSpace, IsGraphKernelTag,
                                             Kokkos::LaunchBounds<1>,
                                             typename ThenPolicyType::work_tag>;
  using wrapper_t      = IndexlessFunctorWrapper<Functor>;
  using base_t = GraphNodeKernelImpl<ExecutionSpace, inner_policy_t, wrapper_t,
                                     ParallelForTag>;

  template <typename Label, typename T>
  GraphNodeThenImpl(Label&& label_, const ExecutionSpace& exec, ThenPolicyType,
                    T&& functor)
      : base_t(std::forward<Label>(label_), exec,
               wrapper_t{std::forward<T>(functor)},
               inner_policy_t(exec, 0, 1)) {}
};

}  // namespace Kokkos::Impl

#endif  // KOKKOS_IMPL_KOKKOS_GRAPHNODETHENIMPL_HPP
