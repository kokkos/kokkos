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

#ifndef KOKKOS_IMPL_KOKKOS_GRAPHNODEDEEPCOPYIMPL_HPP
#define KOKKOS_IMPL_KOKKOS_GRAPHNODEDEEPCOPYIMPL_HPP

#include <Kokkos_ExecPolicy.hpp>
#include <impl/Kokkos_GraphImpl_fwd.hpp>

namespace Kokkos::Impl {

// TODO Refactor the ViewFill functor to use it here. For now, its constructor
//      wants to dispatch the kernel already, unsuitable with graph.
template <typename DstType>
struct NaiveViewFill {
  using value_t = typename DstType::value_type;

  DstType data;
  value_t value;

  template <typename T>
  KOKKOS_FUNCTION void operator()(const T index) const {
    data(index) = value;
  }
};

// For Cuda, HIP and SYCL.
#if defined(KOKKOS_ENABLE_CUDA) ||                                           \
    (defined(KOKKOS_ENABLE_HIP) && defined(KOKKOS_IMPL_HIP_NATIVE_GRAPH)) || \
    (defined(KOKKOS_ENABLE_SYCL) && defined(KOKKOS_IMPL_SYCL_GRAPH_SUPPORT))
template <typename DstType, typename SrcType>
struct GraphNodeDeepCopyImpl<Kokkos::DefaultExecutionSpace, DstType, SrcType>
    : public GraphNodeKernelImpl<
          Kokkos::DefaultExecutionSpace, Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, IsGraphKernelTag>,
          NaiveViewFill<DstType>, ParallelForTag> {

  static_assert(DstType::rank() == 1, "Only rank-1 views are supported.");
  static_assert(std::is_same_v<typename DstType::value_type, SrcType>, "Only 'scalar into view' is supported.");

  using execution_space = Kokkos::DefaultExecutionSpace;

  using policy_t = Kokkos::RangePolicy<execution_space, IsGraphKernelTag>;

  using base_t =
      GraphNodeKernelImpl<execution_space,
                          policy_t,
                          NaiveViewFill<DstType>, ParallelForTag>;

  using functor_t = NaiveViewFill<DstType>;

  template <typename T, typename U>
  GraphNodeDeepCopyImpl(const execution_space& exec, T&& dst, U&& src)
      : base_t(
            exec,
            functor_t{std::forward<T>(dst), std::forward<U>(src)},
            policy_t(exec, 0, dst.size())) {
  }
};

#endif

}  // namespace Kokkos::Impl

#endif  // KOKKOS_IMPL_KOKKOS_GRAPHNODEDEEPCOPYIMPL_HPP
