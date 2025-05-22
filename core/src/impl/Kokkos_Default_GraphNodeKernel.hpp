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

#ifndef KOKKOS_KOKKOS_HOST_GRAPHNODEKERNEL_HPP
#define KOKKOS_KOKKOS_HOST_GRAPHNODEKERNEL_HPP

#include <Kokkos_Macros.hpp>

#include <impl/Kokkos_Default_Graph_fwd.hpp>

#include <Kokkos_Graph.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_Parallel_Reduce.hpp>

namespace Kokkos {
namespace Impl {

//==============================================================================
// <editor-fold desc="GraphNodeKernelImpl"> {{{1

template <class ExecutionSpace>
struct GraphNodeKernelDefaultImpl {
  // TODO @graphs decide if this should use vtable or intrusive erasure via
  //      function pointers like in the rest of the graph interface
  virtual void execute_kernel() = 0;

  GraphNodeKernelDefaultImpl() = default;

  explicit GraphNodeKernelDefaultImpl(ExecutionSpace exec)
      : m_execution_space(std::move(exec)) {}

  ExecutionSpace m_execution_space;
};

// Assume it's better to use std::memset on host.
// Probably a bad assumption.
// Let's see what is done for e.g. ZeroMemset.
template <typename ViewType>
struct Memset {
  static constexpr bool accessible =
      Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                 typename ViewType::memory_space>::accessible;

  template <typename Exec>
  void apply(const Exec &exec) const {
    if constexpr (accessible) {
      exec.fence("fence before 'std::memset'");
      std::memset(dst.data(), value, count);
    } else {
      Kokkos::parallel_for(Kokkos::RangePolicy(exec, 0, count), *this);
    }
  }

  template <typename T>
  KOKKOS_FUNCTION void operator()(const T index) const {
    reinterpret_cast<unsigned char *>(dst.data())[index] =
        static_cast<unsigned char>(value);
  }

  ViewType dst;
  int value;
  size_t count;
};

template <typename ExecutionSpace, typename ViewType>
struct GraphNodeMemsetImpl : public GraphNodeKernelDefaultImpl<ExecutionSpace> {
 public:
  using execute_kernel_vtable_base_t =
      GraphNodeKernelDefaultImpl<ExecutionSpace>;

  using memset_t = Memset<ViewType>;

  template <typename... Args>
  GraphNodeMemsetImpl(ExecutionSpace exec, Args &&...args)
      : execute_kernel_vtable_base_t{std::move(exec)},
        memset{std::forward<Args>(args)...} {}

  void execute_kernel() override final {
    memset.apply(this->m_execution_space);
  }

 private:
  memset_t memset;
};

// TODO Indicate that this kernel specialization is only for the Host somehow?
template <class ExecutionSpace, class PolicyType, class Functor,
          class PatternTag, class... Args>
class GraphNodeKernelImpl
    : public GraphNodeKernelDefaultImpl<ExecutionSpace>,
      public PatternImplSpecializationFromTag<PatternTag, Functor, PolicyType,
                                              Args..., ExecutionSpace>::type {
 public:
  using base_t =
      typename PatternImplSpecializationFromTag<PatternTag, Functor, PolicyType,
                                                Args..., ExecutionSpace>::type;
  using execute_kernel_vtable_base_t =
      GraphNodeKernelDefaultImpl<ExecutionSpace>;
  // We have to use this name here because that's how it was done way back when
  // then implementations of Impl::Parallel*<> were written
  using Policy       = PolicyType;
  using graph_kernel = GraphNodeKernelImpl;

  // TODO @graph kernel name info propagation
  template <class PolicyDeduced, class... ArgsDeduced>
  GraphNodeKernelImpl(std::string const &, ExecutionSpace const &,
                      Functor arg_functor, PolicyDeduced &&arg_policy,
                      ArgsDeduced &&...args)
      : execute_kernel_vtable_base_t(arg_policy.space()),
        base_t(std::move(arg_functor), (PolicyDeduced &&)arg_policy,
               (ArgsDeduced &&)args...) {}

  // FIXME @graph Forward through the instance once that works in the backends
  template <class PolicyDeduced, class... ArgsDeduced>
  GraphNodeKernelImpl(ExecutionSpace const &ex, Functor arg_functor,
                      PolicyDeduced &&arg_policy, ArgsDeduced &&...args)
      : GraphNodeKernelImpl("", ex, std::move(arg_functor),
                            (PolicyDeduced &&)arg_policy,
                            (ArgsDeduced &&)args...) {
    // FIXME This constructor seem unused.
  }

  void execute_kernel() override final { this->base_t::execute(); }
};

// </editor-fold> end GraphNodeKernelImpl }}}1
//==============================================================================

template <class ExecutionSpace>
struct GraphNodeAggregateDefaultImpl
    : GraphNodeKernelDefaultImpl<ExecutionSpace> {
  void execute_kernel() override final {}
};

}  // end namespace Impl
}  // end namespace Kokkos

#endif  // KOKKOS_KOKKOS_HOST_GRAPHNODEKERNEL_HPP
