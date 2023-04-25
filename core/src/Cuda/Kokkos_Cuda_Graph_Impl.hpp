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

#ifndef KOKKOS_KOKKOS_CUDA_GRAPH_IMPL_HPP
#define KOKKOS_KOKKOS_CUDA_GRAPH_IMPL_HPP

#include <Kokkos_Macros.hpp>

#if defined(KOKKOS_ENABLE_CUDA)

#include <Kokkos_Graph_fwd.hpp>

#include <impl/Kokkos_GraphImpl.hpp>  // GraphAccess needs to be complete

// GraphNodeImpl needs to be complete because GraphImpl here is a full
// specialization and not just a partial one
#include <impl/Kokkos_GraphNodeImpl.hpp>
#include <Cuda/Kokkos_Cuda_GraphNode_Impl.hpp>

#include <Cuda/Kokkos_Cuda.hpp>
#include <cuda_runtime_api.h>
#include <Cuda/Kokkos_Cuda_Error.hpp>
#include <Cuda/Kokkos_Cuda_Instance.hpp>

namespace Kokkos {
namespace Impl {

template <>
struct GraphImpl<Kokkos::Cuda> {
 public:
  using execution_space = Kokkos::Cuda;

 private:
  execution_space m_execution_space;
  cudaGraph_t m_graph          = nullptr;
  cudaGraphExec_t m_graph_exec = nullptr;

  using cuda_graph_flags_t = unsigned int;

  using node_details_t = GraphNodeBackendSpecificDetails<Kokkos::Cuda>;

  void _instantiate_graph() {
    constexpr size_t error_log_size = 256;
    cudaGraphNode_t error_node      = nullptr;
    char error_log[error_log_size];
    CudaInternal::singleton()
        .cuda_api_interface_safe_call<cudaGraphExec_t*, cudaGraph_t,
                                      cudaGraphNode_t*, char*, size_t>(
            &cudaGraphInstantiate, &m_graph_exec, m_graph, &error_node,
            error_log, error_log_size);
    // TODO @graphs print out errors
  }

 public:
  using root_node_impl_t =
      GraphNodeImpl<Kokkos::Cuda, Kokkos::Experimental::TypeErasedTag,
                    Kokkos::Experimental::TypeErasedTag>;
  using aggregate_kernel_impl_t = CudaGraphNodeAggregateKernel;
  using aggregate_node_impl_t =
      GraphNodeImpl<Kokkos::Cuda, aggregate_kernel_impl_t,
                    Kokkos::Experimental::TypeErasedTag>;

  // Not moveable or copyable; it spends its whole life as a shared_ptr in the
  // Graph object
  GraphImpl()                 = delete;
  GraphImpl(GraphImpl const&) = delete;
  GraphImpl(GraphImpl&&)      = delete;
  GraphImpl& operator=(GraphImpl const&) = delete;
  GraphImpl& operator=(GraphImpl&&) = delete;
  ~GraphImpl() {
    // TODO @graphs we need to somehow indicate the need for a fence in the
    //              destructor of the GraphImpl object (so that we don't have to
    //              just always do it)
    m_execution_space.fence("Kokkos::GraphImpl::~GraphImpl: Graph Destruction");
    KOKKOS_EXPECTS(bool(m_graph))
    if (bool(m_graph_exec)) {
      CudaInternal::singleton().cuda_api_interface_safe_call<cudaGraphExec_t>(
          &cudaGraphExecDestroy, m_graph_exec);
    }
    CudaInternal::singleton().cuda_api_interface_safe_call<cudaGraph_t>(
        &cudaGraphDestroy, m_graph);
  };

  explicit GraphImpl(Kokkos::Cuda arg_instance)
      : m_execution_space(std::move(arg_instance)) {
    CudaInternal::singleton()
        .cuda_api_interface_safe_call<cudaGraph_t*, unsigned int>(
            &cudaGraphCreate, &m_graph, cuda_graph_flags_t{0});
  }

  void add_node(std::shared_ptr<aggregate_node_impl_t> const& arg_node_ptr) {
    // All of the predecessors are just added as normal, so all we need to
    // do here is add an empty node
    CudaInternal::singleton()
        .cuda_api_interface_safe_call<cudaGraphNode_t*, cudaGraph_t,
                                      const cudaGraphNode_t*, size_t>(
            &cudaGraphAddEmptyNode, &(arg_node_ptr->node_details_t::node),
            m_graph,
            /* dependencies = */ nullptr,
            /* numDependencies = */ 0);
  }

  template <class NodeImpl>
  //  requires NodeImplPtr is a shared_ptr to specialization of GraphNodeImpl
  //  Also requires that the kernel has the graph node tag in it's policy
  void add_node(std::shared_ptr<NodeImpl> const& arg_node_ptr) {
    static_assert(
        NodeImpl::kernel_type::Policy::is_graph_kernel::value,
        "Something has gone horribly wrong, but it's too complicated to "
        "explain here.  Buy Daisy a coffee and she'll explain it to you.");
    KOKKOS_EXPECTS(bool(arg_node_ptr));
    // The Kernel launch from the execute() method has been shimmed to insert
    // the node into the graph
    auto& kernel = arg_node_ptr->get_kernel();
    // note: using arg_node_ptr->node_details_t::node caused an ICE in NVCC 10.1
    auto& cuda_node = static_cast<node_details_t*>(arg_node_ptr.get())->node;
    KOKKOS_EXPECTS(!bool(cuda_node));
    kernel.set_cuda_graph_ptr(&m_graph);
    kernel.set_cuda_graph_node_ptr(&cuda_node);
    kernel.execute();
    KOKKOS_ENSURES(bool(cuda_node));
  }

  template <class NodeImplPtr, class PredecessorRef>
  // requires PredecessorRef is a specialization of GraphNodeRef that has
  // already been added to this graph and NodeImpl is a specialization of
  // GraphNodeImpl that has already been added to this graph.
  void add_predecessor(NodeImplPtr arg_node_ptr, PredecessorRef arg_pred_ref) {
    KOKKOS_EXPECTS(bool(arg_node_ptr))
    auto pred_ptr = GraphAccess::get_node_ptr(arg_pred_ref);
    KOKKOS_EXPECTS(bool(pred_ptr))

    // clang-format off
    // NOTE const-qualifiers below are commented out because of an API break
    // from CUDA 10.0 to CUDA 10.1
    // cudaGraphAddDependencies(cudaGraph_t, cudaGraphNode_t*, cudaGraphNode_t*, size_t)
    // cudaGraphAddDependencies(cudaGraph_t, const cudaGraphNode_t*, const cudaGraphNode_t*, size_t)
    // clang-format on
    auto /*const*/& pred_cuda_node = pred_ptr->node_details_t::node;
    KOKKOS_EXPECTS(bool(pred_cuda_node))

    auto /*const*/& cuda_node = arg_node_ptr->node_details_t::node;
    KOKKOS_EXPECTS(bool(cuda_node))

    CudaInternal::singleton()
        .cuda_api_interface_safe_call<cudaGraph_t, const cudaGraphNode_t*,
                                      const cudaGraphNode_t*, size_t>(
            &cudaGraphAddDependencies, m_graph, &pred_cuda_node, &cuda_node, 1);
  }

  void submit() {
    if (!bool(m_graph_exec)) {
      _instantiate_graph();
    }
    CudaInternal::singleton()
        .cuda_api_interface_safe_call<cudaGraphExec_t, cudaStream_t>(
            &cudaGraphLaunch, m_graph_exec, m_execution_space.cuda_stream());
  }

  execution_space const& get_execution_space() const noexcept {
    return m_execution_space;
  }

  auto create_root_node_ptr() {
    KOKKOS_EXPECTS(bool(m_graph))
    KOKKOS_EXPECTS(!bool(m_graph_exec))
    auto rv = std::make_shared<root_node_impl_t>(
        get_execution_space(), _graph_node_is_root_ctor_tag{});
    CudaInternal::singleton()
        .cuda_api_interface_safe_call<cudaGraphNode_t*, cudaGraph_t,
                                      const cudaGraphNode_t*, size_t>(
            &cudaGraphAddEmptyNode, &(rv->node_details_t::node), m_graph,
            /* dependencies = */ nullptr,
            /* numDependencies = */ 0);
    KOKKOS_ENSURES(bool(rv->node_details_t::node))
    return rv;
  }

  template <class... PredecessorRefs>
  // See requirements/expectations in GraphBuilder
  auto create_aggregate_ptr(PredecessorRefs&&...) {
    // The attachment to predecessors, which is all we really need, happens
    // in the generic layer, which calls through to add_predecessor for
    // each predecessor ref, so all we need to do here is create the (trivial)
    // aggregate node.
    return std::make_shared<aggregate_node_impl_t>(
        m_execution_space, _graph_node_kernel_ctor_tag{},
        aggregate_kernel_impl_t{});
  }
};

}  // end namespace Impl
}  // end namespace Kokkos

#endif  // defined(KOKKOS_ENABLE_CUDA)
#endif  // KOKKOS_KOKKOS_CUDA_GRAPH_IMPL_HPP
