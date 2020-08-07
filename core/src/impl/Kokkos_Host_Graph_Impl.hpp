/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_HOST_GRAPH_IMPL_HPP
#define KOKKOS_HOST_GRAPH_IMPL_HPP

#include <Kokkos_ExecPolicy.hpp>
#include <Kokkos_Graph.hpp>

#include <OpenMP/Kokkos_OpenMP_Parallel.hpp>
#include <Kokkos_Serial.hpp>
#include <Kokkos_OpenMP.hpp>
// FIXME @graph other backends?

#include <impl/Kokkos_OptionalRef.hpp>

#include <set>

namespace Kokkos {
namespace Impl {

template <class ExecutionSpace>
struct HostGraphImpl;

//==============================================================================
// <editor-fold desc="GraphNodeKernelImpl"> {{{1

template <class ExecutionSpace>
struct GraphNodeKernelHostImpl {
  // TODO @graphs decide if this should use vtable or intrusive erasure via
  //      function pointers like in the rest of the graph interface
  virtual void execute_kernel() const = 0;
};

template <class ExecutionSpace, class Policy, class Functor>
class GraphNodeKernelImpl<ExecutionSpace, Policy, Functor,
                          Kokkos::ParallelForTag>
    final : public ParallelFor<Functor, Policy, ExecutionSpace>,
            public GraphNodeKernelHostImpl<ExecutionSpace> {
 public:
  using base_t = ParallelFor<Functor, Policy, ExecutionSpace>;
  using execute_kernel_vtable_base_t = GraphNodeKernelHostImpl<ExecutionSpace>;

  // TODO @graph kernel name info propagation
  template <class PolicyDeduced>
  GraphNodeKernelImpl(std::string, ExecutionSpace const&, Functor arg_functor,
                      PolicyDeduced&& arg_policy)
      : base_t(std::move(arg_functor), (PolicyDeduced &&) arg_policy),
        execute_kernel_vtable_base_t() {}

  // FIXME @graph Forward through the instance once that works in the backends
  template <class PolicyDeduced>
  GraphNodeKernelImpl(ExecutionSpace const& ex, Functor arg_functor,
                      PolicyDeduced&& arg_policy)
      : GraphNodeKernelImpl("", ex, std::move(arg_functor),
                            (PolicyDeduced &&) arg_policy) {}

  void execute_kernel() const final { this->base_t::execute(); }
};

// </editor-fold> end GraphNodeKernelImpl }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="GraphNodeBackendSpecificDetails"> {{{1

template <class ExecutionSpace>
struct GraphNodeBackendSpecificDetails
    : ExecutionSpaceInstanceStorage<ExecutionSpace> {
 private:
  using execution_space_instance_storage_t =
      ExecutionSpaceInstanceStorage<ExecutionSpace>;
  using host_kernel_impl_t = GraphNodeKernelHostImpl<ExecutionSpace>;

  std::shared_ptr<GraphNodeBackendSpecificDetails<ExecutionSpace>>
      m_predecessor = {};

  Kokkos::ObservingRawPtr<host_kernel_impl_t const> m_kernel_ptr = nullptr;

  bool m_has_executed = false;

  template <class>
  friend struct HostGraphImpl;

 protected:
  //----------------------------------------------------------------------------
  // <editor-fold desc="Ctors, destructor, and assignment"> {{{2

  explicit GraphNodeBackendSpecificDetails(ExecutionSpace const& ex) noexcept
      : execution_space_instance_storage_t(ex) {}

  GraphNodeBackendSpecificDetails(ExecutionSpace const& ex,
                                  _graph_node_is_root_ctor_tag) noexcept
      : execution_space_instance_storage_t(ex), m_has_executed(true) {}

  GraphNodeBackendSpecificDetails() noexcept = delete;

  GraphNodeBackendSpecificDetails(GraphNodeBackendSpecificDetails const&) =
      delete;

  GraphNodeBackendSpecificDetails(GraphNodeBackendSpecificDetails&&) noexcept =
      delete;

  GraphNodeBackendSpecificDetails& operator   =(
      GraphNodeBackendSpecificDetails const&) = delete;

  GraphNodeBackendSpecificDetails& operator       =(
      GraphNodeBackendSpecificDetails&&) noexcept = delete;

  ~GraphNodeBackendSpecificDetails() = default;

  // </editor-fold> end Ctors, destructor, and assignment }}}2
  //----------------------------------------------------------------------------

 public:
  void set_kernel(host_kernel_impl_t const& arg_kernel) {
    KOKKOS_EXPECTS(m_kernel_ptr == nullptr)
    m_kernel_ptr = &arg_kernel;
  }

  void set_predecessor(
      std::shared_ptr<GraphNodeBackendSpecificDetails<ExecutionSpace>> const&
          arg_pred_impl) {
    // This method delegates responsibility for executing the predecessor to
    // this node.  Each node can have at most one predecessor (which may be an
    // aggregate).
    KOKKOS_EXPECTS(!bool(m_predecessor))
    KOKKOS_EXPECTS(bool(arg_pred_impl))
    KOKKOS_EXPECTS(!m_has_executed)
    m_predecessor = arg_pred_impl;
  }

  void execute_node() {
    // This node could have already been executed as the predecessor of some
    // other
    KOKKOS_EXPECTS(bool(m_kernel_ptr) || m_has_executed)
    // Just execute the predecessor here, since calling set_predecessor()
    // delegates the responsibility for running it to us.
    if (!m_has_executed) {
      // I'm pretty sure this doesn't need to be atomic under our current
      // supported semantics, but instinct I have feels like it should be...
      m_has_executed = true;
      if (m_predecessor) {
        m_predecessor->execute_node();
      }
      m_kernel_ptr->execute_kernel();
      m_kernel_ptr = nullptr;
    }
    KOKKOS_ENSURES(m_has_executed)
  }
};

// </editor-fold> end GraphNodeBackendSpecificDetails }}}1
//==============================================================================

template <class ExecutionSpace>
struct HostGraphImpl : private ExecutionSpaceInstanceStorage<ExecutionSpace> {
 private:
  using execution_space_instance_storage_base_t =
      ExecutionSpaceInstanceStorage<ExecutionSpace>;
  using node_details_t = GraphNodeBackendSpecificDetails<ExecutionSpace>;

  std::set<std::shared_ptr<node_details_t>> m_sinks;

 public:
  //----------------------------------------------------------------------------
  // <editor-fold desc="Constructors, destructor, and assignment"> {{{2

  // Not moveable or copyable; it spends its whole live as a shared_ptr in the
  // Graph object
  HostGraphImpl()                     = default;
  HostGraphImpl(HostGraphImpl const&) = delete;
  HostGraphImpl(HostGraphImpl&&)      = delete;
  HostGraphImpl& operator=(HostGraphImpl const&) = delete;
  HostGraphImpl& operator=(HostGraphImpl&&) = delete;
  ~HostGraphImpl()                          = default;

  explicit HostGraphImpl(ExecutionSpace arg_space)
      : execution_space_instance_storage_base_t(std::move(arg_space)) {}

  // </editor-fold> end Constructors, destructor, and assignment }}}2
  //----------------------------------------------------------------------------

  template <class NodeImpl>
  //  requires NodeImpl is a specialization of GraphNodeImpl
  void add_node(NodeImpl& arg_node) {
    // Since this is always called before any calls to add_predecessor involving
    // it, we can treat this node as a sink until we discover otherwise.
    arg_node.node_details_t::set_kernel(arg_node.get_kernel());
    auto node_ptr = arg_node.shared_from_this();
    auto spot     = m_sinks.find(node_ptr);
    KOKKOS_ASSERT(spot == m_sinks.end())
    m_sinks.insert(std::move(spot), std::move(node_ptr));
  }

  template <class NodeImpl, class PredecessorRef>
  // requires PredecessorRef is a specialization of GraphNodeRef that has
  // already been added to this graph and NodeImpl is a specialization of
  // GraphNodeImpl that has already been added to this graph.
  void add_predecessor(NodeImpl& arg_node, PredecessorRef arg_pred_ref) {
    auto node_ptr_spot = m_sinks.find(arg_node.shared_from_this());
    auto pred_ref_spot = m_sinks.find(GraphAccess::get_node_ptr(arg_pred_ref));
    KOKKOS_ASSERT(node_ptr_spot != m_sinks.end())
    KOKKOS_ASSERT(pred_ref_spot != m_sinks.end())
    // delegate responsibility for executing the predecessor to arg_node
    // and then remove the predecessor from the set of sinks
    (*node_ptr_spot)->set_predecessor(*pred_ref_spot);
    m_sinks.erase(pred_ref_spot);
  }

  void submit() & {
    for (auto& sink : m_sinks) {
      sink->execute_node();
    }
  }

  // I don't know if we have enough information to do call this overload
  // without incorrectly relying on the determinism of
  // std::shared_ptr::use_count(), but in case we do this is what it should
  // look like.
  void submit() && {
    for (auto& sink : m_sinks) {
      // Swap ownership onto the stack so that it can be destroyed immediately
      // after execution
      std::shared_ptr<node_details_t> node_to_execute = nullptr;
      sink.swap(node_to_execute);
      std::move(*node_to_execute).execute_node();
    }
  }

  ExecutionSpace const& get_execution_space() const {
    return this
        ->execution_space_instance_storage_base_t::execution_space_instance();
  }

  auto create_root_node_ptr() {
    // TODO @graphs encapsulate this better; it's too easy to mess up.
    using root_node_impl_t =
        GraphNodeImpl<ExecutionSpace, Experimental::TypeErasedTag,
                      Experimental::TypeErasedTag>;
    auto rv = std::shared_ptr<root_node_impl_t>{
        new root_node_impl_t{get_execution_space(),
                             _graph_node_is_root_ctor_tag{}},
        typename root_node_impl_t::Deleter{}};
    m_sinks.insert(rv);
    return rv;
  }
};

//==============================================================================
// <editor-fold desc="Explicit specializations for host exec spaces"> {{{1

#ifdef KOKKOS_ENABLE_SERIAL
template <>
struct GraphImpl<Kokkos::Serial> : HostGraphImpl<Kokkos::Serial> {
 private:
  using base_t = HostGraphImpl<Kokkos::Serial>;

 public:
  using base_t::base_t;
};
#endif

#ifdef KOKKOS_ENABLE_OPENMP
template <>
struct GraphImpl<Kokkos::OpenMP> : HostGraphImpl<Kokkos::OpenMP> {
 private:
  using base_t = HostGraphImpl<Kokkos::OpenMP>;

 public:
  using base_t::base_t;
};
#endif

// </editor-fold> end Explicit specializations for host exec spaces }}}1
//==============================================================================

}  // end namespace Impl

}  // end namespace Kokkos

#endif  // KOKKOS_HOST_GRAPH_IMPL_HPP
