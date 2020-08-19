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

#include <impl/Kokkos_Host_Graph_fwd.hpp>

#include <OpenMP/Kokkos_OpenMP_Parallel.hpp>
#include <Kokkos_Serial.hpp>
#include <Kokkos_OpenMP.hpp>
// FIXME @graph other backends?

#include <impl/Kokkos_OptionalRef.hpp>

#include <set>

namespace Kokkos {
namespace Impl {

//==============================================================================
// <editor-fold desc="HostGraphImpl"> {{{1

template <class ExecutionSpace>
struct HostGraphImpl : private ExecutionSpaceInstanceStorage<ExecutionSpace> {
 private:
  using execution_space_instance_storage_base_t =
      ExecutionSpaceInstanceStorage<ExecutionSpace>;
  using node_details_t = GraphNodeBackendSpecificDetails<ExecutionSpace>;

  std::set<std::shared_ptr<node_details_t>> m_sinks;

 public:
  using root_node_impl_t =
      GraphNodeImpl<ExecutionSpace, Experimental::TypeErasedTag,
                    Experimental::TypeErasedTag>;

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

  template <class NodeImplPtr>
  //  requires NodeImplPtr is a shared_ptr to specialization of GraphNodeImpl
  void add_node(NodeImplPtr arg_node_ptr) {
    // Since this is always called before any calls to add_predecessor involving
    // it, we can treat this node as a sink until we discover otherwise.
    arg_node_ptr->node_details_t::set_kernel(arg_node_ptr->get_kernel());
    auto spot = m_sinks.find(arg_node_ptr);
    KOKKOS_ASSERT(spot == m_sinks.end())
    m_sinks.insert(std::move(spot), std::move(arg_node_ptr));
  }

  template <class NodeImplPtr, class PredecessorRef>
  // requires PredecessorRef is a specialization of GraphNodeRef that has
  // already been added to this graph and NodeImpl is a specialization of
  // GraphNodeImpl that has already been added to this graph.
  void add_predecessor(NodeImplPtr arg_node_ptr, PredecessorRef arg_pred_ref) {
    // This is a lot of unnecessary reference count incrementing and
    // decrementing but it doesn't matter because this is super coarse grained
    // anyway.
    auto node_ptr_spot = m_sinks.find(arg_node_ptr);
    auto pred_ptr      = GraphAccess::get_node_ptr(arg_pred_ref);
    auto pred_ref_spot = m_sinks.find(pred_ptr);
    KOKKOS_ASSERT(node_ptr_spot != m_sinks.end())
    if (pred_ref_spot != m_sinks.end()) {
      // delegate responsibility for executing the predecessor to arg_node
      // and then remove the predecessor from the set of sinks
      (*node_ptr_spot)->set_predecessor(std::move(*pred_ref_spot));
      m_sinks.erase(pred_ref_spot);
    } else {
      // We still want to check that it's executed, even though someone else
      // should have executed it before us
      (*node_ptr_spot)->set_predecessor(std::move(pred_ptr));
    }
  }

  template <class... PredecessorRefs>
  // See requirements/expectations in GraphBuilder
  auto create_aggregate_ptr(PredecessorRefs&&... refs) {
    using aggregate_kernel_impl_t =
        GraphNodeAggregateKernelHostImpl<ExecutionSpace>;
    using aggregate_node_impl_t =
        GraphNodeImpl<ExecutionSpace, aggregate_kernel_impl_t,
                      Experimental::TypeErasedTag>;
    return GraphAccess::make_node_shared_ptr_with_deleter(
        new aggregate_node_impl_t{this->execution_space_instance(),
                                  _graph_node_kernel_ctor_tag{},
                                  aggregate_kernel_impl_t{}});
  }

  void submit() & {
    // This reset is gross, but for the purposes of our simple host
    // implementation...
    for (auto& sink : m_sinks) {
      sink->reset_has_executed();
    }
    for (auto& sink : m_sinks) {
      sink->execute_node();
    }
  }

  // I don't know if we have enough information to do call this overload
  // without incorrectly relying on the determinism of
  // std::shared_ptr::use_count(), but in case we do this is what it should
  // look like.
  void submit() && {
    // This reset is gross, but for the purposes of our simple host
    // implementation...
    for (auto& sink : m_sinks) {
      sink->reset_has_executed();
    }
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
    auto rv = Kokkos::Impl::GraphAccess::make_node_shared_ptr_with_deleter(
        new root_node_impl_t{get_execution_space(),
                             _graph_node_is_root_ctor_tag{}});
    m_sinks.insert(rv);
    return rv;
  }
};


// </editor-fold> end HostGraphImpl }}}1
//==============================================================================
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

#include <impl/Kokkos_Host_GraphNodeKernel.hpp>
#include <impl/Kokkos_Host_GraphNode_Impl.hpp>

#endif  // KOKKOS_HOST_GRAPH_IMPL_HPP
