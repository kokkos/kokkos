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

#ifndef KOKKOS_IMPL_KOKKOS_GRAPHIMPL_HPP
#define KOKKOS_IMPL_KOKKOS_GRAPHIMPL_HPP

#include <Kokkos_Macros.hpp>

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Graph_fwd.hpp>

#include <Kokkos_Concepts.hpp>  // is_execution_policy
#include <Kokkos_PointerOwnership.hpp>
#include <impl/Kokkos_GraphImpl_fwd.hpp>

#include <memory>  // std::make_shared

namespace Kokkos {
namespace Impl {

struct GraphAccess {
  template <class ExecutionSpace>
  static Kokkos::Experimental::Graph<ExecutionSpace> construct_graph(
      ExecutionSpace ex) {
    //----------------------------------------//
    return Kokkos::Experimental::Graph<ExecutionSpace>{
        std::make_shared<GraphImpl<ExecutionSpace>>(std::move(ex))};
    //----------------------------------------//
  }
  template <class ExecutionSpace>
  static auto create_root_ref(
      Kokkos::Experimental::Graph<ExecutionSpace>& arg_graph) {
    auto const& graph_impl_ptr = arg_graph.m_impl_ptr;

    auto root_ptr = graph_impl_ptr->create_root_node_ptr();

    return Kokkos::Experimental::GraphNodeRef<ExecutionSpace>{
        graph_impl_ptr, std::move(root_ptr)};
  }

  template <class RootNodeRef>
  // requires remove_cvref_t<RootNodeRef> is a specialization of GraphNodeRef
  static auto create_graph_builder(RootNodeRef&& arg_root) {
    // std::remove_cvref_t can't get here quickly enough...
    using execution_space =
        typename std::remove_cv<typename std::remove_reference<
            RootNodeRef>::type>::type::execution_space;
    return Kokkos::Experimental::GraphBuilder<execution_space>{(RootNodeRef &&)
                                                                   arg_root};
  }

  template <class NodeImpl>
  static auto make_node_shared_ptr_with_deleter(
      Kokkos::OwningRawPtr<NodeImpl> node_impl_ptr) {
    // NodeImpl instances aren't movable, so we have to create them on the call
    // side and take a pointer here. We assume ownership and pass it on to the
    // shared_ptr we're creating
    using new_node_impl_t = typename std::remove_cv<NodeImpl>::type;
    // We can't use make_shared because we have a custom deleter
    return std::shared_ptr<new_node_impl_t>{
        node_impl_ptr, typename new_node_impl_t::Deleter{}};
  }

  template <class GraphImplWeakPtr, class ExecutionSpace, class Kernel,
            class Predecessor>
  static auto make_graph_node_ref(
      GraphImplWeakPtr graph_impl,
      std::shared_ptr<
          Kokkos::Impl::GraphNodeImpl<ExecutionSpace, Kernel, Predecessor>>
          pred_impl) {
    //----------------------------------------
    return Kokkos::Experimental::GraphNodeRef<ExecutionSpace, Kernel,
                                              Predecessor>{
        std::move(graph_impl), std::move(pred_impl)};
    //----------------------------------------
  }
};

template <class Policy>
struct _add_graph_kernel_tag;

template <template <class...> class PolicyTemplate, class... PolicyTraits>
struct _add_graph_kernel_tag<PolicyTemplate<PolicyTraits...>> {
  using type = PolicyTemplate<PolicyTraits..., IsGraphKernelTag>;
};

}  // end namespace Impl

namespace Experimental {  // but not for users, so...

template <class Policy>
// requires ExecutionPolicy<Policy>
constexpr auto require(Policy const& policy,
                       Kokkos::Impl::KernelInGraphProperty) {
  static_assert(Kokkos::is_execution_policy<Policy>::value,
                "Internal implementation error!");
  return typename Kokkos::Impl::_add_graph_kernel_tag<Policy>::type{policy};
}

}  // end namespace Experimental

}  // end namespace Kokkos

#endif  // KOKKOS_IMPL_KOKKOS_GRAPHIMPL_HPP
