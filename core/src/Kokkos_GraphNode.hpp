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

#ifndef KOKKOS_KOKKOS_GRAPHNODE_HPP
#define KOKKOS_KOKKOS_GRAPHNODE_HPP

#include <Kokkos_Macros.hpp>

#include <impl/Kokkos_Error.hpp>  // contract macros

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Graph_fwd.hpp>
#include <impl/Kokkos_GraphImpl_fwd.hpp>

#include <memory>  // std::shared_ptr

namespace Kokkos {
namespace Experimental {

template <class ExecutionSpace, class Kernel /*= TypeErasedTag*/,
          class Predecessor /*= TypeErasedTag*/>
class GraphNodeRef {
 public:
  using execution_space   = ExecutionSpace;
  using graph_kernel      = Kernel;
  using graph_predecessor = Predecessor;

 private:
  template <class, class, class>
  friend class GraphNodeRef;
  template <class>
  friend struct GraphBuilder;
  template <class, class, class>
  friend struct Kokkos::Impl::GraphNodeImpl;
  template <class>
  friend struct Kokkos::Impl::GraphImpl;
  template <class>
  friend struct Kokkos::Impl::GraphNodeBackendSpecificDetails;
  friend struct Kokkos::Impl::GraphAccess;

  using graph_impl_t = Kokkos::Impl::GraphImpl<ExecutionSpace>;

  // TODO figure out if we can get away with a weak reference here
  std::shared_ptr<graph_impl_t> m_graph_impl;

  using node_impl_t =
      Kokkos::Impl::GraphNodeImpl<ExecutionSpace, Kernel, Predecessor>;
  std::shared_ptr<node_impl_t> m_node_impl;

  // Internally, use shallow constness
  node_impl_t& get_node_impl() const { return *m_node_impl.get(); }
  std::shared_ptr<node_impl_t> const& get_node_ptr() const& {
    return m_node_impl;
  }
  std::shared_ptr<node_impl_t> get_node_ptr() && {
    return std::move(m_node_impl);
  }
  graph_impl_t& get_graph_impl() const { return *m_graph_impl; }
  std::shared_ptr<graph_impl_t> const& get_graph_ptr() const {
    return m_graph_impl;
  }

  // TODO kernel name propagation and exposure

  template <class NextKernelDeduced>
  // requires std::remove_cvref_t<NextKernelDeduced> is a specialization of
  // Kokkos::Impl::GraphNodeKernelImpl
  auto _then_kernel(NextKernelDeduced&& arg_kernel) {
    using next_kernel_t = typename std::remove_cv<
        typename std::remove_reference<NextKernelDeduced>::type>::type;
    using return_t = GraphNodeRef<ExecutionSpace, next_kernel_t, GraphNodeRef>;
    auto rv = Kokkos::Impl::GraphAccess::make_graph_node_ref(
        m_graph_impl,
        Kokkos::Impl::GraphAccess::make_node_shared_ptr_with_deleter(
            new typename return_t::node_impl_t{
                m_node_impl->execution_space_instance(),
                Kokkos::Impl::_graph_node_kernel_ctor_tag{},
                (NextKernelDeduced &&) arg_kernel,
                // *this is the predecessor
                Kokkos::Impl::_graph_node_predecessor_ctor_tag{}, *this}));
    // Add the node itself to the backend's graph data structure, now that
    // everything is set up.
    m_graph_impl->add_node(rv.m_node_impl);
    // Add the predecessaor we stored in the constructor above in the backend's
    // data structure, now that everything is set up.
    m_graph_impl->add_predecessor(rv.m_node_impl, *this);
    KOKKOS_ENSURES(bool(rv.m_node_impl) && bool(rv.m_graph_impl))
    return rv;
  }

  GraphNodeRef(std::shared_ptr<graph_impl_t> arg_graph_impl,
               std::shared_ptr<node_impl_t> arg_node_impl)
      : m_graph_impl(std::move(arg_graph_impl)),
        m_node_impl(std::move(arg_node_impl)) {}

 public:
  //----------------------------------------------------------------------------
  // <editor-fold desc="rule of 6 ctors"> {{{2

  // Copyable and movable (basically just shared_ptr semantics
  GraphNodeRef() noexcept               = default;
  GraphNodeRef(GraphNodeRef const&)     = default;
  GraphNodeRef(GraphNodeRef&&) noexcept = default;
  GraphNodeRef& operator=(GraphNodeRef const&) = default;
  GraphNodeRef& operator=(GraphNodeRef&&) noexcept = default;
  ~GraphNodeRef()                                  = default;

  // </editor-fold> end rule of 6 ctors }}}2
  //----------------------------------------------------------------------------

  //----------------------------------------------------------------------------
  // <editor-fold desc="then_parallel_for"> {{{2

  template <class Policy, class Functor>
  // requires ExecutionPolicy<Policy> && DataParallelFunctor<Functor>
  auto then_parallel_for(std::string arg_name, Policy&& policy,
                         Functor&& functor) {
    KOKKOS_EXPECTS(bool(m_graph_impl))
    KOKKOS_EXPECTS(bool(m_node_impl))
    using next_kernel_t =
        Kokkos::Impl::GraphNodeKernelImpl<ExecutionSpace, std::decay_t<Policy>,
                                          std::decay_t<Functor>,
                                          Kokkos::ParallelForTag>;
    return this->_then_kernel(next_kernel_t{std::move(arg_name), policy.space(),
                                            (Functor &&) functor,
                                            (Policy &&) policy});
  }

  template <class Policy, class Functor,
            typename std::enable_if<
                Kokkos::is_execution_policy<typename std::remove_cv<
                    typename std::remove_reference<Policy>::type>::type>::value,
                int>::type = 0>
  // requires ExecutionPolicy<Policy> && DataParallelFunctor<Functor>
  auto then_parallel_for(Policy&& policy, Functor&& functor) {
    return this->then_parallel_for("", (Policy &&) policy,
                                   (Functor &&) functor);
  }

  template <class Functor>
  // requires DataParallelFunctor<Functor>
  auto then_parallel_for(std::string name, std::size_t n, Functor&& functor) {
    return this->then_parallel_for(std::move(name),
                                   Kokkos::RangePolicy<ExecutionSpace>(0, n),
                                   (Functor &&) functor);
  }

  template <class Functor>
  // requires DataParallelFunctor<Functor>
  auto then_parallel_for(std::size_t n, Functor&& functor) {
    return this->then_parallel_for("", n, (Functor &&) functor);
  }

  // </editor-fold> end then_parallel_for }}}2
  //----------------------------------------------------------------------------

  // TODO @graph fill in these overloads
  // template <class Policy, class Functor,
  //    typename std::enable_if<
  //        Kokkos::is_execution_policy<typename std::remove_cv<
  //            typename std::remove_reference<Policy>::type>::type>::value,
  //        int>::type = 0>
  //// requires ExecutionPolicy<Policy> && DataParallelFunctor<Functor>
  // auto then_parallel_reduce(std::string arg_name, Policy&& policy,
  //                       Functor&& functor) {
  //  KOKKOS_EXPECTS(bool(m_graph_impl))
  //  KOKKOS_EXPECTS(bool(m_node_impl))
  //  using next_kernel_t =
  //  Kokkos::Impl::GraphNodeKernelImpl<ExecutionSpace, std::decay_t<Policy>,
  //      std::decay_t<Functor>,
  //      Kokkos::ParallelForTag>;
  //  return this->_then_kernel(next_kernel_t{std::move(arg_name),
  //  policy.space(),
  //                                          (Functor &&) functor,
  //                                          (Policy &&) policy});
  //}
};

}  // end namespace Experimental
}  // end namespace Kokkos

#endif  // KOKKOS_KOKKOS_GRAPHNODE_HPP
