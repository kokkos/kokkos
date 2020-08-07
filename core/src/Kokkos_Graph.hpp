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

#ifndef KOKKOS_GRAPH_HPP
#define KOKKOS_GRAPH_HPP

#include <impl/Kokkos_SimpleTaskScheduler.hpp>  // ExecutionSpaceInstanceStorage

#include <functional>
#include <memory>

namespace Kokkos {

namespace Experimental {

struct TypeErasedTag {};

template <class ExecutionSpace>
struct Graph;

template <class ExecutionSpace, class Kernel = TypeErasedTag,
          class Predecessor = TypeErasedTag>
class GraphNodeRef;

}  // end namespace Experimental

namespace Impl {

template <class ExecutionSpace, class Kernel, class Predecessor>
struct GraphNodeImpl;

template <class ExecutionSpace>
struct GraphImpl;

template <class ExecutionSpace, class Policy, class Functor,
          class KernelTypeTag, class... Args>
class GraphNodeKernelImpl;

template <class ExecutionSpace>
struct GraphNodeBackendSpecificDetails;

struct _execution_space_ctor_tag {};
struct _graph_node_kernel_ctor_tag {};
struct _graph_node_predecessor_ctor_tag {};
struct _graph_node_type_erased_arg_ctor_tag {};
struct _graph_node_is_root_ctor_tag {};

struct GraphAccess {
  template <class ExecutionSpace>
  static Experimental::Graph<ExecutionSpace> construct_graph(
      ExecutionSpace ex) {
    //----------------------------------------//
    return Experimental::Graph<ExecutionSpace>{
        std::make_shared<GraphImpl<ExecutionSpace>>(std::move(ex))};
    //----------------------------------------//
  }
  template <class ExecutionSpace>
  static auto create_root_ref(Experimental::Graph<ExecutionSpace>& arg_graph) {
    auto const& graph_impl_ptr = arg_graph.m_impl_ptr;
    auto root_ptr              = graph_impl_ptr->create_root_node_ptr();
    return Experimental::GraphNodeRef<ExecutionSpace>{graph_impl_ptr,
                                                      std::move(root_ptr)};
  }
  template <class NodeRef>
  // Requires NodeRef is a specialization of GraphNodeRef
  static auto get_node_ptr(NodeRef&& node_ref) { return node_ref.get_node_ptr(); }
};

}  // end namespace Impl

namespace Experimental {

template <class ExecutionSpace, class Kernel /*= TypeErasedTag*/,
          class Predecessor /*= TypeErasedTag*/>
class GraphNodeRef {
 private:
  template <class, class, class>
  friend class GraphNodeRef;
  template <class, class, class>
  friend struct Kokkos::Impl::GraphNodeImpl;
  template <class>
  friend struct Kokkos::Impl::GraphImpl;
  template <class>
  friend struct Kokkos::Impl::GraphNodeBackendSpecificDetails;
  friend struct Kokkos::Impl::GraphAccess;

  using graph_impl_t = Kokkos::Impl::GraphImpl<ExecutionSpace>;
  std::shared_ptr<graph_impl_t> m_graph_impl;

  using node_impl_t =
      Kokkos::Impl::GraphNodeImpl<ExecutionSpace, Kernel, Predecessor>;
  std::shared_ptr<node_impl_t> m_node_impl;

  // Internally, use shallow constness
  node_impl_t& get_node_impl() const { return *m_node_impl.get(); }
  std::shared_ptr<node_impl_t> const& get_node_ptr() const { return m_node_impl; }

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

  // TODO kernel name propagation and exposure

  template <class Policy, class Functor>
  // requires ExecutionPolicy<Policy> && DataParallelFunctor<Functor>
  auto then_parallel_for(std::string arg_name, Policy&& policy,
                         Functor&& functor) {
    KOKKOS_EXPECTS(bool(m_graph_impl))
    KOKKOS_EXPECTS(bool(m_node_impl))
    using kernel_t =
        Kokkos::Impl::GraphNodeKernelImpl<ExecutionSpace, std::decay_t<Policy>,
                                          std::decay_t<Functor>,
                                          Kokkos::ParallelForTag>;
    using return_t = GraphNodeRef<ExecutionSpace, kernel_t, GraphNodeRef>;
    // TODO encapsulate this better
    auto rv = return_t{
        m_graph_impl,
        std::shared_ptr<typename return_t::node_impl_t>{
            // We can't use make_shared because we have a custom deleter
            new typename return_t::node_impl_t{
                m_node_impl->execution_space_instance(),
                // construct the kernel
                Kokkos::Impl::_graph_node_kernel_ctor_tag{},
                kernel_t{std::move(arg_name), policy.space(),
                         (Functor &&) functor, (Policy &&) policy},
                // *this is the predecessor
                Kokkos::Impl::_graph_node_predecessor_ctor_tag{}, *this},
            typename return_t::node_impl_t::Deleter{}}};
    // Add the node itself to the backend's graph data structure, now that
    // everything is set up.
    rv.m_node_impl->add_node(*m_graph_impl);
    // Add the predecessaor we stored in the constructor above in the backend's
    // data structure, now that everything is set up.
    rv.m_node_impl->add_predecessor(*m_graph_impl);
    KOKKOS_ENSURES(bool(rv.m_node_impl) && bool(rv.m_graph_impl))
    return rv;
  }

  template <class Policy, class Functor>
  // requires ExecutionPolicy<Policy> && DataParallelFunctor<Functor>
  auto then_parallel_for(Policy&& policy, Functor&& functor) {
    return this->then_parallel_for("", (Policy &&) policy,
                                   (Functor &&) functor);
  }
};

}  // end namespace Experimental

namespace Impl {
// Customizable for backends
template <class ExecutionSpace>
struct GraphNodeBackendSpecificDetails;

// Customizable for backends
template <class ExecutionSpace, class Kernel, class PredecessorRef>
struct GraphNodeBackendDetailsBeforeTypeErasure {
 protected:
  // Required:
  GraphNodeBackendDetailsBeforeTypeErasure(ExecutionSpace const&, Kernel&,
                                           PredecessorRef const&) noexcept {}
  GraphNodeBackendDetailsBeforeTypeErasure(
      ExecutionSpace const&, _graph_node_is_root_ctor_tag,
      GraphNodeBackendSpecificDetails<ExecutionSpace>&
          this_as_details) noexcept {}

  // Not copyable or movable at the concept level, so the default
  // implementation shouldn't be either.
  GraphNodeBackendDetailsBeforeTypeErasure() noexcept = delete;

  GraphNodeBackendDetailsBeforeTypeErasure(
      GraphNodeBackendDetailsBeforeTypeErasure const&) = delete;

  GraphNodeBackendDetailsBeforeTypeErasure(
      GraphNodeBackendDetailsBeforeTypeErasure&&) noexcept = delete;

  GraphNodeBackendDetailsBeforeTypeErasure& operator   =(
      GraphNodeBackendDetailsBeforeTypeErasure const&) = delete;

  GraphNodeBackendDetailsBeforeTypeErasure& operator       =(
      GraphNodeBackendDetailsBeforeTypeErasure&&) noexcept = delete;

  ~GraphNodeBackendDetailsBeforeTypeErasure() = default;
};

template <class ExecutionSpace>
struct GraphNodeImpl<ExecutionSpace, Experimental::TypeErasedTag,
                     Experimental::TypeErasedTag>
    : GraphNodeBackendSpecificDetails<ExecutionSpace> {
 protected:
  // For now, we're effectively storing our own mini-vtable here in the object
  // representation rather than making a vtable entry in the binary for every
  // kernel used with graphs in the whole application, which we think might
  // not be scalable. We need to test whether this is necessary at scale,
  // though this is not really in a super performance-sensitive part of the
  // code, so it should be fine except for maybe the additional complexity in
  // the code.
  using add_node_callback_t        = void (*)(GraphNodeImpl&,
                                       GraphImpl<ExecutionSpace>&);
  using add_predecessor_callback_t = void (*)(GraphNodeImpl&,
                                              GraphImpl<ExecutionSpace>&);
  using destroy_this_callback_t    = void (*)(GraphNodeImpl&);
  using implementation_base_t = GraphNodeBackendSpecificDetails<ExecutionSpace>;

 private:
  add_node_callback_t m_add_node;
  add_predecessor_callback_t m_add_predecessor;
  destroy_this_callback_t m_destroy_this;

  // For the use case where the backend has no special kernel or predecessor
  // information to stuff in to the root node and thus this is the most-derived
  // type
  static void destroy_this_fn(GraphNodeImpl& arg_this) noexcept {
    auto const& this_ = static_cast<GraphNodeImpl const&>(arg_this);
    this_.~GraphNodeImpl();
  }

 protected:
  GraphNodeImpl(add_node_callback_t arg_add_node,
                add_predecessor_callback_t arg_add_predecessor,
                destroy_this_callback_t arg_destroy_this,
                ExecutionSpace const& ex) noexcept
      : implementation_base_t(ex),
        m_add_node(arg_add_node),
        m_add_predecessor(arg_add_predecessor),
        m_destroy_this(arg_destroy_this) {}

  // Not publicly destructible so that we don't forget to use the vtable entry
  ~GraphNodeImpl() = default;

 public:
  template <class... Args>
  GraphNodeImpl(destroy_this_callback_t arg_destroy_this,
                ExecutionSpace const& ex, _graph_node_is_root_ctor_tag,
                Args&&... args) noexcept
      : implementation_base_t(ex, _graph_node_is_root_ctor_tag{},
                              (Args &&) args...),
        m_add_node(nullptr),
        m_add_predecessor(nullptr),
        m_destroy_this(arg_destroy_this) {}

  // For the use case where the backend has no special kernel or predecessor
  // information to stuff in to the root node and thus this is the most-derived
  // type
  template <class... Args>
  GraphNodeImpl(ExecutionSpace const& ex, _graph_node_is_root_ctor_tag,
                Args&&... args) noexcept
      : GraphNodeImpl(&destroy_this_fn, ex, _graph_node_is_root_ctor_tag{},
                      (Args &&) args...) {}
  struct Deleter {
    void operator()(GraphNodeImpl* ptr) const noexcept {
      (*ptr->m_destroy_this)(*ptr);
    }
  };

  //----------------------------------------------------------------------------
  // <editor-fold desc="no other constructors"> {{{2

  GraphNodeImpl()                     = delete;
  GraphNodeImpl(GraphNodeImpl const&) = delete;
  GraphNodeImpl(GraphNodeImpl&&)      = delete;
  GraphNodeImpl& operator=(GraphNodeImpl const&) = delete;
  GraphNodeImpl& operator=(GraphNodeImpl&&) = delete;

  // </editor-fold> end no other constructors }}}2
  //----------------------------------------------------------------------------

  void add_node(GraphImpl<ExecutionSpace>& graph) {
    (*m_add_node)(*this, graph);
  }

  void add_predecessor(GraphImpl<ExecutionSpace>& graph) {
    (*m_add_predecessor)(*this, graph);
  }

  ExecutionSpace const& execution_space_instance() const {
    return this->implementation_base_t::execution_space_instance();
  }
};

template <class ExecutionSpace, class Kernel>
struct GraphNodeImpl<ExecutionSpace, Kernel, Experimental::TypeErasedTag>
    : GraphNodeImpl<ExecutionSpace, Experimental::TypeErasedTag,
                    Experimental::TypeErasedTag> {
 private:
  Kernel m_kernel;

  using base_t = GraphNodeImpl<ExecutionSpace, Experimental::TypeErasedTag,
                               Experimental::TypeErasedTag>;

  //----------------------------------------------------------------------------
  // <editor-fold desc="Ctors, destructors, and assignment"> {{{2

 protected:
  template <class KernelDeduced>
  GraphNodeImpl(typename base_t::add_node_callback_t arg_add_node,
                typename base_t::add_predecessor_callback_t arg_add_predecessor,
                typename base_t::destroy_this_callback_t arg_destroy_this,
                ExecutionSpace const& ex, _graph_node_kernel_ctor_tag,
                KernelDeduced&& arg_kernel)
      : base_t(arg_add_node, arg_add_predecessor, arg_destroy_this, ex),
        m_kernel((KernelDeduced &&) arg_kernel) {}

 public:
  template <class... Args>
  GraphNodeImpl(ExecutionSpace const& ex, _graph_node_is_root_ctor_tag,
                Args&&... args)
      : base_t(ex, _graph_node_is_root_ctor_tag{}, (Args &&) args...) {}

  Kernel& get_kernel() & { return m_kernel; }
  Kernel const& get_kernel() const& { return m_kernel; }

  // Not copyable or movable
  GraphNodeImpl()                         = delete;
  GraphNodeImpl(GraphNodeImpl const&)     = delete;
  GraphNodeImpl(GraphNodeImpl&&) noexcept = delete;
  GraphNodeImpl& operator=(GraphNodeImpl const&) = delete;
  GraphNodeImpl& operator=(GraphNodeImpl&&) noexcept = delete;
  ~GraphNodeImpl()                                   = default;

  // </editor-fold> end Ctors, destructors, and assignment }}}2
  //----------------------------------------------------------------------------
};

//==============================================================================

template <class ExecutionSpace, class Kernel, class PredecessorRef>
struct GraphNodeImpl
    : GraphNodeImpl<ExecutionSpace, Kernel, Experimental::TypeErasedTag>,
      GraphNodeBackendDetailsBeforeTypeErasure<ExecutionSpace, Kernel,
                                               PredecessorRef>,
      // TODO @graph remove this if it's unnecessary, or document why it is
      std::enable_shared_from_this<
          GraphNodeImpl<ExecutionSpace, Kernel, PredecessorRef>> {
 private:
  PredecessorRef m_predecessor_ref;

  using type_erased_base_t =
      GraphNodeImpl<ExecutionSpace, Experimental::TypeErasedTag,
                    Experimental::TypeErasedTag>;
  using base_t =
      GraphNodeImpl<ExecutionSpace, Kernel, Experimental::TypeErasedTag>;
  using backend_details_base_t =
      GraphNodeBackendDetailsBeforeTypeErasure<ExecutionSpace, Kernel,
                                               PredecessorRef>;

 public:
  //----------------------------------------------------------------------------
  // <editor-fold desc="add_node"> {{{2

  static void add_node_fn(type_erased_base_t& arg_this,
                          GraphImpl<ExecutionSpace>& graph) {
    auto& this_ = static_cast<GraphNodeImpl&>(arg_this);
    this_.add_node(graph);
  }

  // Note: this intentionally shadows the name in the base class
  void add_node(GraphImpl<ExecutionSpace>& graph) {
    graph.add_node(*this);
  }

  // </editor-fold> end add_node }}}2
  //------------------------------------------------------------------------------

  //----------------------------------------------------------------------------
  // <editor-fold desc="add_predecessor"> {{{2

  static void add_predecessor_fn(type_erased_base_t& arg_this,
                                 GraphImpl<ExecutionSpace>& graph) {
    auto& this_ = static_cast<GraphNodeImpl&>(arg_this);
    this_.add_predecessor(graph);
  }

  // Note: this intentionally shadows the name in the base class
  void add_predecessor(GraphImpl<ExecutionSpace>& graph) {
    graph.add_predecessor(*this, m_predecessor_ref);
  }

  // </editor-fold> end add_predecessor }}}2
  //----------------------------------------------------------------------------

  static void destroy_this_fn(type_erased_base_t& arg_this) noexcept {
    auto& this_ = static_cast<GraphNodeImpl&>(arg_this);
    this_.~GraphNodeImpl();
  }

  //----------------------------------------------------------------------------
  // <editor-fold desc="Ctors, destructors, and assignment"> {{{2

  // Not copyable or movable
  GraphNodeImpl()                     = delete;
  GraphNodeImpl(GraphNodeImpl const&) = delete;
  GraphNodeImpl(GraphNodeImpl&&)      = delete;
  GraphNodeImpl& operator=(GraphNodeImpl const&) = delete;
  GraphNodeImpl& operator=(GraphNodeImpl&&) noexcept = delete;
  ~GraphNodeImpl()                                   = default;

  template <class KernelDeduced, class PredecessorPtrDeduced>
  GraphNodeImpl(ExecutionSpace const& ex, _graph_node_kernel_ctor_tag,
                KernelDeduced&& arg_kernel, _graph_node_predecessor_ctor_tag,
                PredecessorPtrDeduced&& arg_predecessor)
      : base_t(&add_node_fn, &add_predecessor_fn, &destroy_this_fn, ex,
               _graph_node_kernel_ctor_tag{}, (KernelDeduced &&) arg_kernel),
        // The backend gets the ability to store (weak, non-owning) references
        // to the kernel in it's final resting place here if it wants. The
        // predecessor is already a pointer, so it doesn't matter that it isn't
        // already at its final address
        backend_details_base_t(ex, this->base_t::get_kernel(), arg_predecessor),
        m_predecessor_ref((PredecessorPtrDeduced &&) arg_predecessor) {}

  template <class... Args>
  GraphNodeImpl(ExecutionSpace const& ex, _graph_node_is_root_ctor_tag,
                Args&&... args)
      : base_t(&destroy_this_fn, ex, _graph_node_is_root_ctor_tag{},
               (Args &&) args...),
        backend_details_base_t(ex, _graph_node_is_root_ctor_tag{}, *this),
        m_predecessor_ref() {}

  // </editor-fold> end Ctors, destructors, and assignment }}}2
  //------------------------------------------------------------------------------
};

// TODO move this to a more appropriate place
struct AlwaysDeduceThisTemplateParameter;

}  // end namespace Impl

//==============================================================================

namespace Experimental {

template <class ExecutionSpace>
struct KOKKOS_ATTRIBUTE_NODISCARD Graph {
 private:
  using impl_t = Kokkos::Impl::GraphImpl<ExecutionSpace>;

  std::shared_ptr<impl_t> m_impl_ptr = nullptr;

  friend struct Kokkos::Impl::GraphAccess;

  explicit Graph(std::shared_ptr<impl_t> arg_impl_ptr)
      : m_impl_ptr(std::move(arg_impl_ptr)) {}

 public:
  ExecutionSpace const& get_execution_space() const {
    return m_impl_ptr->get_execution_space();
  }

  void submit() const {
    KOKKOS_EXPECTS(bool(m_impl_ptr))
    (*m_impl_ptr).submit();
  }
};

template <class ExecutionSpace, class Closure>
Graph<ExecutionSpace> create_graph(ExecutionSpace ex, Closure&& arg_closure) {
  auto rv   = Kokkos::Impl::GraphAccess::construct_graph(ex);
  auto root = Kokkos::Impl::GraphAccess::create_root_ref(rv);
  ((Closure &&) arg_closure)(root);
  return rv;
}

template <class ExecutionSpace = DefaultExecutionSpace,
          class Closure = Kokkos::Impl::AlwaysDeduceThisTemplateParameter>
Graph<ExecutionSpace> create_graph(Closure&& arg_closure) {
  return create_graph(ExecutionSpace{}, (Closure &&) arg_closure);
}

}  // end namespace Experimental

}  // namespace Kokkos

#include "impl/Kokkos_Host_Graph_Impl.hpp"

#endif  // KOKKOS_KOKKOS_GRAPH_HPP
