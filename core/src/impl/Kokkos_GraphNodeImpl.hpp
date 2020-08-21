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

#ifndef KOKKOS_IMPL_GRAPHNODEIMPL_HPP
#define KOKKOS_IMPL_GRAPHNODEIMPL_HPP

#include <Kokkos_Macros.hpp>

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Graph_fwd.hpp>

#include <impl/Kokkos_SimpleTaskScheduler.hpp>  // ExecutionSpaceInstanceStorage
#include <impl/Kokkos_GraphImpl.hpp>

#include <impl/Kokkos_EBO.hpp>

#include <memory>

namespace Kokkos {
namespace Impl {

// Base specialization for the case where both the predecessor and the kernel
// type information is type-erased
template <class ExecutionSpace>
struct GraphNodeImpl<ExecutionSpace, Kokkos::Experimental::TypeErasedTag,
                     Kokkos::Experimental::TypeErasedTag>
    : GraphNodeBackendSpecificDetails<ExecutionSpace>,
      ExecutionSpaceInstanceStorage<ExecutionSpace> {
 public:
  using node_ref_t =
      Kokkos::Experimental::GraphNodeRef<ExecutionSpace,
                                         Kokkos::Experimental::TypeErasedTag,
                                         Kokkos::Experimental::TypeErasedTag>;

 protected:
  // For now, we're effectively storing our own mini-vtable here in the object
  // representation rather than making a vtable entry in the binary for every
  // kernel used with graphs in the whole application, which we think might
  // not be scalable. We need to test whether this is necessary at scale,
  // though this is not really in a super performance-sensitive part of the
  // code, so it should be fine except for maybe the additional complexity in
  // the code.
  using destroy_this_callback_t = void (*)(GraphNodeImpl&);
  using implementation_base_t = GraphNodeBackendSpecificDetails<ExecutionSpace>;
  using execution_space_storage_base_t =
      ExecutionSpaceInstanceStorage<ExecutionSpace>;

 private:
  destroy_this_callback_t m_destroy_this;

  // For the use case where the backend has no special kernel or predecessor
  // information to stuff in to the root node and thus this is the most-derived
  // type
  static void destroy_this_fn(GraphNodeImpl& arg_this) noexcept {
    auto const& this_ = static_cast<GraphNodeImpl const&>(arg_this);
    this_.~GraphNodeImpl();
  }

 protected:
  GraphNodeImpl(destroy_this_callback_t arg_destroy_this,
                ExecutionSpace const& ex) noexcept
      : implementation_base_t(),
        execution_space_storage_base_t(ex),
        m_destroy_this(arg_destroy_this) {}

  // Not publicly destructible so that we don't forget to use the vtable entry
  ~GraphNodeImpl() = default;

 public:
  template <class... Args>
  GraphNodeImpl(destroy_this_callback_t arg_destroy_this,
                ExecutionSpace const& ex, _graph_node_is_root_ctor_tag,
                Args&&... args) noexcept
      : implementation_base_t(_graph_node_is_root_ctor_tag{},
                              (Args &&) args...),
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

  ExecutionSpace const& execution_space_instance() const {
    return this->execution_space_storage_base_t::execution_space_instance();
  }
};

//==============================================================================

// Specialization for the case with the concrete type of the kernel, but the
// predecessor erased.
template <class ExecutionSpace, class Kernel>
struct GraphNodeImpl<ExecutionSpace, Kernel,
                     Kokkos::Experimental::TypeErasedTag>
    : GraphNodeImpl<ExecutionSpace, Kokkos::Experimental::TypeErasedTag,
                    Kokkos::Experimental::TypeErasedTag> {
 public:
  using node_ref_t =
      Kokkos::Experimental::GraphNodeRef<ExecutionSpace, Kernel,
                                         Kokkos::Experimental::TypeErasedTag>;
  using kernel_type = Kernel;

 private:
  Kernel m_kernel;

  using base_t =
      GraphNodeImpl<ExecutionSpace, Kokkos::Experimental::TypeErasedTag,
                    Kokkos::Experimental::TypeErasedTag>;

 public:
  static void destroy_this_fn(base_t& arg_this) noexcept {
    auto& this_ = static_cast<GraphNodeImpl&>(arg_this);
    this_.~GraphNodeImpl();
  }

  //----------------------------------------------------------------------------
  // <editor-fold desc="Ctors, destructors, and assignment"> {{{2

 protected:
  template <class KernelDeduced>
  GraphNodeImpl(typename base_t::destroy_this_callback_t arg_destroy_this,
                ExecutionSpace const& ex, _graph_node_kernel_ctor_tag,
                KernelDeduced&& arg_kernel)
      : base_t(arg_destroy_this, ex), m_kernel((KernelDeduced &&) arg_kernel) {}

 public:
  template <class... Args>
  GraphNodeImpl(ExecutionSpace const& ex, _graph_node_is_root_ctor_tag,
                Args&&... args)
      : base_t(ex, _graph_node_is_root_ctor_tag{}, (Args &&) args...) {}

  template <class KernelDeduced>
  GraphNodeImpl(ExecutionSpace const& ex, _graph_node_kernel_ctor_tag,
                KernelDeduced&& arg_kernel)
      : base_t(&destroy_this_fn, ex), m_kernel((KernelDeduced &&) arg_kernel) {}

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

// Specialization for the case where nothing is type-erased
template <class ExecutionSpace, class Kernel, class PredecessorRef>
struct GraphNodeImpl
    : GraphNodeImpl<ExecutionSpace, Kernel,
                    Kokkos::Experimental::TypeErasedTag>,
      GraphNodeBackendDetailsBeforeTypeErasure<ExecutionSpace, Kernel,
                                               PredecessorRef> {
 public:
  using node_ref_t = Kokkos::Experimental::GraphNodeRef<ExecutionSpace, Kernel,
                                                        PredecessorRef>;

 private:
  PredecessorRef m_predecessor_ref;

  using type_erased_base_t =
      GraphNodeImpl<ExecutionSpace, Kokkos::Experimental::TypeErasedTag,
                    Kokkos::Experimental::TypeErasedTag>;
  using base_t = GraphNodeImpl<ExecutionSpace, Kernel,
                               Kokkos::Experimental::TypeErasedTag>;
  using backend_details_base_t =
      GraphNodeBackendDetailsBeforeTypeErasure<ExecutionSpace, Kernel,
                                               PredecessorRef>;

 public:
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
      : base_t(&destroy_this_fn, ex, _graph_node_kernel_ctor_tag{},
               (KernelDeduced &&) arg_kernel),
        // The backend gets the ability to store (weak, non-owning) references
        // to the kernel in it's final resting place here if it wants. The
        // predecessor is already a pointer, so it doesn't matter that it isn't
        // already at its final address
        backend_details_base_t(ex, this->base_t::get_kernel(), arg_predecessor, *this),
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

}  // end namespace Impl
}  // end namespace Kokkos

#endif  // KOKKOS_IMPL_GRAPHNODEIMPL_HPP
