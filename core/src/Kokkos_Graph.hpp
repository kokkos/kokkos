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

#include <Kokkos_Macros.hpp>
#include <impl/Kokkos_Error.hpp>  // KOKKOS_EXPECTS

#include <Kokkos_Graph_fwd.hpp>
#include <impl/Kokkos_GraphImpl_fwd.hpp>

#include <impl/Kokkos_Utilities.hpp>  // fold emulation

#include <functional>
#include <memory>

namespace Kokkos {
namespace Experimental {

//==============================================================================
// <editor-fold desc="Graph"> {{{1

template <class ExecutionSpace>
struct KOKKOS_ATTRIBUTE_NODISCARD Graph {
 public:
  //----------------------------------------------------------------------------
  // <editor-fold desc="public member types"> {{{2

  using execution_space = ExecutionSpace;
  using graph           = Graph;
  using graph_builder   = GraphBuilder<ExecutionSpace>;

  // </editor-fold> end public member types }}}2
  //----------------------------------------------------------------------------

 private:
  //----------------------------------------------------------------------------
  // <editor-fold desc="friends"> {{{2

  friend struct Kokkos::Impl::GraphAccess;

  // </editor-fold> end friends }}}2
  //----------------------------------------------------------------------------

  //----------------------------------------------------------------------------
  // <editor-fold desc="private data members"> {{{2

  using impl_t                       = Kokkos::Impl::GraphImpl<ExecutionSpace>;
  std::shared_ptr<impl_t> m_impl_ptr = nullptr;

  // </editor-fold> end private data members }}}2
  //----------------------------------------------------------------------------

  //----------------------------------------------------------------------------
  // <editor-fold desc="private ctors"> {{{2

  // Note: only create_graph() uses this constructor, but we can't just make
  // that a friend instead of GraphAccess because of the way that friend
  // function template injection works.
  explicit Graph(std::shared_ptr<impl_t> arg_impl_ptr)
      : m_impl_ptr(std::move(arg_impl_ptr)) {}

  // </editor-fold> end private ctors }}}2
  //----------------------------------------------------------------------------

 public:
  ExecutionSpace const& get_execution_space() const {
    return m_impl_ptr->get_execution_space();
  }

  void submit() const& {
    KOKKOS_EXPECTS(bool(m_impl_ptr))
    (*m_impl_ptr).submit();
  }

  void submit() && {
    KOKKOS_EXPECTS(bool(m_impl_ptr))
    // The graph interface isn't thread-safe, so we can rely on this
    if (m_impl_ptr.use_count() == 1) {
      std::move(*m_impl_ptr).submit();
    } else {
      (*m_impl_ptr).submit();
    }
  }
};

// </editor-fold> end Graph }}}1
//==============================================================================

}  // end namespace Experimental
}  // namespace Kokkos

#endif  // KOKKOS_GRAPH_HPP
