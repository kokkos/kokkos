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

#ifndef KOKKOS_KOKKOS_CUDA_GRAPHNODE_IMPL_HPP
#define KOKKOS_KOKKOS_CUDA_GRAPHNODE_IMPL_HPP

#include <Kokkos_Macros.hpp>

#ifdef KOKKOS_ENABLE_CUDA

#include <Kokkos_Graph_fwd.hpp>

#include <impl/Kokkos_GraphImpl.hpp>  // GraphAccess needs to be complete

#include <Kokkos_Cuda.hpp>
#include <cuda_runtime_api.h>

namespace Kokkos {
namespace Impl {

template <>
struct GraphNodeBackendSpecificDetails<Kokkos::Cuda> {
  cudaGraphNode_t node = nullptr;

  //----------------------------------------------------------------------------
  // <editor-fold desc="Ctors, destructor, and assignment"> {{{2

  explicit GraphNodeBackendSpecificDetails() = default;

  GraphNodeBackendSpecificDetails(_graph_node_is_root_ctor_tag) noexcept {}

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
};

template <class Kernel, class PredecessorRef>
struct GraphNodeBackendDetailsBeforeTypeErasure<Kokkos::Cuda, Kernel,
                                                PredecessorRef> {
 protected:
  //----------------------------------------------------------------------------
  // <editor-fold desc="ctors, destructor, and assignment"> {{{2

  GraphNodeBackendDetailsBeforeTypeErasure(
      Kokkos::Cuda const&, Kernel& kernel, PredecessorRef const&,
      GraphNodeBackendSpecificDetails<Kokkos::Cuda>&) noexcept {}

  GraphNodeBackendDetailsBeforeTypeErasure(
      Kokkos::Cuda const&, _graph_node_is_root_ctor_tag,
      GraphNodeBackendSpecificDetails<Kokkos::Cuda>& this_as_details) noexcept {
  }

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

  // </editor-fold> end ctors, destructor, and assignment }}}2
  //----------------------------------------------------------------------------
};

}  // end namespace Impl
}  // end namespace Kokkos

#include <Cuda/Kokkos_Cuda_GraphNodeKernel.hpp>

#endif  // defined(KOKKOS_ENABLE_CUDA)
#endif  // KOKKOS_KOKKOS_CUDA_GRAPHNODE_IMPL_HPP
