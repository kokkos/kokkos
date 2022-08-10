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

#ifndef KOKKOS_KOKKOS_OPENACC_FUNCTOR_ADAPTER_HPP
#define KOKKOS_KOKKOS_OPENACC_FUNCTOR_ADAPTER_HPP

#include <type_traits>

namespace Kokkos::Experimental::Impl {

template <class Functor, class Policy,
          bool = std::is_void_v<typename Policy::work_tag>>
class FunctorAdapter {
  Functor m_functor;

 public:
  FunctorAdapter(Functor const &functor) : m_functor(functor) {}

  template <class... Args>
  KOKKOS_FUNCTION void operator()(Args &&...args) const {
    m_functor(static_cast<Args &&>(args)...);
  }
};

template <class Functor, class Policy>
class FunctorAdapter<Functor, Policy, false> {
  Functor m_functor;
  using WorkTag = typename Policy::work_tag;

 public:
  FunctorAdapter(Functor const &functor) : m_functor(functor) {}

  template <class... Args>
  KOKKOS_FUNCTION void operator()(Args &&...args) const {
    m_functor(WorkTag(), static_cast<Args &&>(args)...);
  }
};

}  // namespace Kokkos::Experimental::Impl

#endif
