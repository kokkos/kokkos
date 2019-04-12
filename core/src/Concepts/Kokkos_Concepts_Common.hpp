/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
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

#include <Properties/Kokkos_Detection.hpp>

#ifndef KOKKOS_CONCEPTS_KOKKOS_CONCEPTSCOMMON_HPP
#define KOKKOS_CONCEPTS_KOKKOS_CONCEPTSCOMMON_HPP

namespace Kokkos {
namespace Impl {
namespace Concepts {

// A lot of things have execution_space nested types...
KOKKOS_DECLARE_DETECTION_ARCHETYPE_1PARAM(
  _intrusive_execution_space_archetype, T,
  typename T::execution_space
);

// Several concepts use this (at least Functor and Reducer)
KOKKOS_DECLARE_DETECTION_ARCHETYPE_1PARAM(
  _intrusive_value_type, F,
  typename F::value_type
);

// Both ReductionFunctor and Reducer use this archetype
KOKKOS_DECLARE_DETECTION_ARCHETYPE_1PARAM_VARIADIC(
  _intrusive_init_archetype, F, Args,
  decltype(
    Impl::declval<F>().init(Impl::declval<Args>()...)
  )
);

// Both ReductionFunctor and Reducer use this archetype
KOKKOS_DECLARE_DETECTION_ARCHETYPE_1PARAM_VARIADIC(
  _intrusive_join_archetype, F, Args,
  decltype(
    Impl::declval<F>().join(Impl::declval<Args>()...)
  )
);

// Both ReductionFunctor and Reducer use this archetype
KOKKOS_DECLARE_DETECTION_ARCHETYPE_1PARAM_VARIADIC(
  _intrusive_final_archetype, F, Args,
  decltype(
    Impl::declval<F>().final(Impl::declval<Args>()...)
  )
);


} // end namespace Concepts
} // end namespace Impl
} // end namespace Kokkos

#endif //KOKKOS_CONCEPTS_KOKKOS_CONCEPTSCOMMON_HPP
