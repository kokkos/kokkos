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

#ifndef KOKKOS_CONCEPTS_FUNCTOR_IMPL_KOKKOS_FUNCTOR_TEAMSHMEMSIZE_HPP
#define KOKKOS_CONCEPTS_FUNCTOR_IMPL_KOKKOS_FUNCTOR_TEAMSHMEMSIZE_HPP

#include <Concepts/Kokkos_Concepts_Macros.hpp>

#include <Concepts/Functor/impl/Kokkos_ArrayReductionFunctor_ValueCount.hpp>
#include <Concepts/Functor/Kokkos_Functor_Concept.hpp>

#include <Properties/Kokkos_Detection.hpp>

#include <impl/Kokkos_Utilities.hpp> // Kokkos::forward


namespace Kokkos {
namespace Impl {
namespace Concepts {

//==============================================================================
// <editor-fold desc="team_shmem_size detection helpers"> {{{1

KOKKOS_DECLARE_DETECTION_ARCHETYPE_1PARAM(
  _intrusive_shmem_size_archetype, F,
  decltype(
    Impl::declval<F>().shmem_size(int())
  )
);

KOKKOS_DECLARE_DETECTION_ARCHETYPE_1PARAM(
  _intrusive_team_shmem_size_archetype, F,
  decltype(
    Impl::declval<F>().team_shmem_size(int())
  )
);

template <class F>
struct functor_has_intrusive_shmem_size
  : is_detected_convertible_t<size_t, _intrusive_shmem_size_archetype, F> { };


template <class F>
struct functor_has_intrusive_team_shmem_size
  : is_detected_convertible_t<size_t, _intrusive_team_shmem_size_archetype, F> { };


// </editor-fold> end team_shmem_size detection helpers }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="Functor team_shmem_size"> {{{1

template <class F>
KOKKOS_CUSTOMIZATION_POINT_FUNCTION
typename std::enable_if<
  functor_has_intrusive_shmem_size<F>::value,
  size_t
>::type
functor_team_shmem_size(F&& f, int team_size)
  noexcept(noexcept(Impl::declval<F>().shmem_size(team_size)))
{
  return Impl::forward<F>(f).shmem_size(team_size);
}


template <class F>
KOKKOS_CUSTOMIZATION_POINT_FUNCTION
typename std::enable_if<
  functor_has_intrusive_team_shmem_size<F>::value,
  size_t
>::type
functor_team_shmem_size(F&& f, int team_size)
  noexcept(noexcept(Impl::declval<F>().team_shmem_size(team_size)))
{
  return Impl::forward<F>(f).team_shmem_size(team_size);
}

template <class F>
constexpr
KOKKOS_CUSTOMIZATION_POINT_FUNCTION
typename std::enable_if<
  !functor_has_intrusive_shmem_size<F>::value
  && !functor_has_intrusive_team_shmem_size<F>::value,
  size_t
>::type
functor_team_shmem_size(F&&, int) noexcept
{
  return 0;
}

// </editor-fold> end Functor team_shmem_size }}}1
//==============================================================================


} // end namespace Concepts
} // end namespace Impl
} // end namespace Kokkos

#endif //KOKKOS_CONCEPTS_FUNCTOR_IMPL_KOKKOS_FUNCTOR_TEAMSHMEMSIZE_HPP
