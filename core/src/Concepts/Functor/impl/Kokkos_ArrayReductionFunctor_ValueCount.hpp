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

#ifndef KOKKOS_CONCEPTS_FUNCTOR_IMPL_ARRAYREDUCTIONFUNCTOR_VALUECOUNT_HPP
#define KOKKOS_CONCEPTS_FUNCTOR_IMPL_ARRAYREDUCTIONFUNCTOR_VALUECOUNT_HPP

#include <Concepts/Kokkos_Concepts_Macros.hpp>

#include <Properties/Kokkos_Detection.hpp>

namespace Kokkos {
namespace Impl {
namespace Concepts {

//==============================================================================
// <editor-fold desc="ArrayReductionFunctor value_count"> {{{1

// ArrayReductionFunctor has the additional requirement of a value_count data member
KOKKOS_DECLARE_DETECTION_ARCHETYPE_1PARAM(
  _intrusive_value_count_archetype, F,
  decltype(
    Impl::declval<F>().value_count
  )
);

KOKKOS_DECLARE_DETECTION_ARCHETYPE_1PARAM(
  _non_intrusive_value_count_archetype, F,
  decltype(
    value_count(Impl::declval<F>())
  )
);

template <class F>
struct array_reduction_functor_has_intrusive_value_count :
  std::integral_constant<bool,
    is_detected_convertible<unsigned, _intrusive_value_count_archetype, F>::value
  >
{ };

template <class F>
struct value_type_has_intrusive_value_count :
  std::integral_constant<bool,
    is_detected_convertible<unsigned, _intrusive_value_count_archetype, F>::value
  >
{ };

template <class F>
struct value_type_has_non_intrusive_value_count :
  std::integral_constant<bool,
    is_detected_convertible<unsigned, _non_intrusive_value_count_archetype, F>::value
  >
{ };

/**
 *  Overload for functors with an intrusive `value_count` data member.
 *
 *  Returns `f.value_count`.
 *
 */
template <class F, class V>
KOKKOS_CUSTOMIZATION_POINT_FUNCTION
typename std::enable_if<
  array_reduction_functor_has_intrusive_value_count<F>::value,
  unsigned
>::type
array_reduction_functor_value_count(F const& f, V const&)
  noexcept(noexcept(unsigned(f.value_count)))
{
  return f.value_count;
}

/**
 *  Overload for value types with an intrusive `value_count` data member.
 *
 *  Returns `v.value_count`.
 *
 */
template <class F, class V>
KOKKOS_CUSTOMIZATION_POINT_FUNCTION
typename std::enable_if<
  !array_reduction_functor_has_intrusive_value_count<F>::value
  && value_type_has_intrusive_value_count<V>::value,
  unsigned
>::type
array_reduction_functor_value_count(F const&, V const& value)
  noexcept(noexcept(unsigned(value.value_count)))
{
  return value.value_count;
}

/**
 *  Overload for value types with a non-intrusive `value_count`.
 *
 *  Returns `value_count(v)`.
 *
 */
template <class F, class V>
KOKKOS_CUSTOMIZATION_POINT_FUNCTION
typename std::enable_if<
  !array_reduction_functor_has_intrusive_value_count<F>::value
    && !value_type_has_intrusive_value_count<V>::value
    && value_type_has_non_intrusive_value_count<V>::value,
  unsigned
>::type
array_reduction_functor_value_count(F const&, V const& v)
  noexcept(noexcept(unsigned(value_count(v))))
{
  return value_count(v);
}

//------------------------------------------------------------------------------

KOKKOS_DECLARE_DETECTION_ARCHETYPE_2PARAMS(
  _array_reduction_functor_has_value_count_archetype, F, V,
  decltype(
    Concepts::array_reduction_functor_value_count(
      Impl::declval<F>(), Impl::declval<V>()
    )
  )
);

template <class F, class V>
struct array_reduction_functor_has_value_count
  : is_detected_t<
      _array_reduction_functor_has_value_count_archetype, F, V
    > { };

// </editor-fold> end ArrayReductionFunctor value_count }}}1
//==============================================================================


} // end namespace Concepts
} // end namespace Impl
} // end namespace Kokkos

#endif //KOKKOS_CONCEPTS_FUNCTOR_IMPL_ARRAYREDUCTIONFUNCTOR_VALUECOUNT_HPP
