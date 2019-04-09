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

#ifndef KOKKOS_CONCEPTS_FUNCTOR_KOKKOS_REDUCTIONFUNCTOR_HPP
#define KOKKOS_CONCEPTS_FUNCTOR_KOKKOS_REDUCTIONFUNCTOR_HPP

#include <Properties/Kokkos_Detection.hpp>
#include <Concepts/Functor/Kokkos_Functor_Concept.hpp>

#include <impl/Kokkos_Utilities.hpp> // Kokkos::forward


namespace Kokkos {
namespace Impl {
namespace Concepts {

//==============================================================================
// <editor-fold desc="ArrayReductionFunctor value_count"> {{{1

// ArrayReductionFunctor has the additional requirement of a value_count data member
KOKKOS_DECLARE_DETECTION_ARCHETYPE_1PARAM(
  _intrusive_value_count_archetype, F,
  decltype(
  declval<F>().value_count
  )
);

template <class F>
struct array_reduction_functor_has_intrusive_value_count :
  std::integral_constant<bool,
    is_detected_convertible<unsigned, _intrusive_value_count_archetype, F>::value
  >
{ };

template <class F>
typename std::enable_if<
  array_reduction_functor_has_intrusive_value_count<F>::value,
  unsigned
>
array_reduction_functor_value_count(F const& f)
  noexcept(noexcept(unsigned(f.value_count)))
{
  return f.value_count;
}



// </editor-fold> end ArrayReductionFunctor value_count }}}1
//==============================================================================


//==============================================================================
// <editor-fold desc="ReductionFunctor join"> {{{1

KOKKOS_DECLARE_DETECTION_ARCHETYPE_1PARAM_VARIADIC(
  _intrusive_join_archetype, F, Args,
  decltype(
    declval<F>().join(declval<Args>()...)
  )
);

// The inheritance version gives more readable error messages...
// TODO measure the impact of this versus the type alias version on compilation time and provide a macro that switches between them if necessary
template <class F, class... Args>
struct reduction_functor_has_intrusive_join
  : is_detected<_intrusive_join_archetype, F, Args...> { };


KOKKOS_DECLARE_DETECTION_ARCHETYPE_2PARAMS(
  _plus_equal_op_archetype, T, U,
  decltype(
    declval<T>() += declval<U>()
  )
);

template <class T, class U>
struct has_plus_equal_op
  : std::integral_constant<bool,
      is_detected<_plus_equal_op_archetype, T, U>::value
    > { };

KOKKOS_DECLARE_DETECTION_ARCHETYPE_4PARAMS(
  _square_bracket_plus_equal_op_archetype, T, I1, U, I2,
  decltype(
    declval<T>()[declval<I1>()] += declval<U>(declval<I2>())
  )
);

template <class T, class I1, class U, class I2>
struct square_bracket_result_has_plus_equal_op
  : std::integral_constant<bool,
    is_detected<_square_bracket_plus_equal_op_archetype, T, I1, U, I2>::value
  > { };

// TODO @concepts DSH make this a customization point object (so that ADL is never prefered over intrusive customization)
/**
 *  Overload for functors with intrusive join available.
 *
 *  Just calls `f.join(args...)`, but with perfect forwarding
 */
template <class F, class... Args>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
  reduction_functor_has_intrusive_join<F, Args...>::value
>::type
reduction_functor_join(
  F&& f, Args&&... args
) noexcept(noexcept(std::declval<F>().join(std::declval<Args>()...)))
{
  Kokkos::Impl::forward<F>(f).join(
    Kokkos::Impl::forward<Args>(args)...
  );
}

/**
 *  Overload for functors with no intrusive join and value_count, where
 *  we can safely assume this isn't an array reduction.
 */
template <class F, class T, class U>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
  !reduction_functor_has_intrusive_join<F, T&, U const&>::value
  && !array_reduction_functor_has_intrusive_value_count<F>::value
  && has_plus_equal_op<T&, U const&>::value
>::type
reduction_functor_join(
  F&& f, T& dst, U const& src
) noexcept(noexcept(dst += src))
{
  dst += src;
}

/**
 *  Tag version of overload for functors with no intrusive join and value_count,
 *  where we can safely assume this isn't an array reduction.
 *
 *  Note: The tagged versions with no intrusive join all just (effectively)
 *        forward to the non-tagged versions, but in the interest of producing
 *        readable error messages, we write them as separate overloads.  (Also,
 *        we'd have to write things slightly differently, since we can't just
 *        directly forward to the untagged version because it might pick up
 *        an intrusive non-tagged hook.)
 *
 */
template <class F, class Tag, class T, class U>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
  !reduction_functor_has_intrusive_join<F, Tag const&, T&, U const&>::value
    && !array_reduction_functor_has_intrusive_value_count<F>::value
    && has_plus_equal_op<T&, U const&>::value
>::type
reduction_functor_join(
  F&& f, Tag const&, T& dst, U const& src
) noexcept(noexcept(dst += src))
{
  dst += src;
}

/**
 *  Overload for functors with no intrusive join but with intrusive value_count
 *  that takes pointer arguments.
 *
 *  The pattern `T* const&` is to preempt collision with an overload of the form
 *  `U&` where `U` gets deduced as `T*`.
 */
template <class F, class T, class U>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
  !reduction_functor_has_intrusive_join<F, T* const&, U* const&>::value
    && array_reduction_functor_has_intrusive_value_count<F>::value
    && square_bracket_result_has_plus_equal_op<T* const&, unsigned, U* const&, unsigned>::value
>::type
reduction_functor_join(
  F&& f, T* const& dst, U* const& src
) noexcept(noexcept(dst[unsigned()] += src[unsigned()]))
{
  unsigned value_count = Concepts::array_reduction_functor_value_count(f);
  // TODO check if we should be iterating with signed indices here for performance
  for(unsigned i = 0; i < value_count; ++i) {
    dst[i] += src[i];
  }
}

/**
 *  Overload for functors with no intrusive join but with intrusive value_count
 *  that takes pointer arguments.
 *
 *  The pattern `T* const&` is to preempt collision with an overload of the form
 *  `U&` where `U` gets deduced as `T*`.
 *
 *  Note: The tagged versions with no intrusive join all just (effectively)
 *        forward to the non-tagged versions, but in the interest of producing
 *        readable error messages, we write them as separate overloads.  (Also,
 *        we'd have to write things slightly differently, since we can't just
 *        directly forward to the untagged version because it might pick up
 *        an intrusive non-tagged hook.)
 *
 */
template <class F, class Tag, class T, class U>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
  !reduction_functor_has_intrusive_join<F, Tag const&, T* const&, U* const&>::value
    && array_reduction_functor_has_intrusive_value_count<F>::value
    && square_bracket_result_has_plus_equal_op<T* const&, unsigned, U* const&, unsigned>::value
>::type
reduction_functor_join(
  F&& f, Tag const&, T* const& dst, U* const& src
) noexcept(noexcept(dst[unsigned()] += src[unsigned()]))
{
  unsigned value_count = Concepts::array_reduction_functor_value_count(f);
  // TODO check if we should be iterating with signed indices here for performance
  for(unsigned i = 0; i < value_count; ++i) {
    dst[i] += src[i];
  }
}

/**
 *  Overload for functors with no intrusive join but with intrusive value_count,
 *  but that takes non-array arguments.  We must assume that the intrusive value
 *  count is for a version with some other tag
 */
template <class F, class T, class U>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
  !reduction_functor_has_intrusive_join<F, T&, U const&>::value
    && array_reduction_functor_has_intrusive_value_count<F>::value
    && has_plus_equal_op<T&, U const&>::value
    && !std::is_pointer<typename std::remove_cv<T>::type>::value
    && !std::is_pointer<typename std::remove_cv<U>::type>::value
>::type
reduction_functor_join(
  F&& f, T& dst, U const& src
) noexcept(noexcept(dst += src))
{
  dst += src;
}

/**
 *  Overload for functors with no intrusive join but with intrusive value_count,
 *  but that takes non-array arguments.  We must assume that the intrusive value
 *  count is for a version with some other tag
 */
template <class F, class Tag, class T, class U>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
  !reduction_functor_has_intrusive_join<F, Tag const&, T&, U const&>::value
    && array_reduction_functor_has_intrusive_value_count<F>::value
    && has_plus_equal_op<T&, U const&>::value
    && !std::is_pointer<typename std::remove_cv<T>::type>::value
    && !std::is_pointer<typename std::remove_cv<U>::type>::value
>::type
reduction_functor_join(
  F&& f, Tag const&, T& dst, U const& src
) noexcept(noexcept(dst += src))
{
  dst += src;
}

// </editor-fold> end ReductionFunctor join }}}1
//==============================================================================


//==============================================================================
// <editor-fold desc="ReductionFunctor init"> {{{1

KOKKOS_DECLARE_DETECTION_ARCHETYPE_1PARAM_VARIADIC(
  _intrusive_init_archetype, F, Args,
  decltype(
    declval<F>().init(declval<Args>()...)
  )
);

template <class F, class... Args>
struct reduction_functor_has_intrusive_init :
  is_detected<_intrusive_init_archetype, F, Args...> { };

// </editor-fold> end ReductionFunctor init }}}1
//==============================================================================


//==============================================================================
// <editor-fold desc="ReductionFunctor final"> {{{1

KOKKOS_DECLARE_DETECTION_ARCHETYPE_1PARAM_VARIADIC(
  _intrusive_final_archetype, F, Args,
  decltype(
    declval<F>().final(declval<Args>()...)
  )
);

template <class F, class... Args>
struct reduction_functor_has_intrusive_final :
  is_detected<_intrusive_final_archetype, F, Args...> { };

// </editor-fold> end ReductionFunctor final }}}1
//==============================================================================

} // end namespace Concepts
} // end namespace Impl
} // end namespace Kokkos

#endif //KOKKOS_CONCEPTS_FUNCTOR_KOKKOS_REDUCTIONFUNCTOR_HPP
