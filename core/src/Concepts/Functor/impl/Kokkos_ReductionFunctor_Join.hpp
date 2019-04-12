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

#ifndef KOKKOS_CONCEPTS_FUNCTOR_IMPL_KOKKOS_REDUCTIONFUNCTOR_JOIN_HPP
#define KOKKOS_CONCEPTS_FUNCTOR_IMPL_KOKKOS_REDUCTIONFUNCTOR_JOIN_HPP

#include <Concepts/Kokkos_Concepts_Macros.hpp>

#include <Concepts/Functor/impl/Kokkos_ArrayReductionFunctor_ValueCount.hpp>

#include <Concepts/Functor/Kokkos_Functor_Concept.hpp>
#include <Concepts/ExecutionPolicy/Kokkos_ExecutionPolicy_Concept.hpp>

#include <Properties/Kokkos_Detection.hpp>

#include <impl/Kokkos_Utilities.hpp> // Kokkos::forward


namespace Kokkos {
namespace Impl {
namespace Concepts {

//==============================================================================
// <editor-fold desc="join detection helpers"> {{{1


// The inheritance version gives more readable error messages...
// TODO measure the impact of this versus the type alias version on compilation time (here and elsewhere) and provide a macro that switches between them if necessary
template <class Policy, class F, class... Args>
struct reduction_functor_has_intrusive_join :
  // TODO short cirtuit this metafunction?
  std::integral_constant<bool,
    (
      !execution_policy_has_work_tag<Policy>::value
        && is_detected<_intrusive_join_archetype, F, Args...>::value
    )
    || (
      execution_policy_has_work_tag<Policy>::value
        && is_detected<_intrusive_join_archetype, F, execution_policy_work_tag_t<Policy>, Args...>::value
    )
  > { };


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

// </editor-fold> end join detection helpers }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="ReductionFunctor join"> {{{1

// TODO @concepts DSH make this a customization point object (so that ADL is never prefered over intrusive customization)
/**
 *  Overload for functors with intrusive join available.
 *
 *  Just calls `f.join(args...)`, but with perfect forwarding
 */
template <class Policy, class F, class... Args>
KOKKOS_CUSTOMIZATION_POINT_FUNCTION
typename std::enable_if<
  !execution_policy_has_work_tag<Policy>::value
  && reduction_functor_has_intrusive_join<Policy, F, Args...>::value
>::type
reduction_functor_join(
  Policy const&, F&& f, Args&&... args
) noexcept(noexcept(declval<F>().join(declval<Args>()...)))
{
  Impl::forward<F>(f).join(
    Impl::forward<Args>(args)...
  );
}

/**
 *  Tagged version of overload for functors with intrusive join available.
 *
 *  Just calls `f.join(Policy::work_tag{}, args...)`, but with perfect forwarding
 */
template <class Policy, class F, class... Args>
KOKKOS_CUSTOMIZATION_POINT_FUNCTION
typename std::enable_if<
  execution_policy_has_work_tag<remove_cvref_t<Policy>>::value
    && reduction_functor_has_intrusive_join<Policy, F, Args...>::value
>::type
reduction_functor_join(
  Policy const&, F&& f, Args&&... args
) noexcept(noexcept(declval<F>().join(declval<Args>()...)))
{
  Impl::forward<F>(f).join(
    execution_policy_work_tag_t<remove_cvref_t<Policy>>{},
    Impl::forward<Args>(args)...
  );
}

/**
 *  Overload for functors with no intrusive join and value_count, where
 *  we can safely assume this isn't an array reduction.
 */
template <class Policy, class F, class T, class U>
KOKKOS_CUSTOMIZATION_POINT_FUNCTION
typename std::enable_if<
  !reduction_functor_has_intrusive_join<Policy, F, T&, U const&>::value
  && !array_reduction_functor_has_intrusive_value_count<F>::value
  && has_plus_equal_op<T&, U const&>::value
>::type
reduction_functor_join(
  Policy const&, F&& f, T& dst, U const& src
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
template <class Policy, class F, class T, class U>
KOKKOS_CUSTOMIZATION_POINT_FUNCTION
typename std::enable_if<
  // TODO test readability of error message here
  !reduction_functor_has_intrusive_join<Policy, F, T* const&, U* const&>::value
    && make_requirement<decltype(
      Concepts::array_reduction_functor_value_count(Impl::declval<F>(), Impl::declval<T* const&>())
    )>::value
    && square_bracket_result_has_plus_equal_op<T* const&, unsigned, U* const&, unsigned>::value
>::type
reduction_functor_join(
  Policy const&, F&& f, T* const& dst, U* const& src
) noexcept(noexcept(dst[unsigned()] += src[unsigned()]))
{
  unsigned value_count =Concepts::array_reduction_functor_value_count(f, dst);
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
template <class Policy, class F, class T, class U>
KOKKOS_CUSTOMIZATION_POINT_FUNCTION
typename std::enable_if<
  !reduction_functor_has_intrusive_join<Policy, F, T&, U const&>::value
    && array_reduction_functor_has_value_count<F, T&>::value
    && has_plus_equal_op<T&, U const&>::value
    && !std::is_pointer<typename std::remove_cv<T>::type>::value
    && !std::is_pointer<typename std::remove_cv<U>::type>::value
>::type
reduction_functor_join(
  Policy const&, F&& f, T& dst, U const& src
) noexcept(noexcept(dst += src))
{
  dst += src;
}


// </editor-fold> end ReductionFunctor join }}}1
//==============================================================================

} // end namespace Concepts
} // end namespace Impl
} // end namespace Kokkos

#endif //KOKKOS_CONCEPTS_FUNCTOR_IMPL_KOKKOS_REDUCTIONFUNCTOR_JOIN_HPP
