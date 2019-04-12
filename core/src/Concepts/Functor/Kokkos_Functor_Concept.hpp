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

#ifndef KOKKOS_CONCEPTS_FUNCTOR_KOKKOS_FUNCTOR_CONCEPT_HPP
#define KOKKOS_CONCEPTS_FUNCTOR_KOKKOS_FUNCTOR_CONCEPT_HPP

#include <Concepts/Kokkos_Concepts_Common.hpp>
#include <Concepts/ExecutionPolicy/Kokkos_ExecutionPolicy_Concept.hpp>

#include <Properties/Kokkos_Detection.hpp>

namespace Kokkos {
namespace Impl {
namespace Concepts {

//==============================================================================
// <editor-fold desc="functor_invocable_with"> {{{1

KOKKOS_DECLARE_DETECTION_ARCHETYPE_1PARAM_VARIADIC(
  _invocable_with_archetype, F, ArgTypes,
  decltype(
    declval<F>()(declval<ArgTypes>()...)
  )
);

template <class F, class... Args>
struct functor_is_invocable_with
  : std::integral_constant<bool,
      is_detected_t<_invocable_with_archetype, F, Args...>::value
    > { };

template <class F, class... Args>
struct functor_invocation_result
  : is_detected<_invocable_with_archetype, F, Args...>
{ };

template <class F, class... Args>
using functor_invocation_result_t = typename functor_invocation_result<F, Args...>::type;

// </editor-fold> end functor_invocable_with }}}1
//==============================================================================


//==============================================================================
// <editor-fold desc="functor_invoke_with_policy"> {{{1

template <class Policy, class F, class... Args>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if<
  execution_policy_has_work_tag<Policy>::value
  && functor_is_invocable_with<F, execution_policy_work_tag_t<Policy>, Args...>::value,
  functor_invocation_result_t<F, execution_policy_work_tag_t<Policy>, Args...>
>::type
functor_invoke_with_policy(Policy const&, F&& f, Args&&... args)
  noexcept(noexcept(
    Impl::forward<F>(f)(
      execution_policy_work_tag_t<Policy>{}, Impl::forward<Args>(args)...
    )
  ))
{
  return
    Impl::forward<F>(f)(
      execution_policy_work_tag_t<Policy>{}, Impl::forward<Args>(args)...
    );
}

template <class Policy, class F, class... Args>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if<
  !execution_policy_has_work_tag<Policy>::value
    && functor_is_invocable_with<F, Args...>::value,
  functor_invocation_result_t<F, Args...>
>::type
functor_invoke_with_policy(Policy const&, F&& f, Args&&... args)
  noexcept(noexcept(Impl::forward<F>(f)(Impl::forward<Args>(args)...)))
{
  return Impl::forward<F>(f)(Impl::forward<Args>(args)...);
}

// </editor-fold> end functor_invoke_with_policy }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="functor_execution_space"> {{{1

/**
 *  `functor_execution_space<F>::type is `F::execution_space` if valid and an
 *  unspecified type otherwise.
 */
template <class F>
struct functor_execution_space :
  detected_or<Kokkos::DefaultExecutionSpace, _intrusive_execution_space_archetype, F> { };

template <class F>
using functor_execution_space_t = typename functor_execution_space<F>::type;

/**
 *  Mostly just a readability alias for `functor_execution_space`, but in situations
 *  where we want a boolean result.  The readability alias is desirable here
 *  because this will show up in user compilation error messages.
 */
template <class F>
struct functor_has_execution_space :
  std::integral_constant<bool, functor_execution_space<F>::value> { };


// </editor-fold> end functor_execution_space }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="functor_value_type"> {{{1

// Not part of the base concept, but used in multiple derived concepts

/**
 *  `functor_value_type<F>::type` is `F::value_type` if valid and an unspecified
 *  type otherwise.
 */
template <class F>
struct functor_value_type :
  is_detected<_intrusive_value_type, F> { };

/**
 *  Mostly just a readability alias for `functor_value_type`, but in situations
 *  where we want a boolean result.  The readability alias is desirable here
 *  because this will show up in user compilation error messages.
 */
template <class F>
struct functor_has_value_type :
  std::integral_constant<bool, functor_value_type<F>::value> { };


// </editor-fold> end functor_value_type }}}1
//==============================================================================

} // end namespace Concepts
} // end namespace Impl
} // end namespace Kokkos

#include <Concepts/Functor/impl/Kokkos_Functor_TeamShmemSize.hpp>

#endif //KOKKOS_CONCEPTS_FUNCTOR_KOKKOS_FUNCTOR_CONCEPT_HPP
