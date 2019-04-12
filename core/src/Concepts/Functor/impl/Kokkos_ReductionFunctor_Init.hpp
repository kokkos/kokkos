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

#ifndef KOKKOS_CONCEPTS_FUNCTOR_IMPL_KOKKOS_REDUCTIONFUNCTOR_INIT_HPP
#define KOKKOS_CONCEPTS_FUNCTOR_IMPL_KOKKOS_REDUCTIONFUNCTOR_INIT_HPP

#include <Concepts/Kokkos_Concepts_Macros.hpp>

#include <Concepts/Functor/impl/Kokkos_ArrayReductionFunctor_ValueCount.hpp>

#include <Concepts/Functor/Kokkos_Functor_Concept.hpp>
#include <Concepts/ExecutionPolicy/Kokkos_ExecutionPolicy_Concept.hpp>

#include <Properties/Kokkos_Detection.hpp>

#include <impl/Kokkos_Utilities.hpp> // Kokkos::Impl::forward


namespace Kokkos {
namespace Impl {
namespace Concepts {

//==============================================================================
// <editor-fold desc="init function detection helpers"> {{{1

template <class Policy, class F, class... Args>
struct reduction_functor_has_intrusive_init :
  std::integral_constant<bool,
    (
      !execution_policy_has_work_tag<Policy>::value
        && is_detected<_intrusive_init_archetype, F, Args...>::value
    ) || (
      execution_policy_has_work_tag<Policy>::value
        && is_detected<_intrusive_init_archetype, F, execution_policy_work_tag_t<Policy>, Args...>::value
    )
  > { };



// </editor-fold> end init function detection helpers }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="ReductionFunctor init"> {{{1

/**
 *  Overload for functors with intrusive `init` function available.
 *
 *  Just calls `f.init(args...)`, but with perfect forwarding
 *
 */
template <class Policy, class F, class... Args>
KOKKOS_CUSTOMIZATION_POINT_FUNCTION
typename std::enable_if<
  !execution_policy_has_work_tag<Policy>::value
  && reduction_functor_has_intrusive_init<Policy, F, Args...>::value
>::type
reduction_functor_init(
  Policy const&, F&& f, Args&&... args
) noexcept(noexcept(Impl::declval<F>().init(Impl::declval<Args>()...)))
{
  Impl::forward<F>(f).init(Impl::forward<Args>(args)...);
}

/**
 *  Tag overload for functors with an intrusive `init` available
 *
 *  Just calls `f.init(Policy::work_tag{}, args...)`, but with perfect forwarding
 *
 */
template <class Policy, class F, class... Args>
KOKKOS_CUSTOMIZATION_POINT_FUNCTION
typename std::enable_if<
  execution_policy_has_work_tag<Policy>::value
  && reduction_functor_has_intrusive_init<Policy, F, Args...>::value
>::type
reduction_functor_init(
  Policy const&, F&& f, Args&&... args
) noexcept(noexcept(Impl::declval<F>().init(
    execution_policy_work_tag_t<Policy>{},
    Impl::declval<Args>()...
  )))
{
  Impl::forward<F>(f).init(
    execution_policy_work_tag_t<Policy>{},
    Impl::forward<Args>(args)...
  );
}

/**
 *  Array version of overload for functors without an intrusive `init` available
 *
 *  Performs a placement-new default initialization on the argument.
 *
 *  Note: the argument `dst` is *not* expected to be a pointer to a valid
 *        instance of `T` and thus will not be de
 *
 */
template <class Policy, class F, class T>
KOKKOS_CUSTOMIZATION_POINT_FUNCTION
typename std::enable_if<
  !reduction_functor_has_intrusive_init<Policy, F, T*>::value
  && array_reduction_functor_has_value_count<F, T*>::value
>::type
reduction_functor_init(
  Policy const&, F&& f, T* dst
) noexcept(noexcept(new(dst) T()))
{
  auto value_count = Concepts::array_reduction_functor_value_count(f, dst);
  for(unsigned i = 0; i < value_count; ++i) new(dst + i) T();
}

/**
 *  Overload for functors without an intrusive `init` available
 *
 *  Performs a placement-new default initialization on the argument.
 *
 *  Note: the argument `dst` is *not* expected to be a pointer to a valid
 *        instance of `T` and thus will not be de
 *
 */
template <class Policy, class F, class T>
KOKKOS_CUSTOMIZATION_POINT_FUNCTION
typename std::enable_if<
  !reduction_functor_has_intrusive_init<Policy, F, T&>::value
>::type
reduction_functor_init(
  Policy const&, F&& f, T& dst
) noexcept(noexcept(new(&dst) T()))
{
  new(&dst) T();
}



// </editor-fold> end ReductionFunctor init }}}1
//==============================================================================

} // end namespace Concepts
} // end namespace Impl
} // end namespace Kokkos

#endif //KOKKOS_CONCEPTS_FUNCTOR_IMPL_KOKKOS_REDUCTIONFUNCTOR_INIT_HPP
