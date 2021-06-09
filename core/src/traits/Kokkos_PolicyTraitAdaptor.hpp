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

#include <impl/Kokkos_Utilities.hpp>  // type_list

#include <traits/Kokkos_Traits_fwd.hpp>

#ifndef KOKKOS_KOKKOS_POLICYTRAITADAPTOR_HPP
#define KOKKOS_KOKKOS_POLICYTRAITADAPTOR_HPP

namespace Kokkos {
namespace Impl {

//==============================================================================
// <editor-fold desc="Adapter for replacing/adding a trait"> {{{1

//------------------------------------------------------------------------------

// General strategy: given a TraitSpecification, go through the entries in the
// parameter pack of the policy template and find the first one that returns
// `true` for the nested `trait_matches_specification` variable template. If
// that nested variable template is not found these overloads should be safely
// ignored, and the trait can specialize PolicyTraitAdapterImpl to get the
// desired behavior.

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// <editor-fold desc="PolicyTraitMatcher"> {{{2

// To handle the WorkTag case, we need more than just a predicate; we need
// something that we can default to in the unspecialized case, just like we
// do for AnalyzeExecPolicy
template <class TraitSpec, class Trait, class Enable = void>
struct PolicyTraitMatcher : std::false_type {};

template <class TraitSpec, class Trait>
struct PolicyTraitMatcher<
    TraitSpec, Trait,
    std::enable_if_t<
        TraitSpec::template trait_matches_specification<Trait>::value>>
    : std::true_type {};

// </editor-fold> end PolicyTraitMatcher }}}2
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// <editor-fold desc="PolicyTraitAdaptorImpl specializations"> {{{2

// Matching version, replace the trait
template <class TraitSpec, template <class...> class PolicyTemplate,
          class... ProcessedTraits, class MatchingTrait,
          class... ToProcessTraits, class NewTrait>
struct PolicyTraitAdaptorImpl<
    TraitSpec, PolicyTemplate, type_list<ProcessedTraits...>,
    type_list<MatchingTrait, ToProcessTraits...>, NewTrait,
    std::enable_if_t<PolicyTraitMatcher<TraitSpec, MatchingTrait>::value>> {
  static_assert(PolicyTraitMatcher<TraitSpec, NewTrait>::value, "");
  using type = PolicyTemplate<ProcessedTraits..., NewTrait, ToProcessTraits...>;
};

// Non-matching version, check the next option
template <class TraitSpec, template <class...> class PolicyTemplate,
          class... ProcessedTraits, class NonMatchingTrait,
          class... ToProcessTraits, class NewTrait>
struct PolicyTraitAdaptorImpl<
    TraitSpec, PolicyTemplate, type_list<ProcessedTraits...>,
    type_list<NonMatchingTrait, ToProcessTraits...>, NewTrait,
    std::enable_if_t<!PolicyTraitMatcher<TraitSpec, NonMatchingTrait>::value>> {
  using type = typename PolicyTraitAdaptorImpl<
      TraitSpec, PolicyTemplate,
      type_list<ProcessedTraits..., NonMatchingTrait>,
      type_list<ToProcessTraits...>, NewTrait>::type;
};

// Base case: no matches found; just add the trait to the end of the list
template <class TraitSpec, template <class...> class PolicyTemplate,
          class... ProcessedTraits, class NewTrait>
struct PolicyTraitAdaptorImpl<TraitSpec, PolicyTemplate,
                              type_list<ProcessedTraits...>, type_list<>,
                              NewTrait> {
  static_assert(PolicyTraitMatcher<TraitSpec, NewTrait>::value, "");
  using type = PolicyTemplate<ProcessedTraits..., NewTrait>;
};

// </editor-fold> end PolicyTraitAdaptorImpl specializations }}}2
//------------------------------------------------------------------------------

template <class TraitSpec, template <class...> class PolicyTemplate,
          class... Traits, class NewTrait>
struct PolicyTraitAdaptor<TraitSpec, PolicyTemplate<Traits...>, NewTrait>
    : PolicyTraitAdaptorImpl<TraitSpec, PolicyTemplate, type_list<>,
                             type_list<Traits...>, NewTrait> {};

// </editor-fold> end Adapter for replacing/adding a trait }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="CRTP Base class for trait specifications"> {{{1

template <class TraitSpec>
struct TraitSpecificationBase {
  using trait_specification = TraitSpec;
  template <class Policy, class Trait>
  using policy_with_trait =
      typename PolicyTraitAdaptor<TraitSpec, Policy, Trait>::type;
};

// </editor-fold> end CRTP Base class for trait specifications }}}1
//==============================================================================

}  // end namespace Impl
}  // end namespace Kokkos

#endif  // KOKKOS_KOKKOS_POLICYTRAITADAPTOR_HPP
