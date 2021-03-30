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

#ifndef KOKKOS_KOKKOS_VIEWCTOR_EXECSPACE_HPP
#define KOKKOS_KOKKOS_VIEWCTOR_EXECSPACE_HPP

#include <Kokkos_Macros.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <View/Constructor/Kokkos_ViewCtor_fwd.hpp>

#include <View/Constructor/Kokkos_ViewCtor_AllowPadding.hpp>
#include <View/Constructor/Kokkos_ViewCtor_ExecSpace.hpp>
#include <View/Constructor/Kokkos_ViewCtor_Label.hpp>
#include <View/Constructor/Kokkos_ViewCtor_MemSpace.hpp>
#include <View/Constructor/Kokkos_ViewCtor_Pointer.hpp>
#include <View/Constructor/Kokkos_ViewCtor_WithoutInitializing.hpp>

namespace Kokkos {
namespace Impl {

//==============================================================================
// <editor-fold desc="ViewCtorTraitMatcher"/> {{{1

template <class TraitSpec, class Trait, class Enable /*= void*/>
struct ViewCtorTraitMatcher : std::false_type {};

template <class TraitSpec, class Trait>
    struct ViewCtorTraitMatcher<
        TraitSpec, Trait,
        std::enable_if_t<
            TraitSpec::template trait_matches_specification<Trait>::value>>>
    : std::true_type {};

// </editor-fold> end ViewCtorTraitMatcher" }}}1
//==============================================================================

template <class TraitSpecList, class TraitsList, class Enable = void>
struct FindViewCtorSpec;

// Found case
template <class TraitSpec, class... TraitSpecs, class Trait, class... Traits>
struct FindViewCtorSpec<
    type_list<TraitSpec, TraitSpecs...>, type_list<Trait, Traits...>,
    enable_if_t<ViewCtorTraitMatcher<TraitSpec, Trait>::value>> {
  // Mixing in only the found versions (rather than inheriting and passing
  // through in the not found case) limits the size of the
  // hierarchy and avoids extra using base_t::base_t passthroughs
  using mixin_trait = TraitSpec::template mixin_matching_trait<
      Trait, typename FindViewCtorSpec<view_constructor_trait_specifications,
                                       type_list<Traits...>>::mixin_trait>;
};

// Not found case
template <class TraitSpec, class... TraitSpecs, class Trait, class... Traits>
struct FindViewCtorSpec<
    type_list<TraitSpec, TraitSpecs...>, type_list<Trait, Traits...>,
    enable_if_t<!ViewCtorTraitMatcher<TraitSpec, Trait>::value>>
    : FindViewCtorSpec<type_list<TraitSpecs...>, type_list<Trait, Traits...>> {
};

// Base case
template <class... TraitSpecs>
struct FindViewCtorSpec<type_list<TraitSpecs...>, type_list<Traits...>>
    // We don't need to work around the MSVC EBO bug because ViewCtor traits
    // isn't a persistent object so its size doesn't really matter
    : TraitSpecs::base_traits... {};

template <class... Traits>
struct ViewConstructorDescription
    : FindViewCtorSpec<view_constructor_trait_specifications,
                       type_list<Traits...>>::mixin_trait {
  using base_t =
      typename FindViewCtorSpec<view_constructor_trait_specifications,
                                type_list<Traits...>>::mixin_trait;
  using base_t::base_t;
};

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_KOKKOS_VIEWCTOR_EXECSPACE_HPP
