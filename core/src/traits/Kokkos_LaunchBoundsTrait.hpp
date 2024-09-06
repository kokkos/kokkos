//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef KOKKOS_KOKKOS_LAUNCHBOUNDSTRAIT_HPP
#define KOKKOS_KOKKOS_LAUNCHBOUNDSTRAIT_HPP

#include <Kokkos_Macros.hpp>
#include <Kokkos_Concepts.hpp>  // LaunchBounds
#include <traits/Kokkos_PolicyTraitAdaptor.hpp>
#include <traits/Kokkos_Traits_fwd.hpp>

namespace Kokkos {
namespace Impl {

//==============================================================================
// <editor-fold desc="trait specification"> {{{1

struct LaunchBoundsTrait : TraitSpecificationBase<LaunchBoundsTrait> {
  struct base_traits {
    static constexpr bool launch_bounds_is_defaulted = true;

    using launch_bounds = LaunchBounds<>;
    KOKKOS_IMPL_MSVC_NVCC_EBO_WORKAROUND
  };
  template <class LaunchBoundParam, class AnalyzeNextTrait>
  struct mixin_matching_trait : AnalyzeNextTrait {
    using base_t = AnalyzeNextTrait;
    using base_t::base_t;

    static constexpr bool launch_bounds_is_defaulted = false;

    static_assert(base_t::launch_bounds_is_defaulted,
                  "Kokkos Error: More than one launch_bounds given");

    using launch_bounds = LaunchBoundParam;
  };
};

// </editor-fold> end trait specification }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="PolicyTraitMatcher specialization"> {{{1

template <unsigned int maxT, unsigned int minB>
struct PolicyTraitMatcher<LaunchBoundsTrait, LaunchBounds<maxT, minB>>
    : std::true_type {};

// </editor-fold> end PolicyTraitMatcher specialization }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="trait specification"> {{{1

struct SubGroupSizeTrait : TraitSpecificationBase<SubGroupSizeTrait> {
  struct base_traits {
    static constexpr bool subgroup_size_is_defaulted = true;
    static constexpr int subgroup_size               = -1;

    KOKKOS_IMPL_MSVC_NVCC_EBO_WORKAROUND
  };
  template <class SubGroupSizeParam, class AnalyzeNextTrait>
  struct mixin_matching_trait : AnalyzeNextTrait {
    using base_t = AnalyzeNextTrait;
    using base_t::base_t;

    static constexpr bool subgroup_size_is_defaulted = false;

    static_assert(base_t::subgroup_size_is_defaulted,
                  "Kokkos Error: More than one subgroup size given");

    static constexpr int subgroup_size = SubGroupSizeParam::value;
  };
};

struct RegisterFileSizeTrait : TraitSpecificationBase<RegisterFileSizeTrait> {
  struct base_traits {
    static constexpr bool registerfile_size_is_defaulted = true;
    static constexpr int registerfile_size               = 0;

    KOKKOS_IMPL_MSVC_NVCC_EBO_WORKAROUND
  };
  template <class RegisterFileSizeParam, class AnalyzeNextTrait>
  struct mixin_matching_trait : AnalyzeNextTrait {
    using base_t = AnalyzeNextTrait;
    using base_t::base_t;

    static constexpr bool registerfile_size_is_defaulted = false;

    static_assert(base_t::registerfile_size_is_defaulted,
                  "Kokkos Error: More than one subgroup size given");

    static constexpr int registerfile_size = RegisterFileSizeParam::value;
  };
};

// </editor-fold> end trait specification }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="PolicyTraitMatcher specialization"> {{{1

template <int size>
struct PolicyTraitMatcher<SubGroupSizeTrait, SubGroupSize<size>>
    : std::true_type {};

template <int size>
struct PolicyTraitMatcher<RegisterFileSizeTrait, RegisterFileSize<size>>
    : std::true_type {};

// </editor-fold> end PolicyTraitMatcher specialization }}}1
//==============================================================================

}  // end namespace Impl
}  // end namespace Kokkos

#endif  // KOKKOS_KOKKOS_LAUNCHBOUNDSTRAIT_HPP
