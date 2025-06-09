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

#ifndef KOKKOS_KOKKOS_LOOPUNROLLTRAIT_HPP
#define KOKKOS_KOKKOS_LOOPUNROLLTRAIT_HPP

#include <Kokkos_Macros.hpp>
#include <traits/Kokkos_PolicyTraitAdaptor.hpp>
#include <traits/Kokkos_Traits_fwd.hpp>

namespace Kokkos {

namespace Experimental {

template <unsigned int LoopUnrollFactor = 1>
struct LoopUnroll {
  using loop_unroll = LoopUnroll;
  using type        = LoopUnroll<LoopUnrollFactor>;
  constexpr static unsigned int unroll_factor{
      LoopUnrollFactor};  // Default unroll factor is 1
};

}  // namespace Experimental

namespace Impl {

//==============================================================================
// <editor-fold desc="trait specification"> {{{1

struct LoopUnrollTrait : TraitSpecificationBase<LoopUnrollTrait> {
  struct base_traits {
    static constexpr bool loop_unroll_is_defaulted = true;

    using loop_unroll =
        Kokkos::Experimental::LoopUnroll<>;  // Default unroll factor is 1
    KOKKOS_IMPL_MSVC_NVCC_EBO_WORKAROUND
  };
  template <class LoopUnrollParam, class AnalyzeNextTrait>
  struct mixin_matching_trait : AnalyzeNextTrait {
    using base_t = AnalyzeNextTrait;
    using base_t::base_t;

    static constexpr bool loop_unroll_is_defaulted = false;

    static_assert(base_t::loop_unroll_is_defaulted,
                  "Kokkos Error: More than one launch_bounds given");

    using loop_unroll = LoopUnrollParam;
  };
};
}  // namespace Impl

// </editor-fold> end trait specification }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="PolicyTraitMatcher specialization"> {{{1

namespace Impl {

template <unsigned int LoopUnrollFactor>
struct PolicyTraitMatcher<LoopUnrollTrait,
                          Kokkos::Experimental::LoopUnroll<LoopUnrollFactor>>
    : std::true_type {};

// </editor-fold> end PolicyTraitMatcher specialization }}}1
//==============================================================================

}  // end namespace Impl

}  // end namespace Kokkos

#endif  // KOKKOS_KOKKOS_LAUNCHBOUNDSTRAIT_HPP
