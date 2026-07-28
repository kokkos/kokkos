// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_INLINE_HANDLE_TRAIT_HPP
#define KOKKOS_INLINE_HANDLE_TRAIT_HPP

#include <Kokkos_Macros.hpp>
#include <traits/Kokkos_PolicyTraitAdaptor.hpp>
#include <traits/Kokkos_Traits_fwd.hpp>
#include <traits/Kokkos_Traits_fwd.hpp>

namespace Kokkos::Impl {

//==============================================================================
// <editor-fold desc="trait specification"> {{{1

template <class InlineHandle, class AnalyzeNextTrait>
struct InlineHandleMixin : AnalyzeNextTrait {
  using base_t = AnalyzeNextTrait;
  using base_t::base_t;

  static_assert(
      std::is_void_v<typename base_t::inline_handle>,
      "Kokkos Error: More than one InlineHandleTrait specified is given.");
  static constexpr bool inline_handle_is_defaulted = false;
  using inline_handle                              = InlineHandle;
};

struct InlineHandleTrait : TraitSpecificationBase<InlineHandleTrait> {
  struct base_traits {
    static constexpr bool inline_handle_is_defaulted = true;

    using inline_handle = void;
    KOKKOS_IMPL_MSVC_NVCC_EBO_WORKAROUND
  };
  template <class InlineHandle, class AnalyzeNextTrait>
  using mixin_matching_trait =
      InlineHandleMixin<InlineHandle, AnalyzeNextTrait>;
  template <class T>
  using trait_matches_specification = is_inline_handle<T>;
};

// </editor-fold> end trait specification }}}1
//==============================================================================

}  // end namespace Kokkos::Impl

#endif  // KOKKOS_INLINE_HANDLE_TRAIT_HPP
