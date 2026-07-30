// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_THREAD_HANDLE_TRAIT_HPP
#define KOKKOS_THREAD_HANDLE_TRAIT_HPP

#include <Kokkos_Macros.hpp>
#include <traits/Kokkos_PolicyTraitAdaptor.hpp>
#include <traits/Kokkos_Traits_fwd.hpp>

namespace Kokkos::Impl {

//==============================================================================
// <editor-fold desc="trait specification"> {{{1

template <class ThreadHandle, class AnalyzeNextTrait>
struct ThreadHandleMixin : AnalyzeNextTrait {
  using base_t = AnalyzeNextTrait;
  using base_t::base_t;

  static_assert(
      std::is_void_v<typename base_t::thread_handle>,
      "Kokkos Error: More than one ThreadHandleTrait specified is given.");
  static constexpr bool thread_handle_is_defaulted = false;
  using thread_handle                              = ThreadHandle;
};

struct ThreadHandleTrait : TraitSpecificationBase<ThreadHandleTrait> {
  struct base_traits {
    static constexpr bool thread_handle_is_defaulted = true;

    using thread_handle = void;
    KOKKOS_IMPL_MSVC_NVCC_EBO_WORKAROUND
  };
  template <class ThreadHandle, class AnalyzeNextTrait>
  using mixin_matching_trait =
      ThreadHandleMixin<ThreadHandle, AnalyzeNextTrait>;
  template <class T>
  using trait_matches_specification = is_thread_handle<T>;
};

// </editor-fold> end trait specification }}}1
//==============================================================================

}  // end namespace Kokkos::Impl

#endif
