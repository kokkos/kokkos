// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_IMPL_ANALYZE_POLICY_HPP
#define KOKKOS_IMPL_ANALYZE_POLICY_HPP

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Concepts.hpp>  // IndexType
#include <traits/Kokkos_Traits_fwd.hpp>
#include <traits/Kokkos_PolicyTraitAdaptor.hpp>

#include <traits/Kokkos_ExecutionSpaceTrait.hpp>
#include <traits/Kokkos_TeamHandleTrait.hpp>
#include <traits/Kokkos_ThreadHandleTrait.hpp>
#include <traits/Kokkos_InlineHandleTrait.hpp>
#include <traits/Kokkos_GraphKernelTrait.hpp>
#include <traits/Kokkos_IndexTypeTrait.hpp>
#include <traits/Kokkos_IterationPatternTrait.hpp>
#include <traits/Kokkos_LaunchBoundsTrait.hpp>
#include <traits/Kokkos_StaticBatchSizeTrait.hpp>
#include <traits/Kokkos_OccupancyControlTrait.hpp>
#include <traits/Kokkos_ScheduleTrait.hpp>
#include <traits/Kokkos_WorkItemPropertyTrait.hpp>
#include <traits/Kokkos_WorkTagTrait.hpp>

namespace Kokkos {
namespace Impl {

//==============================================================================
// <editor-fold desc="AnalyzePolicyBaseTraits"> {{{1

// Mix in the defaults (base_traits) for the traits that aren't yet handled

//------------------------------------------------------------------------------
// <editor-fold desc="MSVC EBO failure workaround"> {{{2

template <class TraitSpecList>
struct KOKKOS_IMPL_ENFORCE_EMPTY_BASE_OPTIMIZATION AnalyzeExecPolicyBaseTraits;
template <class... TraitSpecifications>
struct KOKKOS_IMPL_ENFORCE_EMPTY_BASE_OPTIMIZATION
    AnalyzeExecPolicyBaseTraits<type_list<TraitSpecifications...>>
    : TraitSpecifications::base_traits... {};

// </editor-fold> end AnalyzePolicyBaseTraits }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="AnalyzeExecPolicy specializations"> {{{1

//------------------------------------------------------------------------------
// Note: unspecialized, so that the default pathway is to fall back to using
// the PolicyTraitMatcher. See AnalyzeExecPolicyUseMatcher below
template <class Enable, class... Traits>
struct AnalyzeExecPolicy
    : AnalyzeExecPolicyUseMatcher<void, execution_policy_trait_specifications,
                                  Traits...> {
  using base_t =
      AnalyzeExecPolicyUseMatcher<void, execution_policy_trait_specifications,
                                  Traits...>;
  using base_t::base_t;
};

//------------------------------------------------------------------------------
// Ignore void for backwards compatibility purposes, though hopefully no one is
// using this in application code
template <class... Traits>
struct AnalyzeExecPolicy<void, void, Traits...>
    : AnalyzeExecPolicy<void, Traits...> {
  using base_t = AnalyzeExecPolicy<void, Traits...>;
  using base_t::base_t;
};

//------------------------------------------------------------------------------
template <>
struct AnalyzeExecPolicy<void>
    : AnalyzeExecPolicyBaseTraits<execution_policy_trait_specifications> {
  // Ensure default constructibility since a converting constructor causes it to
  // be deleted.
  AnalyzeExecPolicy() = default;

  // Base converting constructor and assignment operator: unless an individual
  // policy analysis deletes a constructor, assume it's convertible
  template <class Other>
  AnalyzeExecPolicy(ExecPolicyTraitsWithDefaults<Other> const&) {}

  template <class Other>
  AnalyzeExecPolicy& operator=(ExecPolicyTraitsWithDefaults<Other> const&) {
    return *this;
  }
};

// </editor-fold> end AnalyzeExecPolicy specializations }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="AnalyzeExecPolicyUseMatcher"> {{{1

// We can avoid having to have policies specialize AnalyzeExecPolicy themselves
// by piggy-backing off of the PolicyTraitMatcher that we need to have for
// things like require() anyway. We mixin the effects of the trait using
// the `mixin_matching_trait` nested alias template in the trait specification

// General PolicyTraitMatcher version

// Matching case
template <class TraitSpec, class... TraitSpecs, class Trait, class... Traits>
struct AnalyzeExecPolicyUseMatcher<
    std::enable_if_t<PolicyTraitMatcher<TraitSpec, Trait>::value>,
    type_list<TraitSpec, TraitSpecs...>, Trait, Traits...>
    : TraitSpec::template mixin_matching_trait<
          Trait, AnalyzeExecPolicy<void, Traits...>> {
  using base_t = typename TraitSpec::template mixin_matching_trait<
      Trait, AnalyzeExecPolicy<void, Traits...>>;
  using base_t::base_t;
};

// Non-matching case
template <class TraitSpec, class... TraitSpecs, class Trait, class... Traits>
struct AnalyzeExecPolicyUseMatcher<
    std::enable_if_t<!PolicyTraitMatcher<TraitSpec, Trait>::value>,
    type_list<TraitSpec, TraitSpecs...>, Trait, Traits...>
    : AnalyzeExecPolicyUseMatcher<void, type_list<TraitSpecs...>, Trait,
                                  Traits...> {
  using base_t = AnalyzeExecPolicyUseMatcher<void, type_list<TraitSpecs...>,
                                             Trait, Traits...>;
  using base_t::base_t;
};

// No match found case:
template <class>
struct show_name_of_invalid_execution_policy_trait;
template <class Trait, class... Traits>
struct AnalyzeExecPolicyUseMatcher<void, type_list<>, Trait, Traits...> {
  static constexpr auto trigger_error_message =
      show_name_of_invalid_execution_policy_trait<Trait>{};
  static_assert(
      /* always false: */ std::is_void_v<Trait>,
      "Unknown execution policy trait. Search compiler output for "
      "'show_name_of_invalid_execution_policy_trait' to see the type of the "
      "invalid trait.");
};

// All traits matched case:
template <>
struct AnalyzeExecPolicyUseMatcher<void, type_list<>>
    : AnalyzeExecPolicy<void> {
  using base_t = AnalyzeExecPolicy<void>;
  using base_t::base_t;
};

// </editor-fold> end AnalyzeExecPolicyUseMatcher }}}1
//==============================================================================

// Helpers to choose the default execution space
//   - If TeamHandleTrait exist, type is team_handle::execution_space
//   - Else, type is Kokkos::DefaultExecutionSpace
template <class T>
struct DefaultExecutionSpaceSelector;

template <>
struct DefaultExecutionSpaceSelector<void> {
  using type = Kokkos::DefaultExecutionSpace;
};

template <Kokkos::TeamHandle T>
struct DefaultExecutionSpaceSelector<T> {
  using type = typename T::execution_space;
};

template <Kokkos::ThreadHandleType T>
struct DefaultExecutionSpaceSelector<T> {
  using type = typename T::execution_space;
};

template <Kokkos::InlineHandleType T>
struct DefaultExecutionSpaceSelector<T> {
  using type = typename T::execution_space;
};

// Helper to get the primary handle type for execution space/type deduction
template <class AnalysisResults>
struct HandleSelector {
  using type = std::conditional_t<
      !AnalysisResults::inline_handle_is_defaulted,
      typename AnalysisResults::inline_handle,
      std::conditional_t<
          !AnalysisResults::thread_handle_is_defaulted,
          typename AnalysisResults::thread_handle,
          std::conditional_t<!AnalysisResults::team_handle_is_defaulted,
                             typename AnalysisResults::team_handle, void>>>;
};

//------------------------------------------------------------------------------
// Used for defaults that depend on other analysis results
template <class AnalysisResults>
struct ExecPolicyTraitsWithDefaults : AnalysisResults {
  using base_t = AnalysisResults;
  using base_t::base_t;

  // At most one of ExecSpace, TeamHandle, ThreadHandle, or InlineHandle may be
  // specified
  static_assert(
      (int(!base_t::execution_space_is_defaulted) +
       int(!base_t::team_handle_is_defaulted) +
       int(!base_t::thread_handle_is_defaulted) +
       int(!base_t::inline_handle_is_defaulted)) <= 1,
      "Kokkos Error: Cannot give more than one policy trait (ExecSpace, "
      "TeamHandle, ThreadHandle, or InlineHandle).");

  // Handle used to select the default execution space when the execution space
  // trait is not set explicitly.
  using handle = typename HandleSelector<base_t>::type;

  // Query for the default execution space
  using execution_space = typename std::conditional_t<
      base_t::execution_space_is_defaulted,
      typename DefaultExecutionSpaceSelector<handle>::type,
      typename base_t::execution_space>;

  // Define the "execution_type" to be whichever policy trait was explicitly set
  // (default to execspace if no handle is set)
  using execution_type =
      std::conditional_t<std::is_void_v<handle>, execution_space, handle>;

  // The old code turned this into an integral type for backwards compatibility,
  // so that's what we're doing here. The original comment was:
  //   nasty hack to make index_type into an integral_type
  //   instead of the wrapped IndexType<T> for backwards compatibility
  using index_type = typename std::conditional_t<
      base_t::index_type_is_defaulted,
      Kokkos::IndexType<typename execution_space::index_type>,
      typename base_t::index_type>::type;
};

//------------------------------------------------------------------------------

template <typename... Traits>
struct PolicyTraits
    : ExecPolicyTraitsWithDefaults<AnalyzeExecPolicy<void, Traits...>> {
  using base_t =
      ExecPolicyTraitsWithDefaults<AnalyzeExecPolicy<void, Traits...>>;
  using base_t::base_t;
};

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_IMPL_ANALYZE_POLICY_HPP
