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

#ifndef KOKKOS_IMPL_ANALYZE_POLICY_HPP
#define KOKKOS_IMPL_ANALYZE_POLICY_HPP

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Concepts.hpp>
#include <impl/Kokkos_Tags.hpp>
#include <impl/Kokkos_GraphImpl_fwd.hpp>
#include <impl/Kokkos_Error.hpp>
#include <impl/Kokkos_EBO.hpp>

namespace Kokkos {
namespace Experimental {
struct DesiredOccupancy {
  int m_occ = 100;
  explicit constexpr DesiredOccupancy(int occ) : m_occ(occ) {
    KOKKOS_EXPECTS(0 <= occ && occ <= 100);
  }
  explicit constexpr operator int() const { return m_occ; }
  constexpr int value() const { return m_occ; }
  explicit DesiredOccupancy() = default;
};
struct MaximizeOccupancy {
  explicit MaximizeOccupancy() = default;
};
}  // namespace Experimental

namespace Impl {

//------------------------------------------------------------------------------
template <class Enable, class... TraitsList>
struct AnalyzePolicy;

//------------------------------------------------------------------------------
// ExecutionSpace case
template <class ExecutionSpace, class... Traits>
struct AnalyzePolicy<
    std::enable_if_t<Impl::is_execution_space<ExecutionSpace>::value>,
    ExecutionSpace, Traits...> : AnalyzePolicy<void, Traits...> {
  using base_t = AnalyzePolicy<void, Traits...>;
  static_assert(base_t::execution_space_is_defaulted,
                "Kokkos Error: More than one execution space given");
  static constexpr bool execution_space_is_defaulted = false;
  using execution_space                              = ExecutionSpace;
};

//------------------------------------------------------------------------------
// Schedule case
template <class ScheduleType, class... Traits>
struct AnalyzePolicy<void, Kokkos::Schedule<ScheduleType>, Traits...>
    : AnalyzePolicy<void, Traits...> {
  using base_t = AnalyzePolicy<void, Traits...>;
  static_assert(base_t::schedule_type_is_defaulted,
                "Kokkos Error: More than one schedule type given");
  static constexpr bool schedule_type_is_defaulted = false;
  using schedule_type = Kokkos::Schedule<ScheduleType>;
};

//------------------------------------------------------------------------------
// IndexType case
template <class IntegralIndexType, class... Traits>
struct AnalyzePolicy<void, Kokkos::IndexType<IntegralIndexType>, Traits...>
    : AnalyzePolicy<void, Traits...> {
  using base_t = AnalyzePolicy<void, Traits...>;
  static_assert(base_t::index_type_is_defaulted,
                "Kokkos Error: More than one index type given");
  static constexpr bool index_type_is_defaulted = false;
  using index_type = Kokkos::IndexType<IntegralIndexType>;
};

// IndexType given as an integral type directly
template <class IntegralIndexType, class... Traits>
struct AnalyzePolicy<
    std::enable_if_t<std::is_integral<IntegralIndexType>::value>,
    IntegralIndexType, Traits...> : AnalyzePolicy<void, Traits...> {
  using base_t = AnalyzePolicy<void, Traits...>;
  static_assert(base_t::index_type_is_defaulted,
                "Kokkos Error: More than one index type given");
  static constexpr bool index_type_is_defaulted = false;
  using index_type = Kokkos::IndexType<IntegralIndexType>;
};

//------------------------------------------------------------------------------
// Iteration pattern case
template <class IterationPattern, class... Traits>
struct AnalyzePolicy<
    std::enable_if_t<Impl::is_iteration_pattern<IterationPattern>::value>,
    IterationPattern, Traits...> : AnalyzePolicy<void, Traits...> {
  using base_t = AnalyzePolicy<void, Traits...>;
  static_assert(std::is_void<typename base_t::iteration_pattern>::value,
                "Kokkos Error: More than one iteration pattern given");
  using iteration_pattern = IterationPattern;
};

//------------------------------------------------------------------------------
// Launch bounds case
template <unsigned int... Bounds, class... Traits>
struct AnalyzePolicy<void, Kokkos::LaunchBounds<Bounds...>, Traits...>
    : AnalyzePolicy<void, Traits...> {
  using base_t = AnalyzePolicy<void, Traits...>;
  static_assert(base_t::launch_bounds_is_defaulted,
                "Kokkos Error: More than one launch_bounds given");
  static constexpr bool launch_bounds_is_defaulted = false;
  using launch_bounds = Kokkos::LaunchBounds<Bounds...>;
};

//------------------------------------------------------------------------------
// Work item propoerty case
template <class Property, class... Traits>
struct AnalyzePolicy<
    std::enable_if_t<
        Kokkos::Experimental::is_work_item_property<Property>::value>,
    Property, Traits...> : AnalyzePolicy<void, Traits...> {
  using base_t = AnalyzePolicy<void, Traits...>;
  static_assert(
      std::is_same<typename base_t::work_item_property,
                   Kokkos::Experimental::WorkItemProperty::None_t>::value,
      "Kokkos Error: More than one work item property given");
  using work_item_property = Property;
};

//------------------------------------------------------------------------------
// GraphKernel Tag case
template <class... Traits>
struct AnalyzePolicy<void, Impl::IsGraphKernelTag, Traits...>
    : AnalyzePolicy<void, Traits...> {
  using is_graph_kernel = std::true_type;
};

//------------------------------------------------------------------------------
// Occupancy control case
template <class... Traits>
struct AnalyzePolicy<void, Kokkos::Experimental::DesiredOccupancy, Traits...>
    : AnalyzePolicy<void, Traits...> {
  using occupancy_control = Kokkos::Experimental::DesiredOccupancy;
};

template <class... Traits>
struct AnalyzePolicy<void, Kokkos::Experimental::MaximizeOccupancy, Traits...>
    : AnalyzePolicy<void, Traits...> {
  using occupancy_control = Kokkos::Experimental::MaximizeOccupancy;
};

//------------------------------------------------------------------------------
// Ignore void for backwards compatibility purposes, though hopefully no one is
// using this in application code
template <class... Traits>
struct AnalyzePolicy<void, void, Traits...> : AnalyzePolicy<void, Traits...> {};

//------------------------------------------------------------------------------
// Handle work tag: if nothing else matches, tread the trait as a work tag

// Since we don't have subsumption in pre-C++20, we need to have the work tag
// "trait" handling code be unspecialized, so we handle it instead in a class
// with a different name.
template <class... Traits>
struct AnalyzePolicyHandleWorkTag : AnalyzePolicy<void, Traits...> {};

template <class WorkTag, class... Traits>
struct AnalyzePolicyHandleWorkTag<WorkTag, Traits...>
    : AnalyzePolicy<void, Traits...> {
  using base_t = AnalyzePolicy<void, Traits...>;
  static_assert(std::is_void<typename base_t::work_tag>::value,
                "Kokkos Error: More than one work tag given");
  using work_tag = WorkTag;
};

// This only works if this is not a partial specialization, so we have to
// do the partial specialization elsewhere
template <class Enable, class... Traits>
struct AnalyzePolicy : AnalyzePolicyHandleWorkTag<Traits...> {};

//------------------------------------------------------------------------------
// Defaults, for the traits that aren't yet handled

// A tag class for dependent defaults that must be handled by the
// PolicyTraitsWithDefaults wrapper, since their defaults depend on other traits
struct dependent_policy_trait_default;

template <>
struct AnalyzePolicy<void> {
  static constexpr auto execution_space_is_defaulted = true;
  using execution_space = Kokkos::DefaultExecutionSpace;

  static constexpr bool schedule_type_is_defaulted = true;
  using schedule_type                              = Schedule<Static>;

  static constexpr bool index_type_is_defaulted = true;
  using index_type = dependent_policy_trait_default;

  using iteration_pattern = void;  // TODO set default iteration pattern

  static constexpr bool launch_bounds_is_defaulted = true;
  using launch_bounds                              = LaunchBounds<>;

  using work_item_property = Kokkos::Experimental::WorkItemProperty::None_t;
  using is_graph_kernel    = std::false_type;

  using occupancy_control = Kokkos::Experimental::MaximizeOccupancy;

  using work_tag = void;
};

//------------------------------------------------------------------------------
// Used for defaults that depend on other analysis results
template <class AnalysisResults>
struct PolicyTraitsWithDefaults : AnalysisResults {
  using base_t = AnalysisResults;
  // The old code turned this into an integral type for backwards compatibility,
  // so that's what we're doing here. The original comment was:
  //   nasty hack to make index_type into an integral_type
  //   instead of the wrapped IndexType<T> for backwards compatibility
  using index_type = typename std::conditional_t<
      base_t::index_type_is_defaulted,
      Kokkos::IndexType<typename base_t::execution_space::size_type>,
      typename base_t::index_type>::type;
};

//------------------------------------------------------------------------------
// Data storage for policies that require storage
template <class AnalysisResults>
struct PolicyDataStorage : PolicyTraitsWithDefaults<AnalysisResults>,
                           NoUniqueAddressMemberEmulation<
                               typename AnalysisResults::occupancy_control> {
 private:
  using occupancy_control_t = typename AnalysisResults::occupancy_control;

  using occupancy_control_storage_base_t =
    NoUniqueAddressMemberEmulation<occupancy_control_t>;
 public:

  static constexpr bool experimental_contains_desired_occupancy =
      std::is_same<occupancy_control_t,
          Kokkos::Experimental::DesiredOccupancy>::value;

  PolicyDataStorage() = default;

  // Converting constructors
  template <
      class Other,
      std::enable_if_t<
          experimental_contains_desired_occupancy &&
          PolicyDataStorage<Other>::experimental_contains_desired_occupancy,
          int> = 0>
  PolicyDataStorage(PolicyDataStorage<Other> const &other) {
    this->impl_set_desired_occupancy(other.impl_get_desired_occupancy());
  }

  template <class Other,
      std::enable_if_t<!experimental_contains_desired_occupancy ||
                       !PolicyDataStorage<Other>::
                       experimental_contains_desired_occupancy,
          int> = 0>
  PolicyDataStorage(PolicyDataStorage<Other> const &) {}

  // Converting assignment operators
  template <
      class Other,
      std::enable_if_t<
          experimental_contains_desired_occupancy &&
          PolicyDataStorage<Other>::experimental_contains_desired_occupancy,
          int> = 0>
  PolicyDataStorage &operator=(PolicyDataStorage<Other> const &other) {
    this->impl_set_desired_occupancy(other.impl_get_desired_occupancy());
    return *this;
  }

  template <class Other,
      std::enable_if_t<!experimental_contains_desired_occupancy ||
                       !PolicyDataStorage<Other>::
                       experimental_contains_desired_occupancy,
          int> = 0>
  PolicyDataStorage &operator=(PolicyDataStorage<Other> const &) {
    return *this;
  }

  // Access to desired occupancy (getter and setter)
  template <class Dummy = occupancy_control_t>
  std::enable_if_t<std::is_same<Dummy, occupancy_control_t>::value &&
                   experimental_contains_desired_occupancy,
      Kokkos::Experimental::DesiredOccupancy>
  impl_get_desired_occupancy() const {
    return this
        ->occupancy_control_storage_base_t::no_unique_address_data_member();
  }

  template <class Dummy = occupancy_control_t>
  std::enable_if_t<std::is_same<Dummy, occupancy_control_t>::value &&
                   experimental_contains_desired_occupancy>
  impl_set_desired_occupancy(occupancy_control_t desired_occupancy) {
    this->occupancy_control_storage_base_t::no_unique_address_data_member() =
        desired_occupancy;
  }
};

//------------------------------------------------------------------------------
template <typename... Traits>
struct PolicyTraits
    : PolicyDataStorage<AnalyzePolicy<void, Traits...>> {
  using base_t = PolicyDataStorage<AnalyzePolicy<void, Traits...>>;
  template <class... Args>
  PolicyTraits(PolicyTraits<Args...> const &p) : base_t(p) {}
  PolicyTraits() = default;
};

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_IMPL_ANALYZE_POLICY_HPP
