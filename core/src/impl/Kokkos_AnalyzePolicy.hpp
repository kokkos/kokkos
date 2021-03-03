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

namespace Kokkos {
namespace Experimental {
struct MaximizeOccupancy;
struct DesiredOccupancy {
  int m_occ = 100;
  explicit constexpr DesiredOccupancy(int occ) : m_occ(occ) {
    KOKKOS_EXPECTS(0 <= occ && occ <= 100);
  }
  explicit constexpr operator int() const { return m_occ; }
  constexpr int value() const { return m_occ; }
  DesiredOccupancy() = default;
  explicit DesiredOccupancy(MaximizeOccupancy const&) : DesiredOccupancy() {}
};
struct MaximizeOccupancy {
  MaximizeOccupancy() = default;
};
}  // namespace Experimental

namespace Impl {

//------------------------------------------------------------------------------
template <class Enable, class... TraitsList>
struct AnalyzePolicy;

//------------------------------------------------------------------------------

template <class AnalysisResults>
struct PolicyTraitsWithDefaults;

//------------------------------------------------------------------------------
// ExecutionSpace case
template <class ExecutionSpace, class... Traits>
struct AnalyzePolicy<
    std::enable_if_t<Impl::is_execution_space<ExecutionSpace>::value>,
    ExecutionSpace, Traits...> : AnalyzePolicy<void, Traits...> {
  using base_t = AnalyzePolicy<void, Traits...>;
  using base_t::base_t;
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
  using base_t::base_t;
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
  using base_t::base_t;
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
  using base_t::base_t;
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
  using base_t::base_t;
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
  using base_t::base_t;
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
  using base_t::base_t;
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
  using base_t = AnalyzePolicy<void, Traits...>;
  using base_t::base_t;
  using is_graph_kernel = std::true_type;
};

//------------------------------------------------------------------------------
// Occupancy control case
// The DesiredOccupancy case has runtime storage, so we need to handle copies
// and assignments
template <class... Traits>
struct AnalyzePolicy<void, Kokkos::Experimental::DesiredOccupancy, Traits...>
    : AnalyzePolicy<void, Traits...> {
 public:
  using base_t = AnalyzePolicy<void, Traits...>;
  using base_t::base_t;
  using occupancy_control = Kokkos::Experimental::DesiredOccupancy;
  static constexpr bool experimental_contains_desired_occupancy = true;

 private:
  // storage for a stateful desired occupancy
  occupancy_control m_desired_occupancy = {};

 public:
  // Converting constructor
  // Just rely on the convertibility of occupancy_control to transfer the data
  template <class Other>
  AnalyzePolicy(PolicyTraitsWithDefaults<Other> const& other)
      : base_t(other),
        m_desired_occupancy(other.impl_get_occupancy_control()) {}

  // Converting assignment operator
  // Just rely on the convertibility of occupancy_control to transfer the data
  template <class Other>
  AnalyzePolicy& operator=(PolicyTraitsWithDefaults<Other> const& other) {
    *static_cast<base_t*>(this) = other;
    this->impl_set_desired_occupancy(
        occupancy_control{other.impl_get_occupancy_control()});
    return *this;
  }

  // Access to occupancy control instance, usable in generic context
  constexpr occupancy_control impl_get_occupancy_control() const {
    return m_desired_occupancy;
  }

  // Access to desired occupancy (getter and setter)
  Kokkos::Experimental::DesiredOccupancy impl_get_desired_occupancy() const {
    return m_desired_occupancy;
  }

  void impl_set_desired_occupancy(occupancy_control desired_occupancy) {
    m_desired_occupancy = desired_occupancy;
  }
};

template <class... Traits>
struct AnalyzePolicy<void, Kokkos::Experimental::MaximizeOccupancy, Traits...>
    : AnalyzePolicy<void, Traits...> {
  using base_t = AnalyzePolicy<void, Traits...>;
  using base_t::base_t;
  using occupancy_control = Kokkos::Experimental::MaximizeOccupancy;
  static constexpr bool experimental_contains_desired_occupancy = false;
};

//------------------------------------------------------------------------------
// Ignore void for backwards compatibility purposes, though hopefully no one is
// using this in application code
template <class... Traits>
struct AnalyzePolicy<void, void, Traits...> : AnalyzePolicy<void, Traits...> {
  using base_t = AnalyzePolicy<void, Traits...>;
  using base_t::base_t;
};

//------------------------------------------------------------------------------
// Handle work tag: if nothing else matches, tread the trait as a work tag

// Since we don't have subsumption in pre-C++20, we need to have the work tag
// "trait" handling code be unspecialized, so we handle it instead in a class
// with a different name.
template <class... Traits>
struct AnalyzePolicyHandleWorkTag : AnalyzePolicy<void, Traits...> {
  using base_t = AnalyzePolicy<void, Traits...>;
  using base_t::base_t;
};

template <class WorkTag, class... Traits>
struct AnalyzePolicyHandleWorkTag<WorkTag, Traits...>
    : AnalyzePolicy<void, Traits...> {
  using base_t = AnalyzePolicy<void, Traits...>;
  using base_t::base_t;
  static_assert(std::is_void<typename base_t::work_tag>::value,
                "Kokkos Error: More than one work tag given");
  using work_tag = WorkTag;
};

// This only works if this is not a partial specialization, so we have to
// do the partial specialization elsewhere
template <class Enable, class... Traits>
struct AnalyzePolicy : AnalyzePolicyHandleWorkTag<Traits...> {
  using base_t = AnalyzePolicyHandleWorkTag<Traits...>;
  using base_t::base_t;
};

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
  static constexpr bool experimental_contains_desired_occupancy = false;
  // Default access occupancy_control, for when it is the (stateless) default
  constexpr occupancy_control impl_get_occupancy_control() const {
    return occupancy_control{};
  }

  using work_tag = void;

  AnalyzePolicy() = default;

  // Base converting constructors: unless an individual policy analysis
  // deletes a constructor, assume it's convertible
  template <class Other>
  AnalyzePolicy(PolicyTraitsWithDefaults<Other> const&) {}

  template <class Other>
  AnalyzePolicy& operator=(PolicyTraitsWithDefaults<Other> const&) {}
};

//------------------------------------------------------------------------------
// Used for defaults that depend on other analysis results
template <class AnalysisResults>
struct PolicyTraitsWithDefaults : AnalysisResults {
  using base_t = AnalysisResults;
  using base_t::base_t;
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
template <typename... Traits>
struct PolicyTraits : PolicyTraitsWithDefaults<AnalyzePolicy<void, Traits...>> {
  using base_t = PolicyTraitsWithDefaults<AnalyzePolicy<void, Traits...>>;
  template <class... Args>
  PolicyTraits(PolicyTraits<Args...> const& p) : base_t(p) {}
  PolicyTraits() = default;
};

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_IMPL_ANALYZE_POLICY_HPP
