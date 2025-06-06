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

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#include <Kokkos_Macros.hpp>
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif
#ifndef KOKKOS_KOKKOS_TUNERS_HPP
#define KOKKOS_KOKKOS_TUNERS_HPP

#include <Kokkos_Macros.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_ExecPolicy.hpp>
#include <KokkosExp_MDRangePolicy.hpp>
#include <impl/Kokkos_Profiling_Interface.hpp>

#include <array>
#include <utility>
#include <tuple>
#include <string>
#include <vector>
#include <map>
#include <cassert>

namespace Kokkos {
namespace Tools {

namespace Experimental {

// forward declarations
SetOrRange make_candidate_set(size_t size, int64_t* data);
bool have_tuning_tool();
size_t declare_output_type(const std::string&,
                           Kokkos::Tools::Experimental::VariableInfo);
void request_output_values(size_t, size_t,
                           Kokkos::Tools::Experimental::VariableValue*);
VariableValue make_variable_value(size_t, int64_t);
VariableValue make_variable_value(size_t, double);
SetOrRange make_candidate_range(double lower, double upper, double step,
                                bool openLower, bool openUpper);
SetOrRange make_candidate_range(int64_t lower, int64_t upper, int64_t step,
                                bool openLower, bool openUpper);
size_t get_new_context_id();
void begin_context(size_t context_id);
void end_context(size_t context_id);
namespace Impl {

/** We're going to take in search space descriptions
 * as nested maps, which aren't efficient to
 * iterate across by index. These are very similar
 * to nested maps, but better for index-based lookup
 */
template <typename ValueType, typename ContainedType>
struct ValueHierarchyNode;

template <typename ValueType, typename ContainedType>
struct ValueHierarchyNode {
  std::vector<ValueType> root_values;
  std::vector<ContainedType> sub_values;
  void add_root_value(const ValueType& in) { root_values.push_back(in); }
  void add_sub_container(const ContainedType& in) { sub_values.push_back(in); }
  const ValueType& get_root_value(const size_t index) const {
    return root_values[index];
  }
  const ContainedType& get_sub_value(const size_t index) const {
    return sub_values[index];
  }
};

template <typename ValueType>
struct ValueHierarchyNode<ValueType, void> {
  std::vector<ValueType> root_values;
  explicit ValueHierarchyNode(std::vector<ValueType> rv)
      : root_values(std::move(rv)) {}
  void add_root_value(const ValueType& in) { root_values.push_back(in); }
  const ValueType& get_root_value(const size_t index) const {
    return root_values[index];
  }
};

/** For a given nested map type, we need a way to
 * declare the equivalent ValueHierarchyNode
 * structure
 */

template <class NestedMap>
struct MapTypeConverter;

// Vectors are our lowest-level, no nested values
template <class T>
struct MapTypeConverter<std::vector<T>> {
  using type = ValueHierarchyNode<T, void>;
};

// Maps contain both the "root" types and sub-vectors
template <class K, class V>
struct MapTypeConverter<std::map<K, V>> {
  using type = ValueHierarchyNode<K, typename MapTypeConverter<V>::type>;
};

/**
 * We also need to be able to construct a ValueHierarchyNode set from a
 * map
 */

template <class NestedMap>
struct ValueHierarchyConstructor;

// Vectors are our lowest-level, no nested values. Just fill in the fundamental
// values
template <class T>
struct ValueHierarchyConstructor<std::vector<T>> {
  using return_type = typename MapTypeConverter<std::vector<T>>::type;
  static return_type build(const std::vector<T>& in) { return return_type{in}; }
};

// For maps, we need to fill in the fundamental values, and construct child
// nodes
template <class K, class V>
struct ValueHierarchyConstructor<std::map<K, V>> {
  using return_type = typename MapTypeConverter<std::map<K, V>>::type;
  static return_type build(const std::map<K, V>& in) {
    return_type node_to_build;
    for (auto& entry : in) {
      node_to_build.add_root_value(entry.first);
      node_to_build.add_sub_container(
          ValueHierarchyConstructor<V>::build(entry.second));
    }
    return node_to_build;
  }
};

/**
 * We're going to be declaring a sparse multidimensional
 * tuning space as a set of nested maps. The innermost level
 * will be a vector. The dimensionality of such a space is the number of
 * maps + 1.
 *
 * The following templates implement such logic recursively
 */
template <class InspectForDepth>
struct get_space_dimensionality;

// The dimensionality of a vector is 1
template <class T>
struct get_space_dimensionality<std::vector<T>> {
  static constexpr int value = 1;
};

// The dimensionality of a map is 1 (the map) plus the dimensionality
// of the map's value type
template <class K, class V>
struct get_space_dimensionality<std::map<K, V>> {
  static constexpr int value = 1 + get_space_dimensionality<V>::value;
};

template <class T, int N>
struct n_dimensional_sparse_structure;

template <class T>
struct n_dimensional_sparse_structure<T, 1> {
  using type = std::vector<T>;
};

template <class T, int N>
struct n_dimensional_sparse_structure {
  using type =
      std::map<T, typename n_dimensional_sparse_structure<T, N - 1>::type>;
};

/**
 * This is the ugly part of this implementation: mapping a set of doubles in
 * [0.0,1.0) into a point in this multidimensional space. We're going to
 * implement this concept recursively, building up a tuple at each level.
 */

// First, a helper to get the value in one dimension
template <class Container>
struct DimensionValueExtractor;

// At any given level, just return your value at that level
template <class RootType, class Subtype>
struct DimensionValueExtractor<ValueHierarchyNode<RootType, Subtype>> {
  static RootType get(const ValueHierarchyNode<RootType, Subtype>& dimension,
                      double fraction_to_traverse) {
    size_t index = dimension.root_values.size() * fraction_to_traverse;
    return dimension.get_root_value(index);
  }
};

/** Now we're going to do the full "get a point in the space".
 * At a root level, we'll take in a ValueHierarchyNode and a set of doubles
 * representing the value in [0.0,1.0) we want to pick
 */

// At the bottom level, we have one double and a base-level ValueHierarchyNode

template <class HierarchyNode, class... InterpolationIndices>
struct GetMultidimensionalPoint;

template <class ValueType>
struct GetMultidimensionalPoint<ValueHierarchyNode<ValueType, void>, double> {
  using node_type   = ValueHierarchyNode<ValueType, void>;
  using return_type = std::tuple<ValueType>;
  static return_type build(const node_type& in, double index) {
    return std::make_tuple(DimensionValueExtractor<node_type>::get(in, index));
  }
};

// At levels above the bottom, we tuple_cat the result of our child on the end
// of our own tuple
template <class ValueType, class Subtype, class... Indices>
struct GetMultidimensionalPoint<ValueHierarchyNode<ValueType, Subtype>, double,
                                Indices...> {
  using node_type = ValueHierarchyNode<ValueType, Subtype>;
  using sub_tuple =
      typename GetMultidimensionalPoint<Subtype, Indices...>::return_type;
  using return_type = decltype(std::tuple_cat(
      std::declval<std::tuple<ValueType>>(), std::declval<sub_tuple>()));
  static return_type build(const node_type& in, double fraction_to_traverse,
                           Indices... indices) {
    size_t index         = in.sub_values.size() * fraction_to_traverse;
    auto dimension_value = std::make_tuple(
        DimensionValueExtractor<node_type>::get(in, fraction_to_traverse));
    return std::tuple_cat(dimension_value,
                          GetMultidimensionalPoint<Subtype, Indices...>::build(
                              in.get_sub_value(index), indices...));
  }
};

template <typename PointType, class ArrayType, size_t... Is>
auto get_point_helper(const PointType& in, const ArrayType& indices,
                      std::index_sequence<Is...>) {
  using helper = GetMultidimensionalPoint<
      PointType,
      decltype(std::get<Is>(std::declval<ArrayType>()).value.double_value)...>;
  return helper::build(in, std::get<Is>(indices).value.double_value...);
}

template <typename PointType, typename ArrayType>
struct GetPoint;

template <typename PointType, size_t ArraySize>
struct GetPoint<
    PointType,
    std::array<Kokkos::Tools::Experimental::VariableValue, ArraySize>> {
  using index_set_type =
      std::array<Kokkos::Tools::Experimental::VariableValue, ArraySize>;
  static auto build(const PointType& in, const index_set_type& indices) {
    return get_point_helper(in, indices, std::make_index_sequence<ArraySize>{});
  }
};

template <typename PointType, typename ArrayType>
auto get_point(const PointType& point, const ArrayType& indices) {
  return GetPoint<PointType, ArrayType>::build(point, indices);
}

}  // namespace Impl

template <template <class...> class Container, size_t MaxDimensionSize = 100,
          class... TemplateArguments>
class MultidimensionalSparseTuningProblem {
 public:
  using ProblemSpaceInput = Container<TemplateArguments...>;
  static constexpr int space_dimensionality =
      Impl::get_space_dimensionality<ProblemSpaceInput>::value;
  static constexpr size_t max_space_dimension_size = MaxDimensionSize;
  static constexpr double tuning_min               = 0.0;
  static constexpr double tuning_max               = 0.999;

  // Not declared as static constexpr to work around the following compiler bug
  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=96862
  // where a floating-point expression cannot be constexpr under -frounding-math
  double tuning_step = tuning_max / max_space_dimension_size;

  using StoredProblemSpace =
      typename Impl::MapTypeConverter<ProblemSpaceInput>::type;
  using HierarchyConstructor =
      typename Impl::ValueHierarchyConstructor<Container<TemplateArguments...>>;

  using ValueArray = std::array<Kokkos::Tools::Experimental::VariableValue,
                                space_dimensionality>;
  template <class Key, class Value>
  using extended_map = std::map<Key, Value>;
  template <typename Key>
  using extended_problem =
      MultidimensionalSparseTuningProblem<extended_map, MaxDimensionSize, Key,
                                          ProblemSpaceInput>;
  template <typename Key, typename Value>
  using ExtendedProblemSpace =
      typename Impl::MapTypeConverter<extended_map<Key, Value>>::type;

  template <typename Key>
  auto extend(const std::string& axis_name,
              const std::vector<Key>& new_tuning_axis) const
      -> extended_problem<Key> {
    ExtendedProblemSpace<Key, ProblemSpaceInput> extended_space;
    for (auto& key : new_tuning_axis) {
      extended_space.add_root_value(key);
      extended_space.add_sub_container(m_space);
    }
    std::vector<std::string> extended_names;
    extended_names.reserve(m_variable_names.size() + 1);
    extended_names.push_back(axis_name);
    extended_names.insert(extended_names.end(), m_variable_names.begin(),
                          m_variable_names.end());
    return extended_problem<Key>(extended_space, extended_names);
  }

 private:
  StoredProblemSpace m_space;
  std::array<size_t, space_dimensionality> variable_ids;
  std::vector<std::string> m_variable_names;
  size_t context;

 public:
  MultidimensionalSparseTuningProblem() = default;

  MultidimensionalSparseTuningProblem(StoredProblemSpace space,
                                      const std::vector<std::string>& names)
      : m_space(std::move(space)), m_variable_names(names) {
    KOKKOS_ASSERT(names.size() == space_dimensionality);
    for (unsigned long x = 0; x < names.size(); ++x) {
      VariableInfo info;
      info.type = Kokkos::Tools::Experimental::ValueType::kokkos_value_double;
      info.category = Kokkos::Tools::Experimental::StatisticalCategory::
          kokkos_value_interval;
      info.valueQuantity =
          Kokkos::Tools::Experimental::CandidateValueType::kokkos_value_range;
      info.candidates = Kokkos::Tools::Experimental::make_candidate_range(
          tuning_min, tuning_max, tuning_step, true, true);
      variable_ids[x] = declare_output_type(names[x], info);
    }
  }

  MultidimensionalSparseTuningProblem(ProblemSpaceInput space,
                                      const std::vector<std::string>& names)
      : MultidimensionalSparseTuningProblem(HierarchyConstructor::build(space),
                                            names) {}

  template <typename... Coordinates>
  auto get_point(Coordinates... coordinates) {
    using ArrayType = std::array<Kokkos::Tools::Experimental::VariableValue,
                                 sizeof...(coordinates)>;
    return Impl::get_point(
        m_space, ArrayType({Kokkos::Tools::Experimental::make_variable_value(
                     0, static_cast<double>(coordinates))...}));
  }

  auto begin() {
    context = Kokkos::Tools::Experimental::get_new_context_id();
    ValueArray values;
    for (int x = 0; x < space_dimensionality; ++x) {
      values[x] = Kokkos::Tools::Experimental::make_variable_value(
          variable_ids[x], 0.0);
    }
    begin_context(context);
    request_output_values(context, space_dimensionality, values.data());
    return Impl::get_point(m_space, values);
  }

  auto end() { end_context(context); }
};

template <typename Tuner>
struct ExtendableTunerMixin {
  template <typename Key>
  auto combine(const std::string& axis_name,
               const std::vector<Key>& new_axis) const {
    const auto& sub_tuner = static_cast<const Tuner*>(this)->get_tuner();
    return sub_tuner.extend(axis_name, new_axis);
  }

  template <typename... Coordinates>
  auto get_point(Coordinates... coordinates) {
    const auto& sub_tuner = static_cast<const Tuner*>(this)->get_tuner();
    return sub_tuner.get_point(coordinates...);
  }

 private:
  ExtendableTunerMixin() = default;
  friend Tuner;
};

template <size_t MaxDimensionSize = 100, template <class...> class Container,
          class... TemplateArguments>
auto make_multidimensional_sparse_tuning_problem(
    const Container<TemplateArguments...>& in, std::vector<std::string> names) {
  return MultidimensionalSparseTuningProblem<Container, MaxDimensionSize,
                                             TemplateArguments...>(in, names);
}

class TeamSizeTuner : public ExtendableTunerMixin<TeamSizeTuner> {
 private:
  using SpaceDescription = std::map<int64_t, std::vector<int64_t>>;
  using TunerType = decltype(make_multidimensional_sparse_tuning_problem<20>(
      std::declval<SpaceDescription>(),
      std::declval<std::vector<std::string>>()));
  TunerType tuner;

 public:
  TeamSizeTuner()                                      = default;
  TeamSizeTuner& operator=(const TeamSizeTuner& other) = default;
  TeamSizeTuner(const TeamSizeTuner& other)            = default;
  TeamSizeTuner& operator=(TeamSizeTuner&& other)      = default;
  TeamSizeTuner(TeamSizeTuner&& other)                 = default;
  template <typename ViableConfigurationCalculator, typename Functor,
            typename TagType, typename... Properties>
  TeamSizeTuner(const std::string& name,
                const Kokkos::TeamPolicy<Properties...>& policy_in,
                const Functor& functor, const TagType& tag,
                ViableConfigurationCalculator calc) {
    using PolicyType = Kokkos::TeamPolicy<Properties...>;
    PolicyType policy(policy_in);
    auto initial_vector_length = policy.impl_vector_length();
    if (initial_vector_length < 1) {
      policy.impl_set_vector_length(1);
    }
    /**
     * Here we attempt to enumerate all of the possible configurations
     * to expose to an autotuner. There are three possibilities
     *
     * 1) We're tuning both vector length and team size
     * 2) We're tuning vector length but not team size
     * 3) We're tuning team size but not vector length
     *
     * (In the fourth case where nothing is tuned
     * this function won't be called)
     *
     * The set of valid team sizes is dependent on
     * a vector length, so this leads to three
     * algorithms
     *
     * 1) Loop over vector lengths to get the set
     *    of team sizes for each vector length,
     *    add it all to the set
     * 2) Loop over vector lengths to see if the
     *    provided team size is valid for that
     *    vector length. If so, add it
     * 3) A special case of (1) in which we only
     *    have one vector length
     *
     */
    SpaceDescription space_description;

    auto max_vector_length = PolicyType::vector_length_max();
    std::vector<int64_t> allowed_vector_lengths;

    if (policy.impl_auto_vector_length()) {  // case 1 or 2
      for (int vector_length = max_vector_length; vector_length >= 1;
           vector_length /= 2) {
        policy.impl_set_vector_length(vector_length);
        /**
         * Figuring out whether a vector length is valid depends
         * on whether we're in case 1 (tune everything) or 2 (just tune vector
         * length)
         *
         * If we're tuning everything, all legal vector lengths are valid.
         * If we're just tuning vector length, we need to check that if we
         * set this vector length, the team size provided will be valid.
         *
         * These are the left and right hand sides of the "or" in this
         * conditional, respectively.
         */
        auto max_team_size = calc.get_max_team_size(policy, functor, tag);
        if ((policy.impl_auto_team_size()) ||
            (policy.team_size() <= max_team_size)) {
          allowed_vector_lengths.push_back(vector_length);
        }
      }
    } else {  // case 3, there's only one vector length to care about
      allowed_vector_lengths.push_back(policy.impl_vector_length());
    }

    for (const auto vector_length : allowed_vector_lengths) {
      std::vector<int64_t> allowed_team_sizes;
      policy.impl_set_vector_length(vector_length);
      auto max_team_size = calc.get_max_team_size(policy, functor, tag);
      if (policy.impl_auto_team_size()) {  // case 1 or 3, try all legal team
                                           // sizes
        for (int team_size = max_team_size; team_size >= 1; team_size /= 2) {
          allowed_team_sizes.push_back(team_size);
        }
      } else {  // case 2, just try the provided team size
        allowed_team_sizes.push_back(policy.team_size());
      }
      space_description[vector_length] = allowed_team_sizes;
    }
    tuner = make_multidimensional_sparse_tuning_problem<20>(
        space_description, {std::string(name + "_vector_length"),
                            std::string(name + "_team_size")});
    policy.impl_set_vector_length(initial_vector_length);
  }

  template <typename... Properties>
  auto tune(const Kokkos::TeamPolicy<Properties...>& policy_in) {
    Kokkos::TeamPolicy<Properties...> policy(policy_in);
    if (Kokkos::Tools::Experimental::have_tuning_tool()) {
      auto configuration = tuner.begin();
      auto team_size     = std::get<1>(configuration);
      auto vector_length = std::get<0>(configuration);
      if (vector_length > 0) {
        policy.impl_set_team_size(team_size);
        policy.impl_set_vector_length(vector_length);
      }
    }
    return policy;
  }
  void end() {
    if (Kokkos::Tools::Experimental::have_tuning_tool()) {
      tuner.end();
    }
  }

  TunerType get_tuner() const { return tuner; }
};
namespace Impl {
template <class T>
struct tuning_type_for;

template <>
struct tuning_type_for<double> {
  static constexpr Kokkos::Tools::Experimental::ValueType value =
      Kokkos::Tools::Experimental::ValueType::kokkos_value_double;
  static double get(
      const Kokkos::Tools::Experimental::VariableValue& value_struct) {
    return value_struct.value.double_value;
  }
};
template <>
struct tuning_type_for<int64_t> {
  static constexpr Kokkos::Tools::Experimental::ValueType value =
      Kokkos::Tools::Experimental::ValueType::kokkos_value_int64;
  static int64_t get(
      const Kokkos::Tools::Experimental::VariableValue& value_struct) {
    return value_struct.value.int_value;
  }
};
}  // namespace Impl
template <class Bound>
class SingleDimensionalRangeTuner {
  size_t id;
  size_t context;
  using tuning_util = Impl::tuning_type_for<Bound>;

  Bound default_value;

 public:
  SingleDimensionalRangeTuner() = default;
  SingleDimensionalRangeTuner(
      const std::string& name,
      Kokkos::Tools::Experimental::StatisticalCategory category,
      Bound default_val, Bound lower, Bound upper, Bound step = (Bound)0) {
    default_value = default_val;
    Kokkos::Tools::Experimental::VariableInfo info;
    info.category   = category;
    info.candidates = make_candidate_range(
        static_cast<Bound>(lower), static_cast<Bound>(upper),
        static_cast<Bound>(step), false, false);
    info.valueQuantity =
        Kokkos::Tools::Experimental::CandidateValueType::kokkos_value_range;
    info.type = tuning_util::value;
    id        = Kokkos::Tools::Experimental::declare_output_type(name, info);
  }

  Bound begin() {
    context = Kokkos::Tools::Experimental::get_new_context_id();
    Kokkos::Tools::Experimental::begin_context(context);
    auto tuned_value =
        Kokkos::Tools::Experimental::make_variable_value(id, default_value);
    Kokkos::Tools::Experimental::request_output_values(context, 1,
                                                       &tuned_value);
    return tuning_util::get(tuned_value);
  }

  void end() { Kokkos::Tools::Experimental::end_context(context); }

  template <typename Functor>
  void with_tuned_value(Functor& func) {
    func(begin());
    end();
  }
};

class RangePolicyOccupancyTuner {
 private:
  using TunerType = SingleDimensionalRangeTuner<int64_t>;
  TunerType tuner;

 public:
  RangePolicyOccupancyTuner() = default;
  template <typename ViableConfigurationCalculator, typename Functor,
            typename TagType, typename... Properties>
  RangePolicyOccupancyTuner(const std::string& name,
                            const Kokkos::RangePolicy<Properties...>&,
                            const Functor&, const TagType&,
                            ViableConfigurationCalculator)
      : tuner(TunerType(name,
                        Kokkos::Tools::Experimental::StatisticalCategory::
                            kokkos_value_ratio,
                        100, 5, 100, 5)) {}

  template <typename... Properties>
  auto tune(const Kokkos::RangePolicy<Properties...>& policy_in) {
    Kokkos::RangePolicy<Properties...> policy(policy_in);
    if (Kokkos::Tools::Experimental::have_tuning_tool()) {
      auto occupancy = tuner.begin();
      policy.impl_set_desired_occupancy(
          Kokkos::Experimental::DesiredOccupancy{static_cast<int>(occupancy)});
    }
    return policy;
  }
  void end() {
    if (Kokkos::Tools::Experimental::have_tuning_tool()) {
      tuner.end();
    }
  }

  TunerType get_tuner() const { return tuner; }
};

namespace Impl {

template <typename T>
void fill_tile(std::vector<T>& cont, int tile_size) {
  for (int x = 1; x < tile_size; x *= 2) {
    cont.push_back(x);
  }
}
template <typename T, typename Mapped>
void fill_tile(std::map<T, Mapped>& cont, int tile_size) {
  for (int x = 1; x < tile_size; x *= 2) {
    fill_tile(cont[x], tile_size / x);
  }
}
}  // namespace Impl

template <int MDRangeRank>
struct MDRangeTuner : public ExtendableTunerMixin<MDRangeTuner<MDRangeRank>> {
 private:
  static constexpr int rank       = MDRangeRank;
  static constexpr int max_slices = 15;
  using SpaceDescription =
      typename Impl::n_dimensional_sparse_structure<int, rank>::type;
  using TunerType =
      decltype(make_multidimensional_sparse_tuning_problem<max_slices>(
          std::declval<SpaceDescription>(),
          std::declval<std::vector<std::string>>()));
  TunerType tuner;

 public:
  MDRangeTuner() = default;
  template <typename Functor, typename TagType, typename Calculator,
            typename... Properties>
  MDRangeTuner(const std::string& name,
               const Kokkos::MDRangePolicy<Properties...>& policy,
               const Functor& functor, const TagType& tag, Calculator calc) {
    SpaceDescription desc;
    int max_tile_size =
        calc.get_mdrange_max_tile_size_product(policy, functor, tag);
    Impl::fill_tile(desc, max_tile_size);
    std::vector<std::string> feature_names;
    for (int x = 0; x < rank; ++x) {
      feature_names.push_back(name + "_tile_size_" + std::to_string(x));
    }
    tuner = make_multidimensional_sparse_tuning_problem<max_slices>(
        desc, feature_names);
  }
  template <typename Policy, typename Tuple, size_t... Indices>
  void set_policy_tile(Policy& policy, const Tuple& tuple,
                       const std::index_sequence<Indices...>&) {
    policy.impl_change_tile_size({std::get<Indices>(tuple)...});
  }
  template <typename... Properties>
  auto tune(const Kokkos::MDRangePolicy<Properties...>& policy_in) {
    Kokkos::MDRangePolicy<Properties...> policy(policy_in);
    if (Kokkos::Tools::Experimental::have_tuning_tool()) {
      auto configuration = tuner.begin();
      set_policy_tile(policy, configuration, std::make_index_sequence<rank>{});
    }
    return policy;
  }
  void end() {
    if (Kokkos::Tools::Experimental::have_tuning_tool()) {
      tuner.end();
    }
  }

  TunerType get_tuner() const { return tuner; }
};

template <class Choice>
struct CategoricalTuner {
  using choice_list = std::vector<Choice>;
  choice_list choices;
  size_t context;
  size_t tuning_variable_id;
  CategoricalTuner(std::string name, choice_list m_choices)
      : choices(m_choices) {
    std::vector<int64_t> indices;
    for (typename decltype(choices)::size_type x = 0; x < choices.size(); ++x) {
      indices.push_back(x);
    }
    VariableInfo info;
    info.category      = StatisticalCategory::kokkos_value_categorical;
    info.valueQuantity = CandidateValueType::kokkos_value_set;
    info.type          = ValueType::kokkos_value_int64;
    info.candidates    = make_candidate_set(indices.size(), indices.data());
    tuning_variable_id = declare_output_type(name, info);
  }
  const Choice& begin() {
    context = get_new_context_id();
    begin_context(context);
    VariableValue value = make_variable_value(tuning_variable_id, int64_t(0));
    request_output_values(context, 1, &value);
    return choices[value.value.int_value];
  }
  void end() { end_context(context); }
};

template <typename Choice>
auto make_categorical_tuner(std::string name, std::vector<Choice> choices)
    -> CategoricalTuner<Choice> {
  return CategoricalTuner<Choice>(name, choices);
}

}  // namespace Experimental
}  // namespace Tools
}  // namespace Kokkos

#endif
