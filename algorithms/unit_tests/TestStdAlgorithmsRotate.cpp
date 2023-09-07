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

#include <TestStdAlgorithmsCommon.hpp>
#include <utility>
#include <algorithm>

namespace Test {
namespace stdalgos {
namespace Rotate {

namespace KE = Kokkos::Experimental;

template <class ValueType>
struct UnifDist;

template <>
struct UnifDist<int> {
  using dist_type = std::uniform_int_distribution<int>;
  std::mt19937 m_gen;
  dist_type m_dist;

  UnifDist() : m_dist(-50, 50) { m_gen.seed(1034343); }
  int operator()() { return m_dist(m_gen); }
};

template <>
struct UnifDist<double> {
  using dist_type = std::uniform_real_distribution<double>;
  std::mt19937 m_gen;
  dist_type m_dist;

  UnifDist() : m_dist(-90., 100.) { m_gen.seed(1034343); }

  double operator()() { return m_dist(m_gen); }
};

template <class ViewType>
void fill_view(ViewType dest_view, const std::string& name) {
  using value_type      = typename ViewType::value_type;
  using exe_space       = typename ViewType::execution_space;
  const std::size_t ext = dest_view.extent(0);
  using aux_view_t      = Kokkos::View<value_type*, exe_space>;
  aux_view_t aux_view("aux_view", ext);
  auto v_h = create_mirror_view(Kokkos::HostSpace(), aux_view);

  if (name == "empty") {
    // no op
  }

  else if (name == "one-element-a") {
    v_h(0) = static_cast<value_type>(1);
  }

  else if (name == "one-element-b") {
    v_h(0) = static_cast<value_type>(2);
  }

  else if (name == "two-elements-a") {
    v_h(0) = static_cast<value_type>(1);
    v_h(1) = static_cast<value_type>(2);
  }

  else if (name == "two-elements-b") {
    v_h(0) = static_cast<value_type>(2);
    v_h(1) = static_cast<value_type>(-1);
  }

  else if (name == "small-a") {
    v_h(0)  = static_cast<value_type>(0);
    v_h(1)  = static_cast<value_type>(1);
    v_h(2)  = static_cast<value_type>(1);
    v_h(3)  = static_cast<value_type>(-2);
    v_h(4)  = static_cast<value_type>(3);
    v_h(5)  = static_cast<value_type>(4);
    v_h(6)  = static_cast<value_type>(-40);
    v_h(7)  = static_cast<value_type>(4);
    v_h(8)  = static_cast<value_type>(5);
    v_h(9)  = static_cast<value_type>(62);
    v_h(10) = static_cast<value_type>(6);
  }

  else if (name == "small-b") {
    v_h(0)  = static_cast<value_type>(1);
    v_h(1)  = static_cast<value_type>(1);
    v_h(2)  = static_cast<value_type>(-1);
    v_h(3)  = static_cast<value_type>(2);
    v_h(4)  = static_cast<value_type>(-3);
    v_h(5)  = static_cast<value_type>(4);
    v_h(6)  = static_cast<value_type>(4);
    v_h(7)  = static_cast<value_type>(24);
    v_h(8)  = static_cast<value_type>(5);
    v_h(9)  = static_cast<value_type>(-46);
    v_h(10) = static_cast<value_type>(8);
    v_h(11) = static_cast<value_type>(9);
    v_h(12) = static_cast<value_type>(8);
  }

  else if (name == "medium" || name == "large") {
    UnifDist<value_type> randObj;
    for (std::size_t i = 0; i < ext; ++i) {
      v_h(i) = randObj();
    }
  }

  else {
    throw std::runtime_error("invalid choice");
  }

  Kokkos::deep_copy(aux_view, v_h);
  CopyFunctor<aux_view_t, ViewType> F1(aux_view, dest_view);
  Kokkos::parallel_for("copy", dest_view.extent(0), F1);
}

template <class ViewType, class ResultIt, class ViewHostType>
void verify_data(ResultIt result_it, ViewType view, ViewHostType data_view_host,
                 std::size_t rotation_point) {
  // run std::rotate
  auto n_it = KE::begin(data_view_host) + rotation_point;
  auto std_rit =
      std::rotate(KE::begin(data_view_host), n_it, KE::end(data_view_host));

  // make sure results match
  const auto my_diff  = result_it - KE::begin(view);
  const auto std_diff = std_rit - KE::begin(data_view_host);
  ASSERT_EQ(my_diff, std_diff);

  // check views match
  auto view_h           = create_host_space_copy(view);
  const std::size_t ext = view_h.extent(0);
  for (std::size_t i = 0; i < ext; ++i) {
    ASSERT_EQ(view_h(i), data_view_host[i]);
    // std::cout << "i= " << i << " "
    // 	      << "mine: " << view_h(i) << " "
    // 	      << "std: " << data_view_host(i)
    // 	      << '\n';
  }
}

std::string value_type_to_string(int) { return "int"; }
std::string value_type_to_string(double) { return "double"; }

template <class Tag, class ValueType>
void print_scenario_details(const std::string& name,
                            std::size_t rotation_point) {
  std::cout << "rotate: "
            << " at " << rotation_point << ", " << name << ", "
            << view_tag_to_string(Tag{}) << ", "
            << value_type_to_string(ValueType()) << std::endl;
}

template <class Tag, class ValueType, class InfoType>
void run_single_scenario(const InfoType& scenario_info,
                         std::size_t rotation_point) {
  const auto name            = std::get<0>(scenario_info);
  const std::size_t view_ext = std::get<1>(scenario_info);
  // print_scenario_details<Tag, ValueType>(name, rotation_point);

  {
    auto view = create_view<ValueType>(Tag{}, view_ext, "rotate_data_view");
    fill_view(view, name);
    // create host copy BEFORE rotate or view will be modified
    auto view_h = create_host_space_copy(view);
    auto n_it   = KE::begin(view) + rotation_point;
    auto rit    = KE::rotate(exespace(), KE::begin(view), n_it, KE::end(view));
    verify_data(rit, view, view_h, rotation_point);
  }

  {
    auto view = create_view<ValueType>(Tag{}, view_ext, "rotate_data_view");
    fill_view(view, name);
    // create host copy BEFORE rotate or view will be modified
    auto view_h = create_host_space_copy(view);
    auto n_it   = KE::begin(view) + rotation_point;
    auto rit =
        KE::rotate("label", exespace(), KE::begin(view), n_it, KE::end(view));
    verify_data(rit, view, view_h, rotation_point);
  }

  {
    auto view = create_view<ValueType>(Tag{}, view_ext, "rotate_data_view");
    fill_view(view, name);
    // create host copy BEFORE rotate or view will be modified
    auto view_h = create_host_space_copy(view);
    auto rit    = KE::rotate(exespace(), view, rotation_point);
    // verify_data(rit, view, view_h, rotation_point);
  }

  {
    auto view = create_view<ValueType>(Tag{}, view_ext, "rotate_data_view");
    fill_view(view, name);
    // create host copy BEFORE rotate or view will be modified
    auto view_h = create_host_space_copy(view);
    auto rit    = KE::rotate("label", exespace(), view, rotation_point);
    verify_data(rit, view, view_h, rotation_point);
  }

  Kokkos::fence();
}

template <class Tag, class ValueType>
void run_all_scenarios() {
  const std::map<std::string, std::size_t> scenarios = {
      {"empty", 0},          {"one-element-a", 1},  {"one-element-b", 1},
      {"two-elements-a", 2}, {"two-elements-b", 2}, {"small-a", 11},
      {"small-b", 13},       {"medium", 21103},     {"large", 101513}};

  std::vector<std::size_t> rotation_points = {0,  1,   2,    3,     8,
                                              56, 101, 1003, 101501};

  for (const auto& it : scenarios) {
    for (const auto& it2 : rotation_points) {
      // for each view scenario, we rotate at multiple points
      // but only if the view has an extent that is >= rotation point
      const auto view_ext = it.second;
      if (view_ext >= it2) {
        run_single_scenario<Tag, ValueType>(it, it2);
      }
    }
  }
}

TEST(std_algorithms_mod_seq_ops, rotate) {
  run_all_scenarios<DynamicTag, int>();
  run_all_scenarios<StridedThreeTag, int>();
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedThreeTag, double>();
}

}  // namespace Rotate
}  // namespace stdalgos
}  // namespace Test
