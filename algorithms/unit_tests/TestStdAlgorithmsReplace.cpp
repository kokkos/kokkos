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

namespace Test {
namespace stdalgos {
namespace Replace {

namespace KE = Kokkos::Experimental;

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
    for (std::size_t i = 0; i < ext; ++i) {
      v_h(i) = value_type{-5} + static_cast<value_type>(i + 1);
    }
    v_h(0) = static_cast<value_type>(2);
    v_h(3) = static_cast<value_type>(2);
    v_h(5) = static_cast<value_type>(2);
  }

  else if (name == "small-b") {
    for (std::size_t i = 0; i < ext; ++i) {
      if (i < 4) {
        v_h(i) = static_cast<value_type>(-1);
      } else {
        v_h(i) = static_cast<value_type>(2);
      }
    }
  }

  else if (name == "medium" || name == "large") {
    for (std::size_t i = 0; i < ext; ++i) {
      if (i % 2 == 0) {
        v_h(i) = static_cast<value_type>(-1);
      } else {
        v_h(i) = static_cast<value_type>(2);
      }
    }
  }

  else {
    throw std::runtime_error("invalid choice");
  }

  Kokkos::deep_copy(aux_view, v_h);
  CopyFunctor<aux_view_t, ViewType> F1(aux_view, dest_view);
  Kokkos::parallel_for("copy", dest_view.extent(0), F1);
}

template <class ViewType1, class ValueType>
void verify_data(const std::string& name, ViewType1 test_view,
                 ValueType new_value) {
  //! always careful because views might not be deep copyable
  auto view_dc = create_deep_copyable_compatible_clone(test_view);
  auto view_h  = create_mirror_view_and_copy(Kokkos::HostSpace(), view_dc);

  if (name == "empty") {
    // no op
  }

  else if (name == "one-element-a") {
    ASSERT_EQ(view_h(0), ValueType{1});
  }

  else if (name == "one-element-b") {
    ASSERT_EQ(view_h(0), new_value);
  }

  else if (name == "two-elements-a") {
    ASSERT_EQ(view_h(0), ValueType{1});
    ASSERT_EQ(view_h(1), new_value);
  }

  else if (name == "two-elements-b") {
    ASSERT_EQ(view_h(0), new_value);
    ASSERT_EQ(view_h(1), ValueType{-1});
  }

  else if (name == "small-a") {
    for (std::size_t i = 0; i < view_h.extent(0); ++i) {
      if (i == 0 || i == 3 || i == 5 || i == 6) {
        ASSERT_EQ(view_h(i), new_value);
      } else {
        const auto gold = ValueType{-5} + static_cast<ValueType>(i + 1);
        ASSERT_EQ(view_h(i), gold);
      }
    }
  }

  else if (name == "small-b") {
    for (std::size_t i = 0; i < view_h.extent(0); ++i) {
      if (i < 4) {
        ASSERT_EQ(view_h(i), ValueType{-1});
      } else {
        ASSERT_EQ(view_h(i), new_value);
      }
    }
  }

  else if (name == "medium" || name == "large") {
    for (std::size_t i = 0; i < view_h.extent(0); ++i) {
      if (i % 2 == 0) {
        ASSERT_EQ(view_h(i), ValueType{-1});
      } else {
        ASSERT_EQ(view_h(i), new_value);
      }
    }
  }

  else {
    throw std::runtime_error("invalid choice");
  }
}

std::string value_type_to_string(int) { return "int"; }
std::string value_type_to_string(double) { return "double"; }

template <class Tag, class ValueType, class InfoType>
void run_single_scenario(const InfoType& scenario_info) {
  const auto name            = std::get<0>(scenario_info);
  const std::size_t view_ext = std::get<1>(scenario_info);
  // std::cout << "replace: " << name << ", " << view_tag_to_string(Tag{}) << ",
  // "
  //           << value_type_to_string(ValueType()) << std::endl;

  ValueType old_value{2};
  ValueType new_value{43};
  auto view = create_view<ValueType>(Tag{}, view_ext, "replace");

  {
    fill_view(view, name);
    KE::replace(exespace(), KE::begin(view), KE::end(view), old_value,
                new_value);
    verify_data(name, view, new_value);
  }

  {
    fill_view(view, name);
    KE::replace("label", exespace(), KE::begin(view), KE::end(view), old_value,
                new_value);
    verify_data(name, view, new_value);
  }

  {
    fill_view(view, name);
    KE::replace(exespace(), view, old_value, new_value);
    verify_data(name, view, new_value);
  }

  {
    fill_view(view, name);
    KE::replace("label", exespace(), view, old_value, new_value);
    verify_data(name, view, new_value);
  }

  Kokkos::fence();
}

template <class Tag, class ValueType>
void run_all_scenarios() {
  const std::map<std::string, std::size_t> scenarios = {
      {"empty", 0},          {"one-element-a", 1},  {"one-element-b", 1},
      {"two-elements-a", 2}, {"two-elements-b", 2}, {"small-a", 9},
      {"small-b", 13},       {"medium", 1103},      {"large", 101513}};

  for (const auto& it : scenarios) {
    run_single_scenario<Tag, ValueType>(it);
  }
}

TEST(std_algorithms_replace_ops_test, replace) {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedThreeTag, double>();
  run_all_scenarios<DynamicTag, int>();
  run_all_scenarios<StridedThreeTag, int>();
}

}  // namespace Replace
}  // namespace stdalgos
}  // namespace Test
