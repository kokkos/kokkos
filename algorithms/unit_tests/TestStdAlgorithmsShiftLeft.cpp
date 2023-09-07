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
namespace ShiftLeft {

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

  else {
    UnifDist<value_type> randObj;
    for (std::size_t i = 0; i < ext; ++i) {
      v_h(i) = randObj();
    }
  }

  Kokkos::deep_copy(aux_view, v_h);
  CopyFunctor<aux_view_t, ViewType> F1(aux_view, dest_view);
  Kokkos::parallel_for("copy", dest_view.extent(0), F1);
}

template <class ForwardIterator>
ForwardIterator my_std_shift_left(
    ForwardIterator first, ForwardIterator last,
    typename std::iterator_traits<ForwardIterator>::difference_type n) {
  // copied from
  // https://github.com/llvm/llvm-project/blob/main/libcxx/include/__algorithm/shift_left.h

  if (n == 0) {
    return last;
  }

  ForwardIterator m = first;
  for (; n > 0; --n) {
    if (m == last) {
      return first;
    }
    ++m;
  }
  return std::move(m, last, first);
}

template <class ViewType, class ResultIt, class ViewHostType>
void verify_data(ResultIt result_it, ViewType view, ViewHostType data_view_host,
                 std::size_t shift_value) {
  auto std_rit = my_std_shift_left(KE::begin(data_view_host),
                                   KE::end(data_view_host), shift_value);

  // make sure results match
  const auto my_diff  = result_it - KE::begin(view);
  const auto std_diff = std_rit - KE::begin(data_view_host);
  ASSERT_EQ(my_diff, std_diff);

  // check views match
  auto view_h = create_host_space_copy(view);
  for (std::size_t i = 0; i < (std::size_t)my_diff; ++i) {
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
void print_scenario_details(const std::string& name, std::size_t shift_value) {
  std::cout << "shift_left: "
            << " by " << shift_value << ", " << name << ", "
            << view_tag_to_string(Tag{}) << ", "
            << value_type_to_string(ValueType()) << std::endl;
}

template <class Tag, class ValueType, class InfoType>
void run_single_scenario(const InfoType& scenario_info,
                         std::size_t shift_value) {
  const auto name            = std::get<0>(scenario_info);
  const std::size_t view_ext = std::get<1>(scenario_info);
  // print_scenario_details<Tag, ValueType>(name, shift_value);

  {
    auto view = create_view<ValueType>(Tag{}, view_ext, "shift_left_data_view");
    fill_view(view, name);
    // create host copy BEFORE shift_left or view will be modified
    auto view_h = create_host_space_copy(view);
    auto rit =
        KE::shift_left(exespace(), KE::begin(view), KE::end(view), shift_value);
    verify_data(rit, view, view_h, shift_value);
  }

  {
    auto view = create_view<ValueType>(Tag{}, view_ext, "shift_left_data_view");
    fill_view(view, name);
    // create host copy BEFORE shift_left or view will be modified
    auto view_h = create_host_space_copy(view);
    auto rit    = KE::shift_left("label", exespace(), KE::begin(view),
                              KE::end(view), shift_value);
    verify_data(rit, view, view_h, shift_value);
  }

  {
    auto view = create_view<ValueType>(Tag{}, view_ext, "shift_left_data_view");
    fill_view(view, name);
    // create host copy BEFORE shift_left or view will be modified
    auto view_h = create_host_space_copy(view);
    auto rit    = KE::shift_left(exespace(), view, shift_value);
    verify_data(rit, view, view_h, shift_value);
  }

  {
    auto view = create_view<ValueType>(Tag{}, view_ext, "shift_left_data_view");
    fill_view(view, name);
    // create host copy BEFORE shift_left or view will be modified
    auto view_h = create_host_space_copy(view);
    auto rit    = KE::shift_left("label", exespace(), view, shift_value);
    verify_data(rit, view, view_h, shift_value);
  }

  Kokkos::fence();
}

template <class Tag, class ValueType>
void run_all_scenarios() {
  const std::map<std::string, std::size_t> scenarios = {{"empty", 0},
                                                        {"one-element-a", 1},
                                                        {"one-element-b", 1},
                                                        {"two-elements-a", 2},
                                                        {"two-elements-b", 2},
                                                        {"three-elements-a", 3},
                                                        {"three-elements-b", 3},
                                                        {"small-a", 11},
                                                        {"small-b", 13},
                                                        {"medium", 21103},
                                                        {"large", 101513}};

  // a shift value MUST be non-negative but it does not matter
  // if it is larger than the view, the algorithm is supposed
  // to handle that case too
  std::vector<std::size_t> shifts = {0, 1, 2, 3, 8, 56, 101, 1003, 101501};

  for (const auto& it : scenarios) {
    for (const auto& it2 : shifts) {
      run_single_scenario<Tag, ValueType>(it, it2);
    }
  }
}

TEST(std_algorithms_mod_seq_ops, shift_left) {
  run_all_scenarios<DynamicTag, int>();
  run_all_scenarios<StridedThreeTag, int>();
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedThreeTag, double>();
}

}  // namespace ShiftLeft
}  // namespace stdalgos
}  // namespace Test
