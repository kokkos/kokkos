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
#include <mdspan/mdspan.hpp>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

namespace {

template<class Integral, Integral Value>
using IC = std::integral_constant<Integral, Value>;

template<class ExpectedOutputMdspan, class InputMdspan, class ... Slices>
void test_submdspan_static_slice(
  typename InputMdspan::mapping_type input_mapping,
  [[maybe_unused]] Slices&&... slices)
{
  using input_type = InputMdspan;
  std::vector<typename InputMdspan::value_type> storage(input_mapping.required_span_size());

  input_type input(storage.data(), input_mapping);
  auto output = Kokkos::submdspan(input, std::forward<Slices>(slices)...);
  using output_mdspan_type = decltype(output);
  static_assert(std::is_same<typename output_mdspan_type::extents_type, typename ExpectedOutputMdspan::extents_type>::value, "Extents don't match.");
  static_assert(std::is_same<typename output_mdspan_type::layout_type, typename ExpectedOutputMdspan::layout_type>::value, "Layouts don't match.");
  static_assert(std::is_same<output_mdspan_type, ExpectedOutputMdspan>::value, "submdspan return types don't match.");
}

template<class ExpectedOutputMdspan, class InputMdspan, class ... Slices>
void test_submdspan_static_slice(
  typename InputMdspan::extents_type input_extents,
  [[maybe_unused]] Slices&&... slices)
{
  using input_mapping_type = typename InputMdspan::mapping_type;
  input_mapping_type input_mapping(input_extents);
  test_submdspan_static_slice<ExpectedOutputMdspan, InputMdspan>(input_mapping, std::forward<Slices>(slices)...);
}

TEST(TestMdspan, StaticAsserts) {
  (void) test_info_;
  static_assert(std::is_convertible<IC<std::size_t, 1>, std::size_t>::value, "Just a check.");
  static_assert(std::is_convertible<IC<int32_t, 1>, std::size_t>::value, "Just a check.");
  static_assert(std::is_convertible<IC<uint32_t, 1>, std::size_t>::value, "Just a check.");
}

TEST(TestMdspan, SubmdspanStaticSlice_Left_iddd_FullFullIndex) {
  (void) test_info_;
  using input_extents_type = Kokkos::dextents<int, 3>;
  using input_layout_type = Kokkos::layout_left;
  using input_mdspan_type = Kokkos::mdspan<float, input_extents_type, input_layout_type>;
  input_extents_type input_extents{3, 4, 5};

  {
    using expected_extents_type = Kokkos::dextents<int, 2>;
    using expected_layout_type = Kokkos::layout_left;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&] (auto integralConstant) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(
	input_extents, Kokkos::full_extent, Kokkos::full_extent, integralConstant);
    };
    runTest(int(1));
    runTest(std::size_t(1));
    runTest(std::int64_t(1));
    runTest(std::uint32_t(1));
    runTest(IC<int, 1>{});
    runTest(IC<std::size_t, 1>{});
    runTest(IC<std::int64_t, 1>{});
    runTest(IC<std::uint32_t, 1>{});
  }
}

TEST(TestMdspan, SubmdspanStaticSlice_Left_i345_FullFullIndex) {
  (void) test_info_;
  using input_extents_type = Kokkos::extents<int, 3, 4, 5>;
  using input_layout_type = Kokkos::layout_left;
  using input_mdspan_type = Kokkos::mdspan<float, input_extents_type, input_layout_type>;
  input_extents_type input_extents;

  {
    using expected_extents_type = Kokkos::extents<int, 3, 4>;
    using expected_layout_type = Kokkos::layout_left;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&] (auto integralConstant) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(
	input_extents, Kokkos::full_extent, Kokkos::full_extent, integralConstant);
    };
    runTest(int(1));
    runTest(std::size_t(1));
    runTest(std::int64_t(1));
    runTest(std::uint32_t(1));
    runTest(IC<int, 1>{});
    runTest(IC<std::size_t, 1>{});
    runTest(IC<std::int64_t, 1>{});
    runTest(IC<std::uint32_t, 1>{});
  }
}

TEST(TestMdspan, SubmdspanStaticSlice_Right_iddd_FullFullIndex) {
  (void) test_info_;
  using input_extents_type = Kokkos::dextents<int, 3>;
  using input_layout_type = Kokkos::layout_right;
  using input_mdspan_type = Kokkos::mdspan<float, input_extents_type, input_layout_type>;
  input_extents_type input_extents{3, 4, 5};

  {
    using expected_extents_type = Kokkos::dextents<int, 2>;
    using expected_layout_type = Kokkos::layout_stride;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&] (auto sliceSpec) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(
	input_extents, Kokkos::full_extent, Kokkos::full_extent, sliceSpec);
    };
    runTest(int(1));
    runTest(std::size_t(1));
    runTest(std::int64_t(1));
    runTest(std::uint32_t(1));
    runTest(IC<int, 1>{});
    runTest(IC<std::size_t, 1>{});
    runTest(IC<std::int64_t, 1>{});
    runTest(IC<std::uint32_t, 1>{});
  }
}

TEST(TestMdspan, SubmdspanStaticSlice_Right_i345_FullFullIndex) {
  (void) test_info_;
  using input_extents_type = Kokkos::extents<int, 3, 4, 5>;
  using input_layout_type = Kokkos::layout_right;
  using input_mdspan_type = Kokkos::mdspan<float, input_extents_type, input_layout_type>;
  input_extents_type input_extents;

  {
    using expected_extents_type = Kokkos::extents<int, 3, 4>;
    using expected_layout_type = Kokkos::layout_stride;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&] (auto integralConstant) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(
	input_extents, Kokkos::full_extent, Kokkos::full_extent, integralConstant);
    };
    runTest(int(1));
    runTest(std::size_t(1));
    runTest(std::int64_t(1));
    runTest(std::uint32_t(1));
    runTest(IC<int, 1>{});
    runTest(IC<std::size_t, 1>{});
    runTest(IC<std::int64_t, 1>{});
    runTest(IC<std::uint32_t, 1>{});
  }
}

TEST(TestMdspan, SubmdspanStaticSlice_Left_iddd_FullIndexFull) {
  (void) test_info_;
  using input_extents_type = Kokkos::dextents<int, 3>;
  using input_layout_type = Kokkos::layout_left;
  using input_mdspan_type = Kokkos::mdspan<float, input_extents_type, input_layout_type>;
  input_extents_type input_extents{3, 4, 5};

  {
    using expected_extents_type = Kokkos::dextents<int, 2>;
    using expected_layout_type = Kokkos::Experimental::layout_left_padded<Kokkos::dynamic_extent>;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&] (auto integralConstant) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(
        input_extents, Kokkos::full_extent, integralConstant, Kokkos::full_extent);
    };
    runTest(int(1));
    runTest(std::size_t(1));
    runTest(std::int64_t(1));
    runTest(std::uint32_t(1));
    runTest(IC<int, 1>{});
    runTest(IC<std::size_t, 1>{});
    runTest(IC<std::int64_t, 1>{});
    runTest(IC<std::uint32_t, 1>{});
  }
}

TEST(TestMdspan, SubmdspanStaticSlice_Left_i345_FullIndexFull) {
  (void) test_info_;
  using input_extents_type = Kokkos::extents<int, 3, 4, 5>;
  using input_layout_type = Kokkos::layout_left;
  using input_mdspan_type = Kokkos::mdspan<float, input_extents_type, input_layout_type>;
  input_extents_type input_extents;

  {
    using expected_extents_type = Kokkos::extents<int, 3, 5>;
    using expected_layout_type = Kokkos::Experimental::layout_left_padded<12>;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&] (auto integralConstant) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(
        input_extents, Kokkos::full_extent, integralConstant, Kokkos::full_extent);
    };
    runTest(int(1));
    runTest(std::size_t(1));
    runTest(std::int64_t(1));
    runTest(std::uint32_t(1));
    runTest(IC<int, 1>{});
    runTest(IC<std::size_t, 1>{});
    runTest(IC<std::int64_t, 1>{});
    runTest(IC<std::uint32_t, 1>{});
  }
}

TEST(TestMdspan, SubmdspanStaticSlice_Left_iddd_FullTupleFull) {
  (void) test_info_;
  using std::tuple;
  using input_extents_type = Kokkos::dextents<int, 3>;
  using input_layout_type = Kokkos::layout_left;
  using input_mdspan_type = Kokkos::mdspan<float, input_extents_type, input_layout_type>;
  input_extents_type input_extents{3, 4, 5};

  {
    using expected_extents_type = Kokkos::dextents<int, 3>;
    using expected_layout_type = Kokkos::layout_stride;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&] (auto sliceSpec) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(
        input_extents, Kokkos::full_extent, sliceSpec, Kokkos::full_extent);
    };
    runTest(tuple<int, int>{int(1), int{3}});
    runTest(tuple<std::size_t, std::size_t>{std::size_t(1), std::size_t(3)});
    runTest(tuple<std::int64_t, std::int64_t>{std::int64_t(1), std::int64_t(3)});
    runTest(tuple<std::uint32_t, std::uint32_t>{std::uint32_t(1), std::uint32_t(3)});
  }
  {
    using expected_extents_type = Kokkos::extents<int, Kokkos::dynamic_extent, 2, Kokkos::dynamic_extent>;
    using expected_layout_type = Kokkos::layout_stride;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&] (auto sliceSpec) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(
        input_extents, Kokkos::full_extent, sliceSpec, Kokkos::full_extent);
    };
    runTest(tuple<IC<int, 1>, IC<int, 3>>{IC<int, 1>{}, IC<int, 3>{}});
    runTest(tuple<IC<std::size_t,1 >, IC<std::size_t, 3>>{IC<std::size_t, 1>{}, IC<std::size_t, 3>{}});
    runTest(tuple<IC<std::int64_t, 1>, IC<std::int64_t, 3>>{IC<std::int64_t, 1>{}, IC<std::int64_t, 3>{}});
    runTest(tuple<IC<std::uint32_t, 1>, IC<std::uint32_t, 3>>{IC<std::uint32_t, 1>{}, IC<std::uint32_t, 3>{}});
  }
}

TEST(TestMdspan, SubmdspanStaticSlice_Left_i345_FullTupleFull) {
  (void) test_info_;
  using std::tuple;
  using input_extents_type = Kokkos::extents<int, 3, 4, 5>;
  using input_layout_type = Kokkos::layout_left;
  using input_mdspan_type = Kokkos::mdspan<float, input_extents_type, input_layout_type>;
  input_extents_type input_extents;

  {
    using expected_extents_type = Kokkos::extents<int, 3, Kokkos::dynamic_extent, 5>;
    using expected_layout_type = Kokkos::layout_stride;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&] (auto sliceSpec) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(
        input_extents, Kokkos::full_extent, sliceSpec, Kokkos::full_extent);
    };
    runTest(tuple<int, int>{int(1), int{3}});
    runTest(tuple<std::size_t, std::size_t>{std::size_t(1), std::size_t(3)});
    runTest(tuple<std::int64_t, std::int64_t>{std::int64_t(1), std::int64_t(3)});
    runTest(tuple<std::uint32_t, std::uint32_t>{std::uint32_t(1), std::uint32_t(3)});
  }
  {
    using expected_extents_type = Kokkos::extents<int, 3, 2, 5>;
    using expected_layout_type = Kokkos::layout_stride;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&] (auto sliceSpec) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(
        input_extents, Kokkos::full_extent, sliceSpec, Kokkos::full_extent);
    };
    runTest(tuple<IC<int, 1>, IC<int, 3>>{IC<int, 1>{}, IC<int, 3>{}});
    runTest(tuple<IC<std::size_t,1 >, IC<std::size_t, 3>>{IC<std::size_t, 1>{}, IC<std::size_t, 3>{}});
    runTest(tuple<IC<std::int64_t, 1>, IC<std::int64_t, 3>>{IC<std::int64_t, 1>{}, IC<std::int64_t, 3>{}});
    runTest(tuple<IC<std::uint32_t, 1>, IC<std::uint32_t, 3>>{IC<std::uint32_t, 1>{}, IC<std::uint32_t, 3>{}});
  }
}

TEST(TestMdspan, SubmdspanStaticSlice_Right_iddd_FullTupleFull) {
  (void) test_info_;
  using std::tuple;
  using input_extents_type = Kokkos::dextents<int, 3>;
  using input_layout_type = Kokkos::layout_right;
  using input_mdspan_type = Kokkos::mdspan<float, input_extents_type, input_layout_type>;
  input_extents_type input_extents{3, 4, 5};

  {
    using expected_extents_type = Kokkos::dextents<int, 3>;
    using expected_layout_type = Kokkos::layout_stride;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&] (auto sliceSpec) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(
        input_extents, Kokkos::full_extent, sliceSpec, Kokkos::full_extent);
    };
    runTest(tuple<int, int>{int(1), int{3}});
    runTest(tuple<std::size_t, std::size_t>{std::size_t(1), std::size_t(3)});
    runTest(tuple<std::int64_t, std::int64_t>{std::int64_t(1), std::int64_t(3)});
    runTest(tuple<std::uint32_t, std::uint32_t>{std::uint32_t(1), std::uint32_t(3)});
  }
  {
    using expected_extents_type = Kokkos::extents<int, Kokkos::dynamic_extent, 2, Kokkos::dynamic_extent>;
    using expected_layout_type = Kokkos::layout_stride;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&] (auto sliceSpec) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(
        input_extents, Kokkos::full_extent, sliceSpec, Kokkos::full_extent);
    };
    runTest(tuple<IC<int, 1>, IC<int, 3>>{IC<int, 1>{}, IC<int, 3>{}});
    runTest(tuple<IC<std::size_t,1 >, IC<std::size_t, 3>>{IC<std::size_t, 1>{}, IC<std::size_t, 3>{}});
    runTest(tuple<IC<std::int64_t, 1>, IC<std::int64_t, 3>>{IC<std::int64_t, 1>{}, IC<std::int64_t, 3>{}});
    runTest(tuple<IC<std::uint32_t, 1>, IC<std::uint32_t, 3>>{IC<std::uint32_t, 1>{}, IC<std::uint32_t, 3>{}});
  }
}

TEST(TestMdspan, SubmdspanStaticSlice_Right_i345_FullTupleFull) {
  (void) test_info_;
  using std::tuple;
  using input_extents_type = Kokkos::extents<int, 3, 4, 5>;
  using input_layout_type = Kokkos::layout_right;
  using input_mdspan_type = Kokkos::mdspan<float, input_extents_type, input_layout_type>;
  input_extents_type input_extents;

  {
    using expected_extents_type = Kokkos::extents<int, 3, Kokkos::dynamic_extent, 5>;
    using expected_layout_type = Kokkos::layout_stride;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&] (auto sliceSpec) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(
        input_extents, Kokkos::full_extent, sliceSpec, Kokkos::full_extent);
    };
    runTest(tuple<int, int>{int(1), int{3}});
    runTest(tuple<std::size_t, std::size_t>{std::size_t(1), std::size_t(3)});
    runTest(tuple<std::int64_t, std::int64_t>{std::int64_t(1), std::int64_t(3)});
    runTest(tuple<std::uint32_t, std::uint32_t>{std::uint32_t(1), std::uint32_t(3)});
  }
  {
    using expected_extents_type = Kokkos::extents<int, 3, 2, 5>;
    using expected_layout_type = Kokkos::layout_stride;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&] (auto sliceSpec) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(
        input_extents, Kokkos::full_extent, sliceSpec, Kokkos::full_extent);
    };
    runTest(tuple<IC<int, 1>, IC<int, 3>>{IC<int, 1>{}, IC<int, 3>{}});
    runTest(tuple<IC<std::size_t,1 >, IC<std::size_t, 3>>{IC<std::size_t, 1>{}, IC<std::size_t, 3>{}});
    runTest(tuple<IC<std::int64_t, 1>, IC<std::int64_t, 3>>{IC<std::int64_t, 1>{}, IC<std::int64_t, 3>{}});
    runTest(tuple<IC<std::uint32_t, 1>, IC<std::uint32_t, 3>>{IC<std::uint32_t, 1>{}, IC<std::uint32_t, 3>{}});
  }
}

TEST(TestMdspan, SubmdspanStaticSlice_Left_iddd_TupleFullTuple) {
  (void) test_info_;
  using std::tuple;
  using input_extents_type = Kokkos::dextents<int, 3>;
  using input_layout_type = Kokkos::layout_left;
  using input_mdspan_type = Kokkos::mdspan<float, input_extents_type, input_layout_type>;
  input_extents_type input_extents{3, 4, 5};

  {
    using expected_extents_type = Kokkos::dextents<int, 3>;
    using expected_layout_type = Kokkos::Experimental::layout_left_padded<Kokkos::dynamic_extent>;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&] (auto sliceSpec0, auto sliceSpec1) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(
        input_extents, sliceSpec0, Kokkos::full_extent, sliceSpec1);
    };
    runTest(tuple<int, int>{int{1}, int{3}}, tuple<int, int>{int{1}, int{4}});
    runTest(tuple<std::size_t, std::size_t>{std::size_t{1}, std::size_t{3}}, tuple<std::size_t, std::size_t>{std::size_t{1}, std::size_t{4}});
    runTest(tuple<std::int64_t, std::int64_t>{std::int64_t{1}, std::int64_t{3}}, tuple<std::int64_t, std::int64_t>{std::int64_t{1}, std::int64_t{4}});
    runTest(tuple<std::uint32_t, std::uint32_t>{std::uint32_t{1}, std::uint32_t{3}}, tuple<std::uint32_t, std::uint32_t>{std::uint32_t{1}, std::uint32_t{4}});
  }
  {
    using expected_extents_type = Kokkos::extents<int, 2, Kokkos::dynamic_extent, 3>;
    using expected_layout_type = Kokkos::Experimental::layout_left_padded<Kokkos::dynamic_extent>;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&] (auto sliceSpec0, auto sliceSpec1) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(
        input_extents, sliceSpec0, Kokkos::full_extent, sliceSpec1);
    };
    runTest(tuple<IC<int, 1>, IC<int, 3>>{IC<int, 1>{}, IC<int, 3>{}},
	    tuple<IC<int, 1>, IC<int, 4>>{IC<int, 1>{}, IC<int, 4>{}});
    runTest(tuple<IC<std::size_t, 1>, IC<std::size_t, 3>>{IC<std::size_t, 1>{}, IC<std::size_t, 3>{}},
	    tuple<IC<std::size_t, 1>, IC<std::size_t, 4>>{IC<std::size_t, 1>{}, IC<std::size_t, 4>{}});
    runTest(tuple<IC<std::int64_t, 1>, IC<std::int64_t, 3>>{IC<std::int64_t, 1>{}, IC<std::int64_t, 3>{}},
	    tuple<IC<std::int64_t, 1>, IC<std::int64_t, 4>>{IC<std::int64_t, 1>{}, IC<std::int64_t, 4>{}});
    runTest(tuple<IC<std::uint32_t, 1>, IC<std::uint32_t, 3>>{IC<std::uint32_t, 1>{}, IC<std::uint32_t, 3>{}},
	    tuple<IC<std::uint32_t, 1>, IC<std::uint32_t, 4>>{IC<std::uint32_t, 1>{}, IC<std::uint32_t, 4>{}});
  }
}

TEST(TestMdspan, SubmdspanStaticSlice_Left_i345_TupleFullTuple) {
  (void) test_info_;
  using std::tuple;
  using input_extents_type = Kokkos::extents<int, 3, 4, 5>;
  using input_layout_type = Kokkos::layout_left;
  using input_mdspan_type = Kokkos::mdspan<float, input_extents_type, input_layout_type>;
  input_extents_type input_extents;

  {
    using expected_extents_type = Kokkos::extents<int, Kokkos::dynamic_extent, 4, Kokkos::dynamic_extent>;
    using expected_layout_type = Kokkos::Experimental::layout_left_padded<3>;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&] (auto sliceSpec0, auto sliceSpec1) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(
        input_extents, sliceSpec0, Kokkos::full_extent, sliceSpec1);
    };
    runTest(tuple<int, int>{int{1}, int{3}}, tuple<int, int>{int{1}, int{4}});
    runTest(tuple<std::size_t, std::size_t>{std::size_t{1}, std::size_t{3}}, tuple<std::size_t, std::size_t>{std::size_t{1}, std::size_t{4}});
    runTest(tuple<std::int64_t, std::int64_t>{std::int64_t{1}, std::int64_t{3}}, tuple<std::int64_t, std::int64_t>{std::int64_t{1}, std::int64_t{4}});
    runTest(tuple<std::uint32_t, std::uint32_t>{std::uint32_t{1}, std::uint32_t{3}}, tuple<std::uint32_t, std::uint32_t>{std::uint32_t{1}, std::uint32_t{4}});
  }
  {
    using expected_extents_type = Kokkos::extents<int, 2, 4, 3>;
    using expected_layout_type = Kokkos::Experimental::layout_left_padded<3>;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&] (auto sliceSpec0, auto sliceSpec1) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(
        input_extents, sliceSpec0, Kokkos::full_extent, sliceSpec1);
    };
    runTest(tuple<IC<int, 1>, IC<int, 3>>{IC<int, 1>{}, IC<int, 3>{}},
	    tuple<IC<int, 1>, IC<int, 4>>{IC<int, 1>{}, IC<int, 4>{}});
    runTest(tuple<IC<std::size_t, 1>, IC<std::size_t, 3>>{IC<std::size_t, 1>{}, IC<std::size_t, 3>{}},
	    tuple<IC<std::size_t, 1>, IC<std::size_t, 4>>{IC<std::size_t, 1>{}, IC<std::size_t, 4>{}});
    runTest(tuple<IC<std::int64_t, 1>, IC<std::int64_t, 3>>{IC<std::int64_t, 1>{}, IC<std::int64_t, 3>{}},
	    tuple<IC<std::int64_t, 1>, IC<std::int64_t, 4>>{IC<std::int64_t, 1>{}, IC<std::int64_t, 4>{}});
    runTest(tuple<IC<std::uint32_t, 1>, IC<std::uint32_t, 3>>{IC<std::uint32_t, 1>{}, IC<std::uint32_t, 3>{}},
	    tuple<IC<std::uint32_t, 1>, IC<std::uint32_t, 4>>{IC<std::uint32_t, 1>{}, IC<std::uint32_t, 4>{}});
  }
}

TEST(TestMdspan, SubmdspanStaticSlice_Right_iddd_TupleFullTuple) {
  (void) test_info_;
  using std::tuple;
  using input_extents_type = Kokkos::dextents<int, 3>;
  using input_layout_type = Kokkos::layout_right;
  using input_mdspan_type = Kokkos::mdspan<float, input_extents_type, input_layout_type>;
  input_extents_type input_extents{3, 4, 5};

  {
    using expected_extents_type = Kokkos::dextents<int, 3>;
    using expected_layout_type = Kokkos::Experimental::layout_right_padded<Kokkos::dynamic_extent>;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&] (auto sliceSpec0, auto sliceSpec1) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(
        input_extents, sliceSpec0, Kokkos::full_extent, sliceSpec1);
    };
    runTest(tuple<int, int>{int{1}, int{3}}, tuple<int, int>{int{1}, int{4}});
    runTest(tuple<std::size_t, std::size_t>{std::size_t{1}, std::size_t{3}}, tuple<std::size_t, std::size_t>{std::size_t{1}, std::size_t{4}});
    runTest(tuple<std::int64_t, std::int64_t>{std::int64_t{1}, std::int64_t{3}}, tuple<std::int64_t, std::int64_t>{std::int64_t{1}, std::int64_t{4}});
    runTest(tuple<std::uint32_t, std::uint32_t>{std::uint32_t{1}, std::uint32_t{3}}, tuple<std::uint32_t, std::uint32_t>{std::uint32_t{1}, std::uint32_t{4}});
  }
  {
    using expected_extents_type = Kokkos::extents<int, 2, Kokkos::dynamic_extent, 3>;
    using expected_layout_type = Kokkos::Experimental::layout_right_padded<Kokkos::dynamic_extent>;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&] (auto sliceSpec0, auto sliceSpec1) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(
        input_extents, sliceSpec0, Kokkos::full_extent, sliceSpec1);
    };
    runTest(tuple<IC<int, 1>, IC<int, 3>>{IC<int, 1>{}, IC<int, 3>{}},
	    tuple<IC<int, 1>, IC<int, 4>>{IC<int, 1>{}, IC<int, 4>{}});
    runTest(tuple<IC<std::size_t, 1>, IC<std::size_t, 3>>{IC<std::size_t, 1>{}, IC<std::size_t, 3>{}},
	    tuple<IC<std::size_t, 1>, IC<std::size_t, 4>>{IC<std::size_t, 1>{}, IC<std::size_t, 4>{}});
    runTest(tuple<IC<std::int64_t, 1>, IC<std::int64_t, 3>>{IC<std::int64_t, 1>{}, IC<std::int64_t, 3>{}},
	    tuple<IC<std::int64_t, 1>, IC<std::int64_t, 4>>{IC<std::int64_t, 1>{}, IC<std::int64_t, 4>{}});
    runTest(tuple<IC<std::uint32_t, 1>, IC<std::uint32_t, 3>>{IC<std::uint32_t, 1>{}, IC<std::uint32_t, 3>{}},
	    tuple<IC<std::uint32_t, 1>, IC<std::uint32_t, 4>>{IC<std::uint32_t, 1>{}, IC<std::uint32_t, 4>{}});
  }
}

TEST(TestMdspan, SubmdspanStaticSlice_Right_i345_TupleFullTuple) {
  (void) test_info_;
  using std::tuple;
  using input_extents_type = Kokkos::extents<int, 3, 4, 5>;
  using input_layout_type = Kokkos::layout_right;
  using input_mdspan_type = Kokkos::mdspan<float, input_extents_type, input_layout_type>;
  input_extents_type input_extents;

  {
    using expected_extents_type = Kokkos::extents<int, Kokkos::dynamic_extent, 4, Kokkos::dynamic_extent>;
    using expected_layout_type = Kokkos::Experimental::layout_right_padded<5>;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&] (auto sliceSpec0, auto sliceSpec1) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(
        input_extents, sliceSpec0, Kokkos::full_extent, sliceSpec1);
    };
    runTest(tuple<int, int>{int{1}, int{3}}, tuple<int, int>{int{1}, int{4}});
    runTest(tuple<std::size_t, std::size_t>{std::size_t{1}, std::size_t{3}}, tuple<std::size_t, std::size_t>{std::size_t{1}, std::size_t{4}});
    runTest(tuple<std::int64_t, std::int64_t>{std::int64_t{1}, std::int64_t{3}}, tuple<std::int64_t, std::int64_t>{std::int64_t{1}, std::int64_t{4}});
    runTest(tuple<std::uint32_t, std::uint32_t>{std::uint32_t{1}, std::uint32_t{3}}, tuple<std::uint32_t, std::uint32_t>{std::uint32_t{1}, std::uint32_t{4}});
  }
  {
    using expected_extents_type = Kokkos::extents<int, 2, 4, 3>;
    using expected_layout_type = Kokkos::Experimental::layout_right_padded<5>;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&] (auto sliceSpec0, auto sliceSpec1) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(
        input_extents, sliceSpec0, Kokkos::full_extent, sliceSpec1);
    };
    runTest(tuple<IC<int, 1>, IC<int, 3>>{IC<int, 1>{}, IC<int, 3>{}},
	    tuple<IC<int, 1>, IC<int, 4>>{IC<int, 1>{}, IC<int, 4>{}});
    runTest(tuple<IC<std::size_t, 1>, IC<std::size_t, 3>>{IC<std::size_t, 1>{}, IC<std::size_t, 3>{}},
	    tuple<IC<std::size_t, 1>, IC<std::size_t, 4>>{IC<std::size_t, 1>{}, IC<std::size_t, 4>{}});
    runTest(tuple<IC<std::int64_t, 1>, IC<std::int64_t, 3>>{IC<std::int64_t, 1>{}, IC<std::int64_t, 3>{}},
	    tuple<IC<std::int64_t, 1>, IC<std::int64_t, 4>>{IC<std::int64_t, 1>{}, IC<std::int64_t, 4>{}});
    runTest(tuple<IC<std::uint32_t, 1>, IC<std::uint32_t, 3>>{IC<std::uint32_t, 1>{}, IC<std::uint32_t, 3>{}},
	    tuple<IC<std::uint32_t, 1>, IC<std::uint32_t, 4>>{IC<std::uint32_t, 1>{}, IC<std::uint32_t, 4>{}});
  }
}

// Simpler, rank-2 tests follow.  If you have a build regression,
// it may help to comment out all the tests but the ones below.

TEST(TestMdspan, SubmdspanStaticSlice_Left_idd_FullTuple) {
  (void) test_info_;
  // tuple of two integral_constant is convertible to tuple of two size_t.
  // Nevertheless, submdspan needs special handling to be able to compile
  // with a tuple of two integral_constant as a slice specifier.
  // For example, mdspan needs an overload of
  // __assign_op_slice_handler::operator=
  // in order for the output's extents type to be able to encode
  // the compile-time slice information.
  static_assert(std::is_convertible<std::tuple<IC<std::size_t, 1>, IC<std::size_t, 3>>,
                                    std::tuple<std::size_t, std::size_t>>::value,
                                    "convertibility of tuples of integral constant to size_t is false");

  using input_extents_type = Kokkos::dextents<int, 2>;
  using input_layout_type = Kokkos::layout_left;
  using input_mdspan_type = Kokkos::mdspan<float, input_extents_type, input_layout_type>;

  input_extents_type input_extents{3, 4};
  {
    using expected_extents_type = Kokkos::dextents<int, 2>;
    using expected_layout_type = Kokkos::layout_left;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&](auto sliceSpec) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(input_extents, Kokkos::full_extent, sliceSpec);
    };
    runTest(std::tuple<int, int>{int(1), int(3)});
    runTest(std::tuple<std::size_t, std::size_t>{std::size_t(1), std::size_t(3)});
    runTest(std::tuple<std::uint32_t, std::uint32_t>{std::uint32_t(1), std::uint32_t(3)});
    runTest(std::tuple<std::int64_t, std::int64_t>{std::int64_t(1), std::int64_t(3)});
  }
  {
    using expected_extents_type = Kokkos::extents<int, Kokkos::dynamic_extent, 2>;
    using expected_layout_type = Kokkos::layout_left;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&](auto sliceSpec) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(input_extents, Kokkos::full_extent, sliceSpec);
    };
    runTest(std::tuple<IC<int, 1>, IC<int, 3>> {} );
    runTest(std::tuple<IC<std::size_t, 1>, IC<std::size_t, 3>> {} );
    runTest(std::tuple<IC<uint32_t, 1>, IC<uint32_t, 3>> {} );
    runTest(std::tuple<IC<int64_t, 1>, IC<int64_t, 3>> {} );
  }
}

TEST(TestMdspan, SubmdspanStaticSlice_Right_idd_FullTuple) {
  (void) test_info_;
  using input_extents_type = Kokkos::dextents<int, 2>;
  using input_layout_type = Kokkos::layout_right;
  using input_mdspan_type = Kokkos::mdspan<float, input_extents_type, input_layout_type>;

  input_extents_type input_extents{3, 4};
  {
    using expected_extents_type = Kokkos::dextents<int, 2>;
    using expected_layout_type = Kokkos::Experimental::layout_right_padded<Kokkos::dynamic_extent>;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&](auto sliceSpec) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(input_extents, Kokkos::full_extent, sliceSpec);
    };
    runTest(std::tuple<int, int>{int(1), int(3)});
    runTest(std::tuple<std::size_t, std::size_t>{std::size_t(1), std::size_t(3)});
    runTest(std::tuple<std::uint32_t, std::uint32_t>{std::uint32_t(1), std::uint32_t(3)});
    runTest(std::tuple<std::int64_t, std::int64_t>{std::int64_t(1), std::int64_t(3)});
  }
  {
    using expected_extents_type = Kokkos::extents<int, Kokkos::dynamic_extent, 2>;
    using expected_layout_type = Kokkos::Experimental::layout_right_padded<Kokkos::dynamic_extent>;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&](auto sliceSpec) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(input_extents, Kokkos::full_extent, sliceSpec);
    };
    runTest(std::tuple<IC<int, 1>, IC<int, 3>> {} );
    runTest(std::tuple<IC<std::size_t, 1>, IC<std::size_t, 3>> {} );
    runTest(std::tuple<IC<uint32_t, 1>, IC<uint32_t, 3>> {} );
    runTest(std::tuple<IC<int64_t, 1>, IC<int64_t, 3>> {} );
  }
}

TEST(TestMdspan, SubmdspanStaticSlice_Left_idd_TupleFull) {
  (void) test_info_;
  using input_extents_type = Kokkos::dextents<int, 2>;
  using input_layout_type = Kokkos::layout_left;
  using input_mdspan_type = Kokkos::mdspan<float, input_extents_type, input_layout_type>;

  input_extents_type input_extents{3, 4};
  {
    using expected_extents_type = Kokkos::dextents<int, 2>;
    using expected_layout_type = Kokkos::Experimental::layout_left_padded<Kokkos::dynamic_extent>;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&](auto sliceSpec) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(input_extents, sliceSpec, Kokkos::full_extent);
    };
    runTest(std::tuple<int, int>{int(1), int(3)});
    runTest(std::tuple<std::size_t, std::size_t>{std::size_t(1), std::size_t(3)});
    runTest(std::tuple<std::uint32_t, std::uint32_t>{std::uint32_t(1), std::uint32_t(3)});
    runTest(std::tuple<std::int64_t, std::int64_t>{std::int64_t(1), std::int64_t(3)});
  }
  {
    using expected_extents_type = Kokkos::extents<int, 2, Kokkos::dynamic_extent>;
    using expected_layout_type = Kokkos::Experimental::layout_left_padded<Kokkos::dynamic_extent>;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&](auto sliceSpec) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(input_extents, sliceSpec, Kokkos::full_extent);
    };
    runTest(std::tuple<IC<int, 1>, IC<int, 3>> {} );
    runTest(std::tuple<IC<std::size_t, 1>, IC<std::size_t, 3>> {} );
    runTest(std::tuple<IC<std::uint32_t, 1>, IC<std::uint32_t, 3>> {} );
    runTest(std::tuple<IC<std::int64_t, 1>, IC<std::int64_t, 3>> {} );
  }
}

TEST(TestMdspan, SubmdspanStaticSlice_Right_idd_TupleFull) {
  (void) test_info_;
  using input_extents_type = Kokkos::dextents<int, 2>;
  using input_layout_type = Kokkos::layout_right;
  using input_mdspan_type = Kokkos::mdspan<float, input_extents_type, input_layout_type>;

  input_extents_type input_extents{3, 4};
  {
    using expected_extents_type = Kokkos::dextents<int, 2>;
    using expected_layout_type = Kokkos::layout_right;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&](auto sliceSpec) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(input_extents, sliceSpec, Kokkos::full_extent);
    };
    runTest(std::tuple<int, int>{int(1), int(3)});
    runTest(std::tuple<std::size_t, std::size_t>{std::size_t(1), std::size_t(3)});
    runTest(std::tuple<std::uint32_t, std::uint32_t>{std::uint32_t(1), std::uint32_t(3)});
    runTest(std::tuple<std::int64_t, std::int64_t>{std::int64_t(1), std::int64_t(3)});
  }
  {
    using expected_extents_type = Kokkos::extents<int, 2, Kokkos::dynamic_extent>;
    using expected_layout_type = Kokkos::layout_right;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&](auto sliceSpec) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(input_extents, sliceSpec, Kokkos::full_extent);
    };
    runTest(std::tuple<IC<int, 1>, IC<int, 3>> {} );
    runTest(std::tuple<IC<std::size_t, 1>, IC<std::size_t, 3>> {} );
    runTest(std::tuple<IC<std::uint32_t, 1>, IC<std::uint32_t, 3>> {} );
    runTest(std::tuple<IC<std::int64_t, 1>, IC<std::int64_t, 3>> {} );
  }
}

TEST(TestMdspan, SubmdspanStaticSlice_Stride_idd_TupleFull) {
  (void) test_info_;
  using input_extents_type = Kokkos::dextents<int, 2>;
  using input_layout_type = Kokkos::layout_stride;
  using input_mdspan_type = Kokkos::mdspan<float, input_extents_type, input_layout_type>;

  input_extents_type input_extents{3, 4};
  std::array<int, 2> input_strides{1, 4}; // like layout_left
  using input_mapping_type = typename input_mdspan_type::mapping_type;
  input_mapping_type input_mapping(input_extents, input_strides);
  {
    using expected_extents_type = Kokkos::dextents<int, 2>;
    using expected_layout_type = Kokkos::layout_stride;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&](auto sliceSpec) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(input_mapping, sliceSpec, Kokkos::full_extent);
    };
    runTest(std::tuple<int, int>{int(1), int(3)});
    runTest(std::tuple<std::size_t, std::size_t>{std::size_t(1), std::size_t(3)});
    runTest(std::tuple<std::uint32_t, std::uint32_t>{std::uint32_t(1), std::uint32_t(3)});
    runTest(std::tuple<std::int64_t, std::int64_t>{std::int64_t(1), std::int64_t(3)});
  }
  {
    using expected_extents_type = Kokkos::extents<int, 2, Kokkos::dynamic_extent>;
    using expected_layout_type = Kokkos::layout_stride;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&](auto sliceSpec) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(input_mapping, sliceSpec, Kokkos::full_extent);
    };
    runTest(std::tuple<IC<int, 1>, IC<int, 3>> {} );
    runTest(std::tuple<IC<std::size_t, 1>, IC<std::size_t, 3>> {} );
    runTest(std::tuple<IC<std::uint32_t, 1>, IC<std::uint32_t, 3>> {} );
    runTest(std::tuple<IC<std::int64_t, 1>, IC<std::int64_t, 3>> {} );
  }
}

// Rank 1 tests

TEST(TestMdspan, SubmdspanStaticSlice_Left_id_Tuple) {
  (void) test_info_;
  using input_extents_type = Kokkos::extents<int, Kokkos::dynamic_extent>;
  using input_layout_type = Kokkos::layout_left;
  using input_mdspan_type = Kokkos::mdspan<float, input_extents_type, input_layout_type>;

  input_extents_type input_extents{5};
  {
    using expected_extents_type = Kokkos::extents<int, Kokkos::dynamic_extent>;
    using expected_layout_type = Kokkos::layout_left;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&](auto sliceSpec) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(input_extents, sliceSpec);
    };
    runTest(std::tuple<int, int>{int(1), int(3)});
    runTest(std::tuple<std::size_t, std::size_t>{std::size_t(1), std::size_t(3)});
    runTest(std::tuple<std::uint32_t, std::uint32_t>{std::uint32_t(1), std::uint32_t(3)});
    runTest(std::tuple<std::int64_t, std::int64_t>{std::int64_t(1), std::int64_t(3)});
  }
  {
    using expected_extents_type = Kokkos::extents<int, 2>;
    using expected_layout_type = Kokkos::layout_left;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&](auto sliceSpec) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(input_extents, sliceSpec);
    };
    runTest(std::tuple<IC<int, 1>, IC<int, 3>> {} );
    runTest(std::tuple<IC<std::size_t, 1>, IC<std::size_t, 3>> {} );
    runTest(std::tuple<IC<std::uint32_t, 1>, IC<std::uint32_t, 3>> {} );
    runTest(std::tuple<IC<std::int64_t, 1>, IC<std::int64_t, 3>> {} );
  }
}

TEST(TestMdspan, SubmdspanStaticSlice_Left_i5_Tuple) {
  (void) test_info_;
  using input_extents_type = Kokkos::extents<int, 5>;
  using input_layout_type = Kokkos::layout_left;
  using input_mdspan_type = Kokkos::mdspan<float, input_extents_type, input_layout_type>;

  input_extents_type input_extents;
  {
    using expected_extents_type = Kokkos::extents<int, Kokkos::dynamic_extent>;
    using expected_layout_type = Kokkos::layout_left;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&](auto sliceSpec) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(input_extents, sliceSpec);
    };
    runTest(std::tuple<int, int>{int(1), int(3)});
    runTest(std::tuple<std::size_t, std::size_t>{std::size_t(1), std::size_t(3)});
    runTest(std::tuple<std::uint32_t, std::uint32_t>{std::uint32_t(1), std::uint32_t(3)});
    runTest(std::tuple<std::int64_t, std::int64_t>{std::int64_t(1), std::int64_t(3)});
  }
  {
    using expected_extents_type = Kokkos::extents<int, 2>;
    using expected_layout_type = Kokkos::layout_left;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&](auto sliceSpec) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(input_extents, sliceSpec);
    };
    runTest(std::tuple<IC<int, 1>, IC<int, 3>> {} );
    runTest(std::tuple<IC<std::size_t, 1>, IC<std::size_t, 3>> {} );
    runTest(std::tuple<IC<std::uint32_t, 1>, IC<std::uint32_t, 3>> {} );
    runTest(std::tuple<IC<std::int64_t, 1>, IC<std::int64_t, 3>> {} );
  }
}

TEST(TestMdspan, SubmdspanStaticSlice_Right_id_Tuple) {
  (void) test_info_;
  using input_extents_type = Kokkos::extents<int, Kokkos::dynamic_extent>;
  using input_layout_type = Kokkos::layout_right;
  using input_mdspan_type = Kokkos::mdspan<float, input_extents_type, input_layout_type>;

  input_extents_type input_extents{5};
  {
    using expected_extents_type = Kokkos::extents<int, Kokkos::dynamic_extent>;
    using expected_layout_type = Kokkos::layout_right;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&](auto sliceSpec) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(input_extents, sliceSpec);
    };
    runTest(std::tuple<int, int>{int(1), int(3)});
    runTest(std::tuple<std::size_t, std::size_t>{std::size_t(1), std::size_t(3)});
    runTest(std::tuple<std::uint32_t, std::uint32_t>{std::uint32_t(1), std::uint32_t(3)});
    runTest(std::tuple<std::int64_t, std::int64_t>{std::int64_t(1), std::int64_t(3)});
  }
  {
    using expected_extents_type = Kokkos::extents<int, 2>;
    using expected_layout_type = Kokkos::layout_right;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&](auto sliceSpec) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(input_extents, sliceSpec);
    };
    runTest(std::tuple<IC<int, 1>, IC<int, 3>> {} );
    runTest(std::tuple<IC<std::size_t, 1>, IC<std::size_t, 3>> {} );
    runTest(std::tuple<IC<std::uint32_t, 1>, IC<std::uint32_t, 3>> {} );
    runTest(std::tuple<IC<std::int64_t, 1>, IC<std::int64_t, 3>> {} );
  }
}

TEST(TestMdspan, SubmdspanStaticSlice_Right_i5_Tuple) {
  using input_extents_type = Kokkos::extents<int, 5>;
  using input_layout_type = Kokkos::layout_right;
  using input_mdspan_type = Kokkos::mdspan<float, input_extents_type, input_layout_type>;

  input_extents_type input_extents;
  {
    using expected_extents_type = Kokkos::extents<int, Kokkos::dynamic_extent>;
    using expected_layout_type = Kokkos::layout_right;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&](auto sliceSpec) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(input_extents, sliceSpec);
    };
    runTest(std::tuple<int, int>{int(1), int(3)});
    runTest(std::tuple<std::size_t, std::size_t>{std::size_t(1), std::size_t(3)});
    runTest(std::tuple<std::uint32_t, std::uint32_t>{std::uint32_t(1), std::uint32_t(3)});
    runTest(std::tuple<std::int64_t, std::int64_t>{std::int64_t(1), std::int64_t(3)});
  }
  {
    using expected_extents_type = Kokkos::extents<int, 2>;
    using expected_layout_type = Kokkos::layout_right;
    using expected_output_mdspan_type = Kokkos::mdspan<float, expected_extents_type, expected_layout_type>;

    auto runTest = [&](auto sliceSpec) {
      test_submdspan_static_slice<expected_output_mdspan_type, input_mdspan_type>(input_extents, sliceSpec);
    };
    runTest(std::tuple<IC<int, 1>, IC<int, 3>> {} );
    runTest(std::tuple<IC<std::size_t, 1>, IC<std::size_t, 3>> {} );
    runTest(std::tuple<IC<std::uint32_t, 1>, IC<std::uint32_t, 3>> {} );
    runTest(std::tuple<IC<std::int64_t, 1>, IC<std::int64_t, 3>> {} );
  }
}

} // namespace (anonymous)
