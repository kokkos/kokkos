// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>
#include <impl/Kokkos_GraphNodeCtorProps.hpp>

#include <gtest/gtest.h>

namespace {

constexpr bool test_node_ctor_prop_traits() {
  static_assert(std::is_constructible_v<Kokkos::Impl::NodeCtorProp<std::string>,
                                        const char(&)[6]>);
  return true;
}
static_assert(test_node_ctor_prop_traits());

TEST(TEST_CATEGORY, node_props_empty) {
  using props_t = decltype(Kokkos::Experimental::node_props());

  static_assert(std::same_as<props_t, Kokkos::Impl::NodeCtorProps<>>);

  static_assert(Kokkos::Impl::is_node_props_v<props_t>);

  static_assert(props_t::has<std::string> == false);
  static_assert(props_t::has<TEST_EXECSPACE> == false);
}

TEST(TEST_CATEGORY, node_props_label) {
  using props_t = decltype(Kokkos::Experimental::node_props("label"));

  static_assert(
      std::same_as<props_t, Kokkos::Impl::NodeCtorProps<std::string>>);

  static_assert(Kokkos::Impl::is_node_props_v<props_t>);

  static_assert(props_t::has<std::string> == true);
  static_assert(props_t::has<TEST_EXECSPACE> == false);

  const auto props = Kokkos::Experimental::node_props("label");

  ASSERT_EQ(Kokkos::Impl::get_property<std::string>(props), "label");
}

TEST(TEST_CATEGORY, node_props_device_handle) {
  using props_t = decltype(Kokkos::Experimental::node_props(
      Kokkos::Experimental::get_device_handle(std::declval<TEST_EXECSPACE>())));

  static_assert(
      std::same_as<props_t, Kokkos::Impl::NodeCtorProps<
                                Kokkos::Impl::DeviceHandle<TEST_EXECSPACE>>>);

  static_assert(Kokkos::Impl::is_node_props_v<props_t>);

  static_assert(props_t::has<std::string> == false);
  static_assert(props_t::has<Kokkos::Impl::DeviceHandle<TEST_EXECSPACE>> ==
                true);

  const auto props = Kokkos::Experimental::node_props(
      Kokkos::Experimental::get_device_handle(TEST_EXECSPACE{}));

  ASSERT_EQ(
      Kokkos::Impl::get_property<Kokkos::Impl::DeviceHandle<TEST_EXECSPACE>>(
          props),
      Kokkos::Impl::DeviceHandle<TEST_EXECSPACE>{});
}

TEST(TEST_CATEGORY, node_props_device_handle_label) {
  using props_t = decltype(Kokkos::Experimental::node_props(
      Kokkos::Experimental::get_device_handle(std::declval<TEST_EXECSPACE>()),
      "label"));

  static_assert(
      std::same_as<props_t, Kokkos::Impl::NodeCtorProps<
                                Kokkos::Impl::DeviceHandle<TEST_EXECSPACE>,
                                std::string>>);

  static_assert(Kokkos::Impl::is_node_props_v<props_t>);

  static_assert(props_t::has<std::string> == true);
  static_assert(props_t::has<Kokkos::Impl::DeviceHandle<TEST_EXECSPACE>> ==
                true);

  const auto props = Kokkos::Experimental::node_props(
      Kokkos::Experimental::get_device_handle(TEST_EXECSPACE{}), "label");

  ASSERT_EQ(
      Kokkos::Impl::get_property<Kokkos::Impl::DeviceHandle<TEST_EXECSPACE>>(
          props),
      Kokkos::Impl::DeviceHandle<TEST_EXECSPACE>{});
  ASSERT_EQ(Kokkos::Impl::get_property<std::string>(props), "label");
}

TEST(TEST_CATEGORY, node_props_label_device_handle) {
  using props_t = decltype(Kokkos::Experimental::node_props(
      "label",
      Kokkos::Experimental::get_device_handle(std::declval<TEST_EXECSPACE>())));

  static_assert(
      std::same_as<props_t, Kokkos::Impl::NodeCtorProps<
                                std::string,
                                Kokkos::Impl::DeviceHandle<TEST_EXECSPACE>>>);

  static_assert(Kokkos::Impl::is_node_props_v<props_t>);

  static_assert(props_t::has<std::string> == true);
  static_assert(props_t::has<Kokkos::Impl::DeviceHandle<TEST_EXECSPACE>> ==
                true);

  const auto props = Kokkos::Experimental::node_props(
      "label", Kokkos::Experimental::get_device_handle(TEST_EXECSPACE{}));

  ASSERT_EQ(Kokkos::Impl::get_property<std::string>(props), "label");
  ASSERT_EQ(
      Kokkos::Impl::get_property<Kokkos::Impl::DeviceHandle<TEST_EXECSPACE>>(
          props),
      Kokkos::Impl::DeviceHandle<TEST_EXECSPACE>{});
}

TEST(TEST_CATEGORY, node_props_get_properties_or) {
  const auto [exec_A, exec_B] =
      Kokkos::Experimental::partition_space(TEST_EXECSPACE{}, 1, 1);

  using device_handle_t = Kokkos::Impl::DeviceHandle<TEST_EXECSPACE>;

  const device_handle_t device_handle_A =
      Kokkos::Experimental::get_device_handle(exec_A);
  const device_handle_t device_handle_B =
      Kokkos::Experimental::get_device_handle(exec_B);

  {
    auto empty = Kokkos::Experimental::node_props();

    const auto [d_h_const_lvalue, label_const_lvalue] =
        Kokkos::Impl::get_properties_or<device_handle_t, std::string>(
            std::as_const(empty), device_handle_B, "[unlabeled]");
    const auto [d_h_lvalue, label_lvalue] =
        Kokkos::Impl::get_properties_or<device_handle_t, std::string>(
            empty, device_handle_B, "[unlabeled]");
    const auto [d_h_rvalue, label_rvalue] =
        Kokkos::Impl::get_properties_or<device_handle_t, std::string>(
            std::move(empty), device_handle_B, "[unlabeled]");

    ASSERT_EQ(d_h_const_lvalue, device_handle_B);
    ASSERT_EQ(d_h_lvalue, device_handle_B);
    ASSERT_EQ(d_h_rvalue, device_handle_B);
    ASSERT_EQ(label_const_lvalue, "[unlabeled]");
    ASSERT_EQ(label_lvalue, "[unlabeled]");
    ASSERT_EQ(label_rvalue, "[unlabeled]");
  }
  {
    auto prop_label = Kokkos::Experimental::node_props("label");

    const auto [d_h_const_lvalue, label_const_lvalue] =
        Kokkos::Impl::get_properties_or<device_handle_t, std::string>(
            std::as_const(prop_label), device_handle_B, "[unlabeled]");
    const auto [d_h_lvalue, label_lvalue] =
        Kokkos::Impl::get_properties_or<device_handle_t, std::string>(
            prop_label, device_handle_B, "[unlabeled]");
    const auto [d_h_rvalue, label_rvalue] =
        Kokkos::Impl::get_properties_or<device_handle_t, std::string>(
            std::move(prop_label), device_handle_B, "[unlabeled]");

    ASSERT_EQ(d_h_const_lvalue, device_handle_B);
    ASSERT_EQ(d_h_lvalue, device_handle_B);
    ASSERT_EQ(d_h_rvalue, device_handle_B);
    ASSERT_EQ(label_const_lvalue, "label");
    ASSERT_EQ(label_lvalue, "label");
    ASSERT_EQ(label_rvalue, "label");
  }
  {
    auto prop_device_handle = Kokkos::Experimental::node_props(device_handle_A);

    const auto [d_h_const_lvalue, label_const_lvalue] =
        Kokkos::Impl::get_properties_or<device_handle_t, std::string>(
            std::as_const(prop_device_handle), device_handle_B, "[unlabeled]");
    const auto [d_h_lvalue, label_lvalue] =
        Kokkos::Impl::get_properties_or<device_handle_t, std::string>(
            prop_device_handle, device_handle_B, "[unlabeled]");
    const auto [d_h_rvalue, label_rvalue] =
        Kokkos::Impl::get_properties_or<device_handle_t, std::string>(
            std::move(prop_device_handle), device_handle_B, "[unlabeled]");

    ASSERT_EQ(d_h_const_lvalue, device_handle_A);
    ASSERT_EQ(d_h_lvalue, device_handle_A);
    ASSERT_EQ(d_h_rvalue, device_handle_A);
    ASSERT_EQ(label_const_lvalue, "[unlabeled]");
    ASSERT_EQ(label_lvalue, "[unlabeled]");
    ASSERT_EQ(label_rvalue, "[unlabeled]");
  }
  {
    auto props = Kokkos::Experimental::node_props("label", device_handle_A);

    const auto [d_h_const_lvalue, label_const_lvalue] =
        Kokkos::Impl::get_properties_or<device_handle_t, std::string>(
            std::as_const(props), device_handle_B, "[unlabeled]");
    const auto [d_h_lvalue, label_lvalue] =
        Kokkos::Impl::get_properties_or<device_handle_t, std::string>(
            props, device_handle_B, "[unlabeled]");
    const auto [d_h_rvalue, label_rvalue] =
        Kokkos::Impl::get_properties_or<device_handle_t, std::string>(
            std::move(props), device_handle_B, "[unlabeled]");

    ASSERT_EQ(d_h_const_lvalue, device_handle_A);
    ASSERT_EQ(d_h_lvalue, device_handle_A);
    ASSERT_EQ(d_h_rvalue, device_handle_A);
    ASSERT_EQ(label_const_lvalue, "label");
    ASSERT_EQ(label_lvalue, "label");
    ASSERT_EQ(label_rvalue, "label");
  }
}

}  // end namespace
