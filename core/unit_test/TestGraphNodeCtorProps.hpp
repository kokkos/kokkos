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
  const auto device_handle =
      Kokkos::Experimental::get_device_handle(TEST_EXECSPACE{});

  auto empty              = Kokkos::Experimental::node_props();
  auto prop_label         = Kokkos::Experimental::node_props("label");
  auto prop_device_handle = Kokkos::Experimental::node_props(device_handle);
  auto props = Kokkos::Experimental::node_props("label", device_handle);

  using device_handle_t = Kokkos::Impl::DeviceHandle<TEST_EXECSPACE>;

  {
    const auto [d_h, label] =
        Kokkos::Impl::get_properties_or<device_handle_t, std::string>(
            std::move(empty), device_handle, "[unlabeled]");
    ASSERT_EQ(d_h, device_handle);
    ASSERT_EQ(label, "[unlabeled]");
  }
  {
    const auto [d_h, label] =
        Kokkos::Impl::get_properties_or<device_handle_t, std::string>(
            std::move(prop_label), device_handle, "[unlabeled]");
    ASSERT_EQ(d_h, device_handle);
    ASSERT_EQ(label, "label");
  }
  {
    const auto [d_h, label] =
        Kokkos::Impl::get_properties_or<device_handle_t, std::string>(
            std::move(prop_device_handle), device_handle, "[unlabeled]");
    ASSERT_EQ(d_h, device_handle);
    ASSERT_EQ(label, "[unlabeled]");
  }
  {
    const auto [label, d_h] =
        Kokkos::Impl::get_properties_or<std::string, device_handle_t>(
            std::move(props), "[unlabeled]", device_handle);
    ASSERT_EQ(d_h, device_handle);
    ASSERT_EQ(label, "label");
  }
}

}  // end namespace
