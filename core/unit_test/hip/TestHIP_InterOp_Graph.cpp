// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <filesystem>
#include <fstream>
#include <regex>

#include <TestHIP_Category.hpp>
#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif
#include <Kokkos_Graph.hpp>

#include <gtest/gtest.h>

namespace {

template <typename ViewType>
struct Increment {
  ViewType data;

  KOKKOS_FUNCTION
  void operator()(const int) const { ++data(); }
};

// This test checks the promises of Kokkos::Graph against its
// underlying HIP graph objects.
TEST(TEST_CATEGORY, graph_promises_on_hip_objects) {
  Kokkos::Experimental::Graph<Kokkos::HIP> graph{};
  // Before instantiation, the HIP graph is valid, but the HIP executable
  // graph is still null.
  hipGraph_t hip_graph = graph.hip_graph();

  ASSERT_NE(hip_graph, nullptr);
  ASSERT_EQ(graph.hip_graph_exec(), nullptr);

  // After instantiation, both HIP objects are valid.
  graph.instantiate();

  hipGraphExec_t hip_graph_exec = graph.hip_graph_exec();

  ASSERT_EQ(graph.hip_graph(), hip_graph);
  ASSERT_NE(hip_graph_exec, nullptr);

  // Submission should not affect the underlying objects.
  graph.submit();

  ASSERT_EQ(graph.hip_graph(), hip_graph);
  ASSERT_EQ(graph.hip_graph_exec(), hip_graph_exec);
}

// Use HIP graph to generate a DOT representation.
TEST(TEST_CATEGORY, graph_instantiate_and_debug_dot_print) {
  using view_t = Kokkos::View<int, Kokkos::HIP>;

  const Kokkos::HIP exec{};

  view_t data(Kokkos::view_alloc(exec, "witness"));

  Kokkos::Experimental::Graph graph{
      Kokkos::Experimental::get_device_handle(exec)};

  graph.root_node().then_parallel_for(1, Increment<view_t>{data});

  graph.instantiate();

  size_t num_nodes;

  KOKKOS_IMPL_HIP_SAFE_CALL(
      hipGraphGetNodes(graph.hip_graph(), nullptr, &num_nodes));

  ASSERT_EQ(num_nodes, 2u);

  const auto dot = std::filesystem::temp_directory_path() / "hip_graph.dot";

  KOKKOS_IMPL_HIP_SAFE_CALL(hipGraphDebugDotPrint(
      graph.hip_graph(), dot.c_str(), hipGraphDebugDotFlagsVerbose));

  ASSERT_TRUE(std::filesystem::exists(dot));
  ASSERT_GT(std::filesystem::file_size(dot), 0u);

  // We could write a check against the full kernel's function signature, but
  // it would make the test rely too much on internal implementation details.
  // Therefore, we just look for the functor and policy.
  //
  // The DOT output format may vary across HIP/ROCm versions: some expose a
  // mangled kernel name while others print a demangled C++ signature.
  const std::string expected_mangled(
      "[A-Za-z0-9_]+Increment[A-Za-z0-9_]+RangePolicy");
  const std::string expected_demangled("Increment<.*RangePolicy<");

  std::stringstream buffer;
  buffer << std::ifstream(dot).rdbuf();

  const std::string dot_contents = buffer.str();
  ASSERT_TRUE(std::regex_search(dot_contents, std::regex(expected_mangled)) ||
              std::regex_search(dot_contents, std::regex(expected_demangled)))
      << "Could not find expected signature regex "
      << std::quoted(expected_mangled) << " or "
      << std::quoted(expected_demangled) << " in " << dot;
}

// Build a Kokkos::Graph from an existing hipGraph_t.
TEST(TEST_CATEGORY, graph_construct_from_hip_graph) {
  using view_t = Kokkos::View<int, Kokkos::HIPManagedSpace>;

  hipGraph_t hip_graph = nullptr;
  KOKKOS_IMPL_HIP_SAFE_CALL(hipGraphCreate(&hip_graph, 0));

  const Kokkos::HIP exec{};

  Kokkos::Experimental::Graph graph_from_hip_graph(
      Kokkos::Experimental::get_device_handle(exec), hip_graph);

  ASSERT_EQ(hip_graph, graph_from_hip_graph.hip_graph());

  const view_t data(Kokkos::view_alloc(exec, "witness"));

  graph_from_hip_graph.root_node().then_parallel_for(1,
                                                     Increment<view_t>{data});

  graph_from_hip_graph.submit(exec);

  exec.fence();

  ASSERT_EQ(data(), 1);
}

// Retrieve the underlying HIP node.
TEST(TEST_CATEGORY, interact_with_hip_node) {
  using view_t = Kokkos::View<int, Kokkos::HIPManagedSpace>;

  const Kokkos::HIP exec{};

  view_t data(Kokkos::view_alloc(exec, "witness"));

  Kokkos::Experimental::Graph graph{
      Kokkos::Experimental::get_device_handle(exec)};

  auto node = graph.root_node().then_parallel_for(1, Increment<view_t>{data});

  static_assert(std::same_as<decltype(node.hip_node()), hipGraphNode_t>);

  hipGraphNode_t hip_node = node.hip_node();

  hipGraphNodeType node_type;
  KOKKOS_IMPL_HIP_SAFE_CALL(hipGraphNodeGetType(hip_node, &node_type));

  ASSERT_EQ(node_type, hipGraphNodeTypeKernel);

  ASSERT_EQ(data(), 0);
  graph.submit(exec);
  exec.fence();
  ASSERT_EQ(data(), 1);

  KOKKOS_IMPL_HIP_SAFE_CALL(
      hipGraphNodeSetEnabled(graph.hip_graph_exec(), hip_node, false));

  graph.submit(exec);
  exec.fence();
  ASSERT_EQ(data(), 1);

  KOKKOS_IMPL_HIP_SAFE_CALL(
      hipGraphNodeSetEnabled(graph.hip_graph_exec(), hip_node, true));

  graph.submit(exec);
  exec.fence();
  ASSERT_EQ(data(), 2);
}

template <typename ViewType>
struct CheckValue {
  typename ViewType::const_type data;
  typename ViewType::non_const_value_type value;

  KOKKOS_FUNCTION
  void operator()() const {
    if (data() != value) Kokkos::abort("Wrong value.");
  }
};

template <typename ViewType>
struct CustomHIPNode {
  typename ViewType::const_type data;

  auto operator()() const noexcept {
    hipGraphNodeParams params = {};
    params.type               = hipGraphNodeTypeMemset;
    params.memset.dst =
        const_cast<void*>(static_cast<const void*>(this->data.data()));
    params.memset.pitch       = 0;
    params.memset.value       = 42;
    params.memset.elementSize = 4;
    params.memset.width       = 1;
    params.memset.height      = 1;
    return params;
  }
};

// Add a HIP memset node using the HIP graph node interoperability
// feature.
TEST(TEST_CATEGORY, graph_then_hip_node) {
  const Kokkos::HIP exec{};

  using view_t = Kokkos::View<int, Kokkos::HIPManagedSpace>;

  const view_t data(Kokkos::view_alloc(exec, "witness"));

  Kokkos::Experimental::Graph graph_with_hip_node{
      Kokkos::Experimental::get_device_handle(exec)};

  const auto node_check_zero = graph_with_hip_node.root_node().then(
      Kokkos::Experimental::node_props("check it is zero"),
      CheckValue<view_t>{.data = data, .value = 0});

  ASSERT_EQ(data.use_count(), 2);
  const auto node_memset = node_check_zero.then_hip_node(
      Kokkos::Experimental::node_props("nice interop"),
      CustomHIPNode<view_t>{.data = data});
  ASSERT_EQ(data.use_count(), 3);

  ASSERT_EQ(node_memset.get_node_kind(),
            Kokkos::Experimental::GraphNodeKind::Native);

  const auto node_incr = node_memset.then_parallel_for(
      Kokkos::RangePolicy(exec, 0, 1), Increment{.data = data});

  graph_with_hip_node.submit(exec);

  exec.fence();

  ASSERT_EQ(data(), 43);
}

}  // namespace
