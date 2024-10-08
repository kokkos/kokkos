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

#include <filesystem>
#include <fstream>
#include <regex>

#include <TestCuda_Category.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Graph.hpp>

#include <gtest/gtest.h>

namespace {

template <typename ViewType>
struct Increment {
  ViewType data;

  KOKKOS_FUNCTION
  void operator()(const int) const { ++data(); }
};

class TEST_CATEGORY_FIXTURE(GraphInterOp) : public ::testing::Test {
 public:
  using execution_space = Kokkos::Cuda;
  using view_t =
      Kokkos::View<int, execution_space, Kokkos::MemoryTraits<Kokkos::Atomic>>;
  using graph_t = Kokkos::Experimental::Graph<execution_space>;

  void SetUp() override {
    data = view_t(Kokkos::view_alloc(exec, "witness"));

    graph = Kokkos::Experimental::create_graph(exec, [&](const auto& root) {
      root.then_parallel_for(1, Increment<view_t>{data});
    });
  }

 protected:
  execution_space exec{};
  view_t data;
  std::optional<graph_t> graph;
};

// This test checks the promises of Kokkos::Graph against its
// underlying Cuda native objects.
TEST_F(TEST_CATEGORY_FIXTURE(GraphInterOp), promises_on_native_objects) {
  // Before instantiation, the Cuda graph is valid, but the Cuda executable
  // graph is still null.
  cudaGraph_t cuda_graph = graph->native_graph();

  ASSERT_NE(cuda_graph, nullptr);
  ASSERT_EQ(graph->native_graph_exec(), nullptr);

  // After instantiation, both native objects are valid.
  graph->instantiate();

  cudaGraphExec_t cuda_graph_exec = graph->native_graph_exec();

  ASSERT_EQ(graph->native_graph(), cuda_graph);
  ASSERT_NE(cuda_graph_exec, nullptr);

  // Submission should not affect the underlying objects.
  graph->submit();

  ASSERT_EQ(graph->native_graph(), cuda_graph);
  ASSERT_EQ(graph->native_graph_exec(), cuda_graph_exec);
}

// Count the number of nodes. This is useful to ensure no spurious
// (possibly empty) node is added.
TEST_F(TEST_CATEGORY_FIXTURE(GraphInterOp), count_nodes) {
  graph->instantiate();

  size_t num_nodes;

  KOKKOS_IMPL_CUDA_SAFE_CALL(
      cudaGraphGetNodes(graph->native_graph(), nullptr, &num_nodes));

  ASSERT_EQ(num_nodes, 2u);
}

// Use native Cuda graph to generate a DOT representation.
TEST_F(TEST_CATEGORY_FIXTURE(GraphInterOp), debug_dot_print) {
#if CUDA_VERSION < 11600
  GTEST_SKIP() << "Export a graph to DOT requires Cuda 11.6.";
#elif KOKKOS_COMPILER_GNU < 910
  GTEST_SKIP() << "'filesystem' is not fully supported prior to GCC 9.1.0.";
#elif KOKKOS_COMPILER_CLANG < 1100
  GTEST_SKIP() << "'filesystem' is not fully supported prior to LLVM 11.";
#else
  graph->instantiate();

  const auto dot = std::filesystem::temp_directory_path() / "cuda_graph.dot";

  // Convert path to string then to const char * to make it work on Windows.
  KOKKOS_IMPL_CUDA_SAFE_CALL(
      cudaGraphDebugDotPrint(graph->native_graph(), dot.string().c_str(),
                             cudaGraphDebugDotFlagsVerbose));

  ASSERT_TRUE(std::filesystem::exists(dot));
  ASSERT_GT(std::filesystem::file_size(dot), 0u);

  // We could write a check against the full kernel's function signature, but
  // it would make the test rely too much on internal implementation details.
  // Therefore, we just look for the functor and policy. Note that the
  // signature is mangled in the 'dot' output.
  const std::string expected("[A-Za-z0-9_]+Increment[A-Za-z0-9_]+RangePolicy");

  std::stringstream buffer;
  buffer << std::ifstream(dot).rdbuf();

  ASSERT_TRUE(std::regex_search(buffer.str(), std::regex(expected)))
      << "Could not find expected signature regex " << std::quoted(expected)
      << " in " << dot;
#endif
}

// Ensure that the graph has been instantiated with the default flag.
TEST_F(TEST_CATEGORY_FIXTURE(GraphInterOp), instantiation_flags) {
#if CUDA_VERSION < 12000
  GTEST_SKIP() << "Graph instantiation flag inspection requires Cuda 12.";
#else
  graph->instantiate();
  unsigned long long flags =
      Kokkos::Experimental::finite_max_v<unsigned long long>;
  KOKKOS_IMPL_CUDA_SAFE_CALL(
      cudaGraphExecGetFlags(graph->native_graph_exec(), &flags));

  ASSERT_EQ(flags, 0u);
#endif
}

}  // namespace
