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

#include <TestCuda_Category.hpp>
#include <TestGraph.hpp>

namespace Test {

/**
 * @test This test ensures that the underlying @c CUDA objects can be retrieved
 *       from a @c Kokkos::Graph.
 *
 * It also ensures that these objects can be modified "externally", thereby
 * testing "interoperability".
 */
TEST(TEST_CATEGORY, cuda_graph_instantiate_and_debug_dot_print) {
  //! Build a rank-1 zero-initialized view.
  using view_t   = Kokkos::View<int, TEST_EXECSPACE>;
  using view_h_t = Kokkos::View<int, Kokkos::HostSpace>;
  view_t data("witness");

  //! Build a graph, whose sole node will apply @ref AddOne on @ref data.
  auto graph =
      Kokkos::Experimental::create_graph<TEST_EXECSPACE>([&](auto root) {
        root.then_parallel_for(
            1, SetViewToValueFunctor<TEST_EXECSPACE, int>{data, 1});
      });

  //! Retrieve the @c Kokkos::Impl::Graph.
  auto root_node_ref = Kokkos::Impl::GraphAccess::create_root_ref(graph);
  auto graph_ptr_impl =
      Kokkos::Impl::GraphAccess::get_graph_weak_ptr(root_node_ref).lock();

  /// At this stage, the @c Cuda "executable" graph is null
  /// because it has not been instantiated.
  ASSERT_EQ(graph_ptr_impl->get_cuda_graph_exec(), nullptr);

  //! Instantiate the graph manually.
  graph_ptr_impl->instantiate_graph(cudaGraphInstantiateFlagAutoFreeOnLaunch);

  /// At this stage, the @c Cuda "executable" graph should not be null,
  /// because it has been instantiated.
  ASSERT_NE(graph_ptr_impl->get_cuda_graph_exec(), nullptr);
  const cudaGraphExec_t manual = graph_ptr_impl->get_cuda_graph_exec();

  /// Submit the graph through @c Kokkos::Graph API.
  /// Note that @c Kokkos::Graph::submit should not modify the
  /// @c cudaGraphExec_t object because we instantiated the graph manually.
  graph.submit();

  ASSERT_EQ(graph_ptr_impl->get_cuda_graph_exec(), manual);

  //! The view has been modified and contains a one.
  view_h_t data_h("witness on host");
  Kokkos::deep_copy(graph.get_execution_space(), data_h, data);
  ASSERT_EQ(data_h(), 1);

  //! Export the graph structure as a DOT graph.
  cudaGraphDebugDotPrint(graph_ptr_impl->get_cuda_graph(), "cuda_graph.dot",
                         cudaGraphDebugDotFlagsVerbose);
}

}  // namespace Test
