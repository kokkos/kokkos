// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

module;

#include <Kokkos_Graph.hpp>

export module kokkos.graph;

export {
  namespace Kokkos {
  namespace Experimental {
  using ::Kokkos::Experimental::create_graph;
  using ::Kokkos::Experimental::get_device_handle;
  using ::Kokkos::Experimental::node_props;
  using ::Kokkos::Experimental::when_all;

  using ::Kokkos::Experimental::Graph;
  using ::Kokkos::Experimental::GraphNodeKind;
  using ::Kokkos::Experimental::GraphNodeRef;
  using ::Kokkos::Experimental::GraphNodeRootTag;
  using ::Kokkos::Experimental::ThenPolicy;
  using ::Kokkos::Experimental::TypeErasedTag;
  }  // namespace Experimental
  }  // namespace Kokkos
}
