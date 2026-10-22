// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

module;

#include <Kokkos_Graph.hpp>

export module kokkos.graph_impl;

export {
  namespace Kokkos {
  namespace Impl {
  using ::Kokkos::Impl::is_graph_capture_v;
  using ::Kokkos::Impl::is_graph_kernel_v;
  using ::Kokkos::Impl::is_graph_then_host_v;

  using ::Kokkos::Impl::get_properties_or;
  using ::Kokkos::Impl::get_property;

  using ::Kokkos::Impl::DeviceHandle;
  using ::Kokkos::Impl::GraphImpl;
  using ::Kokkos::Impl::GraphNodeCaptureImpl;
  using ::Kokkos::Impl::GraphNodeKernelImpl;
  using ::Kokkos::Impl::GraphNodeThenHostImpl;
  using ::Kokkos::Impl::GraphNodeThenImpl;
  using ::Kokkos::Impl::IsGraphKernelTag;
  using ::Kokkos::Impl::NodeCtorProps;
  }  // namespace Impl
  }  // namespace Kokkos
}
