// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

module;

#include <Kokkos_ScatterView.hpp>

export module kokkos.scatter_view_impl;

export {
  namespace Kokkos::Experimental::Impl {
  using ::Kokkos::Experimental::Impl::DefaultContribution;
  using ::Kokkos::Experimental::Impl::DefaultDuplication;
  }  // namespace Kokkos::Experimental::Impl
}
