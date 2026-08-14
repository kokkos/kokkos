// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

namespace {

// In a research prototype where we derived from TeamPolicy we saw
// an issue with copy construction that this test reproduces.
struct DerivedPolicy : Kokkos::TeamPolicy<> {
  using base = Kokkos::TeamPolicy<>;

  using base::base;

  explicit DerivedPolicy(const base& policy) : base(policy) {}
};

[[maybe_unused]] DerivedPolicy make_derived_policy(
    const Kokkos::TeamPolicy<>& policy) {
  return DerivedPolicy(policy);
}

}  // namespace
