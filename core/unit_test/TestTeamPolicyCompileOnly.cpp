// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

// For research/prototyping it can be convenient to derive from TeamPolicy

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

namespace {

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
