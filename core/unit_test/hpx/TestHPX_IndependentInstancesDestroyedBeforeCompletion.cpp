// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif
#include <TestHPX_Category.hpp>

namespace {

// Check correct scope handling of independent HPX instances.
TEST(hpx, independent_instance_destroyed_before_completion) {
  Kokkos::View<int, Kokkos::Experimental::HPX> out("out");
  out() = 0;

  {
    Kokkos::Experimental::HPX hpx(
        Kokkos::Experimental::HPX::instance_mode::independent);
    Kokkos::parallel_for(
        "Test::hpx::independent_instances::destroyed_before_completion",
        Kokkos::Experimental::require(
            Kokkos::RangePolicy<Kokkos::Experimental::HPX>(hpx, 0, 1),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        KOKKOS_LAMBDA(int) { out() = 1; });
  }

  Kokkos::fence();
  ASSERT_EQ(1, out());
}

}  // namespace
