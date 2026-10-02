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

// The allocation record's ViewValueFunctor stores the exec used in view_alloc.
// That is a separate HostSharedPtr to instance_data from the one in the handle
// the work is dispatched on. Dispatch itself must not add another owning HPX.
TEST(hpx, independent_instances_view_alloc_same_exec) {
  Kokkos::Experimental::HPX hpx(
      Kokkos::Experimental::HPX::instance_mode::independent);
  ASSERT_EQ(1, hpx.impl_instance_data_use_count());

  const int n = 10;
  Kokkos::View<int*, Kokkos::Experimental::HPX> v(Kokkos::view_alloc("v", hpx),
                                                  n);
  ASSERT_EQ(2, hpx.impl_instance_data_use_count());

  Kokkos::parallel_for(
      "Test::hpx::independent_instances::view_alloc_same_exec",
      Kokkos::RangePolicy<Kokkos::Experimental::HPX>(hpx, 0, n),
      KOKKOS_LAMBDA(const int i) { v(i) = i; });

  ASSERT_EQ(2, hpx.impl_instance_data_use_count());
  hpx.fence();
  ASSERT_EQ(2, hpx.impl_instance_data_use_count());

  for (int i = 0; i < n; ++i) {
    ASSERT_EQ(i, v(i));
  }
}

}  // namespace
