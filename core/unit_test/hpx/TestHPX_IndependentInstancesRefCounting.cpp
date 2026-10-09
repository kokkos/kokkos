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
std::atomic<int> dummy_count;

struct dummy {
  dummy() { ++dummy_count; }
  dummy(dummy &&) { ++dummy_count; }
  dummy(dummy const &) { ++dummy_count; }
  dummy &operator=(dummy &&) { return *this; }
  dummy &operator=(dummy const &) { return *this; }
  ~dummy() { --dummy_count; }
  void f() const {}
};

TEST(hpx, independent_instances_reference_counting) {
  ASSERT_EQ(0, dummy_count);

  {
    dummy d;
    ASSERT_EQ(1, dummy_count);
    Kokkos::Experimental::HPX hpx(
        Kokkos::Experimental::HPX::instance_mode::independent);
    Kokkos::parallel_for(
        "Test::hpx::reference_counting::dummy",
        Kokkos::RangePolicy<Kokkos::Experimental::HPX>(hpx, 0, 1),
        KOKKOS_LAMBDA(int) {
          // Make sure dummy struct is captured.
          d.f();
        });

    hpx.fence();
    ASSERT_EQ(1, dummy_count);
  }

  ASSERT_EQ(0, dummy_count);
}

TEST(hpx, independent_instances_dispatch_use_count) {
  Kokkos::Experimental::HPX hpx(
      Kokkos::Experimental::HPX::instance_mode::independent);
  ASSERT_EQ(1, hpx.impl_instance_data_use_count());

  Kokkos::View<int, Kokkos::Experimental::HPX> out("out");
  out() = 0;

  Kokkos::parallel_for(
      "Test::hpx::reference_counting::dispatch_use_count",
      Kokkos::RangePolicy<Kokkos::Experimental::HPX>(hpx, 0, 1),
      KOKKOS_LAMBDA(int) { out() = 1; });

  ASSERT_EQ(1, hpx.impl_instance_data_use_count());
  hpx.fence();
  ASSERT_EQ(1, out());
  ASSERT_EQ(1, hpx.impl_instance_data_use_count());
}

}  // namespace
