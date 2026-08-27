// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif
#include <cstddef>

namespace {
struct Foo {
  KOKKOS_FUNCTION Foo& operator=(double val_) {
    val = val_;
    return *this;
  }
  float val;
  KOKKOS_FUNCTION bool operator==(double val_) {
    return val_ == static_cast<double>(val);
  }
};

template <class D1, class D2, class... Extents>
void test_deep_copy_assignable_types(Extents... exts) {
  Kokkos::View<D1, TEST_EXECSPACE> a("A", exts...);
  Kokkos::deep_copy(a, 1.5);

  Kokkos::View<D2, TEST_EXECSPACE> b("B", exts...);
  Kokkos::deep_copy(b, a);

  auto h_b = Kokkos::create_mirror_view(b);
  Kokkos::deep_copy(h_b, b);
  ASSERT_TRUE(h_b((exts - 1)...) == 1.5);

  // Check that
  Kokkos::deep_copy(TEST_EXECSPACE(), a, 2.5);
  Kokkos::deep_copy(TEST_EXECSPACE(), b, a);
  Kokkos::deep_copy(TEST_EXECSPACE(), h_b, b);
  TEST_EXECSPACE().fence();

  ASSERT_TRUE(h_b((exts - 1)...) == 2.5);

  // Read b back through its host mirror and check the value.
  auto check_b = [&](double expected) {
    Kokkos::deep_copy(h_b, b);
    ASSERT_TRUE(h_b((exts - 1)...) == expected);
  };

#ifdef KOKKOS_HAS_SHARED_SPACE
  // Gate on accessibility so every kernel below runs on TEST_EXECSPACE;
  // otherwise deep_copy substitutes the view's own execution space.
  if constexpr (Kokkos::SpaceAccessibility<TEST_EXECSPACE,
                                           Kokkos::SharedSpace>::accessible) {
    Kokkos::View<D1, Kokkos::SharedSpace> s("S", exts...);

    // The explicit instance is required; without it this dispatches on
    // SharedSpace::execution_space.
    Kokkos::deep_copy(TEST_EXECSPACE{}, s, 1.5);

    // No exec instance: destination-preferred dispatch resolves to
    // TEST_EXECSPACE because b is the destination.
    Kokkos::deep_copy(b, s);
    check_b(1.5);

    Kokkos::deep_copy(TEST_EXECSPACE{}, a, 3.5);
    Kokkos::deep_copy(TEST_EXECSPACE{}, s, a);
    Kokkos::deep_copy(TEST_EXECSPACE{}, b, s);
    check_b(3.5);

    Kokkos::View<D1, Kokkos::SharedSpace> s2("S2", exts...);
    Kokkos::deep_copy(TEST_EXECSPACE{}, s2, s);
    Kokkos::deep_copy(TEST_EXECSPACE{}, b, s2);
    check_b(3.5);
  }
#endif

#ifdef KOKKOS_HAS_SHARED_HOST_PINNED_SPACE
  // Gate on accessibility so every kernel below runs on TEST_EXECSPACE;
  // otherwise deep_copy substitutes the view's own execution space.
  if constexpr (Kokkos::SpaceAccessibility<
                    TEST_EXECSPACE,
                    Kokkos::SharedHostPinnedSpace>::accessible) {
    Kokkos::View<D1, Kokkos::SharedHostPinnedSpace> sp("SP", exts...);

    // The explicit instance is required; without it this dispatches on
    // SharedHostPinnedSpace::execution_space.
    Kokkos::deep_copy(TEST_EXECSPACE{}, sp, 2.5);

    // No exec instance: destination-preferred dispatch resolves to
    // TEST_EXECSPACE because b is the destination.
    Kokkos::deep_copy(b, sp);
    check_b(2.5);

    Kokkos::deep_copy(TEST_EXECSPACE{}, a, 4.5);
    Kokkos::deep_copy(TEST_EXECSPACE{}, sp, a);
    Kokkos::deep_copy(TEST_EXECSPACE{}, b, sp);
    check_b(4.5);

    Kokkos::View<D1, Kokkos::SharedHostPinnedSpace> sp2("SP2", exts...);
    Kokkos::deep_copy(TEST_EXECSPACE{}, sp2, sp);
    Kokkos::deep_copy(TEST_EXECSPACE{}, b, sp2);
    check_b(4.5);
  }
#endif
}
}  // namespace
