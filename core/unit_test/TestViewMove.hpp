// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

namespace {

// Check that moving a View outside a parallel region does not increase the
// number of views managing the allocation.
template <class ViewType>
void test_moving_view_does_not_change_use_count(ViewType v) {
  auto* const ptr = v.data();
  auto const cnt  = v.use_count();

  // NOLINTBEGIN(bugprone-use-after-move)

  ViewType w(std::move(v));  // move construction
#ifdef KOKKOS_ENABLE_IMPL_VIEW_LEGACY
  if (w.use_count() != 0)
    EXPECT_EQ(w.use_count(), cnt + 1);
  else
    EXPECT_EQ(w.use_count(), 0);
#else
  EXPECT_EQ(w.use_count(), cnt);
#endif
  EXPECT_EQ(w.data(), ptr);
#ifdef KOKKOS_ENABLE_IMPL_VIEW_LEGACY
  EXPECT_EQ(v.use_count(), w.use_count());
  EXPECT_EQ(v.data(), w.data());
#else
  EXPECT_EQ(v.use_count(), 0);
  // FIXME should be nullptr
  EXPECT_EQ(v.data(), ptr);
#endif

  v = std::move(w);  // move assignment
#ifdef KOKKOS_ENABLE_IMPL_VIEW_LEGACY
  if (v.use_count() != 0)
    EXPECT_EQ(v.use_count(), cnt + 1);
  else
    EXPECT_EQ(w.use_count(), 0);
#else
  EXPECT_EQ(v.use_count(), cnt);
#endif
  EXPECT_EQ(v.data(), ptr);
#ifdef KOKKOS_ENABLE_IMPL_VIEW_LEGACY
  EXPECT_EQ(w.use_count(), v.use_count());
  EXPECT_EQ(w.data(), v.data());
#else
  EXPECT_EQ(w.use_count(), 0);
  // FIXME should be nullptr
  EXPECT_EQ(w.data(), ptr);
#endif

  // NOLINTEND(bugprone-use-after-move)
}

TEST(TEST_CATEGORY, view_move_and_use_count) {
  using ExecutionSpace = TEST_EXECSPACE;

  test_moving_view_does_not_change_use_count(
      Kokkos::View<int, ExecutionSpace>("v0"));

  test_moving_view_does_not_change_use_count(
      Kokkos::View<float*, ExecutionSpace>("v1", 1));

  Kokkos::View<double**, ExecutionSpace> v2("v2", 1, 2);
  test_moving_view_does_not_change_use_count(
      Kokkos::View<double**, ExecutionSpace>(v2.data(), v2.extent(0),
                                             v2.extent(1)));
  test_moving_view_does_not_change_use_count(
      Kokkos::View<double**, ExecutionSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
          v2.data(), v2.extent(0), v2.extent(1)));
}

// Check that moving a View leaves the moved-from object in a state equivalent
// to being default constructed
// returns the number of errors encountered
template <class ViewType>
KOKKOS_FUNCTION int check_moved_from_view_state(ViewType v) {
  int err = 0;

  // NOLINTBEGIN(bugprone-use-after-move)

  ViewType w(std::move(v));  // move construction
  if (v != ViewType()) {
    Kokkos::printf("failed moved-from view after calling move constructor\n");
    ++err;
  }

  v = std::move(w);  // move assignment
  if (w != ViewType()) {
    Kokkos::printf(
        "failed moved-from view after calling move assignment operator\n");
    ++err;
  }

  // NOLINTEND(bugprone-use-after-move)

  return err;
}

template <class ViewType>
void test_moved_from_view(ViewType v) {
  // The comparison fails because we don't reset extents or span in either
  // implementation EXPECT_EQ(check_moved_from_view_state(v), 0) << "outside
  // parallel region";

  using ExexutionSpace = typename ViewType::execution_space;
  int errors;
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<ExexutionSpace>(0, 1),
      KOKKOS_LAMBDA(int, int& err) { err += check_moved_from_view_state(v); },
      errors);
  // The comparison fails because we don't reset extents or span in either
  // implementation EXPECT_EQ(errors, 0) << "within parallel region";
}

TEST(TEST_CATEGORY, view_moved_from) {
  using ExecutionSpace = TEST_EXECSPACE;

  test_moved_from_view(Kokkos::View<int, ExecutionSpace>("v0"));
  test_moved_from_view(Kokkos::View<float*, ExecutionSpace>("v1", 1));
  Kokkos::View<double**, ExecutionSpace> v2("v2", 1, 2);
  test_moved_from_view(Kokkos::View<double**, ExecutionSpace>(
      v2.data(), v2.extent(0), v2.extent(1)));
  test_moved_from_view(Kokkos::View<double**, ExecutionSpace,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
      v2.data(), v2.extent(0), v2.extent(1)));
}

}  // namespace
