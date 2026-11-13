// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <tools/include/ToolTestingUtilities.hpp>

namespace {

template <class ViewType, typename... Args>
void test_moving_view_use_count_and_label(const Args&... args) {
  // NOLINTBEGIN(bugprone-use-after-move)

  // Move construction.
  {
    ViewType src(args...);

    auto const ptr = src.data();
    auto const cnt = src.use_count();
    auto const lbl = src.label();

    ViewType dst(std::move(src));

    EXPECT_EQ(dst.use_count(), cnt);
    EXPECT_EQ(dst.data(), ptr);
    EXPECT_EQ(dst.label(), lbl);

    EXPECT_EQ(src.use_count(), 0);
    EXPECT_EQ(src.label(), std::string());
    EXPECT_EQ(src.data(), ptr);
  }

  // Move assignment into an empty view.
  {
    ViewType src(args...);
    ViewType dst;

    auto const ptr = src.data();
    auto const cnt = src.use_count();
    auto const lbl = src.label();

    dst = std::move(src);

    EXPECT_EQ(dst.use_count(), cnt);
    EXPECT_EQ(dst.data(), ptr);
    EXPECT_EQ(dst.label(), lbl);

    EXPECT_EQ(src.use_count(), 0);
    EXPECT_EQ(src.data(), ptr);
    EXPECT_EQ(src.label(), std::string());
  }

  // Move assignment into a non-empty view.
  {
    ViewType src(args...);
    ViewType dst(args...);

    auto const ptr = src.data();
    auto const cnt = src.use_count();
    auto const lbl = src.label();

    dst = std::move(src);

    EXPECT_EQ(dst.use_count(), cnt);
    EXPECT_EQ(dst.data(), ptr);
    EXPECT_EQ(dst.label(), lbl);

    EXPECT_EQ(src.use_count(), 0);
    EXPECT_EQ(src.data(), ptr);
    EXPECT_EQ(src.label(), std::string());
  }

  // NOLINTEND(bugprone-use-after-move)
}

TEST(TEST_CATEGORY, view_move_use_count_and_label) {
  using ExecutionSpace = TEST_EXECSPACE;

  test_moving_view_use_count_and_label<Kokkos::View<int, ExecutionSpace>>("v0");

  test_moving_view_use_count_and_label<Kokkos::View<float*, ExecutionSpace>>(
      "v1", 1);

  Kokkos::View<double**, ExecutionSpace> v2("v2", 1, 2);
  test_moving_view_use_count_and_label<Kokkos::View<double**, ExecutionSpace>>(
      v2.data(), v2.extent(0), v2.extent(1));
  test_moving_view_use_count_and_label<Kokkos::View<
      double**, ExecutionSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>>(
      v2.data(), v2.extent(0), v2.extent(1));
}

// A moved-from view whose use count falls to zero is freed directly.
TEST(TEST_CATEGORY, move_assign_view_is_freed_within_scope) {
  using view_t = Kokkos::View<int, TEST_EXECSPACE>;

  using namespace Kokkos::Test::Tools;
  listen_tool_events(Config::DisableAll(), Config::EnableAllocs(),
                     Config::EnableProfileEvents());

  view_t outer(Kokkos::view_alloc("outer scope view"));

  const bool success = validate_event_set(
      [&]() {
        Kokkos::Tools::markEvent("entering scope");

        view_t inner(Kokkos::view_alloc("inner scope view"));

        inner = std::move(outer);

        Kokkos::Tools::markEvent("exiting scope");
      },
      [](ProfileEvent pe) {
        return MatchDiagnostic{pe.name == "entering scope"};
      },
      [](AllocateDataEvent ade) {
        return MatchDiagnostic{ade.name == "inner scope view"};
      },
      [](DeallocateDataEvent dde) {
        return MatchDiagnostic{dde.name == "inner scope view"};
      },
      [](ProfileEvent pe) {
        return MatchDiagnostic{pe.name == "exiting scope"};
      },
      [](DeallocateDataEvent dde) {
        return MatchDiagnostic{dde.name == "outer scope view"};
      });
  ASSERT_TRUE(success);

  listen_tool_events(Config::DisableAll());
}

template <class ViewType>
KOKKOS_FUNCTION int check_moved_from_view_state(ViewType v) {
  int err = 0;

  constexpr int rank = ViewType::rank();
  std::size_t
      exts[rank + 1];  // add a dummy trailing element to accomodate rank zero
  for (int i = 0; i < rank; ++i) {
    exts[i] = v.extent(i);
  }
  auto* const ptr = v.data();
  auto const span = v.span();

#define CHECK(CONTEXT, VIEW, PTR, SPAN, EXTS)                              \
  for (int i = 0; i < rank; ++i) {                                         \
    if (VIEW.extent(i) != EXTS[i]) {                                       \
      Kokkos::printf(CONTEXT "expected equality of " #VIEW                 \
                             ".extent(%d) which is %d and " #EXTS          \
                             "[%d] which is %d\n",                         \
                     i, VIEW.extent(i), i, EXTS[i]);                       \
      ++err;                                                               \
    }                                                                      \
  }                                                                        \
  if (VIEW.span() != SPAN) {                                               \
    Kokkos::printf(CONTEXT "expected equality of " #VIEW                   \
                           ".span() which is %d and " #SPAN "which is %d", \
                   VIEW.span(), span);                                     \
    ++err;                                                                 \
  }                                                                        \
  if (VIEW.data() != PTR) {                                                \
    Kokkos::printf(CONTEXT "expected equality of " #VIEW                   \
                           ".data() which is %p and " #PTR "which is %p",  \
                   VIEW.data(), ptr);                                      \
    ++err;                                                                 \
  }

  // NOLINTBEGIN(bugprone-use-after-move)

  ViewType w(std::move(v));  // move construction

  CHECK("failed moved-from view after calling move constructor\n", v, ptr, span,
        exts)
  CHECK("failed moved-from view after calling move constructor\n", w, ptr, span,
        exts)

  v = std::move(w);  // move assignment

  CHECK("failed moved-from view after calling move assignment operator\n", v,
        ptr, span, exts)
  CHECK("failed moved-from view after calling move assignment operator\n", w,
        ptr, span, exts)

  // NOLINTEND(bugprone-use-after-move)

  return err;
}

template <class ViewType>
void test_moved_from_view(ViewType v) {
  using ExecutionSpace = typename ViewType::execution_space;
  int errors;
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<ExecutionSpace>(0, 1),
      KOKKOS_LAMBDA(int, int& err) { err += check_moved_from_view_state(v); },
      errors);
  EXPECT_EQ(errors, 0) << "within parallel region";
}

TEST(TEST_CATEGORY, view_moved_from) {
#if defined(KOKKOS_ENABLE_OPENACC) && (KOKKOS_COMPILER_NVHPC > 240500) && \
    (KOKKOS_COMPILER_NVHPC <= 260500)
  // FIXME_OPENACC: Test is known to fail if 24.5 < NVHPC version <= 26.5.
  // Error message: failed moved-from view after calling move constructor
  //                failed moved-from view after calling move assignment
  //                operator
  GTEST_SKIP() << "skipping since the OpenACC backend test fails if 24.5 < "
                  "NVHPC version <= 26.5";
#else
  using ExecutionSpace = TEST_EXECSPACE;

  test_moved_from_view(Kokkos::View<int, ExecutionSpace>("v0"));
  test_moved_from_view(Kokkos::View<float*, ExecutionSpace>("v1", 1));
  Kokkos::View<double**, ExecutionSpace> v2("v2", 1, 2);
  // FIXME_HIP ROCm 7.1 on MI300 triggers an internal compiler error in the
  // CHECK macro
#if !(                                                    \
    defined(KOKKOS_ENABLE_HIP) &&                         \
    (HIP_VERSION_MAJOR == 7 && HIP_VERSION_MINOR == 1) && \
    (defined(KOKKOS_ARCH_AMD_GFX942) || defined(KOKKOS_ARCH_AMD_GFX942_APU)))
  test_moved_from_view(Kokkos::View<double**, ExecutionSpace>(
      v2.data(), v2.extent(0), v2.extent(1)));
#endif
  test_moved_from_view(Kokkos::View<double**, ExecutionSpace,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
      v2.data(), v2.extent(0), v2.extent(1)));
#endif
}

#if !(defined(KOKKOS_COMPILER_NVCC) || defined(KOKKOS_COMPILER_NVHPC) || \
      (defined(KOKKOS_COMPILER_CLANG) && KOKKOS_COMPILER_CLANG < 1600 && \
       defined(KOKKOS_ENABLE_CUDA)))
constexpr bool test_view_is_nothrow_move_constructible() {
  static_assert(
      std::is_nothrow_move_constructible_v<Kokkos::View<int*, TEST_EXECSPACE>>);
  static_assert(
      std::is_nothrow_move_constructible_v<Kokkos::View<
          int*, TEST_EXECSPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>>);
  static_assert(
      std::is_nothrow_move_constructible_v<Kokkos::View<
          int*, TEST_EXECSPACE, Kokkos::MemoryTraits<Kokkos::Atomic>>>);

  return true;
}
static_assert(test_view_is_nothrow_move_constructible());
#endif

}  // namespace
