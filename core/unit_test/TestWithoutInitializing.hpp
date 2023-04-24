//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <Kokkos_Core.hpp>

#include "tools/include/ToolTestingUtilities.hpp"

TEST(TEST_CATEGORY, resize_realloc_no_init) {
  using namespace Kokkos::Test::Tools;
  listen_tool_events(Config::DisableAll(), Config::EnableKernels());
  Kokkos::View<int*** * [1][2][3][4], TEST_EXECSPACE> bla("bla", 5, 6, 7, 8);

  auto success = validate_absence(
      [&]() {
        Kokkos::resize(Kokkos::WithoutInitializing, bla, 5, 6, 7, 9);
        Kokkos::realloc(Kokkos::WithoutInitializing, bla, 8, 8, 8, 8);
        Kokkos::realloc(Kokkos::view_alloc(Kokkos::WithoutInitializing), bla, 5,
                        6, 7, 8);
      },
      [&](BeginParallelForEvent event) {
        if (event.descriptor().find("initialization") != std::string::npos)
          return MatchDiagnostic{true, {"Found begin event"}};
        return MatchDiagnostic{false};
      },
      [&](EndParallelForEvent event) {
        if (event.descriptor().find("initialization") != std::string::npos)
          return MatchDiagnostic{true, {"Found end event"}};
        return MatchDiagnostic{false};
      });
  ASSERT_TRUE(success);
  listen_tool_events(Config::DisableAll());
}

TEST(TEST_CATEGORY, resize_realloc_no_alloc) {
  using namespace Kokkos::Test::Tools;
  listen_tool_events(Config::DisableAll(), Config::EnableKernels(),
                     Config::EnableAllocs());
  Kokkos::View<int*** * [1][2][3][4], TEST_EXECSPACE> bla("bla", 8, 7, 6, 5);

  auto success = validate_absence(
      [&]() {
        Kokkos::resize(bla, 8, 7, 6, 5);
        Kokkos::realloc(Kokkos::WithoutInitializing, bla, 8, 7, 6, 5);
      },
      [&](BeginParallelForEvent) {
        return MatchDiagnostic{true, {"Found begin event"}};
      },
      [&](EndParallelForEvent) {
        return MatchDiagnostic{true, {"Found end event"}};
      },
      [&](AllocateDataEvent) {
        return MatchDiagnostic{true, {"Found alloc event"}};
      },
      [&](DeallocateDataEvent) {
        return MatchDiagnostic{true, {"Found dealloc event"}};
      });
  ASSERT_TRUE(success);
  listen_tool_events(Config::DisableAll());
}

TEST(TEST_CATEGORY, realloc_exec_space) {
#ifdef KOKKOS_ENABLE_CUDA
  if (std::is_same<typename TEST_EXECSPACE::memory_space,
                   Kokkos::CudaUVMSpace>::value)
    GTEST_SKIP() << "skipping since CudaUVMSpace requires additional fences";
#endif
// FIXME_OPENMPTARGET The OpenMPTarget backend doesn't implement allocate taking
// an execution space instance properly so it needs another fence
#ifdef KOKKOS_ENABLE_OPENMPTARGET
  if (std::is_same<TEST_EXECSPACE, Kokkos::Experimental::OpenMPTarget>::value)
    GTEST_SKIP() << "skipping since the OpenMPTarget backend doesn't implement "
                    "allocate taking an execution space instance properly";
#endif

  using namespace Kokkos::Test::Tools;
  listen_tool_events(Config::DisableAll(), Config::EnableFences());
  using view_type = Kokkos::View<int*, TEST_EXECSPACE>;
  view_type outer_view, outer_view2;

  auto success = validate_absence(
      [&]() {
        view_type inner_view(Kokkos::view_alloc(TEST_EXECSPACE{}, "bla"), 8);
        // Avoid testing the destructor
        outer_view = inner_view;
        Kokkos::realloc(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, TEST_EXECSPACE{}),
            inner_view, 10);
        outer_view2 = inner_view;
        Kokkos::realloc(Kokkos::view_alloc(TEST_EXECSPACE{}), inner_view, 10);
      },
      [&](BeginFenceEvent event) {
        if ((event.descriptor().find("Debug Only Check for Execution Error") !=
             std::string::npos) ||
            (event.descriptor().find("HostSpace fence") != std::string::npos))
          return MatchDiagnostic{false};
        return MatchDiagnostic{true, {"Found fence event!"}};
      });
  ASSERT_TRUE(success);
  listen_tool_events(Config::DisableAll());
}

namespace {
struct NonTriviallyCopyable {
  KOKKOS_FUNCTION NonTriviallyCopyable() {}
  KOKKOS_FUNCTION NonTriviallyCopyable(const NonTriviallyCopyable&) {}
};
}  // namespace

TEST(TEST_CATEGORY, view_alloc) {
  using namespace Kokkos::Test::Tools;
  listen_tool_events(Config::DisableAll(), Config::EnableFences());
  using view_type = Kokkos::View<NonTriviallyCopyable*, TEST_EXECSPACE>;
  view_type outer_view;

  auto success = validate_existence(
      [&]() {
        view_type inner_view(Kokkos::view_alloc("bla"), 8);
        // Avoid testing the destructor
        outer_view = inner_view;
      },
      [&](BeginFenceEvent event) {
        return MatchDiagnostic{
            event.descriptor().find(
                "Kokkos::Impl::ViewValueFunctor: View init/destroy fence") !=
            std::string::npos};
      });
  ASSERT_TRUE(success);
  listen_tool_events(Config::DisableAll());
}

TEST(TEST_CATEGORY, view_alloc_exec_space) {
  using namespace Kokkos::Test::Tools;
  listen_tool_events(Config::DisableAll(), Config::EnableFences());
  using view_type = Kokkos::View<NonTriviallyCopyable*, TEST_EXECSPACE>;
  view_type outer_view;

  auto success = validate_absence(
      [&]() {
        view_type inner_view(Kokkos::view_alloc(TEST_EXECSPACE{}, "bla"), 8);
        // Avoid testing the destructor
        outer_view = inner_view;
      },
      [&](BeginFenceEvent event) {
        return MatchDiagnostic{
            event.descriptor().find(
                "Kokkos::Impl::ViewValueFunctor: View init/destroy fence") !=
            std::string::npos};
      });
  ASSERT_TRUE(success);
  listen_tool_events(Config::DisableAll());
}

TEST(TEST_CATEGORY, view_alloc_int) {
  using namespace Kokkos::Test::Tools;
  listen_tool_events(Config::DisableAll(), Config::EnableFences());
  using view_type = Kokkos::View<int*, TEST_EXECSPACE>;
  view_type outer_view;

  auto success = validate_existence(
      [&]() {
        view_type inner_view("bla", 8);
        // Avoid testing the destructor
        outer_view = inner_view;
      },
      [&](BeginFenceEvent event) {
        return MatchDiagnostic{
            event.descriptor().find(
                "Kokkos::Impl::ViewValueFunctor: View init/destroy fence") !=
            std::string::npos};
      });
  ASSERT_TRUE(success);
  listen_tool_events(Config::DisableAll());
}

TEST(TEST_CATEGORY, view_alloc_exec_space_int) {
  using namespace Kokkos::Test::Tools;
  listen_tool_events(Config::DisableAll(), Config::EnableFences());
  using view_type = Kokkos::View<int*, TEST_EXECSPACE>;
  view_type outer_view;

  auto success = validate_absence(
      [&]() {
        view_type inner_view(Kokkos::view_alloc(TEST_EXECSPACE{}, "bla"), 8);
        // Avoid testing the destructor
        outer_view = inner_view;
      },
      [&](BeginFenceEvent event) {
        return MatchDiagnostic{
            event.descriptor().find(
                "Kokkos::Impl::ViewValueFunctor: View init/destroy fence") !=
            std::string::npos};
      });
  ASSERT_TRUE(success);
  listen_tool_events(Config::DisableAll());
}

TEST(TEST_CATEGORY, deep_copy_zero_memset) {
// FIXME_OPENMPTARGET The OpenMPTarget backend doesn't implement ZeroMemset
#ifdef KOKKOS_ENABLE_OPENMPTARGET
  if (std::is_same<TEST_EXECSPACE, Kokkos::Experimental::OpenMPTarget>::value)
    GTEST_SKIP() << "skipping since the OpenMPTarget backend doesn't implement "
                    "ZeroMemset";
#endif
// FIXME_OPENACC: The OpenACC backend doesn't implement ZeroMemset
#ifdef KOKKOS_ENABLE_OPENACC
  if (std::is_same<TEST_EXECSPACE, Kokkos::Experimental::OpenACC>::value)
    GTEST_SKIP() << "skipping since the OpenACC backend doesn't implement "
                    "ZeroMemset";
#endif

  using namespace Kokkos::Test::Tools;
  listen_tool_events(Config::DisableAll(), Config::EnableKernels());
  Kokkos::View<int*, TEST_EXECSPACE> bla("bla", 8);

  auto success =
      validate_absence([&]() { Kokkos::deep_copy(bla, 0); },
                       [&](BeginParallelForEvent) {
                         return MatchDiagnostic{true, {"Found begin event"}};
                       },
                       [&](EndParallelForEvent) {
                         return MatchDiagnostic{true, {"Found end event"}};
                       });
  ASSERT_TRUE(success);
  listen_tool_events(Config::DisableAll());
}

TEST(TEST_CATEGORY, resize_exec_space) {
  using namespace Kokkos::Test::Tools;
  listen_tool_events(Config::DisableAll(), Config::EnableFences(),
                     Config::EnableKernels());
  Kokkos::View<int*** * [1][2][3][4], TEST_EXECSPACE> bla("bla", 8, 7, 6, 5);

  auto success = validate_absence(
      [&]() {
        Kokkos::resize(
            Kokkos::view_alloc(TEST_EXECSPACE{}, Kokkos::WithoutInitializing),
            bla, 5, 6, 7, 8);
      },
      [&](BeginFenceEvent event) {
        if (event.descriptor().find("Kokkos::resize(View)") !=
            std::string::npos)
          return MatchDiagnostic{true, {"Found begin event"}};
        return MatchDiagnostic{false};
      },
      [&](EndFenceEvent event) {
        if (event.descriptor().find("Kokkos::resize(View)") !=
            std::string::npos)
          return MatchDiagnostic{true, {"Found end event"}};
        return MatchDiagnostic{false};
      },
      [&](BeginParallelForEvent event) {
        if (event.descriptor().find("initialization") != std::string::npos)
          return MatchDiagnostic{true, {"Found begin event"}};
        return MatchDiagnostic{false};
      },
      [&](EndParallelForEvent event) {
        if (event.descriptor().find("initialization") != std::string::npos)
          return MatchDiagnostic{true, {"Found end event"}};
        return MatchDiagnostic{false};
      });
  ASSERT_TRUE(success);
  listen_tool_events(Config::DisableAll());
}

TEST(TEST_CATEGORY, view_allocation_int) {
// FIXME_OPENMPTARGET
#ifdef KOKKOS_ENABLE_OPENMPTARGET
  if (std::is_same<TEST_EXECSPACE, Kokkos::Experimental::OpenMPTarget>::value)
    GTEST_SKIP() << "skipping since the OpenMPTarget has unexpected fences";
#endif

  using ExecutionSpace = TEST_EXECSPACE;
  if (Kokkos::SpaceAccessibility<
          /*AccessSpace=*/Kokkos::HostSpace,
          /*MemorySpace=*/ExecutionSpace::memory_space>::accessible) {
    GTEST_SKIP() << "skipping since the fence checked for isn't necessary";
  }
  using namespace Kokkos::Test::Tools;
  listen_tool_events(Config::EnableAll());
  using view_type = Kokkos::View<int*, TEST_EXECSPACE>;
  view_type outer_view;

  auto success = validate_existence(
      [&]() {
        view_type inner_view(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "bla"), 8);
        // Avoid testing the destructor
        outer_view = inner_view;
      },
      [&](BeginFenceEvent event) {
        return MatchDiagnostic{
            event.descriptor().find(
                "fence after copying header from HostSpace") !=
            std::string::npos};
      });
  ASSERT_TRUE(success);
  listen_tool_events(Config::DisableAll());
}

TEST(TEST_CATEGORY, view_allocation_exec_space_int) {
#ifdef KOKKOS_ENABLE_OPENMPTARGET  // FIXME_OPENMPTARGET
  if (std::is_same<TEST_EXECSPACE, Kokkos::Experimental::OpenMPTarget>::value)
    GTEST_SKIP() << "skipping since the OpenMPTarget has unexpected fences";
#endif

#ifdef KOKKOS_ENABLE_CUDA
  if (std::is_same<TEST_EXECSPACE::memory_space, Kokkos::CudaUVMSpace>::value)
    GTEST_SKIP()
        << "skipping since the CudaUVMSpace requires additiional fences";
#endif

  using namespace Kokkos::Test::Tools;
  listen_tool_events(Config::EnableAll());
  using view_type = Kokkos::View<int*, TEST_EXECSPACE>;
  view_type outer_view;

  auto success = validate_absence(
      [&]() {
        view_type inner_view(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                                TEST_EXECSPACE{}, "bla"),
                             8);
        // Avoid testing the destructor
        outer_view = inner_view;
      },
      [&](BeginFenceEvent) { return MatchDiagnostic{true}; });
  ASSERT_TRUE(success);
  listen_tool_events(Config::DisableAll());
}

struct NotDefaultConstructible {
  NotDefaultConstructible() = delete;
};

TEST(TEST_CATEGORY, view_not_default_constructible) {
  using Space = TEST_EXECSPACE;
  Kokkos::View<NotDefaultConstructible, Space> my_view(Kokkos::view_alloc(
      "not_default_constructible", Kokkos::WithoutInitializing));
}
