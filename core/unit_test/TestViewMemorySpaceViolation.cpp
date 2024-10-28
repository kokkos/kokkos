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

#include <gtest/gtest.h>

#include <TestDefaultDeviceType_Category.hpp>

#ifdef KOKKOS_ENABLE_HIP
#if !(HIP_VERSION_MAJOR == 5 && HIP_VERSION_MINOR < 3)
#define KOKKOS_TEST_HAS_SHARED_SPACE
#endif
#else
#ifdef KOKKOS_HAS_SHARED_SPACE
#define KOKKOS_TEST_HAS_SHARED_SPACE
#endif
#endif

TEST(defaultdevicetype_DeathTest, view_memory_space_violation) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";

#if !((defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || \
       defined(KOKKOS_ENABLE_SYCL)) &&                              \
      defined(KOKKOS_ENABLE_DEBUG))
  GTEST_SKIP() << "memory space violation only detected for Cuda, HIP, or SYCL "
                  "with Kokkos_ENABLE_DEBUG";
#else

  auto create_host_view = [](auto view) {
    Kokkos::View<int*, Kokkos::HostSpace> host_unmanaged(view.data(),
                                                         view.size());
  };

  auto create_default_view = [](auto view) {
    Kokkos::View<int*> default_unmanaged(view.data(), view.size());
  };

#ifdef KOKKOS_TEST_HAS_SHARED_SPACE
  auto create_shared_view = [](auto view) {
    Kokkos::View<int*, Kokkos::SharedSpace> shared_unmanaged(view.data(),
                                                             view.size());
  };
#endif

#ifdef KOKKOS_HAS_SHARED_HOST_PINNED_SPACE
  auto create_hostpinned_view = [](auto view) {
    Kokkos::View<int*, Kokkos::SharedHostPinnedSpace> hostpinned_unmanaged(
        view.data(), view.size());
  };
#endif

  int has_real_shared_space = 1;
#ifdef KOKKOS_ENABLE_HIP
  has_real_shared_space     = 0;  // false by default
  KOKKOS_IMPL_HIP_SAFE_CALL(hipDeviceGetAttribute(
      &has_real_shared_space, hipDeviceAttributePageableMemoryAccess,
      Kokkos::HIP{}.hip_device()));
#endif

  {
    Kokkos::View<int*, Kokkos::HostSpace> host_space_view("host_space_view", 1);

    create_host_view(host_space_view);
    ASSERT_DEATH(create_default_view(host_space_view), "");
#ifdef KOKKOS_TEST_HAS_SHARED_SPACE
    if (has_real_shared_space)
      ASSERT_DEATH(create_shared_view(host_space_view), "");
#endif
#ifdef KOKKOS_HAS_SHARED_HOST_PINNED_SPACE
    ASSERT_DEATH(create_hostpinned_view(host_space_view), "");
#endif
  }

  {
    Kokkos::View<int*> default_space_view("default_space_view", 1);

    ASSERT_DEATH(create_host_view(default_space_view), "");
    create_default_view(default_space_view);
#ifdef KOKKOS_TEST_HAS_SHARED_SPACE
    if (has_real_shared_space)
      ASSERT_DEATH(create_shared_view(default_space_view), "");
#endif
#ifdef KOKKOS_HAS_SHARED_HOST_PINNED_SPACE
    ASSERT_DEATH(create_hostpinned_view(default_space_view), "");
#endif
  }

#ifdef KOKKOS_TEST_HAS_SHARED_SPACE
  if (has_real_shared_space) {
    Kokkos::View<int*, Kokkos::SharedSpace> shared_space_view(
        "shared_space_view", 1);

    ASSERT_DEATH(create_host_view(shared_space_view), "");
    ASSERT_DEATH(create_default_view(shared_space_view), "");
    create_shared_view(shared_space_view);
#ifdef KOKKOS_HAS_SHARED_HOST_PINNED_SPACE
    ASSERT_DEATH(create_hostpinned_view(shared_space_view), "");
#endif
  }
#endif

#ifdef KOKKOS_HAS_SHARED_HOST_PINNED_SPACE
  {
    Kokkos::View<int*, Kokkos::SharedHostPinnedSpace> hostpinned_space_view(
        "host_pinned_space_view", 1);

    ASSERT_DEATH(create_host_view(hostpinned_space_view), "");
    ASSERT_DEATH(create_default_view(hostpinned_space_view), "");
#ifdef KOKKOS_TEST_HAS_SHARED_SPACE
    if (has_real_shared_space)
      ASSERT_DEATH(create_shared_view(hostpinned_space_view), "");
#endif
    create_hostpinned_view(hostpinned_space_view);
  }
#endif
#endif
}
