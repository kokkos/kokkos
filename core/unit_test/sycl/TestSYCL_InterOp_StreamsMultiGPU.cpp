// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <TestSYCL_Category.hpp>
#include <TestMultiGPU.hpp>

namespace {

TEST(sycl_multi_gpu, managed_views) {
  auto execs = ::Kokkos::create_device_space();

  auto execs0 = execs.front();
  auto execs1 = execs.back();

  Kokkos::View<int *, TEST_EXECSPACE> view0(Kokkos::view_alloc("v0", execs0),
                                            100);
  Kokkos::View<int *, TEST_EXECSPACE> view1(Kokkos::view_alloc("v1", execs1),
                                            100);

  test_policies(execs0, view0, execs1, view1);
}

TEST(sycl_multi_gpu, unmanaged_views) {
  auto execs = ::Kokkos::create_device_space();

  auto execs0 = execs.front();
  auto execs1 = execs.back();

  int *p0 = sycl::malloc_device<int>(100, execs0.sycl_queue());
  Kokkos::View<int *, TEST_EXECSPACE> view0(p0, 100);

  int *p1 = sycl::malloc_device<int>(100, execs1.sycl_queue());
  Kokkos::View<int *, TEST_EXECSPACE> view1(p1, 100);

  test_policies(execs0, view0, execs1, view1);
  sycl::free(p0, execs0.sycl_queue());
  sycl::free(p1, execs1.sycl_queue());
}

TEST(sycl_multi_gpu, scratch_space) {
  auto execs = ::Kokkos::create_device_space();

  auto execs0 = execs.front();
  auto execs1 = execs.back();

  test_scratch(execs0, execs1);
}
}  // namespace
