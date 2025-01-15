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

#ifndef KOKKOS_TEST_DUALVIEW_HPP
#define KOKKOS_TEST_DUALVIEW_HPP

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <Kokkos_DualView.hpp>

#include <Kokkos_Timer.hpp>

namespace Performance {

namespace Impl {

template <typename Scalar, class ViewType>
struct SumViewEntriesFunctor {
  using value_type = Scalar;
  ViewType view;
  SumViewEntriesFunctor(const ViewType& view_) : view(view_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, value_type& sum) const {
    for (size_t j = 0; j < view.extent(1); ++j) {
      sum += view(i, j);
    }
  }
};

template <typename Scalar, class ViewType>
struct IncrViewEntriesFunctor {
  using value_type = Scalar;
  ViewType view;
  IncrViewEntriesFunctor(const ViewType& view_) : view(view_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    for (size_t j = 0; j < view.extent(1); ++j) {
      view(i, j)++;
    }
  }
};

template <typename Scalar, class Device>
struct test_dualview_with_datacheck {
  using scalar_type     = Scalar;
  using execution_space = Device;

  template <typename ViewType>
  void run(const int n, const int m, double& elapsed_time) {
    ViewType a, b;
    a = ViewType("A", n, m);
    b = ViewType("B", n, m);

    const scalar_type sum_total = scalar_type(n * m);

    Kokkos::deep_copy(a.view_device(), 1);

    Kokkos::Timer timer;

    using device_space = typename ViewType::t_dev::execution_space;
    using host_space   = typename ViewType::t_host::execution_space;

    a.template modify<device_space>();
    a.template sync<host_space>();

    // Check device view is initialized as expected
    scalar_type a_d_sum = 0;

    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<device_space>(0, n),
        SumViewEntriesFunctor<scalar_type, typename ViewType::t_dev>(
            a.view_device()),
        a_d_sum);
    ASSERT_EQ(a_d_sum, sum_total);

    // Use deep_copy
    Kokkos::deep_copy(b, a);
    b.template sync<host_space>();

    // Perform same checks on b as done on a
    // Check device view is initialized as expected
    scalar_type b_d_sum = 0;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<device_space>(0, n),
        SumViewEntriesFunctor<scalar_type, typename ViewType::t_dev>(
            b.view_device()),
        b_d_sum);
    ASSERT_EQ(b_d_sum, sum_total);
    elapsed_time = timer.seconds();
  }
  test_dualview_with_datacheck(const int n, const int m) {
    double elapsed_time = 0;
    run<Kokkos::DualView<Scalar**, Kokkos::LayoutLeft, Device>>(n, m,
                                                                elapsed_time);
    std::cout << " DualView (test_dualview_with_datacheck) timing (sec):"
              << elapsed_time << ", dim0:" << n << ", dim1:" << m << std::endl;
  }
};

template <typename Scalar, class Device>
struct test_dualview_sync {
  using scalar_type     = Scalar;
  using execution_space = Device;

  template <typename ViewType>
  void run(const int n, const int m, const int iters, double& elapsed_time) {
    ViewType a = ViewType("A", n, m);

    using device_space = typename ViewType::t_dev::execution_space;
    using host_space   = typename ViewType::t_host::execution_space;

    Kokkos::deep_copy(a.view_device(), 1);

    int i = iters;

    Kokkos::Timer timer;

    while (i--) {
      // Sync to host
      a.template modify<device_space>();
      a.template sync<host_space>();

      // Update on host
      Kokkos::parallel_for(
          Kokkos::RangePolicy<host_space>(0, n),
          IncrViewEntriesFunctor<scalar_type, typename ViewType::t_host>(
              a.view_host()));

      // Sync to device
      a.template modify<host_space>();
      a.template sync<device_space>();

      // Update on device
      Kokkos::parallel_for(
          Kokkos::RangePolicy<device_space>(0, n),
          IncrViewEntriesFunctor<scalar_type, typename ViewType::t_dev>(
              a.view_device()));
    }
    Kokkos::fence();

    elapsed_time = timer.seconds();

    const scalar_type sum_total = scalar_type(n * m);
    scalar_type a_d_sum         = 0;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<device_space>(0, n),
        SumViewEntriesFunctor<scalar_type, typename ViewType::t_dev>(
            a.view_device()),
        a_d_sum);
    ASSERT_EQ(a_d_sum, sum_total + iters * 2 * sum_total);
  }
  test_dualview_sync(const int n, const int m) {
    double elapsed_time = 0;
    run<Kokkos::DualView<Scalar**, Kokkos::LayoutLeft, Device>>(n, m, 10,
                                                                elapsed_time);
    std::cout << " DualView (test_dualview_sync) timing (sec):" << elapsed_time
              << ", dim0:" << n << ", dim1:" << m << std::endl;
  }
};
}  // namespace Impl

template <typename Scalar, typename Device>
void test_dualview() {
  Impl::test_dualview_with_datacheck<Scalar, Device>(128, 128);
  Impl::test_dualview_with_datacheck<Scalar, Device>(512, 512);
  Impl::test_dualview_with_datacheck<Scalar, Device>(2048, 2048);
  Impl::test_dualview_with_datacheck<Scalar, Device>(8192, 8192);
  Impl::test_dualview_sync<Scalar, Device>(128, 128);
  Impl::test_dualview_sync<Scalar, Device>(512, 512);
  Impl::test_dualview_sync<Scalar, Device>(2048, 2048);
  Impl::test_dualview_sync<Scalar, Device>(8192, 8192);
}
}  // namespace Performance

#endif  // KOKKOS_TEST_DUALVIEW_HPP
