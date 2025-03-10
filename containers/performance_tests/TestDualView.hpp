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

struct times_t {
  double t1;
  double t2;
  double t3;
};

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
  void run(const int n, const int m, times_t& times) {
    ViewType a, b;
    a = ViewType("A", n, m);
    b = ViewType("B", n, m);

    const scalar_type sum_total = scalar_type(n * m);

    Kokkos::deep_copy(a.view_device(), 1);

    Kokkos::Timer timer1, timer2, timer3;

    using device_space = typename ViewType::t_dev::execution_space;
    using host_space   = typename ViewType::t_host::execution_space;

    timer1.reset();
    a.template modify<device_space>();
    a.template sync<host_space>();
    times.t1 = timer1.seconds();

    // Check device view is initialized as expected
    scalar_type a_d_sum = 0;

    timer2.reset();
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<device_space>(0, n),
        SumViewEntriesFunctor<scalar_type, typename ViewType::t_dev>(
            a.view_device()),
        a_d_sum);
    times.t2 = timer2.seconds();
    ASSERT_EQ(a_d_sum, sum_total);

    // Use deep_copy
    Kokkos::deep_copy(b, a);
    timer3.reset();
    b.template sync<host_space>();
    times.t3 += timer3.seconds();

    // Perform same checks on b as done on a
    // Check device view is initialized as expected
    scalar_type b_d_sum = 0;
    timer2.reset();
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<device_space>(0, n),
        SumViewEntriesFunctor<scalar_type, typename ViewType::t_dev>(
            b.view_device()),
        b_d_sum);
    times.t2 += timer2.seconds();
    ASSERT_EQ(b_d_sum, sum_total);
  }
  test_dualview_with_datacheck(const int n, const int m) {
    times_t elapsed_time = {0.0, 0.0, 0.0};
    run<Kokkos::DualView<Scalar**, Kokkos::LayoutLeft, Device>>(n, m,
                                                                elapsed_time);
    std::cout << " DualView (test_dualview_with_datacheck) timing (sec):"
              << elapsed_time.t1 << ", " << elapsed_time.t2 << ", "
              << elapsed_time.t3 << ", dim0:" << n << ", dim1:" << m
              << std::endl;
  }
};

template <typename Scalar, class Device>
struct test_dualview_sync {
  using scalar_type     = Scalar;
  using execution_space = Device;

  template <typename ViewType>
  void run(const int n, const int m, const int iters, times_t& elapsed_time) {
    ViewType a = ViewType("A", n, m);

    using device_space = typename ViewType::t_dev::execution_space;
    using host_space   = typename ViewType::t_host::execution_space;

    Kokkos::deep_copy(a.view_device(), 1);

    int i = iters;

    Kokkos::Timer timer1, timer2, timer3;
    while (i--) {
      // Sync to host
      timer1.reset();
      a.template modify<device_space>();
      a.template sync<host_space>();
      elapsed_time.t1 += timer1.seconds();

      // Update on host
      timer2.reset();
      Kokkos::parallel_for(
          Kokkos::RangePolicy<host_space>(0, n),
          IncrViewEntriesFunctor<scalar_type, typename ViewType::t_host>(
              a.view_host()));
      Kokkos::fence();
      elapsed_time.t2 += timer2.seconds();

      // Sync to device
      timer3.reset();
      a.template modify<host_space>();
      a.template sync<device_space>();
      elapsed_time.t3 += timer3.seconds();

      // Update on device
      timer2.reset();
      Kokkos::parallel_for(
          Kokkos::RangePolicy<device_space>(0, n),
          IncrViewEntriesFunctor<scalar_type, typename ViewType::t_dev>(
              a.view_device()));
      Kokkos::fence();
      elapsed_time.t2 += timer2.seconds();
    }
    Kokkos::fence();

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
    times_t elapsed_time = {0.0, 0.0, 0.0};
    const int iters      = 10;
    run<Kokkos::DualView<Scalar**, Kokkos::LayoutLeft, Device>>(n, m, iters,
                                                                elapsed_time);
    std::cout << " DualView (test_dualview_sync) timing (sec):"
              << elapsed_time.t1 / static_cast<double>(iters) << ", "
              << elapsed_time.t2 / static_cast<double>(iters) << ", "
              << elapsed_time.t3 / static_cast<double>(iters) << ", dim0:" << n
              << ", dim1:" << m << std::endl;
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
