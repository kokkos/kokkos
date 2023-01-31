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
#include <cstdio>
#include <PerfTest_Category.hpp>

namespace Test {

template <class ViewType>
double fill_view(ViewType& a, typename ViewType::const_value_type& val,
                 int repeat) {
  Kokkos::Timer timer;
  for (int i = 0; i < repeat; i++) {
    Kokkos::deep_copy(a, val);
  }
  Kokkos::fence();
  return timer.seconds();
}

template <class Layout>
void run_fillview_tests123(int N, int R) {
  const int N1 = N;
  const int N2 = N1 * N1;
  const int N3 = N2 * N1;
  const int N4 = N2 * N2;
  const int N8 = N4 * N4;

  double time1, time2, time3, time_raw = 100000.0;
  {
    Kokkos::View<double*, Layout> a("A1", N8);
    time1 = fill_view(a, 1.1, R) / R;
  }
  {
    Kokkos::View<double**, Layout> a("A2", N4, N4);
    time2 = fill_view(a, 1.1, R) / R;
  }
  {
    Kokkos::View<double***, Layout> a("A3", N3, N3, N2);
    time3 = fill_view(a, 1.1, R) / R;
  }
#if defined(KOKKOS_ENABLE_CUDA_LAMBDA) || !defined(KOKKOS_ENABLE_CUDA)
  {
    Kokkos::View<double*, Layout> a("A1", N8);
    double* a_ptr = a.data();
    Kokkos::Timer timer;
    for (int r = 0; r < R; r++) {
      Kokkos::parallel_for(
          N8, KOKKOS_LAMBDA(const int& i) { a_ptr[i] = 1.1; });
    }
    Kokkos::fence();
    time_raw = timer.seconds() / R;
  }
#endif
  double size = 1.0 * N8 * 8 / 1024 / 1024;
  printf("   Raw:   %lf s   %lf MB   %lf GB/s\n", time_raw, size,
         size / 1024 / time_raw);
  printf("   Rank1: %lf s   %lf MB   %lf GB/s\n", time1, size,
         size / 1024 / time1);
  printf("   Rank2: %lf s   %lf MB   %lf GB/s\n", time2, size,
         size / 1024 / time2);
  printf("   Rank3: %lf s   %lf MB   %lf GB/s\n", time3, size,
         size / 1024 / time3);
}

template <class Layout>
void run_fillview_tests45(int N, int R) {
  const int N1 = N;
  const int N2 = N1 * N1;
  const int N4 = N2 * N2;
  const int N8 = N4 * N4;

  double time4, time5, time_raw = 100000.0;
  {
    Kokkos::View<double****, Layout> a("A4", N2, N2, N2, N2);
    time4 = fill_view(a, 1.1, R) / R;
  }
  {
    Kokkos::View<double*****, Layout> a("A5", N2, N2, N1, N1, N2);
    time5 = fill_view(a, 1.1, R) / R;
  }
#if defined(KOKKOS_ENABLE_CUDA_LAMBDA) || !defined(KOKKOS_ENABLE_CUDA)
  {
    Kokkos::View<double*, Layout> a("A1", N8);
    double* a_ptr = a.data();
    Kokkos::Timer timer;
    for (int r = 0; r < R; r++) {
      Kokkos::parallel_for(
          N8, KOKKOS_LAMBDA(const int& i) { a_ptr[i] = 1.1; });
    }
    Kokkos::fence();
    time_raw = timer.seconds() / R;
  }
#endif
  double size = 1.0 * N8 * 8 / 1024 / 1024;
  printf("   Raw:   %lf s   %lf MB   %lf GB/s\n", time_raw, size,
         size / 1024 / time_raw);
  printf("   Rank4: %lf s   %lf MB   %lf GB/s\n", time4, size,
         size / 1024 / time4);
  printf("   Rank5: %lf s   %lf MB   %lf GB/s\n", time5, size,
         size / 1024 / time5);
}

template <class Layout>
void run_fillview_tests6(int N, int R) {
  const int N1 = N;
  const int N2 = N1 * N1;
  const int N4 = N2 * N2;
  const int N8 = N4 * N4;

  double time6, time_raw = 100000.0;
  {
    Kokkos::View<double******, Layout> a("A6", N2, N1, N1, N1, N1, N2);
    time6 = fill_view(a, 1.1, R) / R;
  }
#if defined(KOKKOS_ENABLE_CUDA_LAMBDA) || !defined(KOKKOS_ENABLE_CUDA)
  {
    Kokkos::View<double*, Layout> a("A1", N8);
    double* a_ptr = a.data();
    Kokkos::Timer timer;
    for (int r = 0; r < R; r++) {
      Kokkos::parallel_for(
          N8, KOKKOS_LAMBDA(const int& i) { a_ptr[i] = 1.1; });
    }
    Kokkos::fence();
    time_raw = timer.seconds() / R;
  }
#endif
  double size = 1.0 * N8 * 8 / 1024 / 1024;
  printf("   Raw:   %lf s   %lf MB   %lf GB/s\n", time_raw, size,
         size / 1024 / time_raw);
  printf("   Rank6: %lf s   %lf MB   %lf GB/s\n", time6, size,
         size / 1024 / time6);
}

template <class Layout>
void run_fillview_tests7(int N, int R) {
  const int N1 = N;
  const int N2 = N1 * N1;
  const int N4 = N2 * N2;
  const int N8 = N4 * N4;

  double time7, time_raw = 100000.0;
  {
    Kokkos::View<double*******, Layout> a("A7", N2, N1, N1, N1, N1, N1, N1);
    time7 = fill_view(a, 1.1, R) / R;
  }
#if defined(KOKKOS_ENABLE_CUDA_LAMBDA) || !defined(KOKKOS_ENABLE_CUDA)
  {
    Kokkos::View<double*, Layout> a("A1", N8);
    double* a_ptr = a.data();
    Kokkos::Timer timer;
    for (int r = 0; r < R; r++) {
      Kokkos::parallel_for(
          N8, KOKKOS_LAMBDA(const int& i) { a_ptr[i] = 1.1; });
    }
    Kokkos::fence();
    time_raw = timer.seconds() / R;
  }
#endif
  double size = 1.0 * N8 * 8 / 1024 / 1024;
  printf("   Raw:   %lf s   %lf MB   %lf GB/s\n", time_raw, size,
         size / 1024 / time_raw);
  printf("   Rank7: %lf s   %lf MB   %lf GB/s\n", time7, size,
         size / 1024 / time7);
}

template <class Layout>
void run_fillview_tests8(int N, int R) {
  const int N1 = N;
  const int N2 = N1 * N1;
  const int N4 = N2 * N2;
  const int N8 = N4 * N4;

  double time8, time_raw = 100000.0;
  {
    Kokkos::View<double********, Layout> a("A8", N1, N1, N1, N1, N1, N1, N1,
                                           N1);
    time8 = fill_view(a, 1.1, R) / R;
  }
#if defined(KOKKOS_ENABLE_CUDA_LAMBDA) || !defined(KOKKOS_ENABLE_CUDA)
  {
    Kokkos::View<double*, Layout> a("A1", N8);
    double* a_ptr = a.data();
    Kokkos::Timer timer;
    for (int r = 0; r < R; r++) {
      Kokkos::parallel_for(
          N8, KOKKOS_LAMBDA(const int& i) { a_ptr[i] = 1.1; });
    }
    Kokkos::fence();
    time_raw = timer.seconds() / R;
  }
#endif
  double size = 1.0 * N8 * 8 / 1024 / 1024;
  printf("   Raw:   %lf s   %lf MB   %lf GB/s\n", time_raw, size,
         size / 1024 / time_raw);
  printf("   Rank8: %lf s   %lf MB   %lf GB/s\n", time8, size,
         size / 1024 / time8);
}

}  // namespace Test
