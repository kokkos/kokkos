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

#ifndef KOKKOS_TEST_DYNRANKVIEW_HPP
#define KOKKOS_TEST_DYNRANKVIEW_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_DynRankView.hpp>
#include <vector>

#include <Kokkos_Timer.hpp>

// Compare performance of DynRankView to View, specific focus on the parenthesis
// operators

namespace Performance {

// View functor
template <typename DeviceType>
struct InitViewFunctor {
  using inviewtype = Kokkos::View<double ***, DeviceType>;
  inviewtype _inview;

  InitViewFunctor(inviewtype &inview_) : _inview(inview_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    for (unsigned j = 0; j < _inview.extent(1); ++j) {
      for (unsigned k = 0; k < _inview.extent(2); ++k) {
        _inview(i, j, k) = double(i) / 2 - j * j + double(k) / 3;
      }
    }
  }

  struct SumComputationTest {
    using inviewtype = Kokkos::View<double ***, DeviceType>;
    inviewtype _inview;

    using outviewtype = Kokkos::View<double *, DeviceType>;
    outviewtype _outview;

    KOKKOS_INLINE_FUNCTION
    SumComputationTest(inviewtype &inview_, outviewtype &outview_)
        : _inview(inview_), _outview(outview_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i) const {
      for (unsigned j = 0; j < _inview.extent(1); ++j) {
        for (unsigned k = 0; k < _inview.extent(2); ++k) {
          _outview(i) += _inview(i, j, k);
        }
      }
    }
  };
};

template <typename DeviceType>
struct InitStrideViewFunctor {
  using inviewtype = Kokkos::View<double ***, Kokkos::LayoutStride, DeviceType>;
  inviewtype _inview;

  InitStrideViewFunctor(inviewtype &inview_) : _inview(inview_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    for (unsigned j = 0; j < _inview.extent(1); ++j) {
      for (unsigned k = 0; k < _inview.extent(2); ++k) {
        _inview(i, j, k) = double(i) / 2 - j * j + double(k) / 3;
      }
    }
  }
};

template <typename DeviceType>
struct InitViewRank7Functor {
  using inviewtype = Kokkos::View<double *******, DeviceType>;
  inviewtype _inview;

  InitViewRank7Functor(inviewtype &inview_) : _inview(inview_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    for (unsigned j = 0; j < _inview.extent(1); ++j) {
      for (unsigned k = 0; k < _inview.extent(2); ++k) {
        _inview(i, j, k, 0, 0, 0, 0) = double(i) / 2 - j * j + double(k) / 3;
      }
    }
  }
};

// DynRankView functor
template <typename DeviceType>
struct InitDynRankViewFunctor {
  using inviewtype = Kokkos::DynRankView<double, DeviceType>;
  inviewtype _inview;

  InitDynRankViewFunctor(inviewtype &inview_) : _inview(inview_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    for (unsigned j = 0; j < _inview.extent(1); ++j) {
      for (unsigned k = 0; k < _inview.extent(2); ++k) {
        _inview(i, j, k) = double(i) / 2 - j * j + double(k) / 3;
      }
    }
  }

  struct SumComputationTest {
    using inviewtype = Kokkos::DynRankView<double, DeviceType>;
    inviewtype _inview;

    using outviewtype = Kokkos::DynRankView<double, DeviceType>;
    outviewtype _outview;

    KOKKOS_INLINE_FUNCTION
    SumComputationTest(inviewtype &inview_, outviewtype &outview_)
        : _inview(inview_), _outview(outview_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i) const {
      for (unsigned j = 0; j < _inview.extent(1); ++j) {
        for (unsigned k = 0; k < _inview.extent(2); ++k) {
          _outview(i) += _inview(i, j, k);
        }
      }
    }
  };
};

template <typename DeviceType>
void test_dynrankview_op_perf(const int par_size) {
  using execution_space = DeviceType;
  using size_type       = typename execution_space::size_type;
  const size_type dim_2 = 90;
  const size_type dim_3 = 30;

  double elapsed_time_view       = 0;
  double elapsed_time_compview   = 0;
  double elapsed_time_strideview = 0;
  double elapsed_time_view_rank7 = 0;
  double elapsed_time_drview     = 0;
  double elapsed_time_compdrview = 0;
  Kokkos::Timer timer;
  {
    Kokkos::View<double ***, DeviceType> testview("testview", par_size, dim_2,
                                                  dim_3);
    using FunctorType = InitViewFunctor<DeviceType>;

    timer.reset();
    Kokkos::RangePolicy<DeviceType> policy(0, par_size);
    Kokkos::parallel_for(policy, FunctorType(testview));
    DeviceType().fence();
    elapsed_time_view = timer.seconds();
    std::cout << " View time (init only): " << elapsed_time_view << std::endl;

    timer.reset();
    Kokkos::View<double *, DeviceType> sumview("sumview", par_size);
    Kokkos::parallel_for(
        policy, typename FunctorType::SumComputationTest(testview, sumview));
    DeviceType().fence();
    elapsed_time_compview = timer.seconds();
    std::cout << " View sum computation time: " << elapsed_time_view
              << std::endl;

    Kokkos::View<double ***, Kokkos::LayoutStride, DeviceType> teststrideview =
        Kokkos::subview(testview, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    using FunctorStrideType = InitStrideViewFunctor<DeviceType>;

    timer.reset();
    Kokkos::parallel_for(policy, FunctorStrideType(teststrideview));
    DeviceType().fence();
    elapsed_time_strideview = timer.seconds();
    std::cout << " Strided View time (init only): " << elapsed_time_strideview
              << std::endl;
  }
  {
    Kokkos::View<double *******, DeviceType> testview("testview", par_size,
                                                      dim_2, dim_3, 1, 1, 1, 1);
    using FunctorType = InitViewRank7Functor<DeviceType>;

    timer.reset();
    Kokkos::RangePolicy<DeviceType> policy(0, par_size);
    Kokkos::parallel_for(policy, FunctorType(testview));
    DeviceType().fence();
    elapsed_time_view_rank7 = timer.seconds();
    std::cout << " View Rank7 time (init only): " << elapsed_time_view_rank7
              << std::endl;
  }
  {
    Kokkos::DynRankView<double, DeviceType> testdrview("testdrview", par_size,
                                                       dim_2, dim_3);
    using FunctorType = InitDynRankViewFunctor<DeviceType>;

    timer.reset();
    Kokkos::RangePolicy<DeviceType> policy(0, par_size);
    Kokkos::parallel_for(policy, FunctorType(testdrview));
    DeviceType().fence();
    elapsed_time_drview = timer.seconds();
    std::cout << " DynRankView time (init only): " << elapsed_time_drview
              << std::endl;

    timer.reset();
    Kokkos::DynRankView<double, DeviceType> sumview("sumview", par_size);
    Kokkos::parallel_for(
        policy, typename FunctorType::SumComputationTest(testdrview, sumview));
    DeviceType().fence();
    elapsed_time_compdrview = timer.seconds();
    std::cout << " DynRankView sum computation time: "
              << elapsed_time_compdrview << std::endl;
  }

  std::cout << " Ratio of View to DynRankView time: "
            << elapsed_time_view / elapsed_time_drview
            << std::endl;  // expect < 1
  std::cout << " Ratio of View to DynRankView sum computation time: "
            << elapsed_time_compview / elapsed_time_compdrview
            << std::endl;  // expect < 1
  std::cout << " Ratio of View to View Rank7  time: "
            << elapsed_time_view / elapsed_time_view_rank7
            << std::endl;  // expect < 1
  std::cout << " Ratio of StrideView to DynRankView time: "
            << elapsed_time_strideview / elapsed_time_drview
            << std::endl;  // expect < 1
  std::cout << " Ratio of DynRankView to View Rank7  time: "
            << elapsed_time_drview / elapsed_time_view_rank7
            << std::endl;  // expect ?

  timer.reset();

}  // end test_dynrankview

}  // namespace Performance
#endif
