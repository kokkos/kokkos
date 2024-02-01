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

#include "../../algorithms/unit_tests/TestSort.hpp"

#include <gtest/gtest.h>

#include <sstream>
#include <iostream>
#include <time.h>

#include <Kokkos_Core.hpp>

namespace Test {

template <typename ViewType>
bool array_equals(ViewType lhs, ViewType rhs) {
  int result = 1;

  auto reducer = Kokkos::LAnd<int>(result);
  Kokkos::parallel_reduce(
      "compare arrays", lhs.span(),
      KOKKOS_LAMBDA(int i, int local_result) {
        local_result = (lhs.data()[i] == rhs.data()[i]) && local_result;
      },
      reducer);
  return (result);
}

template <typename ViewType>
void array_init(ViewType& view) {
  Kokkos::parallel_for(
      "initialize array", view.span(),
      KOKKOS_LAMBDA(int i) { view.data()[i] = i; });
}

template <typename TeamPolicy>
std::tuple<int, int> compute_thread_work_share(const int N,
                                               TeamPolicy team_policy) {
  auto thread_number = team_policy.league_size();
  auto unitsOfWork   = N / thread_number;
  if (N % thread_number) {
    unitsOfWork += 1;
  }
  auto numberOfBatches = N / unitsOfWork;
  return {unitsOfWork, numberOfBatches};
}

template <typename ViewType>
ViewType create_array(
    std::string label, const int N,
    std::enable_if_t<(unsigned(ViewType::rank) == 1)>* = nullptr) {
  return ViewType(label, N);
}

template <typename ViewType>
ViewType create_array(
    std::string label, const int N,
    std::enable_if_t<(unsigned(ViewType::rank) == 2)>* = nullptr) {
  return ViewType(label, N, N);
}

template <typename ViewType>
ViewType create_array(
    std::string label, const int N,
    std::enable_if_t<(unsigned(ViewType::rank) == 3)>* = nullptr) {
  return ViewType(label, N, N, N);
}

template <typename ViewType>
ViewType create_array(
    std::string label, const int N,
    std::enable_if_t<(unsigned(ViewType::rank) == 4)>* = nullptr) {
  return ViewType(label, N, N, N, N);
}

template <typename ViewType>
ViewType create_array(
    std::string label, const int N,
    std::enable_if_t<(unsigned(ViewType::rank) == 5)>* = nullptr) {
  return ViewType(label, N, N, N, N, N);
}

template <typename ViewType>
ViewType create_array(
    std::string label, const int N,
    std::enable_if_t<(unsigned(ViewType::rank) == 6)>* = nullptr) {
  return ViewType(label, N, N, N, N, N, N);
}

template <typename ViewType>
ViewType create_array(
    std::string label, const int N,
    std::enable_if_t<(unsigned(ViewType::rank) == 7)>* = nullptr) {
  return ViewType(label, N, N, N, N, N, N, N);
}

template <typename ViewType>
ViewType create_array(
    std::string label, const int N,
    std::enable_if_t<(unsigned(ViewType::rank) == 8)>* = nullptr) {
  return ViewType(label, N, N, N, N, N, N, N, N);
}

template <typename ViewType, typename Bounds>
auto extract_subview(
    ViewType& src, int lid, Bounds bounds,
    std::enable_if_t<(unsigned(ViewType::rank) == 2)>* = nullptr) {
  return Kokkos::subview(src, lid, bounds);
}

template <typename ViewType, typename Bounds>
auto extract_subview(
    ViewType& src, int lid, Bounds bounds,
    std::enable_if_t<(unsigned(ViewType::rank) == 3)>* = nullptr) {
  return Kokkos::subview(src, lid, bounds, Kokkos::ALL);
}

template <typename ViewType, typename Bounds>
auto extract_subview(
    ViewType& src, int lid, Bounds bounds,
    std::enable_if_t<(unsigned(ViewType::rank) == 4)>* = nullptr) {
  return Kokkos::subview(src, lid, bounds, Kokkos::ALL, Kokkos::ALL);
}

template <typename ViewType, typename Bounds>
auto extract_subview(
    ViewType& src, int lid, Bounds bounds,
    std::enable_if_t<(unsigned(ViewType::rank) == 5)>* = nullptr) {
  return Kokkos::subview(src, lid, bounds, Kokkos::ALL, Kokkos::ALL,
                         Kokkos::ALL);
}

template <typename ViewType, typename Bounds>
auto extract_subview(
    ViewType& src, int lid, Bounds bounds,
    std::enable_if_t<(unsigned(ViewType::rank) == 6)>* = nullptr) {
  return Kokkos::subview(src, lid, bounds, Kokkos::ALL, Kokkos::ALL,
                         Kokkos::ALL, Kokkos::ALL);
}

template <typename ViewType, typename Bounds>
auto extract_subview(
    ViewType& src, int lid, Bounds bounds,
    std::enable_if_t<(unsigned(ViewType::rank) == 7)>* = nullptr) {
  return Kokkos::subview(src, lid, bounds, Kokkos::ALL, Kokkos::ALL,
                         Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
}

template <typename ViewType, typename Bounds>
auto extract_subview(
    ViewType& src, int lid, Bounds bounds,
    std::enable_if_t<(unsigned(ViewType::rank) == 8)>* = nullptr) {
  return Kokkos::subview(src, lid, bounds, Kokkos::ALL, Kokkos::ALL,
                         Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
}

template <typename ViewType, typename ExecSpace>
class TestLocalDeepCopyRank {
  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

 public:
  TestLocalDeepCopyRank(const int N) : N(N) {
    A = create_array<ViewType>("A", N);
    B = create_array<ViewType>("B", N);

    // Initialize A matrix.
    array_init(A);
  }

  void operator()() {
    test_local_deepcopy_thread();
    reset_b();
    test_local_deepcopy();
    reset_b();
    test_local_deepcopy_scalar();
  }

 private:
  void test_local_deepcopy_thread() {
    // Test local_deep_copy_thread
    // Each thread copies a subview of A into B
    Kokkos::parallel_for(
        team_policy(N, Kokkos::AUTO),
        KOKKOS_LAMBDA(const member_type& teamMember) {
          int lid =
              teamMember.league_rank();  // returns a number between 0 and N
          auto [unitsOfWork, numberOfBatches] =
              compute_thread_work_share(N, teamMember);

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(teamMember, numberOfBatches),
              [=](const int indexWithinBatch) {
                const int idx = indexWithinBatch;

                auto start = idx * unitsOfWork;
                auto stop  = (idx + 1) * unitsOfWork;
                stop       = Kokkos::clamp(stop, 0, N);
                auto subSrc =
                    extract_subview(A, lid, Kokkos::make_pair(start, stop));
                auto subDst =
                    extract_subview(A, lid, Kokkos::make_pair(start, stop));
                Kokkos::Experimental::local_deep_copy_thread(teamMember, subDst,
                                                             subSrc);
                // No wait for local_deep_copy_thread
              });
        });

    ASSERT_TRUE(array_equals(A, B));
  }

  void test_local_deepcopy() {
    // Deep Copy
    Kokkos::parallel_for(
        team_policy(N, Kokkos::AUTO),
        KOKKOS_LAMBDA(const member_type& teamMember) {
          int lid =
              teamMember.league_rank();  // returns a number between 0 and N
          auto subSrc = extract_subview(A, lid, Kokkos::ALL);
          auto subDst = extract_subview(A, lid, Kokkos::ALL);
          Kokkos::Experimental::local_deep_copy(teamMember, subDst, subSrc);
        });

    ASSERT_TRUE(array_equals(A, B));
  }

  void test_local_deepcopy_scalar() {
    Kokkos::parallel_for(
        team_policy(N, Kokkos::AUTO),
        KOKKOS_LAMBDA(const member_type& teamMember) {
          int lid =
              teamMember.league_rank();  // returns a number between 0 and N
          auto subDst = extract_subview(B, lid, Kokkos::ALL);
          Kokkos::Experimental::local_deep_copy(teamMember, subDst, 20.0);
        });

    double sum_all = 0.0;
    Kokkos::parallel_reduce(
        "Check B", B.span(),
        KOKKOS_LAMBDA(int i, double& lsum) { lsum += B.data()[i]; },
        Kokkos::Sum<double>(sum_all));

    auto correct_sum = 20.0;
    for (auto i = 0; i < ViewType::rank; i++) {
      correct_sum *= N;
    }

    ASSERT_EQ(sum_all, correct_sum);
  }

  void reset_b() { Kokkos::deep_copy(B, 0.0); }

  int N;
  ViewType A;
  ViewType B;
};

//-------------------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename ViewType>
void impl_test_local_deepcopy_rangepolicy_rank_1(const int N) {
  // Allocate matrices on device.
  ViewType A("A", N, N, N, N, N, N, N, N);
  ViewType B("B", N, N, N, N, N, N, N, N);

  // Create host mirrors of device views.
  typename ViewType::HostMirror h_A = Kokkos::create_mirror_view(A);
  typename ViewType::HostMirror h_B = Kokkos::create_mirror_view(B);

  // Initialize A matrix.
  auto subA =
      Kokkos::subview(A, 1, 1, 1, 1, 1, 1, Kokkos::ALL(), Kokkos::ALL());
  Kokkos::deep_copy(subA, 10.0);

  // Deep Copy
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(const int& i) {
        auto subSrc = Kokkos::subview(A, 1, 1, 1, 1, 1, 1, i, Kokkos::ALL());
        auto subDst = Kokkos::subview(B, 1, 1, 1, 1, 1, 1, i, Kokkos::ALL());
        Kokkos::Experimental::local_deep_copy(subDst, subSrc);
      });

  Kokkos::deep_copy(h_A, A);
  Kokkos::deep_copy(h_B, B);

  bool test = true;
  for (size_t i = 0; i < A.span(); i++) {
    if (h_A.data()[i] != h_B.data()[i]) {
      test = false;
      break;
    }
  }

  ASSERT_EQ(test, true);

  // Fill
  Kokkos::deep_copy(B, 0.0);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(const int& i) {
        auto subDst = Kokkos::subview(B, 1, 1, 1, 1, 1, 1, i, Kokkos::ALL());
        Kokkos::Experimental::local_deep_copy(subDst, 20.0);
      });

  Kokkos::deep_copy(h_B, B);

  double sum_all = 0.0;
  for (size_t i = 0; i < B.span(); i++) {
    sum_all += h_B.data()[i];
  }

  ASSERT_EQ(sum_all, 20.0 * N * N);
}
//-------------------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename ViewType>
void impl_test_local_deepcopy_rangepolicy_rank_2(const int N) {
  // Allocate matrices on device.
  ViewType A("A", N, N, N, N, N, N, N, N);
  ViewType B("B", N, N, N, N, N, N, N, N);

  // Create host mirrors of device views.
  typename ViewType::HostMirror h_A = Kokkos::create_mirror_view(A);
  typename ViewType::HostMirror h_B = Kokkos::create_mirror_view(B);

  // Initialize A matrix.
  auto subA = Kokkos::subview(A, 1, 1, 1, 1, 1, Kokkos::ALL(), Kokkos::ALL(),
                              Kokkos::ALL());
  Kokkos::deep_copy(subA, 10.0);

  // Deep Copy
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(const int& i) {
        auto subSrc =
            Kokkos::subview(A, 1, 1, 1, 1, 1, i, Kokkos::ALL(), Kokkos::ALL());
        auto subDst =
            Kokkos::subview(B, 1, 1, 1, 1, 1, i, Kokkos::ALL(), Kokkos::ALL());
        Kokkos::Experimental::local_deep_copy(subDst, subSrc);
      });

  Kokkos::deep_copy(h_A, A);
  Kokkos::deep_copy(h_B, B);

  bool test = true;
  for (size_t i = 0; i < A.span(); i++) {
    if (h_A.data()[i] != h_B.data()[i]) {
      test = false;
      break;
    }
  }

  ASSERT_EQ(test, true);

  // Fill
  Kokkos::deep_copy(B, 0.0);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(const int& i) {
        auto subDst =
            Kokkos::subview(B, 1, 1, 1, 1, 1, i, Kokkos::ALL(), Kokkos::ALL());
        Kokkos::Experimental::local_deep_copy(subDst, 20.0);
      });

  Kokkos::deep_copy(h_B, B);

  double sum_all = 0.0;
  for (size_t i = 0; i < B.span(); i++) {
    sum_all += h_B.data()[i];
  }

  ASSERT_EQ(sum_all, 20.0 * N * N * N);
}
//-------------------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename ViewType>
void impl_test_local_deepcopy_rangepolicy_rank_3(const int N) {
  // Allocate matrices on device.
  ViewType A("A", N, N, N, N, N, N, N, N);
  ViewType B("B", N, N, N, N, N, N, N, N);

  // Create host mirrors of device views.
  typename ViewType::HostMirror h_A = Kokkos::create_mirror_view(A);
  typename ViewType::HostMirror h_B = Kokkos::create_mirror_view(B);

  // Initialize A matrix.
  auto subA = Kokkos::subview(A, 1, 1, 1, 1, Kokkos::ALL(), Kokkos::ALL(),
                              Kokkos::ALL(), Kokkos::ALL());
  Kokkos::deep_copy(subA, 10.0);

  // Deep Copy
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(const int& i) {
        auto subSrc = Kokkos::subview(A, 1, 1, 1, 1, i, Kokkos::ALL(),
                                      Kokkos::ALL(), Kokkos::ALL());
        auto subDst = Kokkos::subview(B, 1, 1, 1, 1, i, Kokkos::ALL(),
                                      Kokkos::ALL(), Kokkos::ALL());
        Kokkos::Experimental::local_deep_copy(subDst, subSrc);
      });

  Kokkos::deep_copy(h_A, A);
  Kokkos::deep_copy(h_B, B);

  bool test = true;
  for (size_t i = 0; i < A.span(); i++) {
    if (h_A.data()[i] != h_B.data()[i]) {
      test = false;
      break;
    }
  }

  ASSERT_EQ(test, true);

  // Fill
  Kokkos::deep_copy(B, 0.0);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(const int& i) {
        auto subDst = Kokkos::subview(B, 1, 1, 1, 1, i, Kokkos::ALL(),
                                      Kokkos::ALL(), Kokkos::ALL());
        Kokkos::Experimental::local_deep_copy(subDst, 20.0);
      });

  Kokkos::deep_copy(h_B, B);

  double sum_all = 0.0;
  for (size_t i = 0; i < B.span(); i++) {
    sum_all += h_B.data()[i];
  }

  ASSERT_EQ(sum_all, 20.0 * N * N * N * N);
}
//-------------------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename ViewType>
void impl_test_local_deepcopy_rangepolicy_rank_4(const int N) {
  // Allocate matrices on device.
  ViewType A("A", N, N, N, N, N, N, N, N);
  ViewType B("B", N, N, N, N, N, N, N, N);

  // Create host mirrors of device views.
  typename ViewType::HostMirror h_A = Kokkos::create_mirror_view(A);
  typename ViewType::HostMirror h_B = Kokkos::create_mirror_view(B);

  // Initialize A matrix.
  auto subA = Kokkos::subview(A, 1, 1, 1, Kokkos::ALL(), Kokkos::ALL(),
                              Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
  Kokkos::deep_copy(subA, 10.0);

  // Deep Copy
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(const int& i) {
        auto subSrc =
            Kokkos::subview(A, 1, 1, 1, i, Kokkos::ALL(), Kokkos::ALL(),
                            Kokkos::ALL(), Kokkos::ALL());
        auto subDst =
            Kokkos::subview(B, 1, 1, 1, i, Kokkos::ALL(), Kokkos::ALL(),
                            Kokkos::ALL(), Kokkos::ALL());
        Kokkos::Experimental::local_deep_copy(subDst, subSrc);
      });

  Kokkos::deep_copy(h_A, A);
  Kokkos::deep_copy(h_B, B);

  bool test = true;
  for (size_t i = 0; i < A.span(); i++) {
    if (h_A.data()[i] != h_B.data()[i]) {
      test = false;
      break;
    }
  }

  ASSERT_EQ(test, true);

  // Fill
  Kokkos::deep_copy(B, 0.0);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(const int& i) {
        auto subDst =
            Kokkos::subview(B, 1, 1, 1, i, Kokkos::ALL(), Kokkos::ALL(),
                            Kokkos::ALL(), Kokkos::ALL());
        Kokkos::Experimental::local_deep_copy(subDst, 20.0);
      });

  Kokkos::deep_copy(h_B, B);

  double sum_all = 0.0;
  for (size_t i = 0; i < B.span(); i++) {
    sum_all += h_B.data()[i];
  }

  ASSERT_EQ(sum_all, 20.0 * N * N * N * N * N);
}
//-------------------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename ViewType>
void impl_test_local_deepcopy_rangepolicy_rank_5(const int N) {
  // Allocate matrices on device.
  ViewType A("A", N, N, N, N, N, N, N, N);
  ViewType B("B", N, N, N, N, N, N, N, N);

  // Create host mirrors of device views.
  typename ViewType::HostMirror h_A = Kokkos::create_mirror_view(A);
  typename ViewType::HostMirror h_B = Kokkos::create_mirror_view(B);

  // Initialize A matrix.
  auto subA =
      Kokkos::subview(A, 1, 1, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),
                      Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
  Kokkos::deep_copy(subA, 10.0);

  // Deep Copy
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(const int& i) {
        auto subSrc =
            Kokkos::subview(A, 1, 1, i, Kokkos::ALL(), Kokkos::ALL(),
                            Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
        auto subDst =
            Kokkos::subview(B, 1, 1, i, Kokkos::ALL(), Kokkos::ALL(),
                            Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
        Kokkos::Experimental::local_deep_copy(subDst, subSrc);
      });

  Kokkos::deep_copy(h_A, A);
  Kokkos::deep_copy(h_B, B);

  bool test = true;
  for (size_t i = 0; i < A.span(); i++) {
    if (h_A.data()[i] != h_B.data()[i]) {
      test = false;
      break;
    }
  }

  ASSERT_EQ(test, true);

  // Fill
  Kokkos::deep_copy(B, 0.0);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(const int& i) {
        auto subDst =
            Kokkos::subview(B, 1, 1, i, Kokkos::ALL(), Kokkos::ALL(),
                            Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
        Kokkos::Experimental::local_deep_copy(subDst, 20.0);
      });

  Kokkos::deep_copy(h_B, B);

  double sum_all = 0.0;
  for (size_t i = 0; i < B.span(); i++) {
    sum_all += h_B.data()[i];
  }

  ASSERT_EQ(sum_all, 20.0 * N * N * N * N * N * N);
}
//-------------------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename ViewType>
void impl_test_local_deepcopy_rangepolicy_rank_6(const int N) {
  // Allocate matrices on device.
  ViewType A("A", N, N, N, N, N, N, N, N);
  ViewType B("B", N, N, N, N, N, N, N, N);

  // Create host mirrors of device views.
  typename ViewType::HostMirror h_A = Kokkos::create_mirror_view(A);
  typename ViewType::HostMirror h_B = Kokkos::create_mirror_view(B);

  // Initialize A matrix.
  auto subA = Kokkos::subview(A, 1, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),
                              Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),
                              Kokkos::ALL());
  Kokkos::deep_copy(subA, 10.0);

  // Deep Copy
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(const int& i) {
        auto subSrc = Kokkos::subview(A, 1, i, Kokkos::ALL(), Kokkos::ALL(),
                                      Kokkos::ALL(), Kokkos::ALL(),
                                      Kokkos::ALL(), Kokkos::ALL());
        auto subDst = Kokkos::subview(B, 1, i, Kokkos::ALL(), Kokkos::ALL(),
                                      Kokkos::ALL(), Kokkos::ALL(),
                                      Kokkos::ALL(), Kokkos::ALL());
        Kokkos::Experimental::local_deep_copy(subDst, subSrc);
      });

  Kokkos::deep_copy(h_A, A);
  Kokkos::deep_copy(h_B, B);

  bool test = true;
  for (size_t i = 0; i < A.span(); i++) {
    if (h_A.data()[i] != h_B.data()[i]) {
      test = false;
      break;
    }
  }

  ASSERT_EQ(test, true);

  // Fill
  Kokkos::deep_copy(B, 0.0);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(const int& i) {
        auto subDst = Kokkos::subview(B, 1, i, Kokkos::ALL(), Kokkos::ALL(),
                                      Kokkos::ALL(), Kokkos::ALL(),
                                      Kokkos::ALL(), Kokkos::ALL());
        Kokkos::Experimental::local_deep_copy(subDst, 20.0);
      });

  Kokkos::deep_copy(h_B, B);

  double sum_all = 0.0;
  for (size_t i = 0; i < B.span(); i++) {
    sum_all += h_B.data()[i];
  }

  ASSERT_EQ(sum_all, 20.0 * N * N * N * N * N * N * N);
}
//-------------------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename ViewType>
void impl_test_local_deepcopy_rangepolicy_rank_7(const int N) {
  // Allocate matrices on device.
  ViewType A("A", N, N, N, N, N, N, N, N);
  ViewType B("B", N, N, N, N, N, N, N, N);

  // Create host mirrors of device views.
  typename ViewType::HostMirror h_A = Kokkos::create_mirror_view(A);
  typename ViewType::HostMirror h_B = Kokkos::create_mirror_view(B);

  // Initialize A matrix.
  Kokkos::deep_copy(A, 10.0);

  // Deep Copy
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(const int& i) {
        auto subSrc = Kokkos::subview(
            A, i, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),
            Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
        auto subDst = Kokkos::subview(
            B, i, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),
            Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
        Kokkos::Experimental::local_deep_copy(subDst, subSrc);
      });

  Kokkos::deep_copy(h_A, A);
  Kokkos::deep_copy(h_B, B);

  bool test = true;
  for (size_t i = 0; i < A.span(); i++) {
    if (h_A.data()[i] != h_B.data()[i]) {
      test = false;
      break;
    }
  }

  ASSERT_EQ(test, true);

  // Fill
  Kokkos::deep_copy(B, 0.0);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(const int& i) {
        auto subDst = Kokkos::subview(
            B, i, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),
            Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
        Kokkos::Experimental::local_deep_copy(subDst, 20.0);
      });

  Kokkos::deep_copy(h_B, B);

  double sum_all = 0.0;
  for (size_t i = 0; i < B.span(); i++) {
    sum_all += h_B.data()[i];
  }

  ASSERT_EQ(sum_all, 20.0 * N * N * N * N * N * N * N * N);
}
//-------------------------------------------------------------------------------------------------------------

#if defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)
TEST(TEST_CATEGORY, local_deepcopy_teampolicy_layoutleft) {
  using ExecSpace = TEST_EXECSPACE;
#if defined(KOKKOS_ENABLE_CUDA) && \
    defined(KOKKOS_COMPILER_NVHPC)  // FIXME_NVHPC 23.7
  if (std::is_same_v<ExecSpace, Kokkos::Cuda>)
    GTEST_SKIP()
        << "FIXME_NVHPC : Compiler bug affecting subviews of high rank Views";
#endif
  using Layout = Kokkos::LayoutRight;

  {  // Rank-1
    auto test = TestLocalDeepCopyRank<Kokkos::View<double**, Layout, ExecSpace>,
                                      ExecSpace>(8);
    test();
  }
  {  // Rank-2
    auto test =
        TestLocalDeepCopyRank<Kokkos::View<double***, Layout, ExecSpace>,
                              ExecSpace>(8);
    test();
  }
  {  // Rank-3
    auto test =
        TestLocalDeepCopyRank<Kokkos::View<double****, Layout, ExecSpace>,
                              ExecSpace>(8);
    test();
  }
  {  // Rank-4
    auto test =
        TestLocalDeepCopyRank<Kokkos::View<double*****, Layout, ExecSpace>,
                              ExecSpace>(8);
    test();
  }
  {  // Rank-5
    auto test =
        TestLocalDeepCopyRank<Kokkos::View<double******, Layout, ExecSpace>,
                              ExecSpace>(8);
    test();
  }
  {  // Rank-6
    auto test =
        TestLocalDeepCopyRank<Kokkos::View<double*******, Layout, ExecSpace>,
                              ExecSpace>(8);
    test();
  }
  {  // Rank-7
    auto test =
        TestLocalDeepCopyRank<Kokkos::View<double********, Layout, ExecSpace>,
                              ExecSpace>(8);
    test();
  }
}
//-------------------------------------------------------------------------------------------------------------
TEST(TEST_CATEGORY, local_deepcopy_rangepolicy_layoutleft) {
  using ExecSpace = TEST_EXECSPACE;
#if defined(KOKKOS_ENABLE_CUDA) && \
    defined(KOKKOS_COMPILER_NVHPC)  // FIXME_NVHPC 23.7
  if (std::is_same_v<ExecSpace, Kokkos::Cuda>)
    GTEST_SKIP()
        << "FIXME_NVHPC : Compiler bug affecting subviews of high rank Views";
#endif
  using ViewType = Kokkos::View<double********, Kokkos::LayoutLeft, ExecSpace>;

  {  // Rank-1
    impl_test_local_deepcopy_rangepolicy_rank_1<ExecSpace, ViewType>(8);
  }
  {  // Rank-2
    impl_test_local_deepcopy_rangepolicy_rank_2<ExecSpace, ViewType>(8);
  }
  {  // Rank-3
    impl_test_local_deepcopy_rangepolicy_rank_3<ExecSpace, ViewType>(8);
  }
  {  // Rank-4
    impl_test_local_deepcopy_rangepolicy_rank_4<ExecSpace, ViewType>(8);
  }
  {  // Rank-5
    impl_test_local_deepcopy_rangepolicy_rank_5<ExecSpace, ViewType>(8);
  }
  {  // Rank-6
    impl_test_local_deepcopy_rangepolicy_rank_6<ExecSpace, ViewType>(8);
  }
  {  // Rank-7
    impl_test_local_deepcopy_rangepolicy_rank_7<ExecSpace, ViewType>(8);
  }
}
//-------------------------------------------------------------------------------------------------------------
TEST(TEST_CATEGORY, local_deepcopy_teampolicy_layoutright) {
  using ExecSpace = TEST_EXECSPACE;
#if defined(KOKKOS_ENABLE_CUDA) && \
    defined(KOKKOS_COMPILER_NVHPC)  // FIXME_NVHPC 23.7
  if (std::is_same_v<ExecSpace, Kokkos::Cuda>)
    GTEST_SKIP()
        << "FIXME_NVHPC : Compiler bug affecting subviews of high rank Views";
#endif
  using Layout = Kokkos::LayoutRight;

  {  // Rank-1
    auto test = TestLocalDeepCopyRank<Kokkos::View<double**, Layout, ExecSpace>,
                                      ExecSpace>(8);
    test();
  }
  {  // Rank-2
    auto test =
        TestLocalDeepCopyRank<Kokkos::View<double***, Layout, ExecSpace>,
                              ExecSpace>(8);
    test();
  }
  {  // Rank-3
    auto test =
        TestLocalDeepCopyRank<Kokkos::View<double****, Layout, ExecSpace>,
                              ExecSpace>(8);
    test();
  }
  {  // Rank-4
    auto test =
        TestLocalDeepCopyRank<Kokkos::View<double*****, Layout, ExecSpace>,
                              ExecSpace>(8);
    test();
  }
  {  // Rank-5
    auto test =
        TestLocalDeepCopyRank<Kokkos::View<double******, Layout, ExecSpace>,
                              ExecSpace>(8);
    test();
  }
  {  // Rank-6
    auto test =
        TestLocalDeepCopyRank<Kokkos::View<double*******, Layout, ExecSpace>,
                              ExecSpace>(8);
    test();
  }
  {  // Rank-7
    auto test =
        TestLocalDeepCopyRank<Kokkos::View<double********, Layout, ExecSpace>,
                              ExecSpace>(8);
    test();
  }
}
//-------------------------------------------------------------------------------------------------------------
TEST(TEST_CATEGORY, local_deepcopy_rangepolicy_layoutright) {
  using ExecSpace = TEST_EXECSPACE;
#if defined(KOKKOS_ENABLE_CUDA) && \
    defined(KOKKOS_COMPILER_NVHPC)  // FIXME_NVHPC 23.7
  if (std::is_same_v<ExecSpace, Kokkos::Cuda>)
    GTEST_SKIP()
        << "FIXME_NVHPC : Compiler bug affecting subviews of high rank Views";
#endif

  using ViewType = Kokkos::View<double********, Kokkos::LayoutRight, ExecSpace>;

  {  // Rank-1
    impl_test_local_deepcopy_rangepolicy_rank_1<ExecSpace, ViewType>(8);
  }
  {  // Rank-2
    impl_test_local_deepcopy_rangepolicy_rank_2<ExecSpace, ViewType>(8);
  }
  {  // Rank-3
    impl_test_local_deepcopy_rangepolicy_rank_3<ExecSpace, ViewType>(8);
  }
  {  // Rank-4
    impl_test_local_deepcopy_rangepolicy_rank_4<ExecSpace, ViewType>(8);
  }
  {  // Rank-5
    impl_test_local_deepcopy_rangepolicy_rank_5<ExecSpace, ViewType>(8);
  }
  {  // Rank-6
    impl_test_local_deepcopy_rangepolicy_rank_6<ExecSpace, ViewType>(8);
  }
  {  // Rank-7
    impl_test_local_deepcopy_rangepolicy_rank_7<ExecSpace, ViewType>(8);
  }
}
#endif

namespace Impl {
template <typename T, typename SHMEMTYPE>
using ShMemView =
    Kokkos::View<T, Kokkos::LayoutRight, SHMEMTYPE, Kokkos::MemoryUnmanaged>;

struct DeepCopyScratchFunctor {
  DeepCopyScratchFunctor(
      Kokkos::View<double*, TEST_EXECSPACE::memory_space> check_view_1,
      Kokkos::View<double*, TEST_EXECSPACE::memory_space> check_view_2)
      : check_view_1_(check_view_1),
        check_view_2_(check_view_2),
        N_(check_view_1.extent(0)) {}

  KOKKOS_INLINE_FUNCTION void operator()(
      Kokkos::TeamPolicy<TEST_EXECSPACE,
                         Kokkos::Schedule<Kokkos::Dynamic>>::member_type team)
      const {
    using ShmemType = TEST_EXECSPACE::scratch_memory_space;
    auto shview =
        Impl::ShMemView<double**, ShmemType>(team.team_scratch(1), N_, 1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, N_), KOKKOS_LAMBDA(const size_t& index) {
          auto thread_shview = Kokkos::subview(shview, index, Kokkos::ALL());
          Kokkos::Experimental::local_deep_copy(thread_shview, index);
        });
    Kokkos::Experimental::local_deep_copy(
        team, check_view_1_, Kokkos::subview(shview, Kokkos::ALL(), 0));

    Kokkos::Experimental::local_deep_copy(team, shview, 6.);
    Kokkos::Experimental::local_deep_copy(
        team, check_view_2_, Kokkos::subview(shview, Kokkos::ALL(), 0));
  }

  Kokkos::View<double*, TEST_EXECSPACE::memory_space> check_view_1_;
  Kokkos::View<double*, TEST_EXECSPACE::memory_space> check_view_2_;
  int const N_;
};
}  // namespace Impl

TEST(TEST_CATEGORY, deep_copy_scratch) {
  using TestDeviceTeamPolicy = Kokkos::TeamPolicy<TEST_EXECSPACE>;

  const int N = 8;
  const int bytes_per_team =
      Impl::ShMemView<double**,
                      TEST_EXECSPACE::scratch_memory_space>::shmem_size(N, 1);

  TestDeviceTeamPolicy policy(1, Kokkos::AUTO);
  auto team_exec = policy.set_scratch_size(1, Kokkos::PerTeam(bytes_per_team));

  Kokkos::View<double*, TEST_EXECSPACE::memory_space> check_view_1("check_1",
                                                                   N);
  Kokkos::View<double*, TEST_EXECSPACE::memory_space> check_view_2("check_2",
                                                                   N);

  Kokkos::parallel_for(
      team_exec, Impl::DeepCopyScratchFunctor{check_view_1, check_view_2});
  auto host_copy_1 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), check_view_1);
  auto host_copy_2 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), check_view_2);

  for (unsigned int i = 0; i < N; ++i) {
    ASSERT_EQ(host_copy_1(i), i);
    ASSERT_EQ(host_copy_2(i), 6.0);
  }
}
}  // namespace Test
