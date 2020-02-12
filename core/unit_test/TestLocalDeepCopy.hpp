/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <gtest/gtest.h>

#include <stdexcept>
#include <sstream>
#include <iostream>
#include <time.h>

#include <Kokkos_Core.hpp>

namespace Test {

template <typename ExecSpace, typename ViewType>
void impl_test_local_deepcopy_teampolicy_rank_1(const int N) {
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

  typedef Kokkos::TeamPolicy<ExecSpace> team_policy;
  typedef typename Kokkos::TeamPolicy<ExecSpace>::member_type member_type;

  // Deep Copy
  Kokkos::parallel_for(
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        int lid = teamMember.league_rank();  // returns a number between 0 and N
        auto subSrc = Kokkos::subview(A, 1, 1, 1, 1, 1, 1, lid, Kokkos::ALL());
        auto subDst = Kokkos::subview(B, 1, 1, 1, 1, 1, 1, lid, Kokkos::ALL());
        Kokkos::Experimental::local_deep_copy(teamMember, subDst, subSrc);
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
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        int lid = teamMember.league_rank();  // returns a number between 0 and N
        auto subDst = Kokkos::subview(B, 1, 1, 1, 1, 1, 1, lid, Kokkos::ALL());
        Kokkos::Experimental::local_deep_copy(teamMember, subDst, 20.0);
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
void impl_test_local_deepcopy_teampolicy_rank_2(const int N) {
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

  typedef Kokkos::TeamPolicy<ExecSpace> team_policy;
  typedef typename Kokkos::TeamPolicy<ExecSpace>::member_type member_type;

  // Deep Copy
  Kokkos::parallel_for(
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        int lid = teamMember.league_rank();  // returns a number between 0 and N
        auto subSrc = Kokkos::subview(A, 1, 1, 1, 1, 1, lid, Kokkos::ALL(),
                                      Kokkos::ALL());
        auto subDst = Kokkos::subview(B, 1, 1, 1, 1, 1, lid, Kokkos::ALL(),
                                      Kokkos::ALL());
        Kokkos::Experimental::local_deep_copy(teamMember, subDst, subSrc);
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
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        int lid = teamMember.league_rank();  // returns a number between 0 and N
        auto subDst = Kokkos::subview(B, 1, 1, 1, 1, 1, lid, Kokkos::ALL(),
                                      Kokkos::ALL());
        Kokkos::Experimental::local_deep_copy(teamMember, subDst, 20.0);
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
void impl_test_local_deepcopy_teampolicy_rank_3(const int N) {
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

  typedef Kokkos::TeamPolicy<ExecSpace> team_policy;
  typedef typename Kokkos::TeamPolicy<ExecSpace>::member_type member_type;

  // Deep Copy
  Kokkos::parallel_for(
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        int lid = teamMember.league_rank();  // returns a number between 0 and N
        auto subSrc = Kokkos::subview(A, 1, 1, 1, 1, lid, Kokkos::ALL(),
                                      Kokkos::ALL(), Kokkos::ALL());
        auto subDst = Kokkos::subview(B, 1, 1, 1, 1, lid, Kokkos::ALL(),
                                      Kokkos::ALL(), Kokkos::ALL());
        Kokkos::Experimental::local_deep_copy(teamMember, subDst, subSrc);
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
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        int lid = teamMember.league_rank();  // returns a number between 0 and N
        auto subDst = Kokkos::subview(B, 1, 1, 1, 1, lid, Kokkos::ALL(),
                                      Kokkos::ALL(), Kokkos::ALL());
        Kokkos::Experimental::local_deep_copy(teamMember, subDst, 20.0);
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
void impl_test_local_deepcopy_teampolicy_rank_4(const int N) {
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

  typedef Kokkos::TeamPolicy<ExecSpace> team_policy;
  typedef typename Kokkos::TeamPolicy<ExecSpace>::member_type member_type;

  // Deep Copy
  Kokkos::parallel_for(
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        int lid = teamMember.league_rank();  // returns a number between 0 and N
        auto subSrc =
            Kokkos::subview(A, 1, 1, 1, lid, Kokkos::ALL(), Kokkos::ALL(),
                            Kokkos::ALL(), Kokkos::ALL());
        auto subDst =
            Kokkos::subview(B, 1, 1, 1, lid, Kokkos::ALL(), Kokkos::ALL(),
                            Kokkos::ALL(), Kokkos::ALL());
        Kokkos::Experimental::local_deep_copy(teamMember, subDst, subSrc);
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
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        int lid = teamMember.league_rank();  // returns a number between 0 and N
        auto subDst =
            Kokkos::subview(B, 1, 1, 1, lid, Kokkos::ALL(), Kokkos::ALL(),
                            Kokkos::ALL(), Kokkos::ALL());
        Kokkos::Experimental::local_deep_copy(teamMember, subDst, 20.0);
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
void impl_test_local_deepcopy_teampolicy_rank_5(const int N) {
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

  typedef Kokkos::TeamPolicy<ExecSpace> team_policy;
  typedef typename Kokkos::TeamPolicy<ExecSpace>::member_type member_type;

  // Deep Copy
  Kokkos::parallel_for(
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        int lid = teamMember.league_rank();  // returns a number between 0 and N
        auto subSrc =
            Kokkos::subview(A, 1, 1, lid, Kokkos::ALL(), Kokkos::ALL(),
                            Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
        auto subDst =
            Kokkos::subview(B, 1, 1, lid, Kokkos::ALL(), Kokkos::ALL(),
                            Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
        Kokkos::Experimental::local_deep_copy(teamMember, subDst, subSrc);
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
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        int lid = teamMember.league_rank();  // returns a number between 0 and N
        auto subDst =
            Kokkos::subview(B, 1, 1, lid, Kokkos::ALL(), Kokkos::ALL(),
                            Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
        Kokkos::Experimental::local_deep_copy(teamMember, subDst, 20.0);
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
void impl_test_local_deepcopy_teampolicy_rank_6(const int N) {
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

  typedef Kokkos::TeamPolicy<ExecSpace> team_policy;
  typedef typename Kokkos::TeamPolicy<ExecSpace>::member_type member_type;

  // Deep Copy
  Kokkos::parallel_for(
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        int lid = teamMember.league_rank();  // returns a number between 0 and N
        auto subSrc = Kokkos::subview(A, 1, lid, Kokkos::ALL(), Kokkos::ALL(),
                                      Kokkos::ALL(), Kokkos::ALL(),
                                      Kokkos::ALL(), Kokkos::ALL());
        auto subDst = Kokkos::subview(B, 1, lid, Kokkos::ALL(), Kokkos::ALL(),
                                      Kokkos::ALL(), Kokkos::ALL(),
                                      Kokkos::ALL(), Kokkos::ALL());
        Kokkos::Experimental::local_deep_copy(teamMember, subDst, subSrc);
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
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        int lid = teamMember.league_rank();  // returns a number between 0 and N
        auto subDst = Kokkos::subview(B, 1, lid, Kokkos::ALL(), Kokkos::ALL(),
                                      Kokkos::ALL(), Kokkos::ALL(),
                                      Kokkos::ALL(), Kokkos::ALL());
        Kokkos::Experimental::local_deep_copy(teamMember, subDst, 20.0);
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
void impl_test_local_deepcopy_teampolicy_rank_7(const int N) {
  // Allocate matrices on device.
  ViewType A("A", N, N, N, N, N, N, N, N);
  ViewType B("B", N, N, N, N, N, N, N, N);

  // Create host mirrors of device views.
  typename ViewType::HostMirror h_A = Kokkos::create_mirror_view(A);
  typename ViewType::HostMirror h_B = Kokkos::create_mirror_view(B);

  // Initialize A matrix.
  Kokkos::deep_copy(A, 10.0);

  typedef Kokkos::TeamPolicy<ExecSpace> team_policy;
  typedef typename Kokkos::TeamPolicy<ExecSpace>::member_type member_type;

  // Deep Copy
  Kokkos::parallel_for(
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        int lid = teamMember.league_rank();  // returns a number between 0 and N
        auto subSrc = Kokkos::subview(
            A, lid, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),
            Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
        auto subDst = Kokkos::subview(
            B, lid, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),
            Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
        Kokkos::Experimental::local_deep_copy(teamMember, subDst, subSrc);
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
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        int lid = teamMember.league_rank();  // returns a number between 0 and N
        auto subDst = Kokkos::subview(
            B, lid, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),
            Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
        Kokkos::Experimental::local_deep_copy(teamMember, subDst, 20.0);
      });

  Kokkos::deep_copy(h_B, B);

  double sum_all = 0.0;
  for (size_t i = 0; i < B.span(); i++) {
    sum_all += h_B.data()[i];
  }

  ASSERT_EQ(sum_all, 20.0 * N * N * N * N * N * N * N * N);
}
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
#if !defined(KOKKOS_ENABLE_CUDA) || (8000 <= CUDA_VERSION)
TEST(TEST_CATEGORY, local_deepcopy_teampolicy_layoutleft) {
  typedef TEST_EXECSPACE ExecSpace;
  typedef Kokkos::View<double********, Kokkos::LayoutLeft, ExecSpace> ViewType;

  {  // Rank-1
    impl_test_local_deepcopy_teampolicy_rank_1<ExecSpace, ViewType>(8);
  }
  {  // Rank-2
    impl_test_local_deepcopy_teampolicy_rank_2<ExecSpace, ViewType>(8);
  }
  {  // Rank-3
    impl_test_local_deepcopy_teampolicy_rank_3<ExecSpace, ViewType>(8);
  }
  {  // Rank-4
    impl_test_local_deepcopy_teampolicy_rank_4<ExecSpace, ViewType>(8);
  }
  {  // Rank-5
    impl_test_local_deepcopy_teampolicy_rank_5<ExecSpace, ViewType>(8);
  }
  {  // Rank-6
    impl_test_local_deepcopy_teampolicy_rank_6<ExecSpace, ViewType>(8);
  }
  {  // Rank-7
    impl_test_local_deepcopy_teampolicy_rank_7<ExecSpace, ViewType>(8);
  }
}
//-------------------------------------------------------------------------------------------------------------
TEST(TEST_CATEGORY, local_deepcopy_rangepolicy_layoutleft) {
  typedef TEST_EXECSPACE ExecSpace;
  typedef Kokkos::View<double********, Kokkos::LayoutLeft, ExecSpace> ViewType;

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
  typedef TEST_EXECSPACE ExecSpace;
  typedef Kokkos::View<double********, Kokkos::LayoutRight, ExecSpace> ViewType;

  {  // Rank-1
    impl_test_local_deepcopy_teampolicy_rank_1<ExecSpace, ViewType>(8);
  }
  {  // Rank-2
    impl_test_local_deepcopy_teampolicy_rank_2<ExecSpace, ViewType>(8);
  }
  {  // Rank-3
    impl_test_local_deepcopy_teampolicy_rank_3<ExecSpace, ViewType>(8);
  }
  {  // Rank-4
    impl_test_local_deepcopy_teampolicy_rank_4<ExecSpace, ViewType>(8);
  }
  {  // Rank-5
    impl_test_local_deepcopy_teampolicy_rank_5<ExecSpace, ViewType>(8);
  }
  {  // Rank-6
    impl_test_local_deepcopy_teampolicy_rank_6<ExecSpace, ViewType>(8);
  }
  {  // Rank-7
    impl_test_local_deepcopy_teampolicy_rank_7<ExecSpace, ViewType>(8);
  }
}
//-------------------------------------------------------------------------------------------------------------
TEST(TEST_CATEGORY, local_deepcopy_rangepolicy_layoutright) {
  typedef TEST_EXECSPACE ExecSpace;
  typedef Kokkos::View<double********, Kokkos::LayoutRight, ExecSpace> ViewType;

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
#endif
}  // namespace Test
