// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>

#include <sstream>
#include <iostream>
#include <time.h>

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

namespace Test {

template <typename ExecSpace, typename ViewType>
void impl_test_local_deepcopy_thread_rank_0(const int N) {
  // Allocate matrices on device.
  ViewType A("A", N, N);
  ViewType B("B", N, N);

  // Initialize matrix A
  Kokkos::parallel_for(
      "Init_rank0",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1) {
        A(i0, i1) = 1.0 + i0 + i1 * N;
      });

  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

  // Deep Copy
  Kokkos::parallel_for(
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        int lid = teamMember.league_rank();  // returns a number between 0 and N
        auto subSrc = Kokkos::subview(A, lid, Kokkos::ALL());
        auto subDst = Kokkos::subview(B, lid, Kokkos::ALL());
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(teamMember, N), [=](const int i) {
              auto subSrcThread = Kokkos::subview(subSrc, i);
              auto subDstThread = Kokkos::subview(subDst, i);
              Kokkos::Experimental::deep_copy(
                  Kokkos::Experimental::copy_thread(teamMember), subDstThread,
                  subSrcThread);
            });
      });

  // Verify results
  int errors = 0;
  Kokkos::parallel_reduce(
      "Verify_rank0",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, int& err) {
        double expected = 1.0 + i0 + i1 * N;
        if (B(i0, i1) != expected) {
          err++;
        }
      },
      errors);

  ASSERT_EQ(errors, 0);
}
//-------------------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename ViewType>
void impl_test_local_deepcopy_thread_rank_1(const int N) {
  // Allocate matrices on device.
  ViewType A("A", N, N);
  ViewType B("B", N, N);

  // Initialize matrix A
  Kokkos::parallel_for(
      "Init_rank1",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1) {
        A(i0, i1) = 1.0 + i0 + i1 * N;
      });

  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

  // Deep Copy
  Kokkos::parallel_for(
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        int lid = teamMember.league_rank();  // returns a number between 0 and N
        auto subSrc = Kokkos::subview(A, lid, Kokkos::ALL());
        auto subDst = Kokkos::subview(B, lid, Kokkos::ALL());
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(teamMember, N), [=](const int i) {
              auto subSrcThread =
                  Kokkos::subview(subSrc, Kokkos::make_pair(i, i + 1));
              auto subDstThread =
                  Kokkos::subview(subDst, Kokkos::make_pair(i, i + 1));
              Kokkos::Experimental::deep_copy(
                  Kokkos::Experimental::copy_thread(teamMember), subDstThread,
                  subSrcThread);
            });
      });

  // Verify results
  int errors = 0;
  Kokkos::parallel_reduce(
      "Verify_rank1",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, int& err) {
        double expected = 1.0 + i0 + i1 * N;
        if (B(i0, i1) != expected) {
          err++;
        }
      },
      errors);

  ASSERT_EQ(errors, 0);
}
//-------------------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename ViewType>
void impl_test_local_deepcopy_thread_rank_2(const int N) {
  // Allocate matrices on device.
  ViewType A("A", N, N, N);
  ViewType B("B", N, N, N);

  // Initialize matrix A
  Kokkos::parallel_for(
      "Init_rank2",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {N, N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, const int i2) {
        A(i0, i1, i2) = 1.0 + i0 + i1 * N + i2 * N * N;
      });

  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

  // Deep Copy
  Kokkos::parallel_for(
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        int lid = teamMember.league_rank();  // returns a number between 0 and N
        auto subSrc = Kokkos::subview(A, lid, Kokkos::ALL(), Kokkos::ALL());
        auto subDst = Kokkos::subview(B, lid, Kokkos::ALL(), Kokkos::ALL());
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(teamMember, N), [=](const int i) {
              auto subSrcThread = Kokkos::subview(
                  subSrc, Kokkos::make_pair(i, i + 1), Kokkos::ALL());
              auto subDstThread = Kokkos::subview(
                  subDst, Kokkos::make_pair(i, i + 1), Kokkos::ALL());
              Kokkos::Experimental::deep_copy(
                  Kokkos::Experimental::copy_thread(teamMember), subDstThread,
                  subSrcThread);
            });
      });

  // Verify results
  int errors = 0;
  Kokkos::parallel_reduce(
      "Verify_rank2",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {N, N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, const int i2, int& err) {
        double expected = 1.0 + i0 + i1 * N + i2 * N * N;
        if (B(i0, i1, i2) != expected) {
          err++;
        }
      },
      errors);

  ASSERT_EQ(errors, 0);
}
//-------------------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename ViewType>
void impl_test_local_deepcopy_thread_rank_3(const int N) {
  // Allocate matrices on device.
  ViewType A("A", N, N, N, N);
  ViewType B("B", N, N, N, N);

  // Initialize matrix A
  Kokkos::parallel_for(
      "Init_rank3",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>({0, 0, 0, 0},
                                                        {N, N, N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, const int i2, const int i3) {
        A(i0, i1, i2, i3) = 1.0 + i0 + i1 * N + i2 * N * N + i3 * N * N * N;
      });

  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

  // Deep Copy
  Kokkos::parallel_for(
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        int lid = teamMember.league_rank();  // returns a number between 0 and N
        auto subSrc = Kokkos::subview(A, lid, Kokkos::ALL(), Kokkos::ALL(),
                                      Kokkos::ALL());
        auto subDst = Kokkos::subview(B, lid, Kokkos::ALL(), Kokkos::ALL(),
                                      Kokkos::ALL());
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(teamMember, N), [=](const int i) {
              auto subSrcThread =
                  Kokkos::subview(subSrc, Kokkos::make_pair(i, i + 1),
                                  Kokkos::ALL(), Kokkos::ALL());
              auto subDstThread =
                  Kokkos::subview(subDst, Kokkos::make_pair(i, i + 1),
                                  Kokkos::ALL(), Kokkos::ALL());
              Kokkos::Experimental::deep_copy(
                  Kokkos::Experimental::copy_thread(teamMember), subDstThread,
                  subSrcThread);
            });
      });

  // Verify results
  int errors = 0;
  Kokkos::parallel_reduce(
      "Verify_rank3",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>({0, 0, 0, 0},
                                                        {N, N, N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, const int i2, const int i3,
                    int& err) {
        double expected = 1.0 + i0 + i1 * N + i2 * N * N + i3 * N * N * N;
        if (B(i0, i1, i2, i3) != expected) {
          err++;
        }
      },
      errors);

  ASSERT_EQ(errors, 0);
}
//-------------------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename ViewType>
void impl_test_local_deepcopy_thread_rank_4(const int N) {
  // Allocate matrices on device.
  ViewType A("A", N, N, N, N, N);
  ViewType B("B", N, N, N, N, N);

  // Initialize matrix A
  Kokkos::parallel_for(
      "Init_rank4",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<5>>({0, 0, 0, 0, 0},
                                                        {N, N, N, N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, const int i2, const int i3,
                    const int i4) {
        A(i0, i1, i2, i3, i4) = 1.0 + i0 + i1 * N + i2 * N * N +
                                i3 * N * N * N + i4 * N * N * N * N;
      });

  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

  // Deep Copy
  Kokkos::parallel_for(
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        int lid = teamMember.league_rank();  // returns a number between 0 and N
        auto subSrc = Kokkos::subview(A, lid, Kokkos::ALL(), Kokkos::ALL(),
                                      Kokkos::ALL(), Kokkos::ALL());
        auto subDst = Kokkos::subview(B, lid, Kokkos::ALL(), Kokkos::ALL(),
                                      Kokkos::ALL(), Kokkos::ALL());
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(teamMember, N), [=](const int i) {
              auto subSrcThread =
                  Kokkos::subview(subSrc, Kokkos::make_pair(i, i + 1),
                                  Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
              auto subDstThread =
                  Kokkos::subview(subDst, Kokkos::make_pair(i, i + 1),
                                  Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
              Kokkos::Experimental::deep_copy(
                  Kokkos::Experimental::copy_thread(teamMember), subDstThread,
                  subSrcThread);
            });
      });

  // Verify results
  int errors = 0;
  Kokkos::parallel_reduce(
      "Verify_rank4",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<5>>({0, 0, 0, 0, 0},
                                                        {N, N, N, N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, const int i2, const int i3,
                    const int i4, int& err) {
        double expected = 1.0 + i0 + i1 * N + i2 * N * N + i3 * N * N * N +
                          i4 * N * N * N * N;
        if (B(i0, i1, i2, i3, i4) != expected) {
          err++;
        }
      },
      errors);

  ASSERT_EQ(errors, 0);
}
//-------------------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename ViewType>
void impl_test_local_deepcopy_thread_rank_5(const int N) {
  // Allocate matrices on device.
  ViewType A("A", N, N, N, N, N, N);
  ViewType B("B", N, N, N, N, N, N);

  // Initialize matrix A
  Kokkos::parallel_for(
      "Init_rank5",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<6>>({0, 0, 0, 0, 0, 0},
                                                        {N, N, N, N, N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, const int i2, const int i3,
                    const int i4, const int i5) {
        A(i0, i1, i2, i3, i4, i5) = 1.0 + i0 + i1 * N + i2 * N * N +
                                    i3 * N * N * N + i4 * N * N * N * N +
                                    i5 * N * N * N * N * N;
      });

  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

  // Deep Copy
  Kokkos::parallel_for(
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        int lid = teamMember.league_rank();  // returns a number between 0 and N
        auto subSrc =
            Kokkos::subview(A, lid, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),
                            Kokkos::ALL(), Kokkos::ALL());
        auto subDst =
            Kokkos::subview(B, lid, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),
                            Kokkos::ALL(), Kokkos::ALL());
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(teamMember, N), [=](const int i) {
              auto subSrcThread = Kokkos::subview(
                  subSrc, Kokkos::make_pair(i, i + 1), Kokkos::ALL(),
                  Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
              auto subDstThread = Kokkos::subview(
                  subDst, Kokkos::make_pair(i, i + 1), Kokkos::ALL(),
                  Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
              Kokkos::Experimental::deep_copy(
                  Kokkos::Experimental::copy_thread(teamMember), subDstThread,
                  subSrcThread);
            });
      });

  // Verify results
  int errors = 0;
  Kokkos::parallel_reduce(
      "Verify_rank5",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<6>>({0, 0, 0, 0, 0, 0},
                                                        {N, N, N, N, N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, const int i2, const int i3,
                    const int i4, const int i5, int& err) {
        double expected = 1.0 + i0 + i1 * N + i2 * N * N + i3 * N * N * N +
                          i4 * N * N * N * N + i5 * N * N * N * N * N;
        if (B(i0, i1, i2, i3, i4, i5) != expected) {
          err++;
        }
      },
      errors);

  ASSERT_EQ(errors, 0);
}
//-------------------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename ViewType>
void impl_test_local_deepcopy_thread_rank_6(const int N) {
  // Allocate matrices on device.
  ViewType A("A", N, N, N, N, N, N, N);
  ViewType B("B", N, N, N, N, N, N, N);

  // Initialize matrix A using MDRangePolicy for outer
  // 6 dimensions and nested loop for inner dimension.
  Kokkos::parallel_for(
      "Init_rank6",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<6>>({0, 0, 0, 0, 0, 0},
                                                        {N, N, N, N, N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, const int i2, const int i3,
                    const int i4, const int i5) {
        for (int i6 = 0; i6 < N; i6++) {
          A(i0, i1, i2, i3, i4, i5, i6) = 1.0 + i0 + i1 * N + i2 * N * N +
                                          i3 * N * N * N + i4 * N * N * N * N +
                                          i5 * N * N * N * N * N +
                                          i6 * N * N * N * N * N * N;
        }
      });

  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

  // Deep Copy
  Kokkos::parallel_for(
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        int lid = teamMember.league_rank();  // returns a number between 0 and N
        auto subSrc =
            Kokkos::subview(A, lid, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),
                            Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
        auto subDst =
            Kokkos::subview(B, lid, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),
                            Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(teamMember, N), [=](const int i) {
              auto subSrcThread = Kokkos::subview(
                  subSrc, Kokkos::make_pair(i, i + 1), Kokkos::ALL(),
                  Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
              auto subDstThread = Kokkos::subview(
                  subDst, Kokkos::make_pair(i, i + 1), Kokkos::ALL(),
                  Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
              Kokkos::Experimental::deep_copy(
                  Kokkos::Experimental::copy_thread(teamMember), subDstThread,
                  subSrcThread);
            });
      });

  // Verify results
  int errors = 0;
  Kokkos::parallel_reduce(
      "Verify_rank6",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<6>>({0, 0, 0, 0, 0, 0},
                                                        {N, N, N, N, N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, const int i2, const int i3,
                    const int i4, const int i5, int& err) {
        for (int i6 = 0; i6 < N; i6++) {
          double expected = 1.0 + i0 + i1 * N + i2 * N * N + i3 * N * N * N +
                            i4 * N * N * N * N + i5 * N * N * N * N * N +
                            i6 * N * N * N * N * N * N;
          if (B(i0, i1, i2, i3, i4, i5, i6) != expected) {
            err++;
          }
        }
      },
      errors);

  ASSERT_EQ(errors, 0);
}
//-------------------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename ViewType>
void impl_test_local_deepcopy_thread_rank_7(const int N) {
  // Allocate matrices on device.
  ViewType A("A", N, N, N, N, N, N, N, N);
  ViewType B("B", N, N, N, N, N, N, N, N);

  // Initialize matrix A using MDRangePolicy for outer
  // 6 dimensions and nested loops for inner dimensions.
  Kokkos::parallel_for(
      "Init_rank7",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<6>>({0, 0, 0, 0, 0, 0},
                                                        {N, N, N, N, N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, const int i2, const int i3,
                    const int i4, const int i5) {
        for (int i6 = 0; i6 < N; i6++) {
          for (int i7 = 0; i7 < N; i7++) {
            A(i0, i1, i2, i3, i4, i5, i6, i7) =
                1.0 + i0 + i1 * N + i2 * N * N + i3 * N * N * N +
                i4 * N * N * N * N + i5 * N * N * N * N * N +
                i6 * N * N * N * N * N * N + i7 * N * N * N * N * N * N * N;
          }
        }
      });

  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

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
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(teamMember, N), [=](const int i) {
              auto subSrcThread =
                  Kokkos::subview(subSrc, Kokkos::make_pair(i, i + 1),
                                  Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),
                                  Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
              auto subDstThread =
                  Kokkos::subview(subDst, Kokkos::make_pair(i, i + 1),
                                  Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),
                                  Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
              Kokkos::Experimental::deep_copy(
                  Kokkos::Experimental::copy_thread(teamMember), subDstThread,
                  subSrcThread);
            });
      });

  // Verify results
  int errors = 0;
  Kokkos::parallel_reduce(
      "Verify_rank7",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<6>>({0, 0, 0, 0, 0, 0},
                                                        {N, N, N, N, N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, const int i2, const int i3,
                    const int i4, const int i5, int& err) {
        for (int i6 = 0; i6 < N; i6++) {
          for (int i7 = 0; i7 < N; i7++) {
            double expected = 1.0 + i0 + i1 * N + i2 * N * N + i3 * N * N * N +
                              i4 * N * N * N * N + i5 * N * N * N * N * N +
                              i6 * N * N * N * N * N * N +
                              i7 * N * N * N * N * N * N * N;
            if (B(i0, i1, i2, i3, i4, i5, i6, i7) != expected) {
              err++;
            }
          }
        }
      },
      errors);

  ASSERT_EQ(errors, 0);
}
//-------------------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename ViewType>
void impl_test_local_deepcopy_team_rank_0(const int N) {
  // Allocate matrices on device.
  ViewType A("A", N);
  ViewType B("B", N);

  // Initialize matrix A
  Kokkos::parallel_for(
      "Init_rank0", Kokkos::RangePolicy<ExecSpace>(0, N),
      KOKKOS_LAMBDA(const int i) { A(i) = 1.0 + i; });

  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

  // Deep Copy
  Kokkos::parallel_for(
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        int lid = teamMember.league_rank();  // returns a number between 0 and N
        auto subSrc = Kokkos::subview(A, lid);
        auto subDst = Kokkos::subview(B, lid);
        Kokkos::Experimental::deep_copy(
            Kokkos::Experimental::copy_team(teamMember), subDst, subSrc);
      });

  // Verify results
  int errors = 0;
  Kokkos::parallel_reduce(
      "Verify_rank0", Kokkos::RangePolicy<ExecSpace>(0, N),
      KOKKOS_LAMBDA(const int i, int& err) {
        double expected = 1.0 + i;
        if (B(i) != expected) {
          err++;
        }
      },
      errors);

  ASSERT_EQ(errors, 0);
}
//-------------------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename ViewType>
void impl_test_local_deepcopy_team_rank_1(const int N) {
  // Allocate matrices on device.
  ViewType A("A", N, N);
  ViewType B("B", N, N);

  // Initialize matrix A
  Kokkos::parallel_for(
      "Init_rank1",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1) {
        A(i0, i1) = 1.0 + i0 + i1 * N;
      });

  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

  // Deep Copy
  Kokkos::parallel_for(
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        int lid = teamMember.league_rank();  // returns a number between 0 and N
        auto subSrc = Kokkos::subview(A, lid, Kokkos::ALL());
        auto subDst = Kokkos::subview(B, lid, Kokkos::ALL());
        Kokkos::Experimental::deep_copy(
            Kokkos::Experimental::copy_team(teamMember), subDst, subSrc);
      });

  // Verify results
  int errors = 0;
  Kokkos::parallel_reduce(
      "Verify_rank1",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, int& err) {
        double expected = 1.0 + i0 + i1 * N;
        if (B(i0, i1) != expected) {
          err++;
        }
      },
      errors);

  ASSERT_EQ(errors, 0);
}
//-------------------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename ViewType>
void impl_test_local_deepcopy_team_rank_2(const int N) {
  // Allocate matrices on device.
  ViewType A("A", N, N, N);
  ViewType B("B", N, N, N);

  // Initialize matrix A
  Kokkos::parallel_for(
      "Init_rank2",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {N, N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, const int i2) {
        A(i0, i1, i2) = 1.0 + i0 + i1 * N + i2 * N * N;
      });

  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

  // Deep Copy
  Kokkos::parallel_for(
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        int lid = teamMember.league_rank();  // returns a number between 0 and N
        auto subSrc = Kokkos::subview(A, lid, Kokkos::ALL(), Kokkos::ALL());
        auto subDst = Kokkos::subview(B, lid, Kokkos::ALL(), Kokkos::ALL());
        Kokkos::Experimental::deep_copy(
            Kokkos::Experimental::copy_team(teamMember), subDst, subSrc);
      });

  // Verify results
  int errors = 0;
  Kokkos::parallel_reduce(
      "Verify_rank2",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {N, N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, const int i2, int& err) {
        double expected = 1.0 + i0 + i1 * N + i2 * N * N;
        if (B(i0, i1, i2) != expected) {
          err++;
        }
      },
      errors);

  ASSERT_EQ(errors, 0);
}
//-------------------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename ViewType>
void impl_test_local_deepcopy_team_rank_3(const int N) {
  // Allocate matrices on device.
  ViewType A("A", N, N, N, N);
  ViewType B("B", N, N, N, N);

  // Initialize matrix A
  Kokkos::parallel_for(
      "Init_rank3",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>({0, 0, 0, 0},
                                                        {N, N, N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, const int i2, const int i3) {
        A(i0, i1, i2, i3) = 1.0 + i0 + i1 * N + i2 * N * N + i3 * N * N * N;
      });

  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

  // Deep Copy
  Kokkos::parallel_for(
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        int lid = teamMember.league_rank();  // returns a number between 0 and N
        auto subSrc = Kokkos::subview(A, lid, Kokkos::ALL(), Kokkos::ALL(),
                                      Kokkos::ALL());
        auto subDst = Kokkos::subview(B, lid, Kokkos::ALL(), Kokkos::ALL(),
                                      Kokkos::ALL());
        Kokkos::Experimental::deep_copy(
            Kokkos::Experimental::copy_team(teamMember), subDst, subSrc);
      });

  // Verify results
  int errors = 0;
  Kokkos::parallel_reduce(
      "Verify_rank3",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>({0, 0, 0, 0},
                                                        {N, N, N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, const int i2, const int i3,
                    int& err) {
        double expected = 1.0 + i0 + i1 * N + i2 * N * N + i3 * N * N * N;
        if (B(i0, i1, i2, i3) != expected) {
          err++;
        }
      },
      errors);

  ASSERT_EQ(errors, 0);
}
//-------------------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename ViewType>
void impl_test_local_deepcopy_team_rank_4(const int N) {
  // Allocate matrices on device.
  ViewType A("A", N, N, N, N, N);
  ViewType B("B", N, N, N, N, N);

  // Initialize matrix A
  Kokkos::parallel_for(
      "Init_rank4",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<5>>({0, 0, 0, 0, 0},
                                                        {N, N, N, N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, const int i2, const int i3,
                    const int i4) {
        A(i0, i1, i2, i3, i4) = 1.0 + i0 + i1 * N + i2 * N * N +
                                i3 * N * N * N + i4 * N * N * N * N;
      });

  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

  // Deep Copy
  Kokkos::parallel_for(
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        int lid = teamMember.league_rank();  // returns a number between 0 and N
        auto subSrc = Kokkos::subview(A, lid, Kokkos::ALL(), Kokkos::ALL(),
                                      Kokkos::ALL(), Kokkos::ALL());
        auto subDst = Kokkos::subview(B, lid, Kokkos::ALL(), Kokkos::ALL(),
                                      Kokkos::ALL(), Kokkos::ALL());
        Kokkos::Experimental::deep_copy(
            Kokkos::Experimental::copy_team(teamMember), subDst, subSrc);
      });

  // Verify results
  int errors = 0;
  Kokkos::parallel_reduce(
      "Verify_rank4",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<5>>({0, 0, 0, 0, 0},
                                                        {N, N, N, N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, const int i2, const int i3,
                    const int i4, int& err) {
        double expected = 1.0 + i0 + i1 * N + i2 * N * N + i3 * N * N * N +
                          i4 * N * N * N * N;
        if (B(i0, i1, i2, i3, i4) != expected) {
          err++;
        }
      },
      errors);

  ASSERT_EQ(errors, 0);
}
//-------------------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename ViewType>
void impl_test_local_deepcopy_team_rank_5(const int N) {
  // Allocate matrices on device.
  ViewType A("A", N, N, N, N, N, N);
  ViewType B("B", N, N, N, N, N, N);

  // Initialize matrix A
  Kokkos::parallel_for(
      "Init_rank5",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<6>>({0, 0, 0, 0, 0, 0},
                                                        {N, N, N, N, N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, const int i2, const int i3,
                    const int i4, const int i5) {
        A(i0, i1, i2, i3, i4, i5) = 1.0 + i0 + i1 * N + i2 * N * N +
                                    i3 * N * N * N + i4 * N * N * N * N +
                                    i5 * N * N * N * N * N;
      });

  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

  // Deep Copy
  Kokkos::parallel_for(
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        int lid = teamMember.league_rank();  // returns a number between 0 and N
        auto subSrc =
            Kokkos::subview(A, lid, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),
                            Kokkos::ALL(), Kokkos::ALL());
        auto subDst =
            Kokkos::subview(B, lid, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),
                            Kokkos::ALL(), Kokkos::ALL());
        Kokkos::Experimental::deep_copy(
            Kokkos::Experimental::copy_team(teamMember), subDst, subSrc);
      });

  // Verify results
  int errors = 0;
  Kokkos::parallel_reduce(
      "Verify_rank5",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<6>>({0, 0, 0, 0, 0, 0},
                                                        {N, N, N, N, N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, const int i2, const int i3,
                    const int i4, const int i5, int& err) {
        double expected = 1.0 + i0 + i1 * N + i2 * N * N + i3 * N * N * N +
                          i4 * N * N * N * N + i5 * N * N * N * N * N;
        if (B(i0, i1, i2, i3, i4, i5) != expected) {
          err++;
        }
      },
      errors);

  ASSERT_EQ(errors, 0);
}
//-------------------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename ViewType>
void impl_test_local_deepcopy_team_rank_6(const int N) {
  // Allocate matrices on device.
  ViewType A("A", N, N, N, N, N, N, N);
  ViewType B("B", N, N, N, N, N, N, N);

  // Initialize matrix A using MDRangePolicy for outer
  // 6 dimensions and nested loop for inner dimension.
  Kokkos::parallel_for(
      "Init_rank6",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<6>>({0, 0, 0, 0, 0, 0},
                                                        {N, N, N, N, N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, const int i2, const int i3,
                    const int i4, const int i5) {
        for (int i6 = 0; i6 < N; i6++) {
          A(i0, i1, i2, i3, i4, i5, i6) = 1.0 + i0 + i1 * N + i2 * N * N +
                                          i3 * N * N * N + i4 * N * N * N * N +
                                          i5 * N * N * N * N * N +
                                          i6 * N * N * N * N * N * N;
        }
      });

  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

  // Deep Copy
  Kokkos::parallel_for(
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        int lid = teamMember.league_rank();  // returns a number between 0 and N
        auto subSrc =
            Kokkos::subview(A, lid, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),
                            Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
        auto subDst =
            Kokkos::subview(B, lid, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),
                            Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
        Kokkos::Experimental::deep_copy(
            Kokkos::Experimental::copy_team(teamMember), subDst, subSrc);
      });

  // Verify results
  int errors = 0;
  Kokkos::parallel_reduce(
      "Verify_rank6",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<6>>({0, 0, 0, 0, 0, 0},
                                                        {N, N, N, N, N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, const int i2, const int i3,
                    const int i4, const int i5, int& err) {
        for (int i6 = 0; i6 < N; i6++) {
          double expected = 1.0 + i0 + i1 * N + i2 * N * N + i3 * N * N * N +
                            i4 * N * N * N * N + i5 * N * N * N * N * N +
                            i6 * N * N * N * N * N * N;
          if (B(i0, i1, i2, i3, i4, i5, i6) != expected) {
            err++;
          }
        }
      },
      errors);

  ASSERT_EQ(errors, 0);
}
//-------------------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename ViewType>
void impl_test_local_deepcopy_team_rank_7(const int N) {
  // Allocate matrices on device.
  ViewType A("A", N, N, N, N, N, N, N, N);
  ViewType B("B", N, N, N, N, N, N, N, N);

  // Initialize matrix A using MDRangePolicy for outer
  // 6 dimensions and nested loops for inner dimensions.
  Kokkos::parallel_for(
      "Init_rank7",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<6>>({0, 0, 0, 0, 0, 0},
                                                        {N, N, N, N, N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, const int i2, const int i3,
                    const int i4, const int i5) {
        for (int i6 = 0; i6 < N; i6++) {
          for (int i7 = 0; i7 < N; i7++) {
            A(i0, i1, i2, i3, i4, i5, i6, i7) =
                1.0 + i0 + i1 * N + i2 * N * N + i3 * N * N * N +
                i4 * N * N * N * N + i5 * N * N * N * N * N +
                i6 * N * N * N * N * N * N + i7 * N * N * N * N * N * N * N;
          }
        }
      });

  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

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
        Kokkos::Experimental::deep_copy(
            Kokkos::Experimental::copy_team(teamMember), subDst, subSrc);
      });

  // Verify results
  int errors = 0;
  Kokkos::parallel_reduce(
      "Verify_rank7",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<6>>({0, 0, 0, 0, 0, 0},
                                                        {N, N, N, N, N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, const int i2, const int i3,
                    const int i4, const int i5, int& err) {
        for (int i6 = 0; i6 < N; i6++) {
          for (int i7 = 0; i7 < N; i7++) {
            double expected = 1.0 + i0 + i1 * N + i2 * N * N + i3 * N * N * N +
                              i4 * N * N * N * N + i5 * N * N * N * N * N +
                              i6 * N * N * N * N * N * N +
                              i7 * N * N * N * N * N * N * N;
            if (B(i0, i1, i2, i3, i4, i5, i6, i7) != expected) {
              err++;
            }
          }
        }
      },
      errors);

  ASSERT_EQ(errors, 0);
}
//-------------------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename ViewType>
void impl_test_local_deepcopy_sequential_rank_0(const int N) {
  // Allocate matrices on device.
  ViewType A("A", N);
  ViewType B("B", N);

  // Initialize matrix A
  Kokkos::parallel_for(
      "Init_range_rank0", Kokkos::RangePolicy<ExecSpace>(0, N),
      KOKKOS_LAMBDA(const int i) { A(i) = 1.0 + i; });

  // Deep Copy
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(const int& i) {
        auto subSrc = Kokkos::subview(A, i);
        auto subDst = Kokkos::subview(B, i);
        Kokkos::Experimental::deep_copy(Kokkos::Experimental::copy_seq(),
                                        subDst, subSrc);
      });

  // Verify results
  int errors = 0;
  Kokkos::parallel_reduce(
      "Verify_range_rank0", Kokkos::RangePolicy<ExecSpace>(0, N),
      KOKKOS_LAMBDA(const int i, int& err) {
        double expected = 1.0 + i;
        if (B(i) != expected) {
          err++;
        }
      },
      errors);

  ASSERT_EQ(errors, 0);
}
//-------------------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename ViewType>
void impl_test_local_deepcopy_sequential_rank_1(const int N) {
  // Allocate matrices on device.
  ViewType A("A", N, N);
  ViewType B("B", N, N);

  // Initialize matrix A
  Kokkos::parallel_for(
      "Init_range_rank1",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1) {
        A(i0, i1) = 1.0 + i0 + i1 * N;
      });

  // Deep Copy
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(const int& i) {
        auto subSrc = Kokkos::subview(A, i, Kokkos::ALL());
        auto subDst = Kokkos::subview(B, i, Kokkos::ALL());
        Kokkos::Experimental::deep_copy(Kokkos::Experimental::copy_seq(),
                                        subDst, subSrc);
      });

  // Verify results
  int errors = 0;
  Kokkos::parallel_reduce(
      "Verify_range_rank1",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, int& err) {
        double expected = 1.0 + i0 + i1 * N;
        if (B(i0, i1) != expected) {
          err++;
        }
      },
      errors);

  ASSERT_EQ(errors, 0);
}
//-------------------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename ViewType>
void impl_test_local_deepcopy_sequential_rank_2(const int N) {
  // Allocate matrices on device.
  ViewType A("A", N, N, N);
  ViewType B("B", N, N, N);

  // Initialize matrix A
  Kokkos::parallel_for(
      "Init_range_rank2",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {N, N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, const int i2) {
        A(i0, i1, i2) = 1.0 + i0 + i1 * N + i2 * N * N;
      });

  // Deep Copy
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(const int& i) {
        auto subSrc = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
        auto subDst = Kokkos::subview(B, i, Kokkos::ALL(), Kokkos::ALL());
        Kokkos::Experimental::deep_copy(Kokkos::Experimental::copy_seq(),
                                        subDst, subSrc);
      });

  // Verify results
  int errors = 0;
  Kokkos::parallel_reduce(
      "Verify_range_rank2",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {N, N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, const int i2, int& err) {
        double expected = 1.0 + i0 + i1 * N + i2 * N * N;
        if (B(i0, i1, i2) != expected) {
          err++;
        }
      },
      errors);

  ASSERT_EQ(errors, 0);
}
//-------------------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename ViewType>
void impl_test_local_deepcopy_sequential_rank_3(const int N) {
  // Allocate matrices on device.
  ViewType A("A", N, N, N, N);
  ViewType B("B", N, N, N, N);

  // Initialize matrix A
  Kokkos::parallel_for(
      "Init_range_rank3",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>({0, 0, 0, 0},
                                                        {N, N, N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, const int i2, const int i3) {
        A(i0, i1, i2, i3) = 1.0 + i0 + i1 * N + i2 * N * N + i3 * N * N * N;
      });

  // Deep Copy
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(const int& i) {
        auto subSrc =
            Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
        auto subDst =
            Kokkos::subview(B, i, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
        Kokkos::Experimental::deep_copy(Kokkos::Experimental::copy_seq(),
                                        subDst, subSrc);
      });

  // Verify results
  int errors = 0;
  Kokkos::parallel_reduce(
      "Verify_range_rank3",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>({0, 0, 0, 0},
                                                        {N, N, N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, const int i2, const int i3,
                    int& err) {
        double expected = 1.0 + i0 + i1 * N + i2 * N * N + i3 * N * N * N;
        if (B(i0, i1, i2, i3) != expected) {
          err++;
        }
      },
      errors);

  ASSERT_EQ(errors, 0);
}
//-------------------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename ViewType>
void impl_test_local_deepcopy_sequential_rank_4(const int N) {
  // Allocate matrices on device.
  ViewType A("A", N, N, N, N, N);
  ViewType B("B", N, N, N, N, N);

  // Initialize matrix A
  Kokkos::parallel_for(
      "Init_range_rank4",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<5>>({0, 0, 0, 0, 0},
                                                        {N, N, N, N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, const int i2, const int i3,
                    const int i4) {
        A(i0, i1, i2, i3, i4) = 1.0 + i0 + i1 * N + i2 * N * N +
                                i3 * N * N * N + i4 * N * N * N * N;
      });

  // Deep Copy
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(const int& i) {
        auto subSrc = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL(),
                                      Kokkos::ALL(), Kokkos::ALL());
        auto subDst = Kokkos::subview(B, i, Kokkos::ALL(), Kokkos::ALL(),
                                      Kokkos::ALL(), Kokkos::ALL());
        Kokkos::Experimental::deep_copy(Kokkos::Experimental::copy_seq(),
                                        subDst, subSrc);
      });

  // Verify results
  int errors = 0;
  Kokkos::parallel_reduce(
      "Verify_range_rank4",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<5>>({0, 0, 0, 0, 0},
                                                        {N, N, N, N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, const int i2, const int i3,
                    const int i4, int& err) {
        double expected = 1.0 + i0 + i1 * N + i2 * N * N + i3 * N * N * N +
                          i4 * N * N * N * N;
        if (B(i0, i1, i2, i3, i4) != expected) {
          err++;
        }
      },
      errors);

  ASSERT_EQ(errors, 0);
}
//-------------------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename ViewType>
void impl_test_local_deepcopy_sequential_rank_5(const int N) {
  // Allocate matrices on device.
  ViewType A("A", N, N, N, N, N, N);
  ViewType B("B", N, N, N, N, N, N);

  // Initialize matrix A
  Kokkos::parallel_for(
      "Init_range_rank5",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<6>>({0, 0, 0, 0, 0, 0},
                                                        {N, N, N, N, N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, const int i2, const int i3,
                    const int i4, const int i5) {
        A(i0, i1, i2, i3, i4, i5) = 1.0 + i0 + i1 * N + i2 * N * N +
                                    i3 * N * N * N + i4 * N * N * N * N +
                                    i5 * N * N * N * N * N;
      });

  // Deep Copy
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(const int& i) {
        auto subSrc =
            Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),
                            Kokkos::ALL(), Kokkos::ALL());
        auto subDst =
            Kokkos::subview(B, i, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),
                            Kokkos::ALL(), Kokkos::ALL());
        Kokkos::Experimental::deep_copy(Kokkos::Experimental::copy_seq(),
                                        subDst, subSrc);
      });

  // Verify results
  int errors = 0;
  Kokkos::parallel_reduce(
      "Verify_range_rank5",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<6>>({0, 0, 0, 0, 0, 0},
                                                        {N, N, N, N, N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, const int i2, const int i3,
                    const int i4, const int i5, int& err) {
        double expected = 1.0 + i0 + i1 * N + i2 * N * N + i3 * N * N * N +
                          i4 * N * N * N * N + i5 * N * N * N * N * N;
        if (B(i0, i1, i2, i3, i4, i5) != expected) {
          err++;
        }
      },
      errors);

  ASSERT_EQ(errors, 0);
}
//-------------------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename ViewType>
void impl_test_local_deepcopy_sequential_rank_6(const int N) {
  // Allocate matrices on device.
  ViewType A("A", N, N, N, N, N, N, N);
  ViewType B("B", N, N, N, N, N, N, N);

  // Initialize matrix A using MDRangePolicy for outer
  // 6 dimensions and nested loop for inner dimension.
  Kokkos::parallel_for(
      "Init_range_rank6",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<6>>({0, 0, 0, 0, 0, 0},
                                                        {N, N, N, N, N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, const int i2, const int i3,
                    const int i4, const int i5) {
        for (int i6 = 0; i6 < N; i6++) {
          A(i0, i1, i2, i3, i4, i5, i6) = 1.0 + i0 + i1 * N + i2 * N * N +
                                          i3 * N * N * N + i4 * N * N * N * N +
                                          i5 * N * N * N * N * N +
                                          i6 * N * N * N * N * N * N;
        }
      });

  // Deep Copy
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(const int& i) {
        auto subSrc =
            Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),
                            Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
        auto subDst =
            Kokkos::subview(B, i, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),
                            Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
        Kokkos::Experimental::deep_copy(Kokkos::Experimental::copy_seq(),
                                        subDst, subSrc);
      });

  // Verify results
  int errors = 0;
  Kokkos::parallel_reduce(
      "Verify_range_rank6",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<6>>({0, 0, 0, 0, 0, 0},
                                                        {N, N, N, N, N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, const int i2, const int i3,
                    const int i4, const int i5, int& err) {
        for (int i6 = 0; i6 < N; i6++) {
          double expected = 1.0 + i0 + i1 * N + i2 * N * N + i3 * N * N * N +
                            i4 * N * N * N * N + i5 * N * N * N * N * N +
                            i6 * N * N * N * N * N * N;
          if (B(i0, i1, i2, i3, i4, i5, i6) != expected) {
            err++;
          }
        }
      },
      errors);

  ASSERT_EQ(errors, 0);
}
//-------------------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename ViewType>
void impl_test_local_deepcopy_sequential_rank_7(const int N) {
  // Allocate matrices on device.
  ViewType A("A", N, N, N, N, N, N, N, N);
  ViewType B("B", N, N, N, N, N, N, N, N);

  // Initialize matrix A using MDRangePolicy for outer
  // 6 dimensions and nested loop for inner dimensions.
  Kokkos::parallel_for(
      "Init_range_rank7",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<6>>({0, 0, 0, 0, 0, 0},
                                                        {N, N, N, N, N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, const int i2, const int i3,
                    const int i4, const int i5) {
        for (int i6 = 0; i6 < N; i6++) {
          for (int i7 = 0; i7 < N; i7++) {
            A(i0, i1, i2, i3, i4, i5, i6, i7) =
                1.0 + i0 + i1 * N + i2 * N * N + i3 * N * N * N +
                i4 * N * N * N * N + i5 * N * N * N * N * N +
                i6 * N * N * N * N * N * N + i7 * N * N * N * N * N * N * N;
          }
        }
      });

  // Deep Copy
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(const int& i) {
        auto subSrc = Kokkos::subview(
            A, i, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),
            Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
        auto subDst = Kokkos::subview(
            B, i, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),
            Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
        Kokkos::Experimental::deep_copy(Kokkos::Experimental::copy_seq(),
                                        subDst, subSrc);
      });

  // Verify results
  int errors = 0;
  Kokkos::parallel_reduce(
      "Verify_range_rank7",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<6>>({0, 0, 0, 0, 0, 0},
                                                        {N, N, N, N, N, N}),
      KOKKOS_LAMBDA(const int i0, const int i1, const int i2, const int i3,
                    const int i4, const int i5, int& err) {
        for (int i6 = 0; i6 < N; i6++) {
          for (int i7 = 0; i7 < N; i7++) {
            double expected = 1.0 + i0 + i1 * N + i2 * N * N + i3 * N * N * N +
                              i4 * N * N * N * N + i5 * N * N * N * N * N +
                              i6 * N * N * N * N * N * N +
                              i7 * N * N * N * N * N * N * N;
            if (B(i0, i1, i2, i3, i4, i5, i6, i7) != expected) {
              err++;
            }
          }
        }
      },
      errors);

  ASSERT_EQ(errors, 0);
}
//-------------------------------------------------------------------------------------------------------------
TEST(TEST_CATEGORY, local_deepcopy_thread_layoutleft) {
  using ExecSpace = TEST_EXECSPACE;

  {  // Rank-0
    using ViewType = Kokkos::View<double**, Kokkos::LayoutLeft, ExecSpace>;
    impl_test_local_deepcopy_thread_rank_0<ExecSpace, ViewType>(5);
  }
  {  // Rank-1
    using ViewType = Kokkos::View<double**, Kokkos::LayoutLeft, ExecSpace>;
    impl_test_local_deepcopy_thread_rank_1<ExecSpace, ViewType>(5);
  }
  {  // Rank-2
    using ViewType = Kokkos::View<double***, Kokkos::LayoutLeft, ExecSpace>;
    impl_test_local_deepcopy_thread_rank_2<ExecSpace, ViewType>(5);
  }
  {  // Rank-3
    using ViewType = Kokkos::View<double****, Kokkos::LayoutLeft, ExecSpace>;
    impl_test_local_deepcopy_thread_rank_3<ExecSpace, ViewType>(5);
  }
  {  // Rank-4
    using ViewType = Kokkos::View<double*****, Kokkos::LayoutLeft, ExecSpace>;
    impl_test_local_deepcopy_thread_rank_4<ExecSpace, ViewType>(5);
  }
  {  // Rank-5
    using ViewType = Kokkos::View<double******, Kokkos::LayoutLeft, ExecSpace>;
    impl_test_local_deepcopy_thread_rank_5<ExecSpace, ViewType>(5);
  }
  {  // Rank-6
    using ViewType = Kokkos::View<double*******, Kokkos::LayoutLeft, ExecSpace>;
    impl_test_local_deepcopy_thread_rank_6<ExecSpace, ViewType>(5);
  }
  {  // Rank-7
    using ViewType =
        Kokkos::View<double********, Kokkos::LayoutLeft, ExecSpace>;
    impl_test_local_deepcopy_thread_rank_7<ExecSpace, ViewType>(5);
  }
}
//-------------------------------------------------------------------------------------------------------------
TEST(TEST_CATEGORY, local_deepcopy_team_layoutleft) {
  using ExecSpace = TEST_EXECSPACE;

  {  // Rank-0
    using ViewType = Kokkos::View<double*, Kokkos::LayoutLeft, ExecSpace>;
    impl_test_local_deepcopy_team_rank_0<ExecSpace, ViewType>(5);
  }
  {  // Rank-1
    using ViewType = Kokkos::View<double**, Kokkos::LayoutLeft, ExecSpace>;
    impl_test_local_deepcopy_team_rank_1<ExecSpace, ViewType>(5);
  }
  {  // Rank-2
    using ViewType = Kokkos::View<double***, Kokkos::LayoutLeft, ExecSpace>;
    impl_test_local_deepcopy_team_rank_2<ExecSpace, ViewType>(5);
  }
  {  // Rank-3
    using ViewType = Kokkos::View<double****, Kokkos::LayoutLeft, ExecSpace>;
    impl_test_local_deepcopy_team_rank_3<ExecSpace, ViewType>(5);
  }
  {  // Rank-4
    using ViewType = Kokkos::View<double*****, Kokkos::LayoutLeft, ExecSpace>;
    impl_test_local_deepcopy_team_rank_4<ExecSpace, ViewType>(5);
  }
  {  // Rank-5
    using ViewType = Kokkos::View<double******, Kokkos::LayoutLeft, ExecSpace>;
    impl_test_local_deepcopy_team_rank_5<ExecSpace, ViewType>(5);
  }
  {  // Rank-6
    using ViewType = Kokkos::View<double*******, Kokkos::LayoutLeft, ExecSpace>;
    impl_test_local_deepcopy_team_rank_6<ExecSpace, ViewType>(5);
  }
  {  // Rank-7
    using ViewType =
        Kokkos::View<double********, Kokkos::LayoutLeft, ExecSpace>;
    impl_test_local_deepcopy_team_rank_7<ExecSpace, ViewType>(5);
  }
}
//-------------------------------------------------------------------------------------------------------------
TEST(TEST_CATEGORY, local_deepcopy_sequential_layoutleft) {
  using ExecSpace = TEST_EXECSPACE;

  {  // Rank-0
    using ViewType = Kokkos::View<double*, Kokkos::LayoutLeft, ExecSpace>;
    impl_test_local_deepcopy_sequential_rank_0<ExecSpace, ViewType>(5);
  }
  {  // Rank-1
    using ViewType = Kokkos::View<double**, Kokkos::LayoutLeft, ExecSpace>;
    impl_test_local_deepcopy_sequential_rank_1<ExecSpace, ViewType>(5);
  }
  {  // Rank-2
    using ViewType = Kokkos::View<double***, Kokkos::LayoutLeft, ExecSpace>;
    impl_test_local_deepcopy_sequential_rank_2<ExecSpace, ViewType>(5);
  }
  {  // Rank-3
    using ViewType = Kokkos::View<double****, Kokkos::LayoutLeft, ExecSpace>;
    impl_test_local_deepcopy_sequential_rank_3<ExecSpace, ViewType>(5);
  }
  {  // Rank-4
    using ViewType = Kokkos::View<double*****, Kokkos::LayoutLeft, ExecSpace>;
    impl_test_local_deepcopy_sequential_rank_4<ExecSpace, ViewType>(5);
  }
  {  // Rank-5
    using ViewType = Kokkos::View<double******, Kokkos::LayoutLeft, ExecSpace>;
    impl_test_local_deepcopy_sequential_rank_5<ExecSpace, ViewType>(5);
  }
  {  // Rank-6
    using ViewType = Kokkos::View<double*******, Kokkos::LayoutLeft, ExecSpace>;
    impl_test_local_deepcopy_sequential_rank_6<ExecSpace, ViewType>(5);
  }
  {  // Rank-7
    using ViewType =
        Kokkos::View<double********, Kokkos::LayoutLeft, ExecSpace>;
    impl_test_local_deepcopy_sequential_rank_7<ExecSpace, ViewType>(5);
  }
}
//-------------------------------------------------------------------------------------------------------------
TEST(TEST_CATEGORY, local_deepcopy_thread_layoutright) {
  using ExecSpace = TEST_EXECSPACE;

  {  // Rank-0
    using ViewType = Kokkos::View<double**, Kokkos::LayoutRight, ExecSpace>;
    impl_test_local_deepcopy_thread_rank_0<ExecSpace, ViewType>(5);
  }
  {  // Rank-1
    using ViewType = Kokkos::View<double**, Kokkos::LayoutRight, ExecSpace>;
    impl_test_local_deepcopy_thread_rank_1<ExecSpace, ViewType>(5);
  }
  {  // Rank-2
    using ViewType = Kokkos::View<double***, Kokkos::LayoutRight, ExecSpace>;
    impl_test_local_deepcopy_thread_rank_2<ExecSpace, ViewType>(5);
  }
  {  // Rank-3
    using ViewType = Kokkos::View<double****, Kokkos::LayoutRight, ExecSpace>;
    impl_test_local_deepcopy_thread_rank_3<ExecSpace, ViewType>(5);
  }
  {  // Rank-4
    using ViewType = Kokkos::View<double*****, Kokkos::LayoutRight, ExecSpace>;
    impl_test_local_deepcopy_thread_rank_4<ExecSpace, ViewType>(5);
  }
  {  // Rank-5
    using ViewType = Kokkos::View<double******, Kokkos::LayoutRight, ExecSpace>;
    impl_test_local_deepcopy_thread_rank_5<ExecSpace, ViewType>(5);
  }
  {  // Rank-6
    using ViewType =
        Kokkos::View<double*******, Kokkos::LayoutRight, ExecSpace>;
    impl_test_local_deepcopy_thread_rank_6<ExecSpace, ViewType>(5);
  }
  {  // Rank-7
    using ViewType =
        Kokkos::View<double********, Kokkos::LayoutRight, ExecSpace>;
    impl_test_local_deepcopy_thread_rank_7<ExecSpace, ViewType>(5);
  }
}
//-------------------------------------------------------------------------------------------------------------
TEST(TEST_CATEGORY, local_deepcopy_team_layoutright) {
  using ExecSpace = TEST_EXECSPACE;

  {  // Rank-0
    using ViewType = Kokkos::View<double*, Kokkos::LayoutRight, ExecSpace>;
    impl_test_local_deepcopy_team_rank_0<ExecSpace, ViewType>(5);
  }
  {  // Rank-1
    using ViewType = Kokkos::View<double**, Kokkos::LayoutRight, ExecSpace>;
    impl_test_local_deepcopy_team_rank_1<ExecSpace, ViewType>(5);
  }
  {  // Rank-2
    using ViewType = Kokkos::View<double***, Kokkos::LayoutRight, ExecSpace>;
    impl_test_local_deepcopy_team_rank_2<ExecSpace, ViewType>(5);
  }
  {  // Rank-3
    using ViewType = Kokkos::View<double****, Kokkos::LayoutRight, ExecSpace>;
    impl_test_local_deepcopy_team_rank_3<ExecSpace, ViewType>(5);
  }
  {  // Rank-4
    using ViewType = Kokkos::View<double*****, Kokkos::LayoutRight, ExecSpace>;
    impl_test_local_deepcopy_team_rank_4<ExecSpace, ViewType>(5);
  }
  {  // Rank-5
    using ViewType = Kokkos::View<double******, Kokkos::LayoutRight, ExecSpace>;
    impl_test_local_deepcopy_team_rank_5<ExecSpace, ViewType>(5);
  }
  {  // Rank-6
    using ViewType =
        Kokkos::View<double*******, Kokkos::LayoutRight, ExecSpace>;
    impl_test_local_deepcopy_team_rank_6<ExecSpace, ViewType>(5);
  }
  {  // Rank-7
    using ViewType =
        Kokkos::View<double********, Kokkos::LayoutRight, ExecSpace>;
    impl_test_local_deepcopy_team_rank_7<ExecSpace, ViewType>(5);
  }
}
//-------------------------------------------------------------------------------------------------------------
TEST(TEST_CATEGORY, local_deepcopy_sequential_layoutright) {
  using ExecSpace = TEST_EXECSPACE;

  {  // Rank-0
    using ViewType = Kokkos::View<double*, Kokkos::LayoutRight, ExecSpace>;
    impl_test_local_deepcopy_sequential_rank_0<ExecSpace, ViewType>(5);
  }
  {  // Rank-1
    using ViewType = Kokkos::View<double**, Kokkos::LayoutRight, ExecSpace>;
    impl_test_local_deepcopy_sequential_rank_1<ExecSpace, ViewType>(5);
  }
  {  // Rank-2
    using ViewType = Kokkos::View<double***, Kokkos::LayoutRight, ExecSpace>;
    impl_test_local_deepcopy_sequential_rank_2<ExecSpace, ViewType>(5);
  }
  {  // Rank-3
    using ViewType = Kokkos::View<double****, Kokkos::LayoutRight, ExecSpace>;
    impl_test_local_deepcopy_sequential_rank_3<ExecSpace, ViewType>(5);
  }
  {  // Rank-4
    using ViewType = Kokkos::View<double*****, Kokkos::LayoutRight, ExecSpace>;
    impl_test_local_deepcopy_sequential_rank_4<ExecSpace, ViewType>(5);
  }
  {  // Rank-5
    using ViewType = Kokkos::View<double******, Kokkos::LayoutRight, ExecSpace>;
    impl_test_local_deepcopy_sequential_rank_5<ExecSpace, ViewType>(5);
  }
  {  // Rank-6
    using ViewType =
        Kokkos::View<double*******, Kokkos::LayoutRight, ExecSpace>;
    impl_test_local_deepcopy_sequential_rank_6<ExecSpace, ViewType>(5);
  }
  {  // Rank-7
    using ViewType =
        Kokkos::View<double********, Kokkos::LayoutRight, ExecSpace>;
    impl_test_local_deepcopy_sequential_rank_7<ExecSpace, ViewType>(5);
  }
}

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
          Kokkos::Experimental::deep_copy(
              Kokkos::Experimental::copy_thread(team), thread_shview, index);
        });
    team.team_barrier();
    Kokkos::Experimental::deep_copy(Kokkos::Experimental::copy_team(team),
                                    check_view_1_,
                                    Kokkos::subview(shview, Kokkos::ALL(), 0));
    team.team_barrier();
    Kokkos::Experimental::deep_copy(Kokkos::Experimental::copy_team(team),
                                    shview, 6.);
    team.team_barrier();
    Kokkos::Experimental::deep_copy(Kokkos::Experimental::copy_team(team),
                                    check_view_2_,
                                    Kokkos::subview(shview, Kokkos::ALL(), 0));
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
