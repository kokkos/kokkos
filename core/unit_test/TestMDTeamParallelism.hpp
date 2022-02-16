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
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
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

#include <Kokkos_Core.hpp>

namespace Test {

namespace MDTeamParallelism {

struct FillFlattenedIndex {
  explicit FillFlattenedIndex(int n0, int n1, int n2, int n3 = 1, int n4 = 1,
                              int n5 = 1, int n6 = 1, int n7 = 1)
      : initValue{n0, n1, n2, n3, n4, n5, n6, n7} {}

  KOKKOS_INLINE_FUNCTION
  int operator()(int n0, int n1, int n2, int n3 = 0, int n4 = 0, int n5 = 0,
                 int n6 = 0, int n7 = 0) const {
    return ((((((n7 * initValue[7] + n6) * initValue[6] + n5) * initValue[5] +
               n4) *
                  initValue[4] +
              n3) *
                 initValue[3] +
             n2) *
                initValue[2] +
            n1) *
               initValue[1] +
           n0;
  }

  int initValue[8];
};

struct FillConstant {
  explicit FillConstant(int initValue_) : initValue(initValue_) {}

  int operator()(int, int, int) const { return initValue; }

  int initValue;
};

template <typename ExecSpace>
struct TestMDTeamParallelFor {
  using DataType = int;
  using TeamType = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

  template <typename HostViewType, typename FillFunctor>
  static void check_result_3D(HostViewType h_view,
                              FillFunctor const& fillFunctor) {
    for (size_t i = 0; i < h_view.extent(0); ++i) {
      for (size_t j = 0; j < h_view.extent(1); ++j) {
        for (size_t k = 0; k < h_view.extent(2); ++k) {
          EXPECT_EQ(h_view(i, j, k), fillFunctor(i, j, k));
        }
      }
    }
  }

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_for_3D_MDTeamThreadRange(int* dims,
                                                     const int initValue) {
    using ViewType     = typename Kokkos::View<DataType***, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];

    ViewType v("v", leagueSize, n0, n1);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange = Kokkos::MDTeamThreadRange<Direction>(team, n0, n1);

          Kokkos::parallel_for(teamRange, [=](int i, int j) {
            v(leagueRank, i, j) += fillFlattenedIndex(leagueRank, i, j);
          });
        });

    HostViewType h_view = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultHostExecutionSpace(), v);

    check_result_3D(h_view, fillFlattenedIndex);
  }

  template <typename HostViewType, typename FillFunctor>
  static void check_result_4D(HostViewType h_view, FillFunctor& fillFunctor) {
    for (size_t i = 0; i < h_view.extent(0); ++i) {
      for (size_t j = 0; j < h_view.extent(1); ++j) {
        for (size_t k = 0; k < h_view.extent(2); ++k) {
          for (size_t l = 0; l < h_view.extent(3); ++l) {
            EXPECT_EQ(h_view(i, j, k, l), fillFunctor(i, j, k, l));
          }
        }
      }
    }
  }

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_for_4D_MDTeamThreadRange(int* dims,
                                                     const int initValue) {
    using ViewType     = typename Kokkos::View<DataType****, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];

    ViewType v("v", leagueSize, n0, n1, n2);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange =
              Kokkos::MDTeamThreadRange<Direction>(team, n0, n1, n2);

          Kokkos::parallel_for(teamRange, [=](int i, int j, int k) {
            v(leagueRank, i, j, k) += fillFlattenedIndex(leagueRank, i, j, k);
          });
        });

    HostViewType h_view = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultHostExecutionSpace(), v);

    check_result_4D(h_view, fillFlattenedIndex);
  }

  template <typename HostViewType, typename FillFunctor>
  static void check_result_5D(HostViewType h_view, FillFunctor& fillFunctor) {
    for (size_t i = 0; i < h_view.extent(0); ++i) {
      for (size_t j = 0; j < h_view.extent(1); ++j) {
        for (size_t k = 0; k < h_view.extent(2); ++k) {
          for (size_t l = 0; l < h_view.extent(3); ++l) {
            for (size_t m = 0; m < h_view.extent(4); ++m) {
              EXPECT_EQ(h_view(i, j, k, l, m), fillFunctor(i, j, k, l, m));
            }
          }
        }
      }
    }
  }

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_for_5D_MDTeamThreadRange(int* dims,
                                                     const int initValue) {
    using ViewType     = typename Kokkos::View<DataType*****, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];
    int n3         = dims[4];

    ViewType v("v", leagueSize, n0, n1, n2, n3);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange =
              Kokkos::MDTeamThreadRange<Direction>(team, n0, n1, n2, n3);

          Kokkos::parallel_for(teamRange, [=](int i, int j, int k, int l) {
            v(leagueRank, i, j, k, l) +=
                fillFlattenedIndex(leagueRank, i, j, k, l);
          });
        });

    HostViewType h_view = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultHostExecutionSpace(), v);

    check_result_5D(h_view, fillFlattenedIndex);
  }

  template <typename HostViewType, typename FillFunctor>
  static void check_result_6D(HostViewType h_view, FillFunctor& fillFunctor) {
    for (size_t i = 0; i < h_view.extent(0); ++i) {
      for (size_t j = 0; j < h_view.extent(1); ++j) {
        for (size_t k = 0; k < h_view.extent(2); ++k) {
          for (size_t l = 0; l < h_view.extent(3); ++l) {
            for (size_t m = 0; m < h_view.extent(4); ++m) {
              for (size_t n = 0; n < h_view.extent(5); ++n) {
                EXPECT_EQ(h_view(i, j, k, l, m, n),
                          fillFunctor(i, j, k, l, m, n));
              }
            }
          }
        }
      }
    }
  }

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_for_6D_MDTeamThreadRange(int* dims,
                                                     const int initValue) {
    using ViewType     = typename Kokkos::View<DataType******, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];
    int n3         = dims[4];
    int n4         = dims[5];

    ViewType v("v", leagueSize, n0, n1, n2, n3, n4);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3, n4);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange =
              Kokkos::MDTeamThreadRange<Direction>(team, n0, n1, n2, n3, n4);

          Kokkos::parallel_for(
              teamRange, [=](int i, int j, int k, int l, int m) {
                v(leagueRank, i, j, k, l, m) +=
                    fillFlattenedIndex(leagueRank, i, j, k, l, m);
              });
        });

    HostViewType h_view = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultHostExecutionSpace(), v);

    check_result_6D(h_view, fillFlattenedIndex);
  }

  template <typename HostViewType, typename FillFunctor>
  static void check_result_7D(HostViewType h_view, FillFunctor& fillFunctor) {
    for (size_t i = 0; i < h_view.extent(0); ++i) {
      for (size_t j = 0; j < h_view.extent(1); ++j) {
        for (size_t k = 0; k < h_view.extent(2); ++k) {
          for (size_t l = 0; l < h_view.extent(3); ++l) {
            for (size_t m = 0; m < h_view.extent(4); ++m) {
              for (size_t n = 0; n < h_view.extent(5); ++n) {
                for (size_t o = 0; o < h_view.extent(6); ++o) {
                  EXPECT_EQ(h_view(i, j, k, l, m, n, o),
                            fillFunctor(i, j, k, l, m, n, o));
                }
              }
            }
          }
        }
      }
    }
  }

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_for_7D_MDTeamThreadRange(int* dims,
                                                     const int initValue) {
    using ViewType     = typename Kokkos::View<DataType*******, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];
    int n3         = dims[4];
    int n4         = dims[5];
    int n5         = dims[6];

    ViewType v("v", leagueSize, n0, n1, n2, n3, n4, n5);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3, n4, n5);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange = Kokkos::MDTeamThreadRange<Direction>(team, n0, n1,
                                                                n2, n3, n4, n5);

          Kokkos::parallel_for(
              teamRange, [=](int i, int j, int k, int l, int m, int n) {
                v(leagueRank, i, j, k, l, m, n) +=
                    fillFlattenedIndex(leagueRank, i, j, k, l, m, n);
              });
        });

    HostViewType h_view = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultHostExecutionSpace(), v);

    check_result_7D(h_view, fillFlattenedIndex);
  }

  template <typename HostViewType, typename FillFunctor>
  static void check_result_8D(HostViewType h_view, FillFunctor& fillFunctor) {
    for (size_t i = 0; i < h_view.extent(0); ++i) {
      for (size_t j = 0; j < h_view.extent(1); ++j) {
        for (size_t k = 0; k < h_view.extent(2); ++k) {
          for (size_t l = 0; l < h_view.extent(3); ++l) {
            for (size_t m = 0; m < h_view.extent(4); ++m) {
              for (size_t n = 0; n < h_view.extent(5); ++n) {
                for (size_t o = 0; o < h_view.extent(6); ++o) {
                  for (size_t p = 0; p < h_view.extent(7); ++p) {
                    EXPECT_EQ(h_view(i, j, k, l, m, n, o, p),
                              fillFunctor(i, j, k, l, m, n, o, p));
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_for_8D_MDTeamThreadRange(int* dims,
                                                     const int initValue) {
    using ViewType     = typename Kokkos::View<DataType********, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];
    int n3         = dims[4];
    int n4         = dims[5];
    int n5         = dims[6];
    int n6         = dims[7];

    ViewType v("v", leagueSize, n0, n1, n2, n3, n4, n5, n6);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3, n4, n5,
                                          n6);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange = Kokkos::MDTeamThreadRange<Direction>(
              team, n0, n1, n2, n3, n4, n5, n6);

          Kokkos::parallel_for(
              teamRange, [=](int i, int j, int k, int l, int m, int n, int o) {
                v(leagueRank, i, j, k, l, m, n, o) +=
                    fillFlattenedIndex(leagueRank, i, j, k, l, m, n, o);
              });
        });

    HostViewType h_view = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultHostExecutionSpace(), v);

    check_result_8D(h_view, fillFlattenedIndex);
  }

  template <Kokkos::Iterate OuterDirection = Kokkos::Iterate::Default,
            Kokkos::Iterate InnerDirection = Kokkos::Iterate::Default>
  static void test_parallel_for_4D_MDThreadVectorRange(int* dims,
                                                       const int initValue) {
    using ViewType     = typename Kokkos::View<DataType****, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];

    ViewType v("v", leagueSize, n0, n1, n2);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamThreadRange = Kokkos::TeamThreadRange(team, n0);
          auto teamRange =
              Kokkos::MDThreadVectorRange<OuterDirection, InnerDirection>(
                  team, n1, n2);

          Kokkos::parallel_for(teamThreadRange, [=](int i) {
            Kokkos::parallel_for(teamRange, [=](int j, int k) {
              v(leagueRank, i, j, k) += fillFlattenedIndex(leagueRank, i, j, k);
            });
          });
        });

    HostViewType h_view = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultHostExecutionSpace(), v);

    check_result_4D(h_view, fillFlattenedIndex);
  }

  template <Kokkos::Iterate OuterDirection = Kokkos::Iterate::Default,
            Kokkos::Iterate InnerDirection = Kokkos::Iterate::Default>
  static void test_parallel_for_5D_MDThreadVectorRange(int* dims,
                                                       const int initValue) {
    using ViewType     = typename Kokkos::View<DataType*****, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];
    int n3         = dims[4];

    ViewType v("v", leagueSize, n0, n1, n2, n3);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamThreadRange = Kokkos::TeamThreadRange(team, n0);
          auto teamRange =
              Kokkos::MDThreadVectorRange<OuterDirection, InnerDirection>(
                  team, n1, n2, n3);

          Kokkos::parallel_for(teamThreadRange, [=](int i) {
            Kokkos::parallel_for(teamRange, [=](int j, int k, int l) {
              v(leagueRank, i, j, k, l) +=
                  fillFlattenedIndex(leagueRank, i, j, k, l);
            });
          });
        });

    HostViewType h_view = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultHostExecutionSpace(), v);

    check_result_5D(h_view, fillFlattenedIndex);
  }

  template <Kokkos::Iterate OuterDirection = Kokkos::Iterate::Default,
            Kokkos::Iterate InnerDirection = Kokkos::Iterate::Default>
  static void test_parallel_for_6D_MDThreadVectorRange(int* dims,
                                                       const int initValue) {
    using ViewType     = typename Kokkos::View<DataType******, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];
    int n3         = dims[4];
    int n4         = dims[5];

    ViewType v("v", leagueSize, n0, n1, n2, n3, n4);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3, n4);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamThreadRange = Kokkos::TeamThreadRange(team, n0);
          auto teamRange =
              Kokkos::MDThreadVectorRange<OuterDirection, InnerDirection>(
                  team, n1, n2, n3, n4);

          Kokkos::parallel_for(teamThreadRange, [=](int i) {
            Kokkos::parallel_for(teamRange, [=](int j, int k, int l, int m) {
              v(leagueRank, i, j, k, l, m) +=
                  fillFlattenedIndex(leagueRank, i, j, k, l, m);
            });
          });
        });

    HostViewType h_view = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultHostExecutionSpace(), v);

    check_result_6D(h_view, fillFlattenedIndex);
  }

  template <Kokkos::Iterate OuterDirection = Kokkos::Iterate::Default,
            Kokkos::Iterate InnerDirection = Kokkos::Iterate::Default>
  static void test_parallel_for_7D_MDThreadVectorRange(int* dims,
                                                       const int initValue) {
    using ViewType     = typename Kokkos::View<DataType*******, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];
    int n3         = dims[4];
    int n4         = dims[5];
    int n5         = dims[6];

    ViewType v("v", leagueSize, n0, n1, n2, n3, n4, n5);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3, n4, n5);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamThreadRange = Kokkos::TeamThreadRange(team, n0);
          auto teamRange =
              Kokkos::MDThreadVectorRange<OuterDirection, InnerDirection>(
                  team, n1, n2, n3, n4, n5);

          Kokkos::parallel_for(teamThreadRange, [=](int i) {
            Kokkos::parallel_for(
                teamRange, [=](int j, int k, int l, int m, int n) {
                  v(leagueRank, i, j, k, l, m, n) +=
                      fillFlattenedIndex(leagueRank, i, j, k, l, m, n);
                });
          });
        });

    HostViewType h_view = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultHostExecutionSpace(), v);

    check_result_7D(h_view, fillFlattenedIndex);
  }

  template <Kokkos::Iterate OuterDirection = Kokkos::Iterate::Default,
            Kokkos::Iterate InnerDirection = Kokkos::Iterate::Default>
  static void test_parallel_for_8D_MDThreadVectorRange(int* dims,
                                                       const int initValue) {
    using ViewType     = typename Kokkos::View<DataType********, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];
    int n3         = dims[4];
    int n4         = dims[5];
    int n5         = dims[6];
    int n6         = dims[7];

    ViewType v("v", leagueSize, n0, n1, n2, n3, n4, n5, n6);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3, n4, n5,
                                          n6);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamThreadRange = Kokkos::TeamThreadRange(team, n0);
          auto teamRange =
              Kokkos::MDThreadVectorRange<OuterDirection, InnerDirection>(
                  team, n1, n2, n3, n4, n5, n6);

          Kokkos::parallel_for(teamThreadRange, [=](int i) {
            Kokkos::parallel_for(
                teamRange, [=](int j, int k, int l, int m, int n, int o) {
                  v(leagueRank, i, j, k, l, m, n, o) +=
                      fillFlattenedIndex(leagueRank, i, j, k, l, m, n, o);
                });
          });
        });

    HostViewType h_view = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultHostExecutionSpace(), v);

    check_result_8D(h_view, fillFlattenedIndex);
  }

  template <Kokkos::Iterate OuterDirection = Kokkos::Iterate::Default,
            Kokkos::Iterate InnerDirection = Kokkos::Iterate::Default>
  static void test_parallel_for_3D_MDTeamVectorRange(int* dims,
                                                     const int initValue) {
    using ViewType     = typename Kokkos::View<DataType***, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];

    ViewType v("v", leagueSize, n0, n1);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange =
              Kokkos::MDTeamVectorRange<OuterDirection, InnerDirection>(team,
                                                                        n0, n1);

          Kokkos::parallel_for(teamRange, [=](int i, int j) {
            v(leagueRank, i, j) += fillFlattenedIndex(leagueRank, i, j);
          });
        });

    HostViewType h_view = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultHostExecutionSpace(), v);

    check_result_3D(h_view, fillFlattenedIndex);
  }

  template <Kokkos::Iterate OuterDirection = Kokkos::Iterate::Default,
            Kokkos::Iterate InnerDirection = Kokkos::Iterate::Default>
  static void test_parallel_for_4D_MDTeamVectorRange(int* dims,
                                                     const int initValue) {
    using ViewType     = typename Kokkos::View<DataType****, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];

    ViewType v("v", leagueSize, n0, n1, n2);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange =
              Kokkos::MDTeamVectorRange<OuterDirection, InnerDirection>(
                  team, n0, n1, n2);

          Kokkos::parallel_for(teamRange, [=](int i, int j, int k) {
            v(leagueRank, i, j, k) += fillFlattenedIndex(leagueRank, i, j, k);
          });
        });

    HostViewType h_view = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultHostExecutionSpace(), v);

    check_result_4D(h_view, fillFlattenedIndex);
  }

  template <Kokkos::Iterate OuterDirection = Kokkos::Iterate::Default,
            Kokkos::Iterate InnerDirection = Kokkos::Iterate::Default>
  static void test_parallel_for_5D_MDTeamVectorRange(int* dims,
                                                     const int initValue) {
    using ViewType     = typename Kokkos::View<DataType*****, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];
    int n3         = dims[4];

    ViewType v("v", leagueSize, n0, n1, n2, n3);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange =
              Kokkos::MDTeamVectorRange<OuterDirection, InnerDirection>(
                  team, n0, n1, n2, n3);

          Kokkos::parallel_for(teamRange, [=](int i, int j, int k, int l) {
            v(leagueRank, i, j, k, l) +=
                fillFlattenedIndex(leagueRank, i, j, k, l);
          });
        });

    HostViewType h_view = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultHostExecutionSpace(), v);

    check_result_5D(h_view, fillFlattenedIndex);
  }

  template <Kokkos::Iterate OuterDirection = Kokkos::Iterate::Default,
            Kokkos::Iterate InnerDirection = Kokkos::Iterate::Default>
  static void test_parallel_for_6D_MDTeamVectorRange(int* dims,
                                                     const int initValue) {
    using ViewType     = typename Kokkos::View<DataType******, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];
    int n3         = dims[4];
    int n4         = dims[5];

    ViewType v("v", leagueSize, n0, n1, n2, n3, n4);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3, n4);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange =
              Kokkos::MDTeamVectorRange<OuterDirection, InnerDirection>(
                  team, n0, n1, n2, n3, n4);

          Kokkos::parallel_for(
              teamRange, [=](int i, int j, int k, int l, int m) {
                v(leagueRank, i, j, k, l, m) +=
                    fillFlattenedIndex(leagueRank, i, j, k, l, m);
              });
        });

    HostViewType h_view = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultHostExecutionSpace(), v);

    check_result_6D(h_view, fillFlattenedIndex);
  }

  template <Kokkos::Iterate OuterDirection = Kokkos::Iterate::Default,
            Kokkos::Iterate InnerDirection = Kokkos::Iterate::Default>
  static void test_parallel_for_7D_MDTeamVectorRange(int* dims,
                                                     const int initValue) {
    using ViewType     = typename Kokkos::View<DataType*******, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];
    int n3         = dims[4];
    int n4         = dims[5];
    int n5         = dims[6];

    ViewType v("v", leagueSize, n0, n1, n2, n3, n4, n5);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3, n4, n5);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange =
              Kokkos::MDTeamVectorRange<OuterDirection, InnerDirection>(
                  team, n0, n1, n2, n3, n4, n5);

          Kokkos::parallel_for(
              teamRange, [=](int i, int j, int k, int l, int m, int n) {
                v(leagueRank, i, j, k, l, m, n) +=
                    fillFlattenedIndex(leagueRank, i, j, k, l, m, n);
              });
        });

    HostViewType h_view = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultHostExecutionSpace(), v);

    check_result_7D(h_view, fillFlattenedIndex);
  }

  template <Kokkos::Iterate OuterDirection = Kokkos::Iterate::Default,
            Kokkos::Iterate InnerDirection = Kokkos::Iterate::Default>
  static void test_parallel_for_8D_MDTeamVectorRange(int* dims,
                                                     const int initValue) {
    using ViewType     = typename Kokkos::View<DataType********, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];
    int n3         = dims[4];
    int n4         = dims[5];
    int n5         = dims[6];
    int n6         = dims[7];

    ViewType v("v", leagueSize, n0, n1, n2, n3, n4, n5, n6);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3, n4, n5,
                                          n6);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange =
              Kokkos::MDTeamVectorRange<OuterDirection, InnerDirection>(
                  team, n0, n1, n2, n3, n4, n5, n6);

          Kokkos::parallel_for(
              teamRange, [=](int i, int j, int k, int l, int m, int n, int o) {
                v(leagueRank, i, j, k, l, m, n, o) +=
                    fillFlattenedIndex(leagueRank, i, j, k, l, m, n, o);
              });
        });

    HostViewType h_view = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultHostExecutionSpace(), v);

    check_result_8D(h_view, fillFlattenedIndex);
  }
};

template <typename ExecSpace>
struct TestMDTeamParallelReduce {
  using DataType = int;
  using TeamType = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_reduce_for_3D_MDTeamThreadRange(
      int* dims, const int initValue) {
    using ViewType = typename Kokkos::View<DataType***, ExecSpace>;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];

    ViewType v("v", leagueSize, n0, n1);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1);

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {leagueSize, n0, n1}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
          v(i, j, k) = fillFlattenedIndex(i, j, k);
        });

    int finalSum = 0;

    Kokkos::parallel_reduce(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(TeamType const& team, int& leagueSum) {
          auto leagueRank = team.league_rank();
          int teamSum     = 0;

          Kokkos::parallel_reduce(
              Kokkos::MDTeamThreadRange<Direction>(team, n0, n1),
              [=](const int& i, const int& j, int& threadSum) {
                threadSum += v(leagueRank, i, j);
              },
              teamSum);

          Kokkos::single(Kokkos::PerTeam(team),
                         [&leagueSum, teamSum]() { leagueSum += teamSum; });
        },
        finalSum);

    int firstValue  = 0;
    int lastValue   = (leagueSize * n0 * n1 - 1);
    int numValues   = (leagueSize * n0 * n1);
    int expectedSum = numValues * (firstValue + lastValue) / 2;

    EXPECT_EQ(finalSum, expectedSum);
  }

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_reduce_for_4D_MDTeamThreadRange(
      int* dims, const int initValue) {
    using ViewType = typename Kokkos::View<DataType****, ExecSpace>;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];

    ViewType v("v", leagueSize, n0, n1, n2);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2);

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, 0, 0, 0},
                                               {leagueSize, n0, n1, n2}),
        KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
          v(i, j, k, l) = fillFlattenedIndex(i, j, k, l);
        });

    int finalSum = 0;

    Kokkos::parallel_reduce(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(TeamType const& team, int& leagueSum) {
          auto leagueRank = team.league_rank();
          int teamSum     = 0;

          Kokkos::parallel_reduce(
              Kokkos::MDTeamThreadRange<Direction>(team, n0, n1, n2),
              [=](const int& i, const int& j, const int& k, int& threadSum) {
                threadSum += v(leagueRank, i, j, k);
              },
              teamSum);

          Kokkos::single(Kokkos::PerTeam(team),
                         [&leagueSum, teamSum]() { leagueSum += teamSum; });
        },
        finalSum);

    int firstValue  = 0;
    int lastValue   = (leagueSize * n0 * n1 * n2 - 1);
    int numValues   = (leagueSize * n0 * n1 * n2);
    int expectedSum = numValues * (firstValue + lastValue) / 2;

    EXPECT_EQ(finalSum, expectedSum);
  }

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_reduce_for_5D_MDTeamThreadRange(
      int* dims, const int initValue) {
    using ViewType = typename Kokkos::View<DataType*****, ExecSpace>;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];
    int n3         = dims[4];

    ViewType v("v", leagueSize, n0, n1, n2, n3);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3);

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<5>>({0, 0, 0, 0, 0},
                                               {leagueSize, n0, n1, n2, n3}),
        KOKKOS_LAMBDA(const int i, const int j, const int k, const int l,
                      const int m) {
          v(i, j, k, l, m) = fillFlattenedIndex(i, j, k, l, m);
        });

    int finalSum = 0;

    Kokkos::parallel_reduce(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(TeamType const& team, int& leagueSum) {
          auto leagueRank = team.league_rank();
          int teamSum     = 0;

          Kokkos::parallel_reduce(
              Kokkos::MDTeamThreadRange<Direction>(team, n0, n1, n2, n3),
              [=](const int& i, const int& j, const int& k, const int& l,
                  int& threadSum) { threadSum += v(leagueRank, i, j, k, l); },
              teamSum);

          Kokkos::single(Kokkos::PerTeam(team),
                         [&leagueSum, teamSum]() { leagueSum += teamSum; });
        },
        finalSum);

    int firstValue  = 0;
    int lastValue   = (leagueSize * n0 * n1 * n2 * n3 - 1);
    int numValues   = (leagueSize * n0 * n1 * n2 * n3);
    int expectedSum = numValues * (firstValue + lastValue) / 2;

    EXPECT_EQ(finalSum, expectedSum);
  }

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_reduce_for_6D_MDTeamThreadRange(
      int* dims, const int initValue) {
    using ViewType = typename Kokkos::View<DataType******, ExecSpace>;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];
    int n3         = dims[4];
    int n4         = dims[5];

    ViewType v("v", leagueSize, n0, n1, n2, n3, n4);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3, n4);

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<6>>(
            {0, 0, 0, 0, 0, 0}, {leagueSize, n0, n1, n2, n3, n4}),
        KOKKOS_LAMBDA(const int i, const int j, const int k, const int l,
                      const int m, const int n) {
          v(i, j, k, l, m, n) = fillFlattenedIndex(i, j, k, l, m, n);
        });

    int finalSum = 0;

    Kokkos::parallel_reduce(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(TeamType const& team, int& leagueSum) {
          auto leagueRank = team.league_rank();
          int teamSum     = 0;

          Kokkos::parallel_reduce(
              Kokkos::MDTeamThreadRange<Direction>(team, n0, n1, n2, n3, n4),
              [=](const int& i, const int& j, const int& k, const int& l,
                  const int& m, int& threadSum) {
                threadSum += v(leagueRank, i, j, k, l, m);
              },
              teamSum);

          Kokkos::single(Kokkos::PerTeam(team),
                         [&leagueSum, teamSum]() { leagueSum += teamSum; });
        },
        finalSum);

    int firstValue  = 0;
    int lastValue   = (leagueSize * n0 * n1 * n2 * n3 * n4 - 1);
    int numValues   = (leagueSize * n0 * n1 * n2 * n3 * n4);
    int expectedSum = numValues * (firstValue + lastValue) / 2;

    EXPECT_EQ(finalSum, expectedSum);
  }

  template <Kokkos::Iterate OuterDirection = Kokkos::Iterate::Default,
            Kokkos::Iterate InnerDirection = Kokkos::Iterate::Default>
  static void test_parallel_reduce_for_4D_MDThreadVectorRange(
      int* dims, const int initValue) {
    using ViewType = typename Kokkos::View<DataType****, ExecSpace>;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];

    ViewType v("v", leagueSize, n0, n1, n2);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2);

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, 0, 0, 0},
                                               {leagueSize, n0, n1, n2}),
        KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
          v(i, j, k, l) = fillFlattenedIndex(i, j, k, l);
        });

    int finalSum = 0;

    Kokkos::parallel_reduce(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(TeamType const& team, int& leagueSum) {
          auto leagueRank = team.league_rank();
          int teamSum     = 0;

          auto teamThreadRange = Kokkos::TeamThreadRange(team, n0);
          auto threadVectorRange =
              Kokkos::MDThreadVectorRange<OuterDirection, InnerDirection>(
                  team, n1, n2);

          Kokkos::parallel_for(teamThreadRange, [=, &teamSum](const int& i) {
            int threadSum = 0;
            Kokkos::parallel_reduce(
                threadVectorRange,
                [=](const int& j, const int& k, int& vectorSum) {
                  vectorSum += v(leagueRank, i, j, k);
                },
                threadSum);

            teamSum += threadSum;
          });

          leagueSum += teamSum;
        },
        finalSum);

    int firstValue  = 0;
    int lastValue   = (leagueSize * n0 * n1 * n2 - 1);
    int numValues   = (leagueSize * n0 * n1 * n2);
    int expectedSum = numValues * (firstValue + lastValue) / 2;

    EXPECT_EQ(finalSum, expectedSum);
  }

  template <Kokkos::Iterate OuterDirection = Kokkos::Iterate::Default,
            Kokkos::Iterate InnerDirection = Kokkos::Iterate::Default>
  static void test_parallel_reduce_for_5D_MDThreadVectorRange(
      int* dims, const int initValue) {
    using ViewType = typename Kokkos::View<DataType*****, ExecSpace>;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];
    int n3         = dims[4];

    ViewType v("v", leagueSize, n0, n1, n2, n3);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3);

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<5>>({0, 0, 0, 0, 0},
                                               {leagueSize, n0, n1, n2, n3}),
        KOKKOS_LAMBDA(const int i, const int j, const int k, const int l,
                      const int m) {
          v(i, j, k, l, m) = fillFlattenedIndex(i, j, k, l, m);
        });

    int finalSum = 0;

    Kokkos::parallel_reduce(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(TeamType const& team, int& leagueSum) {
          auto leagueRank = team.league_rank();
          int teamSum     = 0;

          auto teamThreadRange = Kokkos::TeamThreadRange(team, n0);
          auto threadVectorRange =
              Kokkos::MDThreadVectorRange<OuterDirection, InnerDirection>(
                  team, n1, n2, n3);

          Kokkos::parallel_for(teamThreadRange, [=, &teamSum](const int& i) {
            int threadSum = 0;
            Kokkos::parallel_reduce(
                threadVectorRange,
                [=](const int& j, const int& k, const int& l, int& vectorSum) {
                  vectorSum += v(leagueRank, i, j, k, l);
                },
                threadSum);

            teamSum += threadSum;
          });

          leagueSum += teamSum;
        },
        finalSum);

    int firstValue  = 0;
    int lastValue   = (leagueSize * n0 * n1 * n2 * n3 - 1);
    int numValues   = (leagueSize * n0 * n1 * n2 * n3);
    int expectedSum = numValues * (firstValue + lastValue) / 2;

    EXPECT_EQ(finalSum, expectedSum);
  }

  template <Kokkos::Iterate OuterDirection = Kokkos::Iterate::Default,
            Kokkos::Iterate InnerDirection = Kokkos::Iterate::Default>
  static void test_parallel_reduce_for_6D_MDThreadVectorRange(
      int* dims, const int initValue) {
    using ViewType = typename Kokkos::View<DataType******, ExecSpace>;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];
    int n3         = dims[4];
    int n4         = dims[5];

    ViewType v("v", leagueSize, n0, n1, n2, n3, n4);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3, n4);

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<6>>(
            {0, 0, 0, 0, 0, 0}, {leagueSize, n0, n1, n2, n3, n4}),
        KOKKOS_LAMBDA(const int i, const int j, const int k, const int l,
                      const int m, const int n) {
          v(i, j, k, l, m, n) = fillFlattenedIndex(i, j, k, l, m, n);
        });

    int finalSum = 0;

    Kokkos::parallel_reduce(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(TeamType const& team, int& leagueSum) {
          auto leagueRank = team.league_rank();
          int teamSum     = 0;

          auto teamThreadRange = Kokkos::TeamThreadRange(team, n0);
          auto threadVectorRange =
              Kokkos::MDThreadVectorRange<OuterDirection, InnerDirection>(
                  team, n1, n2, n3, n4);

          Kokkos::parallel_for(teamThreadRange, [=, &teamSum](const int& i) {
            int threadSum = 0;
            Kokkos::parallel_reduce(
                threadVectorRange,
                [=](const int& j, const int& k, const int& l, const int& m,
                    int& vectorSum) {
                  vectorSum += v(leagueRank, i, j, k, l, m);
                },
                threadSum);

            teamSum += threadSum;
          });

          leagueSum += teamSum;
        },
        finalSum);

    int firstValue  = 0;
    int lastValue   = (leagueSize * n0 * n1 * n2 * n3 * n4 - 1);
    int numValues   = (leagueSize * n0 * n1 * n2 * n3 * n4);
    int expectedSum = numValues * (firstValue + lastValue) / 2;

    EXPECT_EQ(finalSum, expectedSum);
  }

  template <Kokkos::Iterate OuterDirection = Kokkos::Iterate::Default,
            Kokkos::Iterate InnerDirection = Kokkos::Iterate::Default>
  static void test_parallel_reduce_for_4D_MDTeamVectorRange(
      int* dims, const int initValue) {
    using ViewType = typename Kokkos::View<DataType****, ExecSpace>;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];

    ViewType v("v", leagueSize, n0, n1, n2);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2);

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, 0, 0, 0},
                                               {leagueSize, n0, n1, n2}),
        KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
          v(i, j, k, l) = fillFlattenedIndex(i, j, k, l);
        });

    int finalSum = 0;

    Kokkos::parallel_reduce(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(TeamType const& team, int& leagueSum) {
          auto leagueRank = team.league_rank();
          int teamSum     = 0;

          auto teamVectorRange =
              Kokkos::MDTeamVectorRange<OuterDirection, InnerDirection>(
                  team, n0, n1, n2);

          Kokkos::parallel_reduce(
              teamVectorRange,
              [=](const int& i, const int& j, const int& k, int& vectorSum) {
                vectorSum += v(leagueRank, i, j, k);
              },
              teamSum);

          Kokkos::single(Kokkos::PerTeam(team),
                         [&leagueSum, teamSum]() { leagueSum += teamSum; });
        },
        finalSum);

    int firstValue  = 0;
    int lastValue   = (leagueSize * n0 * n1 * n2 - 1);
    int numValues   = (leagueSize * n0 * n1 * n2);
    int expectedSum = numValues * (firstValue + lastValue) / 2;

    EXPECT_EQ(finalSum, expectedSum);
  }

  template <Kokkos::Iterate OuterDirection = Kokkos::Iterate::Default,
            Kokkos::Iterate InnerDirection = Kokkos::Iterate::Default>
  static void test_parallel_reduce_for_5D_MDTeamVectorRange(
      int* dims, const int initValue) {
    using ViewType = typename Kokkos::View<DataType*****, ExecSpace>;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];
    int n3         = dims[4];

    ViewType v("v", leagueSize, n0, n1, n2, n3);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3);

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<5>>({0, 0, 0, 0, 0},
                                               {leagueSize, n0, n1, n2, n3}),
        KOKKOS_LAMBDA(const int i, const int j, const int k, const int l,
                      const int m) {
          v(i, j, k, l, m) = fillFlattenedIndex(i, j, k, l, m);
        });

    int finalSum = 0;

    Kokkos::parallel_reduce(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(TeamType const& team, int& leagueSum) {
          auto leagueRank = team.league_rank();
          int teamSum     = 0;

          auto teamVectorRange =
              Kokkos::MDTeamVectorRange<OuterDirection, InnerDirection>(
                  team, n0, n1, n2, n3);

          Kokkos::parallel_reduce(
              teamVectorRange,
              [=](const int& i, const int& j, const int& k, const int& l,
                  int& vectorSum) { vectorSum += v(leagueRank, i, j, k, l); },
              teamSum);

          Kokkos::single(Kokkos::PerTeam(team),
                         [&leagueSum, teamSum]() { leagueSum += teamSum; });
        },
        finalSum);

    int firstValue  = 0;
    int lastValue   = (leagueSize * n0 * n1 * n2 * n3 - 1);
    int numValues   = (leagueSize * n0 * n1 * n2 * n3);
    int expectedSum = numValues * (firstValue + lastValue) / 2;

    EXPECT_EQ(finalSum, expectedSum);
  }

  template <Kokkos::Iterate OuterDirection = Kokkos::Iterate::Default,
            Kokkos::Iterate InnerDirection = Kokkos::Iterate::Default>
  static void test_parallel_reduce_for_6D_MDTeamVectorRange(
      int* dims, const int initValue) {
    using ViewType = typename Kokkos::View<DataType******, ExecSpace>;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];
    int n3         = dims[4];
    int n4         = dims[5];

    ViewType v("v", leagueSize, n0, n1, n2, n3, n4);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3, n4);

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<6>>(
            {0, 0, 0, 0, 0, 0}, {leagueSize, n0, n1, n2, n3, n4}),
        KOKKOS_LAMBDA(const int i, const int j, const int k, const int l,
                      const int m, const int n) {
          v(i, j, k, l, m, n) = fillFlattenedIndex(i, j, k, l, m, n);
        });

    int finalSum = 0;

    Kokkos::parallel_reduce(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(TeamType const& team, int& leagueSum) {
          auto leagueRank = team.league_rank();
          int teamSum     = 0;

          auto teamVectorRange =
              Kokkos::MDTeamVectorRange<OuterDirection, InnerDirection>(
                  team, n0, n1, n2, n3, n4);

          Kokkos::parallel_reduce(
              teamVectorRange,
              [=](const int& i, const int& j, const int& k, const int& l,
                  const int& m, int& vectorSum) {
                vectorSum += v(leagueRank, i, j, k, l, m);
              },
              teamSum);

          Kokkos::single(Kokkos::PerTeam(team),
                         [&leagueSum, teamSum]() { leagueSum += teamSum; });
        },
        finalSum);

    int firstValue  = 0;
    int lastValue   = (leagueSize * n0 * n1 * n2 * n3 * n4 - 1);
    int numValues   = (leagueSize * n0 * n1 * n2 * n3 * n4);
    int expectedSum = numValues * (firstValue + lastValue) / 2;

    EXPECT_EQ(finalSum, expectedSum);
  }
};

}  // namespace MDTeamParallelism

}  // namespace Test

/*--------------------------------------------------------------------------*/

namespace Test {

constexpr auto Left  = Kokkos::Iterate::Left;
constexpr auto Right = Kokkos::Iterate::Right;

TEST(TEST_CATEGORY, MDTeamParallelFor) {
  using namespace MDTeamParallelism;
  // int dims[] = {16, 16, 16, 16, 16, 16, 16, 16};
  int dims[] = {4, 4, 4, 4, 4, 4, 4, 4};

  const int initValue = 5;

  {
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_3D_MDTeamThreadRange<Left>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_3D_MDTeamThreadRange<Right>(dims, initValue);

    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_4D_MDTeamThreadRange<Left>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_4D_MDTeamThreadRange<Right>(dims, initValue);

    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_5D_MDTeamThreadRange<Left>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_5D_MDTeamThreadRange<Right>(dims, initValue);

    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_6D_MDTeamThreadRange<Left>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_6D_MDTeamThreadRange<Right>(dims, initValue);

    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_7D_MDTeamThreadRange<Left>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_7D_MDTeamThreadRange<Right>(dims, initValue);

    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_8D_MDTeamThreadRange<Left>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_8D_MDTeamThreadRange<Right>(dims, initValue);
  }

  {
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_4D_MDThreadVectorRange<Left, Left>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_4D_MDThreadVectorRange<Left, Right>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_4D_MDThreadVectorRange<Right, Left>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_4D_MDThreadVectorRange<Right, Right>(dims, initValue);

    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_5D_MDThreadVectorRange<Left, Left>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_5D_MDThreadVectorRange<Left, Right>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_5D_MDThreadVectorRange<Right, Left>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_5D_MDThreadVectorRange<Right, Right>(dims, initValue);

    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_6D_MDThreadVectorRange<Left, Left>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_6D_MDThreadVectorRange<Left, Right>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_6D_MDThreadVectorRange<Right, Left>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_6D_MDThreadVectorRange<Right, Right>(dims, initValue);

    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_7D_MDThreadVectorRange<Left, Left>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_7D_MDThreadVectorRange<Left, Right>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_7D_MDThreadVectorRange<Right, Left>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_7D_MDThreadVectorRange<Right, Right>(dims, initValue);

    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_8D_MDThreadVectorRange<Left, Left>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_8D_MDThreadVectorRange<Left, Right>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_8D_MDThreadVectorRange<Right, Left>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_8D_MDThreadVectorRange<Right, Right>(dims, initValue);
  }

  {
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_3D_MDTeamVectorRange<Left, Left>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_3D_MDTeamVectorRange<Left, Right>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_3D_MDTeamVectorRange<Right, Left>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_3D_MDTeamVectorRange<Right, Right>(dims, initValue);

    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_4D_MDTeamVectorRange<Left, Left>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_4D_MDTeamVectorRange<Left, Right>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_4D_MDTeamVectorRange<Right, Left>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_4D_MDTeamVectorRange<Right, Right>(dims, initValue);

    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_5D_MDTeamVectorRange<Left, Left>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_5D_MDTeamVectorRange<Left, Right>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_5D_MDTeamVectorRange<Right, Left>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_5D_MDTeamVectorRange<Right, Right>(dims, initValue);

    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_6D_MDTeamVectorRange<Left, Left>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_6D_MDTeamVectorRange<Left, Right>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_6D_MDTeamVectorRange<Right, Left>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_6D_MDTeamVectorRange<Right, Right>(dims, initValue);

    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_7D_MDTeamVectorRange<Left, Left>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_7D_MDTeamVectorRange<Left, Right>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_7D_MDTeamVectorRange<Right, Left>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_7D_MDTeamVectorRange<Right, Right>(dims, initValue);

    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_8D_MDTeamVectorRange<Left, Left>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_8D_MDTeamVectorRange<Left, Right>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_8D_MDTeamVectorRange<Right, Left>(dims, initValue);
    TestMDTeamParallelFor<TEST_EXECSPACE>::
        test_parallel_for_8D_MDTeamVectorRange<Right, Right>(dims, initValue);
  }
}  // namespace Test

TEST(TEST_CATEGORY, MDTeamParallelReduce) {
  using namespace MDTeamParallelism;
  // int dims[] = {16, 16, 16, 16, 16, 16, 16, 16};
  int dims[]          = {4, 4, 4, 4, 4, 4, 4, 4};
  const int initValue = 5;

  {
    TestMDTeamParallelReduce<TEST_EXECSPACE>::
        test_parallel_reduce_for_3D_MDTeamThreadRange<Left>(dims, initValue);
    TestMDTeamParallelReduce<TEST_EXECSPACE>::
        test_parallel_reduce_for_3D_MDTeamThreadRange<Right>(dims, initValue);

    TestMDTeamParallelReduce<TEST_EXECSPACE>::
        test_parallel_reduce_for_4D_MDTeamThreadRange<Left>(dims, initValue);
    TestMDTeamParallelReduce<TEST_EXECSPACE>::
        test_parallel_reduce_for_4D_MDTeamThreadRange<Right>(dims, initValue);

    TestMDTeamParallelReduce<TEST_EXECSPACE>::
        test_parallel_reduce_for_5D_MDTeamThreadRange<Left>(dims, initValue);
    TestMDTeamParallelReduce<TEST_EXECSPACE>::
        test_parallel_reduce_for_5D_MDTeamThreadRange<Right>(dims, initValue);

    TestMDTeamParallelReduce<TEST_EXECSPACE>::
        test_parallel_reduce_for_6D_MDTeamThreadRange<Left>(dims, initValue);
    TestMDTeamParallelReduce<TEST_EXECSPACE>::
        test_parallel_reduce_for_6D_MDTeamThreadRange<Right>(dims, initValue);
  }

  {
    TestMDTeamParallelReduce<TEST_EXECSPACE>::
        test_parallel_reduce_for_4D_MDThreadVectorRange<Left, Left>(dims,
                                                                    initValue);
    TestMDTeamParallelReduce<TEST_EXECSPACE>::
        test_parallel_reduce_for_4D_MDThreadVectorRange<Left, Right>(dims,
                                                                     initValue);
    TestMDTeamParallelReduce<TEST_EXECSPACE>::
        test_parallel_reduce_for_4D_MDThreadVectorRange<Right, Left>(dims,
                                                                     initValue);
    TestMDTeamParallelReduce<TEST_EXECSPACE>::
        test_parallel_reduce_for_4D_MDThreadVectorRange<Right, Right>(
            dims, initValue);

    TestMDTeamParallelReduce<TEST_EXECSPACE>::
        test_parallel_reduce_for_5D_MDThreadVectorRange<Left, Left>(dims,
                                                                    initValue);
    TestMDTeamParallelReduce<TEST_EXECSPACE>::
        test_parallel_reduce_for_5D_MDThreadVectorRange<Left, Right>(dims,
                                                                     initValue);
    TestMDTeamParallelReduce<TEST_EXECSPACE>::
        test_parallel_reduce_for_5D_MDThreadVectorRange<Right, Left>(dims,
                                                                     initValue);
    TestMDTeamParallelReduce<TEST_EXECSPACE>::
        test_parallel_reduce_for_5D_MDThreadVectorRange<Right, Right>(
            dims, initValue);

    TestMDTeamParallelReduce<TEST_EXECSPACE>::
        test_parallel_reduce_for_6D_MDThreadVectorRange<Left, Left>(dims,
                                                                    initValue);
    TestMDTeamParallelReduce<TEST_EXECSPACE>::
        test_parallel_reduce_for_6D_MDThreadVectorRange<Left, Right>(dims,
                                                                     initValue);
    TestMDTeamParallelReduce<TEST_EXECSPACE>::
        test_parallel_reduce_for_6D_MDThreadVectorRange<Right, Left>(dims,
                                                                     initValue);
    TestMDTeamParallelReduce<TEST_EXECSPACE>::
        test_parallel_reduce_for_6D_MDThreadVectorRange<Right, Right>(
            dims, initValue);
  }

  {
    TestMDTeamParallelReduce<TEST_EXECSPACE>::
        test_parallel_reduce_for_4D_MDTeamVectorRange<Left, Left>(dims,
                                                                  initValue);
    TestMDTeamParallelReduce<TEST_EXECSPACE>::
        test_parallel_reduce_for_4D_MDTeamVectorRange<Left, Right>(dims,
                                                                   initValue);
    TestMDTeamParallelReduce<TEST_EXECSPACE>::
        test_parallel_reduce_for_4D_MDTeamVectorRange<Right, Left>(dims,
                                                                   initValue);
    TestMDTeamParallelReduce<TEST_EXECSPACE>::
        test_parallel_reduce_for_4D_MDTeamVectorRange<Right, Right>(dims,
                                                                    initValue);

    TestMDTeamParallelReduce<TEST_EXECSPACE>::
        test_parallel_reduce_for_5D_MDTeamVectorRange<Left, Left>(dims,
                                                                  initValue);
    TestMDTeamParallelReduce<TEST_EXECSPACE>::
        test_parallel_reduce_for_5D_MDTeamVectorRange<Left, Right>(dims,
                                                                   initValue);
    TestMDTeamParallelReduce<TEST_EXECSPACE>::
        test_parallel_reduce_for_5D_MDTeamVectorRange<Right, Left>(dims,
                                                                   initValue);
    TestMDTeamParallelReduce<TEST_EXECSPACE>::
        test_parallel_reduce_for_5D_MDTeamVectorRange<Right, Right>(dims,
                                                                    initValue);

    TestMDTeamParallelReduce<TEST_EXECSPACE>::
        test_parallel_reduce_for_6D_MDTeamVectorRange<Left, Left>(dims,
                                                                  initValue);
    TestMDTeamParallelReduce<TEST_EXECSPACE>::
        test_parallel_reduce_for_6D_MDTeamVectorRange<Left, Right>(dims,
                                                                   initValue);
    TestMDTeamParallelReduce<TEST_EXECSPACE>::
        test_parallel_reduce_for_6D_MDTeamVectorRange<Right, Left>(dims,
                                                                   initValue);
    TestMDTeamParallelReduce<TEST_EXECSPACE>::
        test_parallel_reduce_for_6D_MDTeamVectorRange<Right, Right>(dims,
                                                                    initValue);
  }
}

}  // namespace Test
