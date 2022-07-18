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

namespace MDTeamRange {

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

struct TestMDTeamParallelFor {
  using DataType = int;
  using DimsType = int[8];

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
};

template <typename ExecSpace>
struct TestMDTeamThreadRangeParallelFor : public TestMDTeamParallelFor {
  using TeamType = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_for_3D_MDTeamThreadRange(DimsType const& dims) {
    using ViewType     = typename Kokkos::View<DataType***, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];

    ViewType v("v", leagueSize, n0, n1);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange =
              Kokkos::MDTeamThreadRange<Direction, 2, TeamType>(team, n0, n1);

          Kokkos::parallel_for(teamRange, [=](int i, int j) {
            v(leagueRank, i, j) += fillFlattenedIndex(leagueRank, i, j);
          });
        });

    HostViewType h_view = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultHostExecutionSpace(), v);

    check_result_3D(h_view, fillFlattenedIndex);
  }

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_for_4D_MDTeamThreadRange(DimsType const& dims) {
    using ViewType     = typename Kokkos::View<DataType****, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];

    ViewType v("v", leagueSize, n0, n1, n2);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange = Kokkos::MDTeamThreadRange<Direction, 3, TeamType>(
              team, n0, n1, n2);

          Kokkos::parallel_for(teamRange, [=](int i, int j, int k) {
            v(leagueRank, i, j, k) += fillFlattenedIndex(leagueRank, i, j, k);
          });
        });

    HostViewType h_view = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultHostExecutionSpace(), v);

    check_result_4D(h_view, fillFlattenedIndex);
  }

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_for_5D_MDTeamThreadRange(DimsType const& dims) {
    using ViewType     = typename Kokkos::View<DataType*****, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];
    int n3         = dims[4];

    ViewType v("v", leagueSize, n0, n1, n2, n3);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange = Kokkos::MDTeamThreadRange<Direction, 4, TeamType>(
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

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_for_6D_MDTeamThreadRange(DimsType const& dims) {
    using ViewType     = typename Kokkos::View<DataType******, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];
    int n3         = dims[4];
    int n4         = dims[5];

    ViewType v("v", leagueSize, n0, n1, n2, n3, n4);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3, n4);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange = Kokkos::MDTeamThreadRange<Direction, 5, TeamType>(
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

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_for_7D_MDTeamThreadRange(DimsType const& dims) {
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
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3, n4, n5);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange = Kokkos::MDTeamThreadRange<Direction, 6, TeamType>(
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

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_for_8D_MDTeamThreadRange(DimsType const& dims) {
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
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3, n4, n5,
                                          n6);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange = Kokkos::MDTeamThreadRange<Direction, 7, TeamType>(
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

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_single_direction_test(DimsType const& dims) {
    using ViewType     = typename Kokkos::View<DataType***, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int n0 = dims[0];
    int n1 = dims[1];
    int n2 = dims[2];

    ViewType v("v", n0, n1, n2);
    FillFlattenedIndex fillFlattenedIndex(n0, n1, n2);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(1, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          auto teamRange = Kokkos::MDTeamThreadRange<Direction, 3, TeamType>(
              team, n0, n1, n2);

          Kokkos::parallel_for(teamRange, [=](int i, int j, int k) {
            v(i, j, k) += fillFlattenedIndex(i, j, k);
          });
        });

    HostViewType h_view = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultHostExecutionSpace(), v);

    check_result_3D(h_view, fillFlattenedIndex);
  }
};

template <typename ExecSpace>
struct TestMDThreadVectorRangeParallelFor : public TestMDTeamParallelFor {
  using TeamType = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_for_4D_MDThreadVectorRange(DimsType const& dims) {
    using ViewType     = typename Kokkos::View<DataType****, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];

    ViewType v("v", leagueSize, n0, n1, n2);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamThreadRange = Kokkos::TeamThreadRange(team, n0);
          auto teamRange =
              Kokkos::MDThreadVectorRange<Direction, 2, TeamType>(team, n1, n2);

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

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_for_5D_MDThreadVectorRange(DimsType const& dims) {
    using ViewType     = typename Kokkos::View<DataType*****, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];
    int n3         = dims[4];

    ViewType v("v", leagueSize, n0, n1, n2, n3);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamThreadRange = Kokkos::TeamThreadRange(team, n0);
          auto teamRange = Kokkos::MDThreadVectorRange<Direction, 3, TeamType>(
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

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_for_6D_MDThreadVectorRange(DimsType const& dims) {
    using ViewType     = typename Kokkos::View<DataType******, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];
    int n3         = dims[4];
    int n4         = dims[5];

    ViewType v("v", leagueSize, n0, n1, n2, n3, n4);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3, n4);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamThreadRange = Kokkos::TeamThreadRange(team, n0);
          auto teamRange = Kokkos::MDThreadVectorRange<Direction, 4, TeamType>(
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

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_for_7D_MDThreadVectorRange(DimsType const& dims) {
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
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3, n4, n5);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamThreadRange = Kokkos::TeamThreadRange(team, n0);
          auto teamRange = Kokkos::MDThreadVectorRange<Direction, 5, TeamType>(
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

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_for_8D_MDThreadVectorRange(DimsType const& dims) {
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
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3, n4, n5,
                                          n6);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamThreadRange = Kokkos::TeamThreadRange(team, n0);
          auto teamRange = Kokkos::MDThreadVectorRange<Direction, 6, TeamType>(
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
};

template <typename ExecSpace>
struct TestMDTeamVectorRangeParallelFor : public TestMDTeamParallelFor {
  using TeamType = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_for_3D_MDTeamVectorRange(DimsType const& dims) {
    using ViewType     = typename Kokkos::View<DataType***, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];

    ViewType v("v", leagueSize, n0, n1);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange =
              Kokkos::MDTeamVectorRange<Direction, 2, TeamType>(team, n0, n1);

          Kokkos::parallel_for(teamRange, [=](int i, int j) {
            v(leagueRank, i, j) += fillFlattenedIndex(leagueRank, i, j);
          });
        });

    HostViewType h_view = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultHostExecutionSpace(), v);

    check_result_3D(h_view, fillFlattenedIndex);
  }

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_for_4D_MDTeamVectorRange(DimsType const& dims) {
    using ViewType     = typename Kokkos::View<DataType****, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];

    ViewType v("v", leagueSize, n0, n1, n2);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange = Kokkos::MDTeamVectorRange<Direction, 3, TeamType>(
              team, n0, n1, n2);

          Kokkos::parallel_for(teamRange, [=](int i, int j, int k) {
            v(leagueRank, i, j, k) += fillFlattenedIndex(leagueRank, i, j, k);
          });
        });

    HostViewType h_view = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultHostExecutionSpace(), v);

    check_result_4D(h_view, fillFlattenedIndex);
  }

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_for_5D_MDTeamVectorRange(DimsType const& dims) {
    using ViewType     = typename Kokkos::View<DataType*****, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];
    int n3         = dims[4];

    ViewType v("v", leagueSize, n0, n1, n2, n3);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange = Kokkos::MDTeamVectorRange<Direction, 4, TeamType>(
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

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_for_6D_MDTeamVectorRange(DimsType const& dims) {
    using ViewType     = typename Kokkos::View<DataType******, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];
    int n3         = dims[4];
    int n4         = dims[5];

    ViewType v("v", leagueSize, n0, n1, n2, n3, n4);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3, n4);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange = Kokkos::MDTeamVectorRange<Direction, 5, TeamType>(
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

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_for_7D_MDTeamVectorRange(DimsType const& dims) {
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
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3, n4, n5);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange = Kokkos::MDTeamVectorRange<Direction, 6, TeamType>(
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

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_for_8D_MDTeamVectorRange(DimsType const& dims) {
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
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3, n4, n5,
                                          n6);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange = Kokkos::MDTeamVectorRange<Direction, 7, TeamType>(
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

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_double_direction_test(DimsType const& dims) {
    using ViewType     = typename Kokkos::View<DataType****, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int n0 = dims[0];
    int n1 = dims[1];
    int n2 = dims[2];
    int n3 = dims[3];

    ViewType v("v", n0, n1, n2, n3);
    FillFlattenedIndex fillFlattenedIndex(n0, n1, n2, n3);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(1, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          auto teamRange = Kokkos::MDTeamVectorRange<Direction, 4, TeamType>(
              team, n0, n1, n2, n3);

          Kokkos::parallel_for(teamRange, [=](int i, int j, int k, int l) {
            v(i, j, k, l) += fillFlattenedIndex(i, j, k, l);
          });
        });

    HostViewType h_view = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultHostExecutionSpace(), v);

    check_result_4D(h_view, fillFlattenedIndex);
  }
};

struct TestMDTeamParallelReduce {
  using DataType = int;
  using DimsType = int[8];

  template <typename F>
  constexpr static auto get_expected_partial_sum(DimsType const& dims,
                                                 size_t maxRank, F const& f,
                                                 DimsType& indices,
                                                 size_t rank) {
    if (rank == maxRank) {
      return f(indices[0], indices[1], indices[2], indices[3], indices[4],
               indices[5], indices[6], indices[7]);
    }

    auto& index       = indices[rank];
    DataType accValue = 0;
    for (index = 0; index < dims[rank]; ++index) {
      accValue += get_expected_partial_sum(dims, maxRank, f, indices, rank + 1);
    }

    return accValue;
  }

  template <typename F>
  static DataType get_expected_sum(DimsType const& dims, size_t maxRank,
                                   F const& f) {
    DimsType indices = {};
    return get_expected_partial_sum(dims, maxRank, f, indices, 0);
  }
};

template <typename ExecSpace>
struct TestMDTeamThreadRangeParallelReduce : public TestMDTeamParallelReduce {
  using TeamType = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_reduce_for_3D_MDTeamThreadRange(
      DimsType const& dims) {
    using ViewType = typename Kokkos::View<DataType***, ExecSpace>;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];

    ViewType v("v", leagueSize, n0, n1);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1);

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0},
                                                          {leagueSize, n0, n1}),
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
              Kokkos::MDTeamThreadRange<Direction, 2, TeamType>(team, n0, n1),
              [=](const int& i, const int& j, int& threadSum) {
                threadSum += v(leagueRank, i, j);
              },
              teamSum);

          Kokkos::single(Kokkos::PerTeam(team),
                         [&leagueSum, teamSum]() { leagueSum += teamSum; });
        },
        finalSum);

    int expectedSum = get_expected_sum(dims, 3, fillFlattenedIndex);

    EXPECT_EQ(finalSum, expectedSum);
  }

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_reduce_for_4D_MDTeamThreadRange(
      DimsType const& dims) {
    using ViewType = typename Kokkos::View<DataType****, ExecSpace>;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];

    ViewType v("v", leagueSize, n0, n1, n2);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2);

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>(
            {0, 0, 0, 0}, {leagueSize, n0, n1, n2}),
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
              Kokkos::MDTeamThreadRange<Direction, 3, TeamType>(team, n0, n1,
                                                                n2),
              [=](const int& i, const int& j, const int& k, int& threadSum) {
                threadSum += v(leagueRank, i, j, k);
              },
              teamSum);

          Kokkos::single(Kokkos::PerTeam(team),
                         [&leagueSum, teamSum]() { leagueSum += teamSum; });
        },
        finalSum);

    int expectedSum = get_expected_sum(dims, 4, fillFlattenedIndex);

    EXPECT_EQ(finalSum, expectedSum);
  }

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_reduce_for_5D_MDTeamThreadRange(
      DimsType const& dims) {
    using ViewType = typename Kokkos::View<DataType*****, ExecSpace>;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];
    int n3         = dims[4];

    ViewType v("v", leagueSize, n0, n1, n2, n3);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3);

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<5>>(
            {0, 0, 0, 0, 0}, {leagueSize, n0, n1, n2, n3}),
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
              Kokkos::MDTeamThreadRange<Direction, 4, TeamType>(team, n0, n1,
                                                                n2, n3),
              [=](const int& i, const int& j, const int& k, const int& l,
                  int& threadSum) { threadSum += v(leagueRank, i, j, k, l); },
              teamSum);

          Kokkos::single(Kokkos::PerTeam(team),
                         [&leagueSum, teamSum]() { leagueSum += teamSum; });
        },
        finalSum);

    int expectedSum = get_expected_sum(dims, 5, fillFlattenedIndex);

    EXPECT_EQ(finalSum, expectedSum);
  }

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_reduce_for_6D_MDTeamThreadRange(
      DimsType const& dims) {
    using ViewType = typename Kokkos::View<DataType******, ExecSpace>;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];
    int n3         = dims[4];
    int n4         = dims[5];

    ViewType v("v", leagueSize, n0, n1, n2, n3, n4);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3, n4);

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<6>>(
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
              Kokkos::MDTeamThreadRange<Direction, 5, TeamType>(team, n0, n1,
                                                                n2, n3, n4),
              [=](const int& i, const int& j, const int& k, const int& l,
                  const int& m, int& threadSum) {
                threadSum += v(leagueRank, i, j, k, l, m);
              },
              teamSum);

          Kokkos::single(Kokkos::PerTeam(team),
                         [&leagueSum, teamSum]() { leagueSum += teamSum; });
        },
        finalSum);

    int expectedSum = get_expected_sum(dims, 6, fillFlattenedIndex);

    EXPECT_EQ(finalSum, expectedSum);
  }

  // MDRangePolicy only allows up to rank of 6. Because of this, expectedSum
  // array had to be constructed from a nested parallel_for loop.
  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_reduce_for_7D_MDTeamThreadRange(
      DimsType const& dims) {
    using ViewType = typename Kokkos::View<DataType*******, ExecSpace>;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];
    int n3         = dims[4];
    int n4         = dims[5];
    int n5         = dims[6];

    ViewType v("v", leagueSize, n0, n1, n2, n3, n4, n5);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3, n4, n5);
    auto mdRangePolicy = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<6>>(
        {0, 0, 0, 0, 0, 0}, {leagueSize, n0, n1, n2, n3, n4});

    Kokkos::parallel_for(
        mdRangePolicy,
        KOKKOS_LAMBDA(const int leagueRank, const int i, const int j,
                      const int k, const int l, const int m) {
          for (int n = 0; n < n5; ++n) {
            v(leagueRank, i, j, k, l, m, n) =
                fillFlattenedIndex(leagueRank, i, j, k, l, m, n);
          }
        });

    int finalSum = 0;

    Kokkos::parallel_reduce(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(TeamType const& team, int& leagueSum) {
          auto leagueRank = team.league_rank();
          int teamSum     = 0;

          Kokkos::parallel_reduce(
              Kokkos::MDTeamThreadRange<Direction, 6, TeamType>(team, n0, n1,
                                                                n2, n3, n4, n5),
              [=](const int& i, const int& j, const int& k, const int& l,
                  const int& m, const int& n, int& threadSum) {
                threadSum += v(leagueRank, i, j, k, l, m, n);
              },
              teamSum);

          Kokkos::single(Kokkos::PerTeam(team),
                         [&leagueSum, teamSum]() { leagueSum += teamSum; });
        },
        finalSum);

    int expectedSum = get_expected_sum(dims, 7, fillFlattenedIndex);

    EXPECT_EQ(finalSum, expectedSum);
  }

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_reduce_for_8D_MDTeamThreadRange(
      DimsType const& dims) {
    using ViewType = typename Kokkos::View<DataType********, ExecSpace>;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];
    int n3         = dims[4];
    int n4         = dims[5];
    int n5         = dims[6];
    int n6         = dims[7];

    ViewType v("v", leagueSize, n0, n1, n2, n3, n4, n5, n6);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3, n4, n5,
                                          n6);
    auto mdRangePolicy = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<6>>(
        {0, 0, 0, 0, 0, 0}, {leagueSize, n0, n1, n2, n3, n4});

    Kokkos::parallel_for(
        mdRangePolicy,
        KOKKOS_LAMBDA(const int leagueRank, const int i, const int j,
                      const int k, const int l, const int m) {
          for (int n = 0; n < n5; ++n) {
            for (int o = 0; o < n6; ++o) {
              v(leagueRank, i, j, k, l, m, n, o) =
                  fillFlattenedIndex(leagueRank, i, j, k, l, m, n, o);
            }
          }
        });

    int finalSum = 0;

    Kokkos::parallel_reduce(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(TeamType const& team, int& leagueSum) {
          auto leagueRank = team.league_rank();
          int teamSum     = 0;

          Kokkos::parallel_reduce(
              Kokkos::MDTeamThreadRange<Direction, 7, TeamType>(
                  team, n0, n1, n2, n3, n4, n5, n6),
              [=](const int& i, const int& j, const int& k, const int& l,
                  const int& m, const int& n, const int& o, int& threadSum) {
                threadSum += v(leagueRank, i, j, k, l, m, n, o);
              },
              teamSum);

          Kokkos::single(Kokkos::PerTeam(team),
                         [&leagueSum, teamSum]() { leagueSum += teamSum; });
        },
        finalSum);

    int expectedSum = get_expected_sum(dims, 8, fillFlattenedIndex);

    EXPECT_EQ(finalSum, expectedSum);
  }
};

template <typename ExecSpace>
struct TestMDThreadVectorRangeParallelReduce : public TestMDTeamParallelReduce {
  using TeamType = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_reduce_for_4D_MDThreadVectorRange(
      DimsType const& dims) {
    using ViewType = typename Kokkos::View<DataType****, ExecSpace>;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];

    ViewType v("v", leagueSize, n0, n1, n2);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2);

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>(
            {0, 0, 0, 0}, {leagueSize, n0, n1, n2}),
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
              Kokkos::MDThreadVectorRange<Direction, 2, TeamType>(team, n1, n2);

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

    int expectedSum = get_expected_sum(dims, 4, fillFlattenedIndex);

    EXPECT_EQ(finalSum, expectedSum);
  }

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_reduce_for_5D_MDThreadVectorRange(
      DimsType const& dims) {
    using ViewType = typename Kokkos::View<DataType*****, ExecSpace>;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];
    int n3         = dims[4];

    ViewType v("v", leagueSize, n0, n1, n2, n3);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3);

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<5>>(
            {0, 0, 0, 0, 0}, {leagueSize, n0, n1, n2, n3}),
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
              Kokkos::MDThreadVectorRange<Direction, 3, TeamType>(team, n1, n2,
                                                                  n3);

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

    int expectedSum = get_expected_sum(dims, 5, fillFlattenedIndex);

    EXPECT_EQ(finalSum, expectedSum);
  }

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_reduce_for_6D_MDThreadVectorRange(
      DimsType const& dims) {
    using ViewType = typename Kokkos::View<DataType******, ExecSpace>;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];
    int n3         = dims[4];
    int n4         = dims[5];

    ViewType v("v", leagueSize, n0, n1, n2, n3, n4);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3, n4);

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<6>>(
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
              Kokkos::MDThreadVectorRange<Direction, 4, TeamType>(team, n1, n2,
                                                                  n3, n4);

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

    int expectedSum = get_expected_sum(dims, 6, fillFlattenedIndex);

    EXPECT_EQ(finalSum, expectedSum);
  }

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_reduce_for_7D_MDThreadVectorRange(
      DimsType const& dims) {
    using ViewType = typename Kokkos::View<DataType*******, ExecSpace>;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];
    int n3         = dims[4];
    int n4         = dims[5];
    int n5         = dims[6];

    ViewType v("v", leagueSize, n0, n1, n2, n3, n4, n5);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3, n4, n5);
    auto mdRangePolicy = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<6>>(
        {0, 0, 0, 0, 0, 0}, {leagueSize, n0, n1, n2, n3, n4});

    Kokkos::parallel_for(
        mdRangePolicy,
        KOKKOS_LAMBDA(const int leagueRank, const int i, const int j,
                      const int k, const int l, const int m) {
          for (int n = 0; n < n5; ++n) {
            v(leagueRank, i, j, k, l, m, n) =
                fillFlattenedIndex(leagueRank, i, j, k, l, m, n);
          }
        });

    int finalSum = 0;

    Kokkos::parallel_reduce(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(TeamType const& team, int& leagueSum) {
          auto leagueRank = team.league_rank();
          int teamSum     = 0;

          auto teamThreadRange = Kokkos::TeamThreadRange(team, n0);
          auto threadVectorRange =
              Kokkos::MDThreadVectorRange<Direction, 5, TeamType>(team, n1, n2,
                                                                  n3, n4, n5);

          Kokkos::parallel_for(teamThreadRange, [=, &teamSum](const int& i) {
            int threadSum = 0;
            Kokkos::parallel_reduce(
                threadVectorRange,
                [=](const int& j, const int& k, const int& l, const int& m,
                    const int& n, int& vectorSum) {
                  vectorSum += v(leagueRank, i, j, k, l, m, n);
                },
                threadSum);

            teamSum += threadSum;
          });

          leagueSum += teamSum;
        },
        finalSum);

    int expectedSum = get_expected_sum(dims, 7, fillFlattenedIndex);

    EXPECT_EQ(finalSum, expectedSum);
  }

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_reduce_for_8D_MDThreadVectorRange(
      DimsType const& dims) {
    using ViewType = typename Kokkos::View<DataType********, ExecSpace>;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];
    int n3         = dims[4];
    int n4         = dims[5];
    int n5         = dims[6];
    int n6         = dims[7];

    ViewType v("v", leagueSize, n0, n1, n2, n3, n4, n5, n6);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3, n4, n5,
                                          n6);
    auto mdRangePolicy = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<6>>(
        {0, 0, 0, 0, 0, 0}, {leagueSize, n0, n1, n2, n3, n4});

    Kokkos::parallel_for(
        mdRangePolicy,
        KOKKOS_LAMBDA(const int leagueRank, const int i, const int j,
                      const int k, const int l, const int m) {
          for (int n = 0; n < n5; ++n) {
            for (int o = 0; o < n6; ++o) {
              v(leagueRank, i, j, k, l, m, n, o) =
                  fillFlattenedIndex(leagueRank, i, j, k, l, m, n, o);
            }
          }
        });

    int finalSum = 0;

    Kokkos::parallel_reduce(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(TeamType const& team, int& leagueSum) {
          auto leagueRank = team.league_rank();
          int teamSum     = 0;

          auto teamThreadRange = Kokkos::TeamThreadRange(team, n0);
          auto threadVectorRange =
              Kokkos::MDThreadVectorRange<Direction, 6, TeamType>(
                  team, n1, n2, n3, n4, n5, n6);

          Kokkos::parallel_for(teamThreadRange, [=, &teamSum](const int& i) {
            int threadSum = 0;
            Kokkos::parallel_reduce(
                threadVectorRange,
                [=](const int& j, const int& k, const int& l, const int& m,
                    const int& n, const int& o, int& vectorSum) {
                  vectorSum += v(leagueRank, i, j, k, l, m, n, o);
                },
                threadSum);

            teamSum += threadSum;
          });

          leagueSum += teamSum;
        },
        finalSum);

    int expectedSum = get_expected_sum(dims, 8, fillFlattenedIndex);

    EXPECT_EQ(finalSum, expectedSum);
  }
};

template <typename ExecSpace>
struct TestMDTeamVectorRangeParallelReduce : public TestMDTeamParallelReduce {
  using TeamType = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_reduce_for_4D_MDTeamVectorRange(
      DimsType const& dims) {
    using ViewType = typename Kokkos::View<DataType****, ExecSpace>;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];

    ViewType v("v", leagueSize, n0, n1, n2);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2);

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>(
            {0, 0, 0, 0}, {leagueSize, n0, n1, n2}),
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
              Kokkos::MDTeamVectorRange<Direction, 3, TeamType>(team, n0, n1,
                                                                n2);

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

    int expectedSum = get_expected_sum(dims, 4, fillFlattenedIndex);

    EXPECT_EQ(finalSum, expectedSum);
  }

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_reduce_for_5D_MDTeamVectorRange(
      DimsType const& dims) {
    using ViewType = typename Kokkos::View<DataType*****, ExecSpace>;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];
    int n3         = dims[4];

    ViewType v("v", leagueSize, n0, n1, n2, n3);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3);

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<5>>(
            {0, 0, 0, 0, 0}, {leagueSize, n0, n1, n2, n3}),
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
              Kokkos::MDTeamVectorRange<Direction, 4, TeamType>(team, n0, n1,
                                                                n2, n3);

          Kokkos::parallel_reduce(
              teamVectorRange,
              [=](const int& i, const int& j, const int& k, const int& l,
                  int& vectorSum) { vectorSum += v(leagueRank, i, j, k, l); },
              teamSum);

          Kokkos::single(Kokkos::PerTeam(team),
                         [&leagueSum, teamSum]() { leagueSum += teamSum; });
        },
        finalSum);

    int expectedSum = get_expected_sum(dims, 5, fillFlattenedIndex);

    EXPECT_EQ(finalSum, expectedSum);
  }

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_reduce_for_6D_MDTeamVectorRange(
      DimsType const& dims) {
    using ViewType = typename Kokkos::View<DataType******, ExecSpace>;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];
    int n3         = dims[4];
    int n4         = dims[5];

    ViewType v("v", leagueSize, n0, n1, n2, n3, n4);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3, n4);

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<6>>(
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
              Kokkos::MDTeamVectorRange<Direction, 5, TeamType>(team, n0, n1,
                                                                n2, n3, n4);

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

    int expectedSum = get_expected_sum(dims, 6, fillFlattenedIndex);

    EXPECT_EQ(finalSum, expectedSum);
  }

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_reduce_for_7D_MDTeamVectorRange(
      DimsType const& dims) {
    using ViewType = typename Kokkos::View<DataType*******, ExecSpace>;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];
    int n3         = dims[4];
    int n4         = dims[5];
    int n5         = dims[6];

    ViewType v("v", leagueSize, n0, n1, n2, n3, n4, n5);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3, n4, n5);
    auto mdRangePolicy = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<6>>(
        {0, 0, 0, 0, 0, 0}, {leagueSize, n0, n1, n2, n3, n4});

    Kokkos::parallel_for(
        mdRangePolicy,
        KOKKOS_LAMBDA(const int leagueRank, const int i, const int j,
                      const int k, const int l, const int m) {
          for (int n = 0; n < n5; ++n) {
            v(leagueRank, i, j, k, l, m, n) =
                fillFlattenedIndex(leagueRank, i, j, k, l, m, n);
          }
        });

    int finalSum = 0;

    Kokkos::parallel_reduce(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(TeamType const& team, int& leagueSum) {
          auto leagueRank = team.league_rank();
          int teamSum     = 0;

          auto teamVectorRange =
              Kokkos::MDTeamVectorRange<Direction, 6, TeamType>(team, n0, n1,
                                                                n2, n3, n4, n5);

          Kokkos::parallel_reduce(
              teamVectorRange,
              [=](const int& i, const int& j, const int& k, const int& l,
                  const int& m, const int& n, int& vectorSum) {
                vectorSum += v(leagueRank, i, j, k, l, m, n);
              },
              teamSum);

          Kokkos::single(Kokkos::PerTeam(team),
                         [&leagueSum, teamSum]() { leagueSum += teamSum; });
        },
        finalSum);

    int expectedSum = get_expected_sum(dims, 7, fillFlattenedIndex);

    EXPECT_EQ(finalSum, expectedSum);
  }

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_reduce_for_8D_MDTeamVectorRange(
      DimsType const& dims) {
    using ViewType = typename Kokkos::View<DataType********, ExecSpace>;

    int leagueSize = dims[0];
    int n0         = dims[1];
    int n1         = dims[2];
    int n2         = dims[3];
    int n3         = dims[4];
    int n4         = dims[5];
    int n5         = dims[6];
    int n6         = dims[7];

    ViewType v("v", leagueSize, n0, n1, n2, n3, n4, n5, n6);
    FillFlattenedIndex fillFlattenedIndex(leagueSize, n0, n1, n2, n3, n4, n5,
                                          n6);
    auto mdRangePolicy = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<6>>(
        {0, 0, 0, 0, 0, 0}, {leagueSize, n0, n1, n2, n3, n4});

    Kokkos::parallel_for(
        mdRangePolicy,
        KOKKOS_LAMBDA(const int leagueRank, const int i, const int j,
                      const int k, const int l, const int m) {
          for (int n = 0; n < n5; ++n) {
            for (int o = 0; o < n6; ++o) {
              v(leagueRank, i, j, k, l, m, n, o) =
                  fillFlattenedIndex(leagueRank, i, j, k, l, m, n, o);
            }
          }
        });

    int finalSum = 0;

    Kokkos::parallel_reduce(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(TeamType const& team, int& leagueSum) {
          auto leagueRank = team.league_rank();
          int teamSum     = 0;

          auto teamVectorRange =
              Kokkos::MDTeamVectorRange<Direction, 7, TeamType>(
                  team, n0, n1, n2, n3, n4, n5, n6);

          Kokkos::parallel_reduce(
              teamVectorRange,
              [=](const int& i, const int& j, const int& k, const int& l,
                  const int& m, const int& n, const int& o, int& vectorSum) {
                vectorSum += v(leagueRank, i, j, k, l, m, n, o);
              },
              teamSum);

          Kokkos::single(Kokkos::PerTeam(team),
                         [&leagueSum, teamSum]() { leagueSum += teamSum; });
        },
        finalSum);

    int expectedSum = get_expected_sum(dims, 8, fillFlattenedIndex);

    EXPECT_EQ(finalSum, expectedSum);
  }
};
/*--------------------------------------------------------------------------*/

constexpr auto Left  = Kokkos::Iterate::Left;
constexpr auto Right = Kokkos::Iterate::Right;

// Using prime numbers makes debugging easier
// small dimensions were needed for larger dimensions to reduce test run time
int dims[]      = {3, 5, 7, 11, 13, 17, 19, 23};
int smallDims[] = {2, 3, 2, 3, 5, 2, 3, 5};

TEST(TEST_CATEGORY, MDTeamThreadRangeParallelFor) {
  TestMDTeamThreadRangeParallelFor<
      TEST_EXECSPACE>::test_parallel_for_3D_MDTeamThreadRange<Left>(dims);
  TestMDTeamThreadRangeParallelFor<
      TEST_EXECSPACE>::test_parallel_for_3D_MDTeamThreadRange<Right>(dims);

  TestMDTeamThreadRangeParallelFor<
      TEST_EXECSPACE>::test_parallel_for_4D_MDTeamThreadRange<Left>(dims);
  TestMDTeamThreadRangeParallelFor<
      TEST_EXECSPACE>::test_parallel_for_4D_MDTeamThreadRange<Right>(dims);

  TestMDTeamThreadRangeParallelFor<
      TEST_EXECSPACE>::test_parallel_for_5D_MDTeamThreadRange<Left>(dims);
  TestMDTeamThreadRangeParallelFor<
      TEST_EXECSPACE>::test_parallel_for_5D_MDTeamThreadRange<Right>(dims);

  TestMDTeamThreadRangeParallelFor<
      TEST_EXECSPACE>::test_parallel_for_6D_MDTeamThreadRange<Left>(dims);
  TestMDTeamThreadRangeParallelFor<
      TEST_EXECSPACE>::test_parallel_for_6D_MDTeamThreadRange<Right>(dims);

  TestMDTeamThreadRangeParallelFor<
      TEST_EXECSPACE>::test_parallel_for_7D_MDTeamThreadRange<Left>(smallDims);
  TestMDTeamThreadRangeParallelFor<
      TEST_EXECSPACE>::test_parallel_for_7D_MDTeamThreadRange<Right>(smallDims);

  TestMDTeamThreadRangeParallelFor<
      TEST_EXECSPACE>::test_parallel_for_8D_MDTeamThreadRange<Left>(smallDims);
  TestMDTeamThreadRangeParallelFor<
      TEST_EXECSPACE>::test_parallel_for_8D_MDTeamThreadRange<Right>(smallDims);
  TestMDTeamThreadRangeParallelFor<
      TEST_EXECSPACE>::test_parallel_single_direction_test<Left>(dims);
  TestMDTeamThreadRangeParallelFor<
      TEST_EXECSPACE>::test_parallel_single_direction_test<Right>(dims);
}

TEST(TEST_CATEGORY, MDThreadVectorRangeParallelFor) {
  TestMDThreadVectorRangeParallelFor<
      TEST_EXECSPACE>::test_parallel_for_4D_MDThreadVectorRange<Left>(dims);
  TestMDThreadVectorRangeParallelFor<
      TEST_EXECSPACE>::test_parallel_for_4D_MDThreadVectorRange<Right>(dims);

  TestMDThreadVectorRangeParallelFor<
      TEST_EXECSPACE>::test_parallel_for_5D_MDThreadVectorRange<Left>(dims);
  TestMDThreadVectorRangeParallelFor<
      TEST_EXECSPACE>::test_parallel_for_5D_MDThreadVectorRange<Right>(dims);

  TestMDThreadVectorRangeParallelFor<
      TEST_EXECSPACE>::test_parallel_for_6D_MDThreadVectorRange<Left>(dims);
  TestMDThreadVectorRangeParallelFor<
      TEST_EXECSPACE>::test_parallel_for_6D_MDThreadVectorRange<Right>(dims);

  TestMDThreadVectorRangeParallelFor<TEST_EXECSPACE>::
      test_parallel_for_7D_MDThreadVectorRange<Left>(smallDims);
  TestMDThreadVectorRangeParallelFor<TEST_EXECSPACE>::
      test_parallel_for_7D_MDThreadVectorRange<Right>(smallDims);

  TestMDThreadVectorRangeParallelFor<TEST_EXECSPACE>::
      test_parallel_for_8D_MDThreadVectorRange<Left>(smallDims);
  TestMDThreadVectorRangeParallelFor<TEST_EXECSPACE>::
      test_parallel_for_8D_MDThreadVectorRange<Right>(smallDims);
}

TEST(TEST_CATEGORY, MDTeamVectorRangeParallelFor) {
  TestMDTeamVectorRangeParallelFor<
      TEST_EXECSPACE>::test_parallel_for_3D_MDTeamVectorRange<Left>(dims);
  TestMDTeamVectorRangeParallelFor<
      TEST_EXECSPACE>::test_parallel_for_3D_MDTeamVectorRange<Right>(dims);

  TestMDTeamVectorRangeParallelFor<
      TEST_EXECSPACE>::test_parallel_for_4D_MDTeamVectorRange<Left>(dims);
  TestMDTeamVectorRangeParallelFor<
      TEST_EXECSPACE>::test_parallel_for_4D_MDTeamVectorRange<Right>(dims);

  TestMDTeamVectorRangeParallelFor<
      TEST_EXECSPACE>::test_parallel_for_5D_MDTeamVectorRange<Left>(dims);
  TestMDTeamVectorRangeParallelFor<
      TEST_EXECSPACE>::test_parallel_for_5D_MDTeamVectorRange<Right>(dims);

  TestMDTeamVectorRangeParallelFor<
      TEST_EXECSPACE>::test_parallel_for_6D_MDTeamVectorRange<Left>(dims);
  TestMDTeamVectorRangeParallelFor<
      TEST_EXECSPACE>::test_parallel_for_6D_MDTeamVectorRange<Right>(dims);

  TestMDTeamVectorRangeParallelFor<
      TEST_EXECSPACE>::test_parallel_for_7D_MDTeamVectorRange<Left>(smallDims);
  TestMDTeamVectorRangeParallelFor<
      TEST_EXECSPACE>::test_parallel_for_7D_MDTeamVectorRange<Right>(smallDims);

  TestMDTeamVectorRangeParallelFor<
      TEST_EXECSPACE>::test_parallel_for_8D_MDTeamVectorRange<Left>(smallDims);
  TestMDTeamVectorRangeParallelFor<
      TEST_EXECSPACE>::test_parallel_for_8D_MDTeamVectorRange<Right>(smallDims);

  TestMDTeamVectorRangeParallelFor<
      TEST_EXECSPACE>::test_parallel_double_direction_test<Left>(dims);
  TestMDTeamVectorRangeParallelFor<
      TEST_EXECSPACE>::test_parallel_double_direction_test<Right>(dims);
}

TEST(TEST_CATEGORY, MDTeamThreadRangeParallelReduce) {
  TestMDTeamThreadRangeParallelReduce<TEST_EXECSPACE>::
      test_parallel_reduce_for_3D_MDTeamThreadRange<Left>(dims);
  TestMDTeamThreadRangeParallelReduce<TEST_EXECSPACE>::
      test_parallel_reduce_for_3D_MDTeamThreadRange<Right>(dims);

  TestMDTeamThreadRangeParallelReduce<TEST_EXECSPACE>::
      test_parallel_reduce_for_4D_MDTeamThreadRange<Left>(dims);
  TestMDTeamThreadRangeParallelReduce<TEST_EXECSPACE>::
      test_parallel_reduce_for_4D_MDTeamThreadRange<Right>(dims);

  TestMDTeamThreadRangeParallelReduce<TEST_EXECSPACE>::
      test_parallel_reduce_for_5D_MDTeamThreadRange<Left>(dims);
  TestMDTeamThreadRangeParallelReduce<TEST_EXECSPACE>::
      test_parallel_reduce_for_5D_MDTeamThreadRange<Right>(dims);

  TestMDTeamThreadRangeParallelReduce<TEST_EXECSPACE>::
      test_parallel_reduce_for_6D_MDTeamThreadRange<Left>(dims);
  TestMDTeamThreadRangeParallelReduce<TEST_EXECSPACE>::
      test_parallel_reduce_for_6D_MDTeamThreadRange<Right>(dims);

  TestMDTeamThreadRangeParallelReduce<TEST_EXECSPACE>::
      test_parallel_reduce_for_7D_MDTeamThreadRange<Left>(smallDims);
  TestMDTeamThreadRangeParallelReduce<TEST_EXECSPACE>::
      test_parallel_reduce_for_7D_MDTeamThreadRange<Right>(smallDims);

  TestMDTeamThreadRangeParallelReduce<TEST_EXECSPACE>::
      test_parallel_reduce_for_8D_MDTeamThreadRange<Left>(smallDims);
  TestMDTeamThreadRangeParallelReduce<TEST_EXECSPACE>::
      test_parallel_reduce_for_8D_MDTeamThreadRange<Right>(smallDims);
}

TEST(TEST_CATEGORY, MDThreadVectorRangeParallelReduce) {
  TestMDThreadVectorRangeParallelReduce<TEST_EXECSPACE>::
      test_parallel_reduce_for_4D_MDThreadVectorRange<Left>(dims);
  TestMDThreadVectorRangeParallelReduce<TEST_EXECSPACE>::
      test_parallel_reduce_for_4D_MDThreadVectorRange<Right>(dims);

  TestMDThreadVectorRangeParallelReduce<TEST_EXECSPACE>::
      test_parallel_reduce_for_5D_MDThreadVectorRange<Left>(dims);
  TestMDThreadVectorRangeParallelReduce<TEST_EXECSPACE>::
      test_parallel_reduce_for_5D_MDThreadVectorRange<Right>(dims);

  TestMDThreadVectorRangeParallelReduce<TEST_EXECSPACE>::
      test_parallel_reduce_for_6D_MDThreadVectorRange<Left>(dims);
  TestMDThreadVectorRangeParallelReduce<TEST_EXECSPACE>::
      test_parallel_reduce_for_6D_MDThreadVectorRange<Right>(dims);

  TestMDThreadVectorRangeParallelReduce<TEST_EXECSPACE>::
      test_parallel_reduce_for_7D_MDThreadVectorRange<Left>(smallDims);
  TestMDThreadVectorRangeParallelReduce<TEST_EXECSPACE>::
      test_parallel_reduce_for_7D_MDThreadVectorRange<Right>(smallDims);

  TestMDThreadVectorRangeParallelReduce<TEST_EXECSPACE>::
      test_parallel_reduce_for_8D_MDThreadVectorRange<Left>(smallDims);
  TestMDThreadVectorRangeParallelReduce<TEST_EXECSPACE>::
      test_parallel_reduce_for_8D_MDThreadVectorRange<Right>(smallDims);
}

TEST(TEST_CATEGORY, MDTeamVectorRangeParallelReduce) {
  TestMDTeamVectorRangeParallelReduce<TEST_EXECSPACE>::
      test_parallel_reduce_for_4D_MDTeamVectorRange<Left>(dims);
  TestMDTeamVectorRangeParallelReduce<TEST_EXECSPACE>::
      test_parallel_reduce_for_4D_MDTeamVectorRange<Right>(dims);

  TestMDTeamVectorRangeParallelReduce<TEST_EXECSPACE>::
      test_parallel_reduce_for_5D_MDTeamVectorRange<Left>(dims);
  TestMDTeamVectorRangeParallelReduce<TEST_EXECSPACE>::
      test_parallel_reduce_for_5D_MDTeamVectorRange<Right>(dims);

  TestMDTeamVectorRangeParallelReduce<TEST_EXECSPACE>::
      test_parallel_reduce_for_6D_MDTeamVectorRange<Left>(dims);
  TestMDTeamVectorRangeParallelReduce<TEST_EXECSPACE>::
      test_parallel_reduce_for_6D_MDTeamVectorRange<Right>(dims);

  TestMDTeamVectorRangeParallelReduce<TEST_EXECSPACE>::
      test_parallel_reduce_for_7D_MDTeamVectorRange<Left>(smallDims);
  TestMDTeamVectorRangeParallelReduce<TEST_EXECSPACE>::
      test_parallel_reduce_for_7D_MDTeamVectorRange<Right>(smallDims);

  TestMDTeamVectorRangeParallelReduce<TEST_EXECSPACE>::
      test_parallel_reduce_for_8D_MDTeamVectorRange<Left>(smallDims);
  TestMDTeamVectorRangeParallelReduce<TEST_EXECSPACE>::
      test_parallel_reduce_for_8D_MDTeamVectorRange<Right>(smallDims);
}
}  // namespace MDTeamRange
}  // namespace Test