// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <iostream>
#include <limits>

#include <benchmark/benchmark.h>

#include "PerfTest_Category.hpp"
#include <Kokkos_Core.hpp>

namespace Test {

template <typename ScalarType, typename ViewType>
void check_computation(const ViewType& A, const ViewType& B) {
  int numErrors = 0;
  auto Ahost    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A);
  auto Bhost    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), B);

  ScalarType epsilon = std::numeric_limits<ScalarType>::epsilon() * 100;
  if constexpr (ViewType::rank == 3) {
    const int n0 = Ahost.extent_int(0) - 2, n1 = Ahost.extent_int(1) - 2,
              n2 = Ahost.extent_int(2) - 2;
    for (int i0 = 0; i0 < n0; ++i0) {
      for (int i1 = 1; i1 < n1 + 1; ++i1) {
        for (int i2 = 1; i2 < n2 + 1; ++i2) {
          ScalarType check =
              0.25 *
              (ScalarType)(Bhost(i0, i1 + 1, i2) + Bhost(i0, i1 - 1, i2) +
                           Bhost(i0, i1, i2 + 1) + Bhost(i0, i1, i2 - 1) +
                           Bhost(i0, i1, i2));
          if (Kokkos::abs(Ahost(i0, i1, i2) - check) > epsilon) {
            ++numErrors;
            std::cerr << "Correctness error at index: " << i0 << "," << i1
                      << "," << i2 << ", got " << Ahost(i0, i1, i2)
                      << ", expected " << check << "\n";
          }
        }
      }
    }
  } else if constexpr (ViewType::rank == 4) {
    const int n0 = Ahost.extent_int(0) - 2, n1 = Ahost.extent_int(1) - 2,
              n2 = Ahost.extent_int(2) - 2, n3 = Ahost.extent_int(3) - 2;
    for (int i0 = 0; i0 < n0; ++i0) {
      for (int i1 = 1; i1 < n1 + 1; ++i1) {
        for (int i2 = 1; i2 < n2 + 1; ++i2) {
          for (int i3 = 1; i3 < n3 + 1; ++i3) {
            ScalarType check = 0.25 * (ScalarType)(Bhost(i0, i1 + 1, i2, i3) +
                                                   Bhost(i0, i1 - 1, i2, i3) +
                                                   Bhost(i0, i1, i2 + 1, i3) +
                                                   Bhost(i0, i1, i2 - 1, i3) +
                                                   Bhost(i0, i1, i2, i3 + 1) +
                                                   Bhost(i0, i1, i2, i3 - 1) +
                                                   Bhost(i0, i1, i2, i3));
            if (Kokkos::abs(Ahost(i0, i1, i2, i3) - check) > epsilon) {
              ++numErrors;
              std::cerr << "Correctness error at index: " << i0 << "," << i1
                        << "," << i2 << "," << i3 << ", got "
                        << Ahost(i0, i1, i2, i3) << ", expected " << check
                        << "\n";
            }
          }
        }
      }
    }
    if (numErrors != 0) {
      std::cerr << "Detected some errors for a run with dimensions "
                << Ahost.extent(0);
      for (std::size_t i = 1; i < Ahost.rank(); i++) {
        std::cerr << "x" << Ahost.extent(i);
      }
      std::cerr << std::endl;
    }
  } else if constexpr (ViewType::rank == 5) {
    const int n0 = Ahost.extent_int(0) - 2, n1 = Ahost.extent_int(1) - 2,
              n2 = Ahost.extent_int(2) - 2, n3 = Ahost.extent_int(3) - 2,
              n4 = Ahost.extent_int(4) - 2;
    for (int i0 = 0; i0 < n0; ++i0) {
      for (int i1 = 1; i1 < n1 + 1; ++i1) {
        for (int i2 = 1; i2 < n2 + 1; ++i2) {
          for (int i3 = 1; i3 < n3 + 1; ++i3) {
            for (int i4 = 1; i4 < n4 + 1; ++i4) {
              ScalarType check =
                  0.25 * (ScalarType)(Bhost(i0, i1 + 1, i2, i3, i4) +
                                      Bhost(i0, i1 - 1, i2, i3, i4) +
                                      Bhost(i0, i1, i2 + 1, i3, i4) +
                                      Bhost(i0, i1, i2 - 1, i3, i4) +
                                      Bhost(i0, i1, i2, i3 + 1, i4) +
                                      Bhost(i0, i1, i2, i3 - 1, i4) +
                                      Bhost(i0, i1, i2, i3, i4 + 1) +
                                      Bhost(i0, i1, i2, i3, i4 - 1) +
                                      Bhost(i0, i1, i2, i3, i4));
              if (Kokkos::abs(Ahost(i0, i1, i2, i3, i4) - check) > epsilon) {
                ++numErrors;
                std::cerr << "Correctness error at index: " << i0 << "," << i1
                          << "," << i2 << "," << i3 << "," << i4 << ", got "
                          << Ahost(i0, i1, i2, i3, i4) << ", expected " << check
                          << "\n";
              }
            }
          }
        }
      }
    }
    if (numErrors != 0) {
      std::cerr << "Detected some errors for a run with dimensions "
                << Ahost.extent(0);
      for (std::size_t i = 1; i < Ahost.rank(); i++) {
        std::cerr << "x" << Ahost.extent(i);
      }
      std::cerr << std::endl;
    }
  }
}

template <typename T, std::size_t Rank>
struct add_pointer_n {
  using type = typename add_pointer_n<T*, Rank - 1>::type;
};

template <typename T>
struct add_pointer_n<T, 0> {
  using type = T;
};

template <typename Layout>
struct LayoutToIterationPattern {};

template <>
struct LayoutToIterationPattern<Kokkos::LayoutRight> {
  static constexpr Kokkos::Iterate pattern = Kokkos::Iterate::Right;
};

template <>
struct LayoutToIterationPattern<Kokkos::LayoutLeft> {
  static constexpr Kokkos::Iterate pattern = Kokkos::Iterate::Left;
};

template <typename T, std::size_t Rank>
using add_pointer_n_t = typename add_pointer_n<T, Rank>::type;

template <int Dimension, class DeviceType,
          typename Layout = Kokkos::LayoutRight, typename ScalarType = double>
struct TeamThreadMDRangeStencilBase {
  using execution_space = DeviceType;
  using scalar_type     = ScalarType;
  using team_policy     = Kokkos::TeamPolicy<execution_space>;
  using team_member     = typename team_policy::member_type;
  using view_type = Kokkos::View<add_pointer_n_t<ScalarType, Dimension + 1>,
                                 Layout, DeviceType>;

  static constexpr Kokkos::Iterate direction =
      LayoutToIterationPattern<Layout>::pattern;
  using rank_type           = Kokkos::Rank<Dimension, direction, direction>;
  using team_thread_mdrange = Kokkos::TeamThreadMDRange<rank_type, team_member>;

  static constexpr int dimension = Dimension;

  view_type A;
  view_type B;
  const Kokkos::Array<int, dimension> ranges;

  TeamThreadMDRangeStencilBase(const view_type& A_, const view_type& B_,
                               const Kokkos::Array<int, dimension>& dims)
      : A(A_), B(B_), ranges(dims) {}

  static auto get_policy(int league_size) {
    return team_policy(league_size, 32);
  }
};

template <int Dimension, class DeviceType,
          typename Layout = Kokkos::LayoutRight, typename ScalarType = double>
struct TeamThreadMDRangeStencil;

template <class DeviceType, typename Layout, typename ScalarType>
struct TeamThreadMDRangeStencil<2, DeviceType, Layout, ScalarType>
    : TeamThreadMDRangeStencilBase<2, DeviceType, Layout, ScalarType> {
  using Base = TeamThreadMDRangeStencilBase<2, DeviceType, Layout, ScalarType>;
  using Base::Base;
  using typename Base::team_member;
  using typename Base::team_thread_mdrange;

  KOKKOS_INLINE_FUNCTION
  void operator()(const team_member& team) const {
    const int league_rank = team.league_rank();

    auto team_range =
        team_thread_mdrange(team, this->ranges[0], this->ranges[1]);
    Kokkos::parallel_for(team_range, [=, this](int i0, int i1) {
      i0++;
      i1++;
      this->A(league_rank, i0, i1) =
          0.25 * (ScalarType)(this->B(league_rank, i0 + 1, i1) +
                              this->B(league_rank, i0 - 1, i1) +
                              this->B(league_rank, i0, i1 + 1) +
                              this->B(league_rank, i0, i1 - 1) +
                              this->B(league_rank, i0, i1));
    });
  }
};

template <class DeviceType, typename Layout, typename ScalarType>
struct TeamThreadMDRangeStencil<3, DeviceType, Layout, ScalarType>
    : TeamThreadMDRangeStencilBase<3, DeviceType, Layout, ScalarType> {
  using Base = TeamThreadMDRangeStencilBase<3, DeviceType, Layout, ScalarType>;
  using Base::Base;
  using typename Base::team_member;
  using typename Base::team_thread_mdrange;

  KOKKOS_INLINE_FUNCTION
  void operator()(const team_member& team) const {
    const int league_rank = team.league_rank();

    auto team_range = team_thread_mdrange(team, this->ranges[0],
                                          this->ranges[1], this->ranges[2]);
    Kokkos::parallel_for(team_range, [=, this](int i0, int i1, int i2) {
      i0++;
      i1++;
      i2++;
      this->A(league_rank, i0, i1, i2) =
          0.25 * (ScalarType)(this->B(league_rank, i0 + 1, i1, i2) +
                              this->B(league_rank, i0 - 1, i1, i2) +
                              this->B(league_rank, i0, i1 + 1, i2) +
                              this->B(league_rank, i0, i1 - 1, i2) +
                              this->B(league_rank, i0, i1, i2 + 1) +
                              this->B(league_rank, i0, i1, i2 - 1) +
                              this->B(league_rank, i0, i1, i2));
    });
  }
};

template <class DeviceType, typename Layout, typename ScalarType>
struct TeamThreadMDRangeStencil<4, DeviceType, Layout, ScalarType>
    : TeamThreadMDRangeStencilBase<4, DeviceType, Layout, ScalarType> {
  using Base = TeamThreadMDRangeStencilBase<4, DeviceType, Layout, ScalarType>;
  using Base::Base;
  using typename Base::team_member;
  using typename Base::team_thread_mdrange;

  KOKKOS_INLINE_FUNCTION
  void operator()(const team_member& team) const {
    const int league_rank = team.league_rank();

    auto team_range =
        team_thread_mdrange(team, this->ranges[0], this->ranges[1],
                            this->ranges[2], this->ranges[3]);
    Kokkos::parallel_for(team_range, [=, this](int i0, int i1, int i2, int i3) {
      i0++;
      i1++;
      i2++;
      i3++;
      this->A(league_rank, i0, i1, i2, i3) =
          0.25 * (ScalarType)(this->B(league_rank, i0 + 1, i1, i2, i3) +
                              this->B(league_rank, i0 - 1, i1, i2, i3) +
                              this->B(league_rank, i0, i1 + 1, i2, i3) +
                              this->B(league_rank, i0, i1 - 1, i2, i3) +
                              this->B(league_rank, i0, i1, i2 + 1, i3) +
                              this->B(league_rank, i0, i1, i2 - 1, i3) +
                              this->B(league_rank, i0, i1, i2, i3 + 1) +
                              this->B(league_rank, i0, i1, i2, i3 - 1) +
                              this->B(league_rank, i0, i1, i2, i3));
    });
  }
};

template <int Dimension, class DeviceType,
          typename Layout = Kokkos::LayoutRight, typename ScalarType = double>
struct TeamVectorMDRangeStencilBase {
  using execution_space = DeviceType;
  using scalar_type     = ScalarType;
  using team_policy     = Kokkos::TeamPolicy<execution_space>;
  using team_member     = typename team_policy::member_type;
  using view_type = Kokkos::View<add_pointer_n_t<ScalarType, Dimension + 1>,
                                 Layout, DeviceType>;

  static constexpr Kokkos::Iterate direction =
      LayoutToIterationPattern<Layout>::pattern;
  using rank_type           = Kokkos::Rank<Dimension, direction, direction>;
  using team_vector_mdrange = Kokkos::TeamVectorMDRange<rank_type, team_member>;

  static constexpr int dimension = Dimension;

  view_type A;
  view_type B;
  const Kokkos::Array<int, dimension> ranges;

  TeamVectorMDRangeStencilBase(const view_type& A_, const view_type& B_,
                               const Kokkos::Array<int, dimension>& dims)
      : A(A_), B(B_), ranges(dims) {}

  static auto get_policy(int league_size) {
    return team_policy(league_size, 32);
  }
};

template <int Dimension, class DeviceType,
          typename Layout = Kokkos::LayoutRight, typename ScalarType = double>
struct TeamVectorMDRangeStencil;

template <class DeviceType, typename Layout, typename ScalarType>
struct TeamVectorMDRangeStencil<2, DeviceType, Layout, ScalarType>
    : TeamVectorMDRangeStencilBase<2, DeviceType, Layout, ScalarType> {
  using Base = TeamVectorMDRangeStencilBase<2, DeviceType, Layout, ScalarType>;
  using Base::Base;
  using typename Base::team_member;
  using typename Base::team_vector_mdrange;

  KOKKOS_INLINE_FUNCTION
  void operator()(const team_member& team) const {
    const int league_rank = team.league_rank();

    auto team_range =
        team_vector_mdrange(team, this->ranges[0], this->ranges[1]);
    Kokkos::parallel_for(team_range, [=, this](int i0, int i1) {
      i0++;
      i1++;
      this->A(league_rank, i0, i1) =
          0.25 * (ScalarType)(this->B(league_rank, i0 + 1, i1) +
                              this->B(league_rank, i0 - 1, i1) +
                              this->B(league_rank, i0, i1 + 1) +
                              this->B(league_rank, i0, i1 - 1) +
                              this->B(league_rank, i0, i1));
    });
  }
};

template <class DeviceType, typename Layout, typename ScalarType>
struct TeamVectorMDRangeStencil<3, DeviceType, Layout, ScalarType>
    : TeamVectorMDRangeStencilBase<3, DeviceType, Layout, ScalarType> {
  using Base = TeamVectorMDRangeStencilBase<3, DeviceType, Layout, ScalarType>;
  using Base::Base;
  using typename Base::team_member;
  using typename Base::team_vector_mdrange;

  KOKKOS_INLINE_FUNCTION
  void operator()(const team_member& team) const {
    const int league_rank = team.league_rank();

    auto team_range = team_vector_mdrange(team, this->ranges[0],
                                          this->ranges[1], this->ranges[2]);
    Kokkos::parallel_for(team_range, [=, this](int i0, int i1, int i2) {
      i0++;
      i1++;
      i2++;
      this->A(league_rank, i0, i1, i2) =
          0.25 * (ScalarType)(this->B(league_rank, i0 + 1, i1, i2) +
                              this->B(league_rank, i0 - 1, i1, i2) +
                              this->B(league_rank, i0, i1 + 1, i2) +
                              this->B(league_rank, i0, i1 - 1, i2) +
                              this->B(league_rank, i0, i1, i2 + 1) +
                              this->B(league_rank, i0, i1, i2 - 1) +
                              this->B(league_rank, i0, i1, i2));
    });
  }
};

template <class DeviceType, typename Layout, typename ScalarType>
struct TeamVectorMDRangeStencil<4, DeviceType, Layout, ScalarType>
    : TeamVectorMDRangeStencilBase<4, DeviceType, Layout, ScalarType> {
  using Base = TeamVectorMDRangeStencilBase<4, DeviceType, Layout, ScalarType>;
  using Base::Base;
  using typename Base::team_member;
  using typename Base::team_vector_mdrange;

  KOKKOS_INLINE_FUNCTION
  void operator()(const team_member& team) const {
    const int league_rank = team.league_rank();

    auto team_range =
        team_vector_mdrange(team, this->ranges[0], this->ranges[1],
                            this->ranges[2], this->ranges[3]);
    Kokkos::parallel_for(team_range, [=, this](int i0, int i1, int i2, int i3) {
      i0++;
      i1++;
      i2++;
      i3++;
      this->A(league_rank, i0, i1, i2, i3) =
          0.25 * (ScalarType)(this->B(league_rank, i0 + 1, i1, i2, i3) +
                              this->B(league_rank, i0 - 1, i1, i2, i3) +
                              this->B(league_rank, i0, i1 + 1, i2, i3) +
                              this->B(league_rank, i0, i1 - 1, i2, i3) +
                              this->B(league_rank, i0, i1, i2 + 1, i3) +
                              this->B(league_rank, i0, i1, i2 - 1, i3) +
                              this->B(league_rank, i0, i1, i2, i3 + 1) +
                              this->B(league_rank, i0, i1, i2, i3 - 1) +
                              this->B(league_rank, i0, i1, i2, i3));
    });
  }
};

template <int Dimension, class DeviceType,
          typename Layout = Kokkos::LayoutRight, typename ScalarType = double>
struct ThreadVectorMDRangeStencilBase {
  using execution_space = DeviceType;
  using scalar_type     = ScalarType;
  using team_policy     = Kokkos::TeamPolicy<execution_space>;
  using team_member     = typename team_policy::member_type;
  using view_type = Kokkos::View<add_pointer_n_t<ScalarType, Dimension + 1>,
                                 Layout, DeviceType>;

  static constexpr Kokkos::Iterate direction =
      LayoutToIterationPattern<Layout>::pattern;
  using rank_type = Kokkos::Rank<Dimension, direction, direction>;
  using thread_vector_mdrange =
      Kokkos::ThreadVectorMDRange<rank_type, team_member>;

  static constexpr int dimension = Dimension;

  view_type A;
  view_type B;
  const Kokkos::Array<int, dimension> ranges;

  ThreadVectorMDRangeStencilBase(const view_type& A_, const view_type& B_,
                                 const Kokkos::Array<int, dimension>& dims)
      : A(A_), B(B_), ranges(dims) {}

  static auto get_policy(int league_size) {
    assert(league_size % 32 == 0);
    return team_policy(league_size / 32, 32);
  }
};

template <int Dimension, class DeviceType,
          typename Layout = Kokkos::LayoutRight, typename ScalarType = double>
struct ThreadVectorMDRangeStencil;

template <class DeviceType, typename Layout, typename ScalarType>
struct ThreadVectorMDRangeStencil<2, DeviceType, Layout, ScalarType>
    : ThreadVectorMDRangeStencilBase<2, DeviceType, Layout, ScalarType> {
  using Base =
      ThreadVectorMDRangeStencilBase<2, DeviceType, Layout, ScalarType>;
  using Base::Base;
  using typename Base::team_member;
  using typename Base::thread_vector_mdrange;

  KOKKOS_INLINE_FUNCTION
  void operator()(const team_member& team) const {
    const int league_rank = team.league_rank();

    auto team_thread_range = Kokkos::TeamThreadRange(team, 32);
    Kokkos::parallel_for(team_thread_range, [=, this](int i0) {
      const auto i = league_rank * 32 + i0;

      auto vector_range =
          thread_vector_mdrange(team, this->ranges[0], this->ranges[1]);
      Kokkos::parallel_for(vector_range, [=, this](int i1, int i2) {
        i1++;
        i2++;
        this->A(i, i1, i2) =
            0.25 *
            (ScalarType)(this->B(i, i1 + 1, i2) + this->B(i, i1 - 1, i2) +
                         this->B(i, i1, i2 + 1) + this->B(i, i1, i2 - 1) +
                         this->B(i, i1, i2));
      });
    });
  }
};

template <class DeviceType, typename Layout, typename ScalarType>
struct ThreadVectorMDRangeStencil<3, DeviceType, Layout, ScalarType>
    : ThreadVectorMDRangeStencilBase<3, DeviceType, Layout, ScalarType> {
  using Base =
      ThreadVectorMDRangeStencilBase<3, DeviceType, Layout, ScalarType>;
  using Base::Base;
  using typename Base::team_member;
  using typename Base::thread_vector_mdrange;

  KOKKOS_INLINE_FUNCTION
  void operator()(const team_member& team) const {
    const int league_rank = team.league_rank();

    auto team_thread_range = Kokkos::TeamThreadRange(team, 32);
    Kokkos::parallel_for(team_thread_range, [=, this](int i0) {
      const auto i = league_rank * 32 + i0;

      auto vector_range = thread_vector_mdrange(
          team, this->ranges[0], this->ranges[1], this->ranges[2]);
      Kokkos::parallel_for(vector_range, [=, this](int i1, int i2, int i3) {
        i1++;
        i2++;
        i3++;
        this->A(i, i1, i2, i3) =
            0.25 *
            (ScalarType)(this->B(i, i1 + 1, i2, i3) +
                         this->B(i, i1 - 1, i2, i3) +
                         this->B(i, i1, i2 + 1, i3) +
                         this->B(i, i1, i2 - 1, i3) +
                         this->B(i, i1, i2, i3 + 1) +
                         this->B(i, i1, i2, i3 - 1) + this->B(i, i1, i2, i3));
      });
    });
  }
};

template <class DeviceType, typename Layout, typename ScalarType>
struct ThreadVectorMDRangeStencil<4, DeviceType, Layout, ScalarType>
    : ThreadVectorMDRangeStencilBase<4, DeviceType, Layout, ScalarType> {
  using Base =
      ThreadVectorMDRangeStencilBase<4, DeviceType, Layout, ScalarType>;
  using Base::Base;
  using typename Base::team_member;
  using typename Base::thread_vector_mdrange;

  KOKKOS_INLINE_FUNCTION
  void operator()(const team_member& team) const {
    const int league_rank = team.league_rank();

    auto team_thread_range = Kokkos::TeamThreadRange(team, 32);
    Kokkos::parallel_for(team_thread_range, [=, this](int i0) {
      const auto i = league_rank * 32 + i0;

      auto vector_range =
          thread_vector_mdrange(team, this->ranges[0], this->ranges[1],
                                this->ranges[2], this->ranges[3]);
      Kokkos::parallel_for(
          vector_range, [=, this](int i1, int i2, int i3, int i4) {
            i1++;
            i2++;
            i3++;
            i4++;
            this->A(i, i1, i2, i3, i4) =
                0.25 * (ScalarType)(this->B(i, i1 + 1, i2, i3, i4) +
                                    this->B(i, i1 - 1, i2, i3, i4) +
                                    this->B(i, i1, i2 + 1, i3, i4) +
                                    this->B(i, i1, i2 - 1, i3, i4) +
                                    this->B(i, i1, i2, i3 + 1, i4) +
                                    this->B(i, i1, i2, i3 - 1, i4) +
                                    this->B(i, i1, i2, i3, i4 + 1) +
                                    this->B(i, i1, i2, i3, i4 - 1) +
                                    this->B(i, i1, i2, i3, i4));
          });
    });
  }
};

template <typename FunctorType, std::size_t... Idx>
void bench_team_mdrange(benchmark::State& state, std::index_sequence<Idx...>) {
  using execution_space = typename FunctorType::execution_space;
  using view_type       = typename FunctorType::view_type;

  const int league_size = static_cast<int>(state.range(0));

  Kokkos::Array<int, FunctorType::dimension> dims;
  for (std::size_t i = 0; i < dims.size(); i++) {
    dims[i] = state.range(1);
  }

  state.counters["league_size"] = league_size;
  state.counters["dim_size"]    = dims[0];

  view_type Atest("Atest", league_size, (dims[Idx] + 2)...);
  view_type Btest("Btest", league_size, (dims[Idx] + 2)...);

  Kokkos::deep_copy(Atest, 1.0);
  execution_space().fence();
  Kokkos::deep_copy(Btest, 1.0);
  execution_space().fence();

  const auto policy = FunctorType::get_policy(league_size);

  for (auto _ : state) {
    Kokkos::Timer timer;
    Kokkos::parallel_for(policy, FunctorType(Atest, Btest, dims));
    execution_space().fence();
    const double dt = timer.seconds();
    state.SetIterationTime(dt);
  }

  check_computation<typename FunctorType::scalar_type>(Atest, Btest);
}

template <typename FunctorType>
void bench_team_mdrange(benchmark::State& state) {
  bench_team_mdrange<FunctorType>(
      state, std::make_index_sequence<FunctorType::dimension>());
}

#define TEAM_MDRANGE_STENCIL_BENCHMARK(functor, dim, fn, ...) \
  BENCHMARK(fn<functor<dim, TEST_EXECSPACE>>)                 \
      ->UseManualTime()                                       \
      ->Unit(benchmark::kMillisecond)                         \
      ->Name("TeamMDRangeStencil_" #dim "D_" #functor)        \
      ->ArgNames({"league_size", "size"})                     \
      ->ArgsProduct({__VA_ARGS__});

#define LEAGUE_SIZES \
  { 256 }
#define SIZES_2D \
  { 128, 256, 512 }
#define SIZES_3D \
  { 16, 32, 64 }
#define SIZES_4D \
  { 4, 8, 16 }

TEAM_MDRANGE_STENCIL_BENCHMARK(TeamThreadMDRangeStencil, 2, bench_team_mdrange,
                               LEAGUE_SIZES, SIZES_2D)
TEAM_MDRANGE_STENCIL_BENCHMARK(TeamThreadMDRangeStencil, 3, bench_team_mdrange,
                               LEAGUE_SIZES, SIZES_3D)
TEAM_MDRANGE_STENCIL_BENCHMARK(TeamThreadMDRangeStencil, 4, bench_team_mdrange,
                               LEAGUE_SIZES, SIZES_4D)

TEAM_MDRANGE_STENCIL_BENCHMARK(TeamVectorMDRangeStencil, 2, bench_team_mdrange,
                               LEAGUE_SIZES, SIZES_2D)
TEAM_MDRANGE_STENCIL_BENCHMARK(TeamVectorMDRangeStencil, 3, bench_team_mdrange,
                               LEAGUE_SIZES, SIZES_3D)
TEAM_MDRANGE_STENCIL_BENCHMARK(TeamVectorMDRangeStencil, 4, bench_team_mdrange,
                               LEAGUE_SIZES, SIZES_4D)

TEAM_MDRANGE_STENCIL_BENCHMARK(ThreadVectorMDRangeStencil, 2,
                               bench_team_mdrange, LEAGUE_SIZES, SIZES_2D)
TEAM_MDRANGE_STENCIL_BENCHMARK(ThreadVectorMDRangeStencil, 3,
                               bench_team_mdrange, LEAGUE_SIZES, SIZES_3D)
TEAM_MDRANGE_STENCIL_BENCHMARK(ThreadVectorMDRangeStencil, 4,
                               bench_team_mdrange, LEAGUE_SIZES, SIZES_4D)

#undef LEAGUE_SIZES
#undef SIZES_2D
#undef SIZES_3D
#undef SIZES_4D

#undef TEAM_MDRANGE_STENCIL_BENCHMARK

}  // namespace Test
