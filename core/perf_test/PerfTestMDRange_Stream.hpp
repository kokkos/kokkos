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

#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>
#include "Benchmark_Context.hpp"
#include "PerfTest_Category.hpp"

template <int R, typename ScalarType, typename Layout, class ExecutionSpace>
struct view_type_rank {};

template <typename ExecutionSpace, typename ScalarType, typename Layout>
struct view_type_rank<2, ScalarType, Layout, ExecutionSpace> {
  using type = Kokkos::View<ScalarType **, Layout, ExecutionSpace>;
};

template <typename ExecutionSpace, typename ScalarType, typename Layout>
struct view_type_rank<3, ScalarType, Layout, ExecutionSpace> {
  using type = Kokkos::View<ScalarType ***, Layout, ExecutionSpace>;
};

template <typename ExecutionSpace, typename ScalarType, typename Layout>
struct view_type_rank<4, ScalarType, Layout, ExecutionSpace> {
  using type = Kokkos::View<ScalarType ****, Layout, ExecutionSpace>;
};

template <typename ExecutionSpace, typename ScalarType, typename Layout>
struct view_type_rank<5, ScalarType, Layout, ExecutionSpace> {
  using type = Kokkos::View<ScalarType *****, Layout, ExecutionSpace>;
};

template <typename ExecutionSpace, typename ScalarType, typename Layout>
struct view_type_rank<6, ScalarType, Layout, ExecutionSpace> {
  using type = Kokkos::View<ScalarType ******, Layout, ExecutionSpace>;
};

namespace Benchmark {

// Not used
template <typename ViewType>
struct Functor_Set {
  ViewType tensor;
  static constexpr bool need_initialize = false;
};

template <typename ViewType>
struct Functor_Copy {
  ViewType tensor;
  static constexpr bool need_initialize = true;
};

template <typename ViewType>
struct Functor_Scale {
  ViewType tensor;
  static constexpr bool need_initialize = true;
};

template <typename ViewType>
struct Functor_Add {
  ViewType tensor;
  static constexpr bool need_initialize = true;
};

template <typename ViewType>
struct Functor_Triad {
  ViewType tensor;
  static constexpr bool need_initialize = true;
};

template <class ExecutionSpace, int Rank, typename ScalarType = double,
          typename IndexType = Kokkos::IndexType<int32_t>>
struct MDRangePolicyTriad {
  using execution_space = ExecutionSpace;
  using prefered_layout = typename ExecutionSpace::array_layout;
  using scalar_type     = ScalarType;
  using view_type = typename view_type_rank<Rank, scalar_type, prefered_layout,
                                            execution_space>::type;

  static const Kokkos::Iterate outer_iter =
      Kokkos::Impl::layout_iterate_type_selector<
          prefered_layout>::outer_iteration_pattern;
  static const Kokkos::Iterate inner_iter =
      Kokkos::Impl::layout_iterate_type_selector<
          prefered_layout>::inner_iteration_pattern;
  using rank_type = Kokkos::Rank<Rank, outer_iter, inner_iter>;
  using policy_type =
      Kokkos::MDRangePolicy<ExecutionSpace, rank_type, IndexType>;

  view_type A;
  view_type B;
  view_type C;
  ScalarType scalar;
  int N;

  template <typename... Args>
  MDRangePolicyTriad(const view_type &A_, const view_type &B_,
                     const view_type &C_, ScalarType scalar_, int N_)
      : A(A_), B(B_), C(C_), scalar(scalar_), N(N_) {
    static_assert(Rank >= 2 && Rank <= 6,
                  "MDRangePolicyTriad: Only ranks 2 to 6 are supported");
  }

  template <typename... Args>
  KOKKOS_INLINE_FUNCTION void operator()(Args... args) const {
    C(args...) = A(args...) + scalar * B(args...);
  }

  struct Init {
    view_type tensor;
    scalar_type value;

    template <typename... Args>
    Init(const view_type &tensor_, const scalar_type &value_)
        : tensor(tensor_), value(value_) {}

    template <typename... Indices>
    KOKKOS_INLINE_FUNCTION void operator()(Indices... indices) const {
      tensor(indices...) = value;
    }
  };

  static double test_triad(int N, const int iterations = 1) {
    // Use constexpr to create views
    auto create_view = [](const char *name, int dim) {
      int N1 = dim;
      int N2 = N1 * N1;
      int N3 = N2 * N1;
      if constexpr (Rank == 2) {
        return view_type(name, N3, N3);
      } else if constexpr (Rank == 3) {
        return view_type(name, N2, N2, N2);
      } else if constexpr (Rank == 4) {
        return view_type(name, N2, N2, N1, N1);
      } else if constexpr (Rank == 5) {
        return view_type(name, N2, N1, N1, N1, N1);
      } else if constexpr (Rank == 6) {
        return view_type(name, N1, N1, N1, N1, N1, N1);
      }
    };

    view_type A_test = create_view("A_test", N);
    view_type B_test = create_view("B_test", N);
    view_type C_test = create_view("C_test", N);

    scalar_type scalar = 1.0 / static_cast<scalar_type>(N);

    using FunctorType =
        MDRangePolicyTriad<ExecutionSpace, Rank, ScalarType, IndexType>;

    typename policy_type::point_type lower_bounds, upper_bounds;
    for (int i = 0; i < Rank; ++i) {
      lower_bounds[i] = 0;
      upper_bounds[i] = A_test.extent(i);
    }

    policy_type init_policy(lower_bounds, upper_bounds);
    Kokkos::parallel_for("init_A", init_policy,
                         Init(A_test, static_cast<ScalarType>(1.0)));
    Kokkos::parallel_for("init_B", init_policy,
                         Init(B_test, static_cast<ScalarType>(2.0)));
    Kokkos::parallel_for("init_C", init_policy,
                         Init(C_test, static_cast<ScalarType>(0.0)));
    execution_space().fence();

    policy_type compute_policy(lower_bounds, upper_bounds);
    double dt = 0;
    for (int i = 0; i < iterations; ++i) {
      Kokkos::Timer timer;
      Kokkos::parallel_for(compute_policy,
                           FunctorType(A_test, B_test, C_test, scalar, N));
      execution_space().fence();
      dt += timer.seconds();
    }
    dt /= iterations;
    return dt;
  }
};

}  // namespace Benchmark
