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
#include <benchmark/benchmark.h>
#include "PerfTest_Category.hpp"

namespace Benchmark {

struct Tag_Set {};
struct Tag_Copy {};
struct Tag_Scale {};
struct Tag_Add {};
struct Tag_Triad {};

template <int Rank, typename ScalarType, typename Layout, typename MemorySpace>
struct ViewTypeRank {};

template <typename ScalarType, typename Layout, typename MemorySpace>
struct ViewTypeRank<2, ScalarType, Layout, MemorySpace> {
  using type = Kokkos::View<ScalarType **, Layout, MemorySpace>;
};

template <typename ScalarType, typename Layout, typename MemorySpace>
struct ViewTypeRank<3, ScalarType, Layout, MemorySpace> {
  using type = Kokkos::View<ScalarType ***, Layout, MemorySpace>;
};

template <typename ScalarType, typename Layout, typename MemorySpace>
struct ViewTypeRank<4, ScalarType, Layout, MemorySpace> {
  using type = Kokkos::View<ScalarType ****, Layout, MemorySpace>;
};

template <typename ScalarType, typename Layout, typename MemorySpace>
struct ViewTypeRank<5, ScalarType, Layout, MemorySpace> {
  using type = Kokkos::View<ScalarType *****, Layout, MemorySpace>;
};

template <typename ScalarType, typename Layout, typename MemorySpace>
struct ViewTypeRank<6, ScalarType, Layout, MemorySpace> {
  using type = Kokkos::View<ScalarType ******, Layout, MemorySpace>;
};

// Functor for stream test (copy, scale, add, triad).
// The problem size is N^6, meaning that each view will have the size of N^6
// whatever the rank is.
template <class ExecutionSpace, int Rank, typename ScalarType = double,
          typename IndexType = Kokkos::IndexType<uint32_t>>
struct MDRangePolicy_StreamTest {
  using execution_space  = ExecutionSpace;
  using memory_space     = typename execution_space::memory_space;
  using preferred_layout = typename ExecutionSpace::array_layout;
  using scalar_type      = ScalarType;
  using view_type = typename ViewTypeRank<Rank, scalar_type, preferred_layout,
                                          memory_space>::type;

  static const Kokkos::Iterate outer_iter =
      Kokkos::Impl::layout_iterate_type_selector<
          preferred_layout>::outer_iteration_pattern;
  static const Kokkos::Iterate inner_iter =
      Kokkos::Impl::layout_iterate_type_selector<
          preferred_layout>::inner_iteration_pattern;
  using rank_type = Kokkos::Rank<Rank, outer_iter, inner_iter>;
  using policy_type =
      Kokkos::MDRangePolicy<ExecutionSpace, rank_type, IndexType>;
  using FunctorType =
      MDRangePolicy_StreamTest<ExecutionSpace, Rank, ScalarType, IndexType>;

  view_type view_A;
  view_type view_B;
  view_type view_C;
  ScalarType scalar;
  int N;

  MDRangePolicy_StreamTest(const view_type &A_, const view_type &B_,
                           const view_type &C_, ScalarType scalar_, int N_)
      : view_A(A_), view_B(B_), view_C(C_), scalar(scalar_), N(N_) {
    static_assert(Rank >= 2 && Rank <= 6,
                  "MDRangePolicy_StreamTest: Only ranks 2 to 6 supported");
  }

  // Tagged operator()
  template <typename... Args>
  KOKKOS_INLINE_FUNCTION void operator()(const Tag_Set &, Args... args) const {
    view_A(args...) = static_cast<ScalarType>(scalar);
  }

  template <typename... Args>
  KOKKOS_INLINE_FUNCTION void operator()(const Tag_Copy &, Args... args) const {
    view_B(args...) = view_A(args...);
  }

  template <typename... Args>
  KOKKOS_INLINE_FUNCTION void operator()(const Tag_Scale &,
                                         Args... args) const {
    view_B(args...) = scalar * view_A(args...);
  }

  template <typename... Args>
  KOKKOS_INLINE_FUNCTION void operator()(const Tag_Add &, Args... args) const {
    view_C(args...) = view_A(args...) + view_B(args...);
  }

  template <typename... Args>
  KOKKOS_INLINE_FUNCTION void operator()(const Tag_Triad &,
                                         Args... args) const {
    view_C(args...) = view_A(args...) + scalar * view_B(args...);
  }

  static view_type create_test_view(const char *name, int dim) {
    long N1 = dim;
    long N2 = N1 * N1;
    long N3 = N2 * N1;
    std::string view_name(name);
    if constexpr (Rank == 2) {
      return view_type(view_name, N3, N3);
    } else if constexpr (Rank == 3) {
      return view_type(view_name, N2, N2, N2);
    } else if constexpr (Rank == 4) {
      return view_type(view_name, N2, N2, N1, N1);
    } else if constexpr (Rank == 5) {
      return view_type(view_name, N2, N1, N1, N1, N1);
    } else if constexpr (Rank == 6) {
      return view_type(view_name, N1, N1, N1, N1, N1, N1);
    }
  }

  struct Init {
    view_type m_tensor;
    scalar_type m_value;

    Init(const view_type &tensor, const scalar_type &value)
        : m_tensor(tensor), m_value(value) {}

    template <typename... Indices>
    KOKKOS_INLINE_FUNCTION void operator()(Indices... indices) const {
      m_tensor(indices...) = m_value;
    }
  };

  template <typename Tag>
  static double run_test(const int N, const int iterations) {
    view_type view_A_test =
        create_test_view("MDRangePolicy_StreamTest::view_A", N);
    view_type view_B_test =
        create_test_view("MDRangePolicy_StreamTest::view_B", N);
    view_type view_C_test =
        create_test_view("MDRangePolicy_StreamTest::view_C", N);
    scalar_type scalar = static_cast<scalar_type>(2.718281828);

    using policy_test_type =
        Kokkos::MDRangePolicy<ExecutionSpace, rank_type, IndexType, Tag>;
    typename policy_test_type::point_type lower_bounds, upper_bounds;
    for (int i = 0; i < Rank; ++i) {
      lower_bounds[i] = 0;
      upper_bounds[i] = view_A_test.extent(i);
    }

    policy_type init_policy(lower_bounds, upper_bounds);
    Kokkos::parallel_for(init_policy,
                         Init(view_A_test, static_cast<ScalarType>(1.0)));
    Kokkos::parallel_for(init_policy,
                         Init(view_B_test, static_cast<ScalarType>(2.0)));
    Kokkos::parallel_for(init_policy,
                         Init(view_C_test, static_cast<ScalarType>(0.0)));
    execution_space().fence();

    policy_test_type compute_policy(lower_bounds, upper_bounds);
    double total_time = 0.0;
    for (int i = 0; i < iterations; ++i) {
      Kokkos::Timer timer;
      Kokkos::parallel_for(compute_policy, FunctorType(view_A_test, view_B_test,
                                                       view_C_test, scalar, N));
      execution_space().fence();
      total_time += timer.seconds();
    }
    total_time /= iterations;
    return total_time;
  }

  static double test_set(int N, int iterations = 1) {
    return run_test<Tag_Set>(N, iterations);
  }

  static double test_copy(int N, int iterations = 1) {
    return run_test<Tag_Copy>(N, iterations);
  }

  static double test_scale(int N, int iterations = 1) {
    return run_test<Tag_Scale>(N, iterations);
  }

  static double test_add(int N, int iterations = 1) {
    return run_test<Tag_Add>(N, iterations);
  }

  static double test_triad(int N, int iterations = 1) {
    return run_test<Tag_Triad>(N, iterations);
  }
};

}  // namespace Benchmark
