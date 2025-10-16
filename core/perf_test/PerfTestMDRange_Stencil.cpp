// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <functional>
#include <iostream>
#include <numeric>

#include <benchmark/benchmark.h>

#include "PerfTest_Category.hpp"
#include <Kokkos_Core.hpp>

namespace Test {

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

template <typename ScalarType, typename ViewType>
void check_computation(const ViewType& A, const ViewType& B) {
  int numErrors = 0;
  auto Ahost    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A);
  auto Bhost    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), B);

  // On KNL, this may vectorize - add print statement to prevent
  // Also, compare against epsilon, as vectorization can change bitwise
  // answer
  if constexpr (ViewType::rank == 2) {
    for (int i = 0; i < Ahost.extent_int(0); ++i) {
      for (int j = 0; j < Ahost.extent_int(1); ++j) {
        ScalarType check =
            0.25 *
            (ScalarType)(Bhost(i + 2, j) + Bhost(i + 1, j) + Bhost(i, j + 2) +
                         Bhost(i, j + 1) + Bhost(i, j));
        if (Ahost(i, j) - check != 0) {
          ++numErrors;
          std::cout << "Correctness error at index: " << i << "," << j
                    << ", got " << Ahost(i, j) << ", expected " << check
                    << "\n";
        }
      }
    }
  } else if constexpr (ViewType::rank == 3) {
    for (int i = 0; i < Ahost.extent_int(0); ++i) {
      for (int j = 0; j < Ahost.extent_int(1); ++j) {
        for (int k = 0; k < Ahost.extent_int(2); ++k) {
          ScalarType check =
              0.25 * (ScalarType)(Bhost(i + 2, j, k) + Bhost(i + 1, j, k) +
                                  Bhost(i, j + 2, k) + Bhost(i, j + 1, k) +
                                  Bhost(i, j, k + 2) + Bhost(i, j, k + 1) +
                                  Bhost(i, j, k));
          if (Ahost(i, j, k) - check != 0) {
            ++numErrors;
            std::cout << "Correctness error at index: " << i << "," << j << ","
                      << k << ", got " << Ahost(i, j, k) << ", expected "
                      << check << "\n";
          }
        }
      }
    }
  } else if constexpr (ViewType::rank == 4) {
    for (int i = 0; i < Ahost.extent_int(0); ++i) {
      for (int j = 0; j < Ahost.extent_int(1); ++j) {
        for (int k = 0; k < Ahost.extent_int(2); ++k) {
          for (int u = 0; u < Ahost.extent_int(3); ++u) {
            ScalarType check =
                0.25 *
                (ScalarType)(Bhost(i + 2, j, k, u) + Bhost(i + 1, j, k, u) +
                             Bhost(i, j + 2, k, u) + Bhost(i, j + 1, k, u) +
                             Bhost(i, j, k + 2, u) + Bhost(i, j, k + 1, u) +
                             Bhost(i, j, k, u + 2) + Bhost(i, j, k, u + 1) +
                             Bhost(i, j, k, u));
            if (Ahost(i, j, k, u) - check != 0) {
              ++numErrors;
              std::cout << "Correctness error at index: " << i << "," << j
                        << "," << k << "," << u << ", got " << Ahost(i, j, k, u)
                        << ", expected " << check << "\n";
            }
          }
        }
      }
    }
    if (numErrors != 0) {
      std::cout << "Detected some errors for a run with dimensions "
                << Ahost.extent(0);
      for (std::size_t i = 1; i < Ahost.rank(); i++) {
        std::cout << "x" << Ahost.extent(i);
      }
      std::cout << std::endl;
    }
  }
}

template <typename FunctorType, std::size_t... Idx>
void bench_mdrange(benchmark::State& state, std::index_sequence<Idx...>) {
  using execution_space = typename FunctorType::execution_space;
  using view_type       = typename FunctorType::view_type;

  Kokkos::Array<int, FunctorType::dimension> dims, tiles;
  for (std::size_t i = 0; i < dims.size(); i++) {
    dims[i]  = state.range(0);
    tiles[i] = state.range(1);
  }

  const auto policy = FunctorType::get_policy(dims, tiles);

  bool using_default_tiling = false;
  for (std::size_t i = 0; i < tiles.size(); i++) {
    state.counters[std::string("tile_") + std::to_string(i)] = tiles[i];
    using_default_tiling |= tiles[i] != state.range(1);
  }
  state.counters["default_tiling"] = using_default_tiling;

  view_type Atest("Atest", dims[Idx]...);
  view_type Btest("Btest", (dims[Idx] + 2)...);

  Kokkos::deep_copy(Atest, 1.0);
  execution_space().fence();
  Kokkos::deep_copy(Btest, 1.0);
  execution_space().fence();

  for (auto _ : state) {
    Kokkos::Timer timer;
    Kokkos::parallel_for(policy, FunctorType(Atest, Btest, dims));
    execution_space().fence();
    const double dt = timer.seconds();
    state.SetIterationTime(dt);
  }
  // Correctness check
  check_computation<typename FunctorType::scalar_type>(Atest, Btest);
}

template <typename FunctorType>
void bench_mdrange(benchmark::State& state) {
  bench_mdrange<FunctorType>(
      state, std::make_index_sequence<FunctorType::dimension>());
}

template <typename T, std::size_t Rank>
struct add_pointer_n {
  using type = typename add_pointer_n<T*, Rank - 1>::type;
};

template <typename T>
struct add_pointer_n<T, 0> {
  using type = T;
};

template <typename T, std::size_t Rank>
using add_pointer_n_t = typename add_pointer_n<T, Rank>::type;

template <class DeviceType, int Dimension,
          typename TestLayout = Kokkos::LayoutRight,
          typename ScalarType = double>
struct MDRange {
  using execution_space = DeviceType;
  using scalar_type     = ScalarType;
  using size_type       = typename execution_space::size_type;
  using view_type       = Kokkos::View<add_pointer_n_t<ScalarType, Dimension>,
                                 TestLayout, DeviceType>;

  static constexpr int dimension = Dimension;

  view_type A;
  view_type B;
  const Kokkos::Array<int, dimension> ranges;

  template <typename... Dims>
  MDRange(const view_type& A_, const view_type& B_,
          const Kokkos::Array<int, dimension>& dims)
      : A(A_), B(B_), ranges(dims) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, const int j) const
    requires(dimension == 2)
  {
    A(i, j) = 0.25 * (ScalarType)(B(i + 2, j) + B(i + 1, j) + B(i, j + 2) +
                                  B(i, j + 1) + B(i, j));
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, const int j, const int k) const
    requires(dimension == 3)
  {
    A(i, j, k) =
        0.25 * (ScalarType)(B(i + 2, j, k) + B(i + 1, j, k) + B(i, j + 2, k) +
                            B(i, j + 1, k) + B(i, j, k + 2) + B(i, j, k + 1) +
                            B(i, j, k));
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, const int j, const int k, const int u) const
    requires(dimension == 4)
  {
    A(i, j, k, u) =
        0.25 *
        (ScalarType)(B(i + 2, j, k, u) + B(i + 1, j, k, u) + B(i, j + 2, k, u) +
                     B(i, j + 1, k, u) + B(i, j, k + 2, u) + B(i, j, k + 1, u) +
                     B(i, j, k, u + 2) + B(i, j, k, u + 1) + B(i, j, k, u));
  }

  static auto get_policy(const Kokkos::Array<int, dimension>& end,
                         Kokkos::Array<int, dimension>& tile) {
    constexpr Kokkos::Iterate iteration_pattern =
        LayoutToIterationPattern<TestLayout>::pattern;
    const Kokkos::MDRangePolicy<
        Kokkos::Rank<dimension, iteration_pattern, iteration_pattern>,
        execution_space>
        policy(Kokkos::Array<int, dimension>{}, end, tile);

    for (int i = 0; i < dimension; i++) {
      tile[i] = policy.m_tile[i];
    }

    return policy;
  }
};

template <class DeviceType, int Dimension,
          typename TestLayout = Kokkos::LayoutRight,
          typename ScalarType = double>
struct CollapseTwo {
  // RangePolicy for ND range, but will collapse only 2 dims; unroll 2 dims in
  // one-dim

  using execution_space = DeviceType;
  using scalar_type     = ScalarType;
  using size_type       = typename execution_space::size_type;
  using view_type       = Kokkos::View<add_pointer_n_t<ScalarType, Dimension>,
                                 TestLayout, DeviceType>;

  static constexpr int dimension = Dimension;

  view_type A;
  view_type B;
  const Kokkos::Array<int, dimension> ranges;

  CollapseTwo(view_type& A_, const view_type& B_,
              const Kokkos::Array<int, dimension>& dims)
      : A(A_), B(B_), ranges(dims) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int r) const
    requires(dimension == 3)
  {
    if constexpr (std::is_same_v<TestLayout, Kokkos::LayoutRight>) {
      int i = r / ranges[1];
      int j = r - i * ranges[1];
      for (int k = 0; k < ranges[2]; ++k) {
        A(i, j, k) =
            0.25 * (ScalarType)(B(i + 2, j, k) + B(i + 1, j, k) +
                                B(i, j + 2, k) + B(i, j + 1, k) +
                                B(i, j, k + 2) + B(i, j, k + 1) + B(i, j, k));
      }
    } else if constexpr (std::is_same_v<TestLayout, Kokkos::LayoutLeft>) {
      int k = r / ranges[1];
      int j = r - k * ranges[1];
      for (int i = 0; i < ranges[0]; ++i) {
        A(i, j, k) =
            0.25 * (ScalarType)(B(i + 2, j, k) + B(i + 1, j, k) +
                                B(i, j + 2, k) + B(i, j + 1, k) +
                                B(i, j, k + 2) + B(i, j, k + 1) + B(i, j, k));
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int r) const
    requires(dimension == 4)
  {
    if constexpr (std::is_same_v<TestLayout, Kokkos::LayoutRight>) {
      int i = r / (ranges[1] * ranges[2]);
      int j = (r - i * ranges[1] * ranges[2]) / ranges[2];
      int k = r - i * ranges[1] * ranges[2] - j * ranges[2];
      for (int u = 0; u < ranges[3]; ++u) {
        A(i, j, k, u) =
            0.25 *
            (ScalarType)(B(i + 2, j, k, u) + B(i + 1, j, k, u) +
                         B(i, j + 2, k, u) + B(i, j + 1, k, u) +
                         B(i, j, k + 2, u) + B(i, j, k + 1, u) +
                         B(i, j, k, u + 2) + B(i, j, k, u + 1) + B(i, j, k, u));
      }
    } else if constexpr (std::is_same_v<TestLayout, Kokkos::LayoutLeft>) {
      int u = r / (ranges[1] * ranges[2]);
      int k = (r - u * ranges[1] * ranges[2]) / ranges[1];
      int j = r - u * ranges[1] * ranges[2] - k * ranges[1];
      for (int i = 0; i < ranges[0]; ++i) {
        A(i, j, k, u) =
            0.25 *
            (ScalarType)(B(i + 2, j, k, u) + B(i + 1, j, k, u) +
                         B(i, j + 2, k, u) + B(i, j + 1, k, u) +
                         B(i, j, k + 2, u) + B(i, j, k + 1, u) +
                         B(i, j, k, u + 2) + B(i, j, k, u + 1) + B(i, j, k, u));
      }
    }
  }

  static auto get_policy(const Kokkos::Array<int, dimension>& dims,
                         const Kokkos::Array<int, dimension>&) {
    int collapse_index_rangeA = 0;
    if constexpr (std::is_same_v<TestLayout, Kokkos::LayoutRight>) {
      collapse_index_rangeA = std::reduce(Kokkos::begin(dims),
                                          Kokkos::begin(dims) + (dimension - 1),
                                          1, std::multiplies<int>{});
    } else if constexpr (std::is_same_v<TestLayout, Kokkos::LayoutLeft>) {
      collapse_index_rangeA =
          std::reduce(Kokkos::begin(dims) + 1, Kokkos::end(dims), 1,
                      std::multiplies<int>{});
    } else {
      static_assert(!(std::is_same_v<TestLayout, Kokkos::LayoutRight> ||
                      std::is_same_v<TestLayout, Kokkos::LayoutLeft>),
                    "LayoutRight or LayoutLeft required");
    }

    return Kokkos::RangePolicy<execution_space>(0, collapse_index_rangeA);
  }
};

template <class DeviceType, int Dimension,
          typename TestLayout = Kokkos::LayoutRight,
          typename ScalarType = double>
struct CollapseAll {
  // RangePolicy for ND range, but will collapse all dims

  using execution_space = DeviceType;
  using scalar_type     = ScalarType;
  using size_type       = typename execution_space::size_type;
  using view_type       = Kokkos::View<add_pointer_n_t<ScalarType, Dimension>,
                                 TestLayout, DeviceType>;

  static constexpr int dimension = Dimension;

  view_type A;
  view_type B;
  const Kokkos::Array<int, dimension> ranges;

  template <typename... Dims>
  CollapseAll(view_type& A_, const view_type& B_,
              const Kokkos::Array<int, dimension>& dims)
      : A(A_), B(B_), ranges(dims) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int r) const
    requires(dimension == 2)
  {
    if constexpr (std::is_same_v<TestLayout, Kokkos::LayoutRight>) {
      int i   = r / ranges[1];
      int j   = r - i * ranges[1];
      A(i, j) = 0.25 * (ScalarType)(B(i + 2, j) + B(i + 1, j) + B(i, j + 2) +
                                    B(i, j + 1) + B(i, j));
    } else if constexpr (std::is_same_v<TestLayout, Kokkos::LayoutLeft>) {
      int j   = r / ranges[0];
      int i   = r - j * ranges[0];
      A(i, j) = 0.25 * (ScalarType)(B(i + 2, j) + B(i + 1, j) + B(i, j + 2) +
                                    B(i, j + 1) + B(i, j));
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int r) const
    requires(dimension == 3)
  {
    if constexpr (std::is_same_v<TestLayout, Kokkos::LayoutRight>) {
      int i = r / (ranges[1] * ranges[2]);
      int j = (r - i * ranges[1] * ranges[2]) / ranges[2];
      int k = r - i * ranges[1] * ranges[2] - j * ranges[2];
      A(i, j, k) =
          0.25 * (ScalarType)(B(i + 2, j, k) + B(i + 1, j, k) + B(i, j + 2, k) +
                              B(i, j + 1, k) + B(i, j, k + 2) + B(i, j, k + 1) +
                              B(i, j, k));
    } else if constexpr (std::is_same_v<TestLayout, Kokkos::LayoutLeft>) {
      int k = r / (ranges[0] * ranges[1]);
      int j = (r - k * ranges[0] * ranges[1]) / ranges[0];
      int i = r - k * ranges[0] * ranges[1] - j * ranges[0];
      A(i, j, k) =
          0.25 * (ScalarType)(B(i + 2, j, k) + B(i + 1, j, k) + B(i, j + 2, k) +
                              B(i, j + 1, k) + B(i, j, k + 2) + B(i, j, k + 1) +
                              B(i, j, k));
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int r) const
    requires(dimension == 4)
  {
    if constexpr (std::is_same_v<TestLayout, Kokkos::LayoutRight>) {
      // TODO: store the strides in variables
      int i = r / (ranges[1] * ranges[2] * ranges[3]);
      int j =
          (r - i * ranges[1] * ranges[2] * ranges[3]) / (ranges[2] * ranges[3]);
      int k = (r - i * ranges[1] * ranges[2] * ranges[3] -
               j * ranges[2] * ranges[3]) /
              ranges[3];
      int u = r - i * ranges[1] * ranges[2] * ranges[3] -
              j * ranges[2] * ranges[3] - k * ranges[3];
      A(i, j, k, u) =
          0.25 *
          (ScalarType)(B(i + 2, j, k, u) + B(i + 1, j, k, u) +
                       B(i, j + 2, k, u) + B(i, j + 1, k, u) +
                       B(i, j, k + 2, u) + B(i, j, k + 1, u) +
                       B(i, j, k, u + 2) + B(i, j, k, u + 1) + B(i, j, k, u));
    } else if constexpr (std::is_same_v<TestLayout, Kokkos::LayoutLeft>) {
      int u = r / (ranges[0] * ranges[1] * ranges[2]);
      int k =
          (r - u * ranges[0] * ranges[1] * ranges[2]) / (ranges[1] * ranges[2]);
      int j = (r - u * ranges[0] * ranges[1] * ranges[2] -
               k * ranges[1] * ranges[2]) /
              ranges[2];
      int i = r - u * ranges[0] * ranges[1] * ranges[2] -
              k * ranges[1] * ranges[2] - j * ranges[2];
      A(i, j, k, u) =
          0.25 *
          (ScalarType)(B(i + 2, j, k, u) + B(i + 1, j, k, u) +
                       B(i, j + 2, k, u) + B(i, j + 1, k, u) +
                       B(i, j, k + 2, u) + B(i, j, k + 1, u) +
                       B(i, j, k, u + 2) + B(i, j, k, u + 1) + B(i, j, k, u));
    }
  }

  static auto get_policy(const Kokkos::Array<int, dimension>& dims,
                         const Kokkos::Array<int, dimension>&) {
    const int flat_index_range = std::reduce(
        Kokkos::begin(dims), Kokkos::end(dims), 1, std::multiplies<int>{});
    return Kokkos::RangePolicy<execution_space>(0, flat_index_range);
  }
};

#define MDRANGE_STENCIL_BENCHMARK(functor, dim, layout, ...)                \
  BENCHMARK(bench_mdrange<functor<TEST_EXECSPACE, dim, Kokkos::layout>>)    \
      ->UseManualTime()                                                     \
      ->Unit(benchmark::kMillisecond)                                       \
      ->Name("bench_mdrange_stencil_" #dim "d_" #functor "_" #layout)       \
      ->ArgNames({"size", "tile_size"})                                     \
      ->ArgsProduct(                                                        \
          {benchmark::CreateRange(min_size_##dim##D, max_size_##dim##D, 2), \
           __VA_ARGS__});

// 2D benchmarks
constexpr int min_size_2D = 1 << 9;
constexpr int max_size_2D = 1 << 13;

MDRANGE_STENCIL_BENCHMARK(MDRange, 2, LayoutRight, {0, 1})
MDRANGE_STENCIL_BENCHMARK(CollapseAll, 2, LayoutRight, {-1})
MDRANGE_STENCIL_BENCHMARK(MDRange, 2, LayoutLeft, {0, 1})
MDRANGE_STENCIL_BENCHMARK(CollapseAll, 2, LayoutLeft, {-1})

// 3D benchmarks
constexpr int min_size_3D = 1 << 7;
constexpr int max_size_3D = 1 << 9;

MDRANGE_STENCIL_BENCHMARK(MDRange, 3, LayoutRight, {0, 1})
MDRANGE_STENCIL_BENCHMARK(CollapseTwo, 3, LayoutRight, {-1})
MDRANGE_STENCIL_BENCHMARK(CollapseAll, 3, LayoutRight, {-1})
MDRANGE_STENCIL_BENCHMARK(MDRange, 3, LayoutLeft, {0, 1})
MDRANGE_STENCIL_BENCHMARK(CollapseTwo, 3, LayoutLeft, {-1})
MDRANGE_STENCIL_BENCHMARK(CollapseAll, 3, LayoutLeft, {-1})

// 4D benchmarks
constexpr int min_size_4D = 1 << 5;
constexpr int max_size_4D = 96;

MDRANGE_STENCIL_BENCHMARK(MDRange, 4, LayoutRight, {0, 1})
MDRANGE_STENCIL_BENCHMARK(CollapseTwo, 4, LayoutRight, {-1})
MDRANGE_STENCIL_BENCHMARK(CollapseAll, 4, LayoutRight, {-1})
MDRANGE_STENCIL_BENCHMARK(MDRange, 4, LayoutLeft, {0, 1})
MDRANGE_STENCIL_BENCHMARK(CollapseTwo, 4, LayoutLeft, {-1})
MDRANGE_STENCIL_BENCHMARK(CollapseAll, 4, LayoutLeft, {-1})

#undef MDRANGE_STENCIL_BENCHMARK

}  // end namespace Test
