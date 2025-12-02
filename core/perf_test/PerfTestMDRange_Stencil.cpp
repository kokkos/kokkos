// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <functional>
#include <iostream>
#include <numeric>

#include <benchmark/benchmark.h>

#include "PerfTest_Category.hpp"
#include "impl/Kokkos_Command_Line_Parsing.hpp"
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
          std::cerr << "Correctness error at index: " << i << "," << j
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
            std::cerr << "Correctness error at index: " << i << "," << j << ","
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
              std::cerr << "Correctness error at index: " << i << "," << j
                        << "," << k << "," << u << ", got " << Ahost(i, j, k, u)
                        << ", expected " << check << "\n";
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
  void operator()(const int i, const int j, const int k, const int l) const
    requires(dimension == 4)
  {
    A(i, j, k, l) =
        0.25 *
        (ScalarType)(B(i + 2, j, k, l) + B(i + 1, j, k, l) + B(i, j + 2, k, l) +
                     B(i, j + 1, k, l) + B(i, j, k + 2, l) + B(i, j, k + 1, l) +
                     B(i, j, k, l + 2) + B(i, j, k, l + 1) + B(i, j, k, l));
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
    if constexpr (std::is_same_v<TestLayout, Kokkos::LayoutLeft>) {
      int i = r % ranges[0], j = r / ranges[0];
      for (int k = 0; k < ranges[2]; ++k) {
        A(i, j, k) =
            0.25 * (ScalarType)(B(i + 2, j, k) + B(i + 1, j, k) +
                                B(i, j + 2, k) + B(i, j + 1, k) +
                                B(i, j, k + 2) + B(i, j, k + 1) + B(i, j, k));
      }
    } else {
      int k = r % ranges[2], j = r / ranges[2];
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
    if constexpr (std::is_same_v<TestLayout, Kokkos::LayoutLeft>) {
      int i = r % ranges[0], jk = r / ranges[0];
      int j = jk % ranges[1], k = jk / ranges[1];
      for (int l = 0; l < ranges[3]; ++l) {
        A(i, j, k, l) =
            0.25 *
            (ScalarType)(B(i + 2, j, k, l) + B(i + 1, j, k, l) +
                         B(i, j + 2, k, l) + B(i, j + 1, k, l) +
                         B(i, j, k + 2, l) + B(i, j, k + 1, l) +
                         B(i, j, k, l + 2) + B(i, j, k, l + 1) + B(i, j, k, l));
      }
    } else {
      int l = r % ranges[3], jk = r / ranges[3];
      int k = jk % ranges[2], j = jk / ranges[2];
      for (int i = 0; i < ranges[0]; ++i) {
        A(i, j, k, l) =
            0.25 *
            (ScalarType)(B(i + 2, j, k, l) + B(i + 1, j, k, l) +
                         B(i, j + 2, k, l) + B(i, j + 1, k, l) +
                         B(i, j, k + 2, l) + B(i, j, k + 1, l) +
                         B(i, j, k, l + 2) + B(i, j, k, l + 1) + B(i, j, k, l));
      }
    }
  }

  static auto get_policy(const Kokkos::Array<int, dimension>& dims,
                         const Kokkos::Array<int, dimension>&) {
    int collapse_index_rangeA = 0;
    if constexpr (std::is_same_v<TestLayout, Kokkos::LayoutLeft>) {
      collapse_index_rangeA = std::reduce(Kokkos::begin(dims),
                                          Kokkos::begin(dims) + (dimension - 1),
                                          1, std::multiplies<int>{});
    } else if constexpr (std::is_same_v<TestLayout, Kokkos::LayoutRight>) {
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
    if constexpr (std::is_same_v<TestLayout, Kokkos::LayoutLeft>) {
      int i = r % ranges[0], j = r / ranges[0];
      A(i, j) = 0.25 * (ScalarType)(B(i + 2, j) + B(i + 1, j) + B(i, j + 2) +
                                    B(i, j + 1) + B(i, j));
    } else {
      int j = r % ranges[1], i = r / ranges[1];
      A(i, j) = 0.25 * (ScalarType)(B(i + 2, j) + B(i + 1, j) + B(i, j + 2) +
                                    B(i, j + 1) + B(i, j));
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int r) const
    requires(dimension == 3)
  {
    if constexpr (std::is_same_v<TestLayout, Kokkos::LayoutLeft>) {
      int i = r % ranges[0], jk = r / ranges[0];
      int j = jk % ranges[1], k = jk / ranges[1];
      A(i, j, k) =
          0.25 * (ScalarType)(B(i + 2, j, k) + B(i + 1, j, k) + B(i, j + 2, k) +
                              B(i, j + 1, k) + B(i, j, k + 2) + B(i, j, k + 1) +
                              B(i, j, k));
    } else {
      int k = r % ranges[2], ji = r / ranges[2];
      int j = ji % ranges[1], i = ji / ranges[1];
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
    if constexpr (std::is_same_v<TestLayout, Kokkos::LayoutLeft>) {
      int i = r % ranges[0], jkl = r / ranges[0];
      int j = jkl % ranges[1], kl = jkl / ranges[1];
      int k = kl % ranges[2], l = kl / ranges[2];

      A(i, j, k, l) =
          0.25 *
          (ScalarType)(B(i + 2, j, k, l) + B(i + 1, j, k, l) +
                       B(i, j + 2, k, l) + B(i, j + 1, k, l) +
                       B(i, j, k + 2, l) + B(i, j, k + 1, l) +
                       B(i, j, k, l + 2) + B(i, j, k, l + 1) + B(i, j, k, l));
    } else {
      int l = r % ranges[3], ijk = r / ranges[3];
      int k = ijk % ranges[2], ij = ijk / ranges[2];
      int j = ij % ranges[1], i = ij / ranges[1];
      A(i, j, k, l) =
          0.25 *
          (ScalarType)(B(i + 2, j, k, l) + B(i + 1, j, k, l) +
                       B(i, j + 2, k, l) + B(i, j + 1, k, l) +
                       B(i, j, k + 2, l) + B(i, j, k + 1, l) +
                       B(i, j, k, l + 2) + B(i, j, k, l + 1) + B(i, j, k, l));
    }
  }

  static auto get_policy(const Kokkos::Array<int, dimension>& dims,
                         const Kokkos::Array<int, dimension>&) {
    const int flat_index_range = std::reduce(
        Kokkos::begin(dims), Kokkos::end(dims), 1, std::multiplies<int>{});
    return Kokkos::RangePolicy<execution_space>(0, flat_index_range);
  }
};

#define MDRANGE_STENCIL_BENCHMARK(functor, dim, layout, sizes, ...)      \
  BENCHMARK(bench_mdrange<functor<TEST_EXECSPACE, dim, Kokkos::layout>>) \
      ->UseManualTime()                                                  \
      ->Unit(benchmark::kMillisecond)                                    \
      ->Name("MDRangeStencil_" #dim "D_" #functor "_" #layout)           \
      ->ArgNames({"size", "tile_size"})                                  \
      ->ArgsProduct({sizes, __VA_ARGS__});

int declare_benchmarks() {
  std::vector<int64_t> size_2d{512};
  std::vector<int64_t> size_3d{128};
  std::vector<int64_t> size_4d{32};
  std::vector<int64_t> tile_sizes{0};

#if defined(KOKKOS_ENABLE_COMPILE_AND_RUN_LONG_BENCHMARKS)
  size_2d.push_back(1024);
  size_2d.push_back(2048);
  size_2d.push_back(4096);
  size_2d.push_back(8192);
  size_3d.push_back(192);
  size_3d.push_back(256);
  size_3d.push_back(512);
  size_4d.push_back(64);
  size_4d.push_back(96);
  tile_sizes.push_back(1);
#endif

  MDRANGE_STENCIL_BENCHMARK(MDRange, 2, LayoutRight, size_2d, tile_sizes)
  MDRANGE_STENCIL_BENCHMARK(MDRange, 3, LayoutLeft, size_3d, tile_sizes)
  MDRANGE_STENCIL_BENCHMARK(MDRange, 4, LayoutRight, size_4d, tile_sizes)

#if defined(KOKKOS_ENABLE_COMPILE_AND_RUN_LONG_BENCHMARKS)
  MDRANGE_STENCIL_BENCHMARK(MDRange, 2, LayoutLeft, size_2d, tile_sizes)
  MDRANGE_STENCIL_BENCHMARK(MDRange, 3, LayoutRight, size_3d, tile_sizes)
  MDRANGE_STENCIL_BENCHMARK(MDRange, 4, LayoutLeft, size_4d, tile_sizes)

  MDRANGE_STENCIL_BENCHMARK(CollapseTwo, 3, LayoutRight, size_3d, {-1})
  MDRANGE_STENCIL_BENCHMARK(CollapseTwo, 3, LayoutLeft, size_3d, {-1})
  MDRANGE_STENCIL_BENCHMARK(CollapseTwo, 4, LayoutRight, size_4d, {-1})
  MDRANGE_STENCIL_BENCHMARK(CollapseTwo, 4, LayoutLeft, size_4d, {-1})

  MDRANGE_STENCIL_BENCHMARK(CollapseAll, 2, LayoutRight, size_2d, {-1})
  MDRANGE_STENCIL_BENCHMARK(CollapseAll, 2, LayoutLeft, size_2d, {-1})
  MDRANGE_STENCIL_BENCHMARK(CollapseAll, 3, LayoutRight, size_3d, {-1})
  MDRANGE_STENCIL_BENCHMARK(CollapseAll, 3, LayoutLeft, size_3d, {-1})
  MDRANGE_STENCIL_BENCHMARK(CollapseAll, 4, LayoutRight, size_4d, {-1})
  MDRANGE_STENCIL_BENCHMARK(CollapseAll, 4, LayoutLeft, size_4d, {-1})
#endif

  return 0;
}

static int unused [[maybe_unused]] = declare_benchmarks();

#undef MDRANGE_STENCIL_BENCHMARK

}  // end namespace Test
