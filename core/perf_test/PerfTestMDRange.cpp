// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>
#include <benchmark/benchmark.h>
#include <iostream>

#include "PerfTest_Category.hpp"

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
void check_computation(const ViewType &A, const ViewType &B) {
  long numErrors = 0;
  auto Ahost     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A);
  auto Bhost     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), B);

  const long icount = Ahost.extent(0);
  const long jcount = Ahost.extent(1);
  const long kcount = Ahost.extent(2);
  // On KNL, this may vectorize - add print statement to prevent
  // Also, compare against epsilon, as vectorization can change bitwise
  // answer
  for (long i = 0; i < icount; ++i) {
    for (long j = 0; j < jcount; ++j) {
      for (long k = 0; k < kcount; ++k) {
        ScalarType check =
            0.25 * (ScalarType)(Bhost(i + 2, j, k) + Bhost(i + 1, j, k) +
                                Bhost(i, j + 2, k) + Bhost(i, j + 1, k) +
                                Bhost(i, j, k + 2) + Bhost(i, j, k + 1) +
                                Bhost(i, j, k));
        if (Ahost(i, j, k) - check != 0) {
          ++numErrors;
          std::cout << "  Correctness error at index: " << i << "," << j << ","
                    << k << "\n"
                    << "  multi Ahost = " << Ahost(i, j, k)
                    << "  expected = " << check
                    << "  multi Bhost(ijk) = " << Bhost(i, j, k)
                    << "  multi Bhost(i+1jk) = " << Bhost(i + 1, j, k)
                    << "  multi Bhost(i+2jk) = " << Bhost(i + 2, j, k)
                    << "  multi Bhost(ij+1k) = " << Bhost(i, j + 1, k)
                    << "  multi Bhost(ij+2k) = " << Bhost(i, j + 2, k)
                    << "  multi Bhost(ijk+1) = " << Bhost(i, j, k + 1)
                    << "  multi Bhost(ijk+2) = " << Bhost(i, j, k + 2)
                    << std::endl;
        }
      }
    }
  }
  if (numErrors != 0) {
    std::cout << " LL multi run: errors " << numErrors << "  range product "
              << icount * jcount * kcount << "  LL " << jcount * kcount
              << "  LR " << icount * jcount << std::endl;
  }
}

template <typename FunctorType>
void bench_mdrange(benchmark::State &state) {
  using execution_space = FunctorType::execution_space;
  using view_type       = FunctorType::view_type;

  int icount = state.range(0);
  int jcount = state.range(0);
  int kcount = state.range(0);
  int Ti     = state.range(1);
  int Tj     = state.range(1);
  int Tk     = state.range(1);

  view_type Atest("Atest", icount, jcount, kcount);
  view_type Btest("Btest", icount + 2, jcount + 2, kcount + 2);

  Kokkos::deep_copy(Atest, 1.0);
  execution_space().fence();
  Kokkos::deep_copy(Btest, 1.0);
  execution_space().fence();

  const auto policy =
      FunctorType::get_policy(icount, jcount, kcount, Ti, Tj, Tk);

  int i = 0;

  for (auto _ : state) {
    Kokkos::Timer timer;
    Kokkos::parallel_for(policy,
                         FunctorType(Atest, Btest, icount, jcount, kcount));
    execution_space().fence();
    const double dt = timer.seconds();
    state.SetIterationTime(dt);

    // Correctness check - only the first run
    if (0 == i++) {
      check_computation<typename FunctorType::scalar_type>(Atest, Btest);
    }
  }  // end for
}

template <class DeviceType, typename ScalarType = double,
          typename TestLayout = Kokkos::LayoutRight>
struct MDRange3D {
  using execution_space = DeviceType;
  using scalar_type     = ScalarType;
  using size_type       = typename execution_space::size_type;
  using view_type       = Kokkos::View<ScalarType ***, TestLayout, DeviceType>;

  view_type A;
  view_type B;
  const long irange;
  const long jrange;
  const long krange;

  MDRange3D(const view_type &A_, const view_type &B_, const long &irange_,
            const long &jrange_, const long &krange_)
      : A(A_), B(B_), irange(irange_), jrange(jrange_), krange(krange_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const long i, const long j, const long k) const {
    A(i, j, k) =
        0.25 * (ScalarType)(B(i + 2, j, k) + B(i + 1, j, k) + B(i, j + 2, k) +
                            B(i, j + 1, k) + B(i, j, k + 2) + B(i, j, k + 1) +
                            B(i, j, k));
  }

  static auto get_policy(const unsigned int icount, const unsigned int jcount,
                         const unsigned int kcount, const unsigned int Ti = 1,
                         const unsigned int Tj = 1, const unsigned int Tk = 1) {
    constexpr Kokkos::Iterate iteration_pattern =
        LayoutToIterationPattern<TestLayout>::pattern;
    return Kokkos::MDRangePolicy<
        Kokkos::Rank<3, iteration_pattern, iteration_pattern>, execution_space>(
        {0, 0, 0}, {icount, jcount, kcount}, {Ti, Tj, Tk});
  }
};

template <class DeviceType, typename ScalarType = double,
          typename TestLayout = Kokkos::LayoutRight>
struct RangePolicyCollapseTwo {
  // RangePolicy for 3D range, but will collapse only 2 dims => like Rank<2> for
  // multi-dim; unroll 2 dims in one-dim

  using execution_space = DeviceType;
  using scalar_type     = ScalarType;
  using size_type       = typename execution_space::size_type;
  using view_type       = Kokkos::View<ScalarType ***, TestLayout, DeviceType>;

  view_type A;
  view_type B;
  const long irange;
  const long jrange;
  const long krange;

  RangePolicyCollapseTwo(view_type &A_, const view_type &B_,
                         const long &irange_, const long &jrange_,
                         const long &krange_)
      : A(A_), B(B_), irange(irange_), jrange(jrange_), krange(krange_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const long r) const {
    if constexpr (std::is_same_v<TestLayout, Kokkos::LayoutRight>) {
      // id(i,j,k) = k + j*Nk + i*Nk*Nj = k + Nk*(j + i*Nj) = k + Nk*r
      // r = j + i*Nj
      long i = int(r / jrange);
      long j = int(r - i * jrange);
      for (int k = 0; k < krange; ++k) {
        A(i, j, k) =
            0.25 * (ScalarType)(B(i + 2, j, k) + B(i + 1, j, k) +
                                B(i, j + 2, k) + B(i, j + 1, k) +
                                B(i, j, k + 2) + B(i, j, k + 1) + B(i, j, k));
      }
    } else if constexpr (std::is_same_v<TestLayout, Kokkos::LayoutLeft>) {
      // id(i,j,k) = i + j*Ni + k*Ni*Nj = i + Ni*(j + k*Nj) = i + Ni*r
      // r = j + k*Nj
      long k = int(r / jrange);
      long j = int(r - k * jrange);
      for (int i = 0; i < irange; ++i) {
        A(i, j, k) =
            0.25 * (ScalarType)(B(i + 2, j, k) + B(i + 1, j, k) +
                                B(i, j + 2, k) + B(i, j + 1, k) +
                                B(i, j, k + 2) + B(i, j, k + 1) + B(i, j, k));
      }
    }
  }

  static auto get_policy(const unsigned int icount, const unsigned int jcount,
                         const unsigned int kcount, const unsigned int,
                         const unsigned int, const unsigned int) {
    long collapse_index_rangeA = 0;
    if constexpr (std::is_same_v<TestLayout, Kokkos::LayoutRight>) {
      collapse_index_rangeA = static_cast<long>(icount) * jcount;
    } else if constexpr (std::is_same_v<TestLayout, Kokkos::LayoutLeft>) {
      collapse_index_rangeA = static_cast<long>(kcount) * jcount;
    } else {
      static_assert(false, "LayoutRight or LayoutLeft required");
    }

    return Kokkos::RangePolicy<execution_space>(0, (collapse_index_rangeA));
  }
};

template <class DeviceType, typename ScalarType = double,
          typename TestLayout = Kokkos::LayoutRight>
struct RangePolicyCollapseAll {
  // RangePolicy for 3D range, but will collapse all dims

  using execution_space = DeviceType;
  using scalar_type     = ScalarType;
  using size_type       = typename execution_space::size_type;
  using view_type       = Kokkos::View<ScalarType ***, TestLayout, DeviceType>;

  view_type A;
  view_type B;
  const long irange;
  const long jrange;
  const long krange;

  RangePolicyCollapseAll(view_type &A_, const view_type &B_,
                         const long &irange_, const long &jrange_,
                         const long &krange_)
      : A(A_), B(B_), irange(irange_), jrange(jrange_), krange(krange_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const long r) const {
    if constexpr (std::is_same_v<TestLayout, Kokkos::LayoutRight>) {
      long i = int(r / (jrange * krange));
      long j = int((r - i * jrange * krange) / krange);
      long k = int(r - i * jrange * krange - j * krange);
      A(i, j, k) =
          0.25 * (ScalarType)(B(i + 2, j, k) + B(i + 1, j, k) + B(i, j + 2, k) +
                              B(i, j + 1, k) + B(i, j, k + 2) + B(i, j, k + 1) +
                              B(i, j, k));
    } else if constexpr (std::is_same_v<TestLayout, Kokkos::LayoutLeft>) {
      long k = int(r / (irange * jrange));
      long j = int((r - k * irange * jrange) / irange);
      long i = int(r - k * irange * jrange - j * irange);
      A(i, j, k) =
          0.25 * (ScalarType)(B(i + 2, j, k) + B(i + 1, j, k) + B(i, j + 2, k) +
                              B(i, j + 1, k) + B(i, j, k + 2) + B(i, j, k + 1) +
                              B(i, j, k));
    }
  }

  static auto get_policy(const unsigned int icount, const unsigned int jcount,
                         const unsigned int kcount, const unsigned int,
                         const unsigned int, const unsigned int) {
    const long flat_index_range = icount * static_cast<long>(jcount) * kcount;
    return Kokkos::RangePolicy<execution_space>(0, flat_index_range);
  }
};

BENCHMARK(bench_mdrange<MDRange3D<TEST_EXECSPACE>>)
    ->Iterations(10)
    ->ArgNames({"size", "tile_size"})
    ->ArgsProduct({benchmark::CreateRange(1 << 7, 1 << 9, 2),
                   benchmark::CreateDenseRange(1, 8, 1)});

BENCHMARK(bench_mdrange<RangePolicyCollapseTwo<TEST_EXECSPACE>>)
    ->Iterations(10)
    ->ArgNames({"size", "tile_size"})
    ->ArgsProduct({benchmark::CreateRange(1 << 7, 1 << 9, 2), {1}});

BENCHMARK(bench_mdrange<RangePolicyCollapseAll<TEST_EXECSPACE>>)
    ->Iterations(10)
    ->ArgNames({"size", "tile_size"})
    ->ArgsProduct({benchmark::CreateRange(1 << 7, 1 << 9, 2), {1}});

}  // end namespace Test
