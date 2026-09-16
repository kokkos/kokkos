// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif
#include <TestHIP_Category.hpp>

#include <cmath>
#include <vector>

namespace Test {

namespace {

constexpr int NREG  = 128;
constexpr int NITER = 4;

struct TagFor {};
struct TagForLB {};
struct TagReduce {};

struct HeavyKernel {
  Kokkos::View<double*, TEST_EXECSPACE> out;

  KOKKOS_INLINE_FUNCTION double work(const int i) const {
    double r[NREG];
#pragma unroll
    for (int k = 0; k < NREG; ++k) r[k] = 1.0 + 1e-6 * (i + k);

    for (int it = 0; it < NITER; ++it) {
#pragma unroll
      for (int k = 0; k < NREG; ++k) {
        r[k] = r[k] * 1.0000001 + r[(k + 1) % NREG] * 0.9999999;
      }
    }

    double s = 0.0;
#pragma unroll
    for (int k = 0; k < NREG; ++k) s += r[k];
    return s;
  }

  KOKKOS_INLINE_FUNCTION void operator()(TagFor, const int i) const {
    out(i) = work(i);
  }

  KOKKOS_INLINE_FUNCTION void operator()(TagForLB, const int i) const {
    out(i) = work(i);
  }

  KOKKOS_INLINE_FUNCTION void operator()(TagReduce, const int i,
                                         double& lsum) const {
    lsum += work(i);
  }
};

unsigned bit_ceil_u(unsigned v) {
  unsigned b = 1;
  while (b < v) b <<= 1;
  return b;
}

unsigned predicted_block(const size_t nwork, const unsigned cus,
                         const size_t concurrency) {
  if (nwork == 0 || nwork >= concurrency) return 0;
  const unsigned per_eu =
      static_cast<unsigned>((nwork + cus - 1) / cus);  // ceil(nwork / cus)
  unsigned b = bit_ceil_u(per_eu);
  if (b < 256u) b = 256u;
  if (b > 1024u) b = 1024u;
  return b;
}

double view_sum(const Kokkos::View<double*, TEST_EXECSPACE>& view,
                const size_t n) {
  double sum = 0.0;
  Kokkos::parallel_reduce(
      "sum_view", Kokkos::RangePolicy<TEST_EXECSPACE>(0, n),
      KOKKOS_LAMBDA(const int i, double& lsum) { lsum += view(i); }, sum);
  Kokkos::fence();
  return sum;
}

}  // namespace

TEST(hip, blocksize_range_regression) {
  using exec   = TEST_EXECSPACE;
  using policy = Kokkos::RangePolicy<exec>;

  const size_t concurrency = exec().concurrency();
  ASSERT_GT(concurrency, size_t(0));

  hipDeviceProp_t prop;
  int dev = 0;
  KOKKOS_IMPL_HIP_SAFE_CALL(hipGetDevice(&dev));
  KOKKOS_IMPL_HIP_SAFE_CALL(hipGetDeviceProperties(&prop, dev));
  const unsigned cus = prop.multiProcessorCount;
  ASSERT_GT(cus, 0u);

  std::vector<size_t> sweep;
  for (double f : {0.25, 0.50, 0.90, 0.99, 1.01, 1.10, 1.50, 2.00}) {
    sweep.push_back(static_cast<size_t>(f * cus * 512.0));
  }

  bool exercised_1024_heuristic = false;
  for (size_t n : sweep) {
    const unsigned pb = predicted_block(n, cus, concurrency);
    if (pb == 1024u) exercised_1024_heuristic = true;

    Kokkos::View<double*, exec> out_for(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "out_for"), n);
    Kokkos::View<double*, exec> out_for_lb(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "out_for_lb"), n);

    HeavyKernel f_for{out_for};
    HeavyKernel f_for_lb{out_for_lb};
    HeavyKernel f_reduce{out_for};

    Kokkos::parallel_for("for", Kokkos::RangePolicy<exec, TagFor>(0, n),
                         f_for);
    Kokkos::fence();

    Kokkos::parallel_for(
        "for_lb",
        Kokkos::RangePolicy<exec, TagForLB, Kokkos::LaunchBounds<256, 1>>(0,
                                                                           n),
        f_for_lb);
    Kokkos::fence();

    double sum_reduce = 0.0;
    Kokkos::parallel_reduce("reduce",
                            Kokkos::RangePolicy<exec, TagReduce>(0, n),
                            f_reduce, sum_reduce);
    Kokkos::fence();

    const double sum_for    = view_sum(out_for, n);
    const double sum_for_lb = view_sum(out_for_lb, n);
    const double scale      = std::max(1.0, std::fabs(sum_for));
    const double tol        = 1e-10 * scale;

    ASSERT_NEAR(sum_for_lb, sum_for, tol);
    ASSERT_NEAR(sum_reduce, sum_for, tol);
  }

  ASSERT_TRUE(exercised_1024_heuristic);
}

}  // namespace Test
