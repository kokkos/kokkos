// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_SIMD_TENSORCORE_EXAMPLES_UTILS_HPP
#define KOKKOS_SIMD_TENSORCORE_EXAMPLES_UTILS_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Random.hpp>

#include <cmath>

#include <Kokkos_SIMD.hpp>

using ExecSpace = Kokkos::Cuda;
using Layout    = Kokkos::LayoutLeft;
using Scalar    = double;

constexpr int WARP_SIZE = 32;
constexpr int WMMA_M    = 8;
constexpr int WMMA_N    = 8;
constexpr int WMMA_K    = 4;
constexpr int BM        = 64;
constexpr int BN        = 32;
constexpr int BK        = 32;
constexpr int M         = 256;
constexpr int N         = 256;
constexpr int K         = 256;

constexpr Kokkos::Experimental::PrecisionType InputPrecision =
    Kokkos::Experimental::PrecisionType::Double;
constexpr Kokkos::Experimental::PrecisionType AccumPrecision =
    Kokkos::Experimental::PrecisionType::Double;

using Matrix        = Kokkos::View<Scalar**, Layout, ExecSpace>;
using TeamPolicy    = Kokkos::TeamPolicy<ExecSpace>;
using MemberType    = TeamPolicy::member_type;
using ScratchSpace  = MemberType::scratch_memory_space;
using ScratchMatrix = Kokkos::View<Scalar**, Layout, ScratchSpace,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using RandPool      = Kokkos::Random_XorShift64_Pool<ExecSpace>;
using Range2D       = Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>;

inline void fill_matrix(Matrix mat, Scalar value) {
  Kokkos::parallel_for(
      "fill_matrix", Range2D({0, 0}, {mat.extent(0), mat.extent(1)}),
      KOKKOS_LAMBDA(const int i, const int j) { mat(i, j) = value; });
}

inline void random_matrix(Matrix mat, RandPool pool) {
  Kokkos::parallel_for(
      "random_matrix", Range2D({0, 0}, {mat.extent(0), mat.extent(1)}),
      KOKKOS_LAMBDA(const int i, const int j) {
        auto gen  = pool.get_state();
        mat(i, j) = gen.frand();
        pool.free_state(gen);
      });
}

inline void reference_matmul(Matrix A, Matrix B, Matrix C) {
  Kokkos::parallel_for(
      "reference_matmul", Range2D({0, 0}, {M, N}),
      KOKKOS_LAMBDA(const int i, const int j) {
        Scalar sum = 0.0;
        for (int k = 0; k < K; ++k) {
          sum += A(i, k) * B(k, j);
        }
        C(i, j) = sum;
      });
}

inline double relative_error(Matrix result, Matrix reference) {
  double err  = 0.0;
  double norm = 0.0;

  Kokkos::parallel_reduce(
      "relative_error", Range2D({0, 0}, {M, N}),
      KOKKOS_LAMBDA(const int i, const int j, double& err_l, double& norm_l) {
        const double e = double(result(i, j)) - double(reference(i, j));
        const double r = double(reference(i, j));
        err_l += e * e;
        norm_l += r * r;
      },
      err, norm);

  return std::sqrt(err) / std::sqrt(norm);
}

#endif
