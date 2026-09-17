// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_SIMD_MMA_EXAMPLES_UTILS_HPP
#define KOKKOS_SIMD_MMA_EXAMPLES_UTILS_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Random.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>

#include <Kokkos_SIMD.hpp>

#if defined(KOKKOS_ENABLE_EXPERIMENTAL_SIMD_AMX)
using ExecSpace = Kokkos::DefaultHostExecutionSpace;
#elif defined(KOKKOS_ENABLE_CUDA)
using ExecSpace = Kokkos::Cuda;
#elif defined(KOKKOS_ENABLE_HIP)
using ExecSpace = Kokkos::HIP;
#else
#error "Kokkos SIMD MMA examples require AMX, CUDA, or HIP"
#endif

using Layout = Kokkos::LayoutLeft;

#if defined(KOKKOS_ENABLE_EXPERIMENTAL_SIMD_AMX)
using Scalar            = float;
constexpr int WARP_SIZE = 1;
constexpr int MMA_M    = 16;
constexpr int MMA_N    = 16;
constexpr int MMA_K    = 32;
constexpr int BM        = 64;
constexpr int BN        = 64;
constexpr int BK        = 64;
constexpr Kokkos::Experimental::PrecisionType InputPrecision =
    Kokkos::Experimental::PrecisionType::BF16;
constexpr Kokkos::Experimental::PrecisionType AccumPrecision =
    Kokkos::Experimental::PrecisionType::Float;
#elif defined(KOKKOS_ENABLE_HIP)
using Scalar            = double;
constexpr int WARP_SIZE = 64;
constexpr int MMA_M    = 16;
constexpr int MMA_N    = 16;
constexpr int MMA_K    = 4;
constexpr int BM        = 64;
constexpr int BN        = 64;
constexpr int BK        = 32;
constexpr Kokkos::Experimental::PrecisionType InputPrecision =
    Kokkos::Experimental::PrecisionType::Double;
constexpr Kokkos::Experimental::PrecisionType AccumPrecision =
    Kokkos::Experimental::PrecisionType::Double;
#else
using Scalar            = double;
constexpr int WARP_SIZE = 32;
constexpr int MMA_M    = 8;
constexpr int MMA_N    = 8;
constexpr int MMA_K    = 4;
constexpr int BM        = 64;
constexpr int BN        = 32;
constexpr int BK        = 32;
constexpr Kokkos::Experimental::PrecisionType InputPrecision =
    Kokkos::Experimental::PrecisionType::Double;
constexpr Kokkos::Experimental::PrecisionType AccumPrecision =
    Kokkos::Experimental::PrecisionType::Double;
#endif

constexpr int M = 256;
constexpr int N = 256;
constexpr int K = 256;

using Matrix        = Kokkos::View<Scalar**, Layout, ExecSpace>;
using TeamPolicy    = Kokkos::TeamPolicy<ExecSpace>;
using MemberType    = TeamPolicy::member_type;
using ScratchSpace  = MemberType::scratch_memory_space;
using ScratchMatrix = Kokkos::View<Scalar**, Layout, ScratchSpace,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using RandPool      = Kokkos::Random_XorShift64_Pool<ExecSpace>;
using Range2D       = Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>;
using Range3D       = Kokkos::MDRangePolicy<Kokkos::Rank<3>, ExecSpace>;
using Range4D       = Kokkos::MDRangePolicy<Kokkos::Rank<4>, ExecSpace>;

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

template <class TensorT>
inline void random_rank3_view(TensorT tensor, RandPool pool) {
  Kokkos::parallel_for(
      "random_rank3_view",
      Range3D({0, 0, 0},
              {tensor.extent(0), tensor.extent(1), tensor.extent(2)}),
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        auto gen        = pool.get_state();
        tensor(i, j, k) = gen.frand();
        pool.free_state(gen);
      });
}

KOKKOS_INLINE_FUNCTION uint16_t float_to_bf16_bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));

  const uint32_t lsb           = (bits >> 16) & 1;
  const uint32_t rounding_bias = 0x7fff + lsb;
  return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

KOKKOS_INLINE_FUNCTION float bf16_bits_to_float(uint16_t bits) {
  uint32_t word = uint32_t(bits) << 16;
  float value   = 0.0f;
  std::memcpy(&value, &word, sizeof(value));
  return value;
}

KOKKOS_INLINE_FUNCTION float round_to_bf16_float(float value) {
  return bf16_bits_to_float(float_to_bf16_bits(value));
}

inline void reference_matmul(Matrix A, Matrix B, Matrix C) {
  Kokkos::parallel_for(
      "reference_matmul", Range2D({0, 0}, {M, N}),
      KOKKOS_LAMBDA(const int i, const int j) {
        Scalar sum = 0.0;
        for (int k = 0; k < K; ++k) {
#if defined(KOKKOS_ENABLE_EXPERIMENTAL_SIMD_AMX)
          sum += round_to_bf16_float(A(i, k)) * round_to_bf16_float(B(k, j));
#else
          sum += A(i, k) * B(k, j);
#endif
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

template <class TensorT>
inline double relative_error_rank4(TensorT result, TensorT reference) {
  double err  = 0.0;
  double norm = 0.0;

  Kokkos::parallel_reduce(
      "relative_error_rank4",
      Range4D({0, 0, 0, 0}, {result.extent(0), result.extent(1),
                             result.extent(2), result.extent(3)}),
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l,
                    double& err_l, double& norm_l) {
        const double e =
            double(result(i, j, k, l)) - double(reference(i, j, k, l));
        const double r = double(reference(i, j, k, l));
        err_l += e * e;
        norm_l += r * r;
      },
      err, norm);

  return std::sqrt(err) / std::sqrt(norm);
}

#endif
