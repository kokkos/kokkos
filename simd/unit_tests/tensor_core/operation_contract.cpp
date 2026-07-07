// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <Kokkos_SIMD.hpp>

#if defined(KOKKOS_ENABLE_CUDA)
using ExecSpace = Kokkos::Cuda;
#elif defined(KOKKOS_ENABLE_HIP)
using ExecSpace = Kokkos::HIP;
#else
#error "Kokkos SIMD tensor-core operation contract tests require CUDA or HIP"
#endif

using Scalar            = double;
#if defined(KOKKOS_ENABLE_HIP)
constexpr int WARP_SIZE = 64;
constexpr int WMMA_M    = 16;
constexpr int WMMA_N    = 16;
constexpr int WMMA_K    = 4;
#else
constexpr int WARP_SIZE = 32;
constexpr int WMMA_M    = 8;
constexpr int WMMA_N    = 8;
constexpr int WMMA_K    = 4;
#endif
constexpr Kokkos::Experimental::PrecisionType InputPrecision =
    Kokkos::Experimental::PrecisionType::Double;
constexpr Kokkos::Experimental::PrecisionType AccumPrecision =
    Kokkos::Experimental::PrecisionType::Double;

using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
using MemberType = TeamPolicy::member_type;
using Range2D    = Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>;

template <class Layout>
using Matrix = Kokkos::View<Scalar**, Layout, ExecSpace>;

template <class Layout>
struct OperandLayout;

template <>
struct OperandLayout<Kokkos::LayoutLeft> {
  using type = Kokkos::layout_left;
};

template <>
struct OperandLayout<Kokkos::LayoutRight> {
  using type = Kokkos::layout_right;
};

using InputFragDType =
    typename Kokkos::Experimental::FragmentDType<ExecSpace,
                                                 InputPrecision>::type;
using AccumFragDType =
    typename Kokkos::Experimental::FragmentDType<ExecSpace,
                                                 AccumPrecision>::type;
using MMAShape = Kokkos::Experimental::mma_shape<WMMA_M, WMMA_N, WMMA_K>;

template <class OperandLayoutT>
using AFrag = Kokkos::Experimental::fragment<
    InputFragDType, Kokkos::Experimental::matrix_a_extents<WMMA_M, WMMA_K>,
    OperandLayoutT,
    Kokkos::Experimental::mma_policy<MMAShape, Kokkos::Experimental::matrix_a,
                                     OperandLayoutT, ExecSpace>>;

template <class OperandLayoutT>
using BFrag = Kokkos::Experimental::fragment<
    InputFragDType, Kokkos::Experimental::matrix_b_extents<WMMA_K, WMMA_N>,
    OperandLayoutT,
    Kokkos::Experimental::mma_policy<MMAShape, Kokkos::Experimental::matrix_b,
                                     OperandLayoutT, ExecSpace>>;

using CFrag = Kokkos::Experimental::fragment<
    AccumFragDType, Kokkos::Experimental::accumulator_extents<WMMA_M, WMMA_N>,
    Kokkos::layout_right,
    Kokkos::Experimental::mma_policy<MMAShape,
                                     Kokkos::Experimental::accumulator,
                                     Kokkos::layout_left, ExecSpace>>;

KOKKOS_INLINE_FUNCTION Scalar reference_operand_value(Scalar value) {
  return value;
}

template <class Layout>
void fill_operands(Matrix<Layout> a, Matrix<Layout> b) {
  Kokkos::parallel_for(
      "fill_a_contract", Range2D({0, 0}, {WMMA_M, WMMA_K}),
      KOKKOS_LAMBDA(const int i, const int k) {
        a(i, k) = Scalar(0.125) * Scalar((i + 1) + 2 * (k + 1));
      });

  Kokkos::parallel_for(
      "fill_b_contract", Range2D({0, 0}, {WMMA_K, WMMA_N}),
      KOKKOS_LAMBDA(const int k, const int j) {
        b(k, j) = Scalar(0.0625) * Scalar((k + 1) - (j + 1));
      });
}

template <class Layout>
void fill_constant(Matrix<Layout> matrix, const Scalar value) {
  Kokkos::parallel_for(
      "fill_constant_contract", Range2D({0, 0}, {WMMA_M, WMMA_N}),
      KOKKOS_LAMBDA(const int i, const int j) { matrix(i, j) = value; });
}

template <class Layout>
void reference_matmul(Matrix<Layout> a, Matrix<Layout> b, Matrix<Layout> c,
                      const Scalar initial_value = Scalar(0.0)) {
  Kokkos::parallel_for(
      "reference_contract_matmul", Range2D({0, 0}, {WMMA_M, WMMA_N}),
      KOKKOS_LAMBDA(const int i, const int j) {
        Scalar sum = initial_value;
        for (int k = 0; k < WMMA_K; ++k) {
          sum += reference_operand_value(a(i, k)) *
                 reference_operand_value(b(k, j));
        }
        c(i, j) = sum;
      });
}

template <class Layout>
double relative_error(Matrix<Layout> result, Matrix<Layout> reference) {
  double err  = 0.0;
  double norm = 0.0;
  Kokkos::parallel_reduce(
      "operation_contract_relative_error", Range2D({0, 0}, {WMMA_M, WMMA_N}),
      KOKKOS_LAMBDA(const int i, const int j, double& err_l, double& norm_l) {
        const double e = double(result(i, j)) - double(reference(i, j));
        const double r = double(reference(i, j));
        err_l += e * e;
        norm_l += r * r;
      },
      err, norm);

  return std::sqrt(err) / std::sqrt(norm);
}

template <class Layout, class OperandLayoutT>
struct OneTileMatmul {
  Matrix<Layout> A;
  Matrix<Layout> B;
  Matrix<Layout> C;

  KOKKOS_INLINE_FUNCTION void operator()(const MemberType&) const {
    AFrag<OperandLayoutT> a_frag;
    BFrag<OperandLayoutT> b_frag;
    CFrag c_frag;

    Kokkos::Experimental::fill_fragment(c_frag, Scalar(0.0));

    auto a_tile = Kokkos::subview(A, Kokkos::pair<int, int>(0, WMMA_M),
                                  Kokkos::pair<int, int>(0, WMMA_K));
    auto b_tile = Kokkos::subview(B, Kokkos::pair<int, int>(0, WMMA_K),
                                  Kokkos::pair<int, int>(0, WMMA_N));
    auto c_tile = Kokkos::subview(C, Kokkos::pair<int, int>(0, WMMA_M),
                                  Kokkos::pair<int, int>(0, WMMA_N));

    Kokkos::Experimental::load_matrix_sync(a_frag, a_tile);
    Kokkos::Experimental::load_matrix_sync(b_frag, b_tile);
    Kokkos::Experimental::mma_sync(c_frag, a_frag, b_frag, c_frag);
    Kokkos::Experimental::store_matrix_sync(c_tile, c_frag);
  }
};

template <class Layout>
bool run_valid_layout_case(const char* name) {
  using OperandLayoutT = typename OperandLayout<Layout>::type;

  Matrix<Layout> a("A", WMMA_M, WMMA_K);
  Matrix<Layout> b("B", WMMA_K, WMMA_N);
  Matrix<Layout> c("C", WMMA_M, WMMA_N);
  Matrix<Layout> reference("Reference", WMMA_M, WMMA_N);

  fill_operands(a, b);
  reference_matmul(a, b, reference);

  OneTileMatmul<Layout, OperandLayoutT> functor{a, b, c};
  Kokkos::parallel_for("operation_contract_valid_layout",
                       TeamPolicy(1, 1, WARP_SIZE), functor);
  ExecSpace().fence();

#if defined(KOKKOS_ENABLE_HIP)
  constexpr double tol = 1e-7;
#else
  constexpr double tol = 1e-15;
#endif

  const double rel_err = relative_error(c, reference);
  const bool success   = rel_err < tol;
  if (!success) {
    std::printf(
        "Kokkos SIMD tensor-core operation contract %s failed: rel_err=%.4e "
        "tol=%.4e\n",
        name, rel_err, tol);
  }
  return success;
}

using I8Matrix  = Kokkos::View<std::int8_t**, Kokkos::LayoutLeft, ExecSpace>;
using I32Matrix = Kokkos::View<std::int32_t**, Kokkos::LayoutLeft, ExecSpace>;

using Int8InputFragDType = typename Kokkos::Experimental::FragmentDType<
    ExecSpace, Kokkos::Experimental::PrecisionType::Int8>::type;
using Int8AccumFragDType = typename Kokkos::Experimental::FragmentDType<
    ExecSpace, Kokkos::Experimental::PrecisionType::Int32>::type;
using Int8MMAShape = Kokkos::Experimental::mma_shape<16, 16, 16>;

using Int8AFrag = Kokkos::Experimental::fragment<
    Int8InputFragDType, Kokkos::Experimental::matrix_a_extents<16, 16>,
    Kokkos::layout_left,
    Kokkos::Experimental::mma_policy<Int8MMAShape,
                                     Kokkos::Experimental::matrix_a,
                                     Kokkos::layout_left, ExecSpace>>;

using Int8BFrag = Kokkos::Experimental::fragment<
    Int8InputFragDType, Kokkos::Experimental::matrix_b_extents<16, 16>,
    Kokkos::layout_left,
    Kokkos::Experimental::mma_policy<Int8MMAShape,
                                     Kokkos::Experimental::matrix_b,
                                     Kokkos::layout_left, ExecSpace>>;

using Int8CFrag = Kokkos::Experimental::fragment<
    Int8AccumFragDType, Kokkos::Experimental::accumulator_extents<16, 16>,
    Kokkos::layout_right,
    Kokkos::Experimental::mma_policy<Int8MMAShape,
                                     Kokkos::Experimental::accumulator,
                                     Kokkos::layout_left, ExecSpace>>;

void fill_int8_operands(I8Matrix a, I8Matrix b) {
  Kokkos::parallel_for(
      "fill_int8_a_contract", Range2D({0, 0}, {16, 16}),
      KOKKOS_LAMBDA(const int i, const int k) {
        a(i, k) = static_cast<std::int8_t>(((i + 2 * k) % 9) - 4);
      });

  Kokkos::parallel_for(
      "fill_int8_b_contract", Range2D({0, 0}, {16, 16}),
      KOKKOS_LAMBDA(const int k, const int j) {
        b(k, j) = static_cast<std::int8_t>(((3 * k - j) % 11) - 5);
      });
}

void reference_int8_matmul(I8Matrix a, I8Matrix b, I32Matrix c) {
  Kokkos::parallel_for(
      "reference_int8_contract_matmul", Range2D({0, 0}, {16, 16}),
      KOKKOS_LAMBDA(const int i, const int j) {
        std::int32_t sum = 0;
        for (int k = 0; k < 16; ++k) {
          sum += static_cast<std::int32_t>(a(i, k)) *
                 static_cast<std::int32_t>(b(k, j));
        }
        c(i, j) = sum;
      });
}

struct Int8OneTileMatmul {
  I8Matrix A;
  I8Matrix B;
  I32Matrix C;

  KOKKOS_INLINE_FUNCTION void operator()(const MemberType&) const {
    Int8AFrag a_frag;
    Int8BFrag b_frag;
    Int8CFrag c_frag;

    Kokkos::Experimental::fill_fragment(c_frag, std::int32_t(0));

    auto a_tile = Kokkos::subview(A, Kokkos::pair<int, int>(0, 16),
                                  Kokkos::pair<int, int>(0, 16));
    auto b_tile = Kokkos::subview(B, Kokkos::pair<int, int>(0, 16),
                                  Kokkos::pair<int, int>(0, 16));
    auto c_tile = Kokkos::subview(C, Kokkos::pair<int, int>(0, 16),
                                  Kokkos::pair<int, int>(0, 16));

    Kokkos::Experimental::load_matrix_sync(a_frag, a_tile);
    Kokkos::Experimental::load_matrix_sync(b_frag, b_tile);
    Kokkos::Experimental::mma_sync(c_frag, a_frag, b_frag, c_frag);
    Kokkos::Experimental::store_matrix_sync(c_tile, c_frag);
  }
};

bool run_int8_matmul_case() {
  I8Matrix a("A_int8_contract", 16, 16);
  I8Matrix b("B_int8_contract", 16, 16);
  I32Matrix c("C_int8_contract", 16, 16);
  I32Matrix reference("Reference_int8_contract", 16, 16);

  fill_int8_operands(a, b);
  reference_int8_matmul(a, b, reference);

  Int8OneTileMatmul functor{a, b, c};
  Kokkos::parallel_for("operation_contract_int8_matmul",
                       TeamPolicy(1, 1, WARP_SIZE), functor);
  ExecSpace().fence();

  int max_abs_err = 0;
  Kokkos::parallel_reduce(
      "operation_contract_int8_error", Range2D({0, 0}, {16, 16}),
      KOKKOS_LAMBDA(const int i, const int j, int& err_l) {
        const int diff     = int(c(i, j)) - int(reference(i, j));
        const int abs_diff = diff < 0 ? -diff : diff;
        if (abs_diff > err_l) err_l = abs_diff;
      },
      Kokkos::Max<int>(max_abs_err));

  if (max_abs_err != 0) {
    std::printf(
        "Kokkos SIMD tensor-core INT8 contract failed: max_abs_err=%d\n",
        max_abs_err);
  }
  return max_abs_err == 0;
}

bool run_mode(const char* mode) {
  if (std::strcmp(mode, "valid-left") == 0) {
    return run_valid_layout_case<Kokkos::LayoutLeft>("valid-left");
  }

  if (std::strcmp(mode, "valid-right") == 0) {
    return run_valid_layout_case<Kokkos::LayoutRight>("valid-right");
  }

  if (std::strcmp(mode, "int8-matmul") == 0) {
    return run_int8_matmul_case();
  }

  std::printf("unsupported operation contract mode: %s\n", mode);
  return false;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::printf("usage: %s MODE\n", argv[0]);
    return 1;
  }

  Kokkos::initialize(argc, argv);
  const bool success = run_mode(argv[1]);
  Kokkos::finalize();

  if (!success) {
    std::printf("Kokkos SIMD tensor-core operation contract test FAILED: %s\n",
                argv[1]);
    return 1;
  }

  std::printf("Kokkos SIMD tensor-core operation contract test PASSED: %s\n",
              argv[1]);
  return 0;
}
