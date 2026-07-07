// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Random.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#include <Kokkos_SIMD.hpp>

#if defined(KOKKOS_ENABLE_EXPERIMENTAL_SIMD_AMX)
using ExecSpace = Kokkos::DefaultHostExecutionSpace;
#elif defined(KOKKOS_ENABLE_CUDA)
using ExecSpace = Kokkos::Cuda;
#elif defined(KOKKOS_ENABLE_HIP)
using ExecSpace = Kokkos::HIP;
#endif

#if defined(KOKKOS_ENABLE_EXPERIMENTAL_SIMD_AMX)
using Scalar            = float;
constexpr int WARP_SIZE = 1;
constexpr int WMMA_M    = 16;
constexpr int WMMA_N    = 16;
constexpr int WMMA_K    = 32;
constexpr Kokkos::Experimental::PrecisionType InputPrecision =
    Kokkos::Experimental::PrecisionType::BF16;
constexpr Kokkos::Experimental::PrecisionType AccumPrecision =
    Kokkos::Experimental::PrecisionType::Float;
#elif defined(KOKKOS_ENABLE_HIP)
using Scalar            = double;
constexpr int WARP_SIZE = 64;
constexpr int WMMA_M    = 16;
constexpr int WMMA_N    = 16;
constexpr int WMMA_K    = 4;
constexpr Kokkos::Experimental::PrecisionType InputPrecision =
    Kokkos::Experimental::PrecisionType::Double;
constexpr Kokkos::Experimental::PrecisionType AccumPrecision =
    Kokkos::Experimental::PrecisionType::Double;
#elif defined(KOKKOS_ENABLE_CUDA)
using Scalar            = double;
constexpr int WARP_SIZE = 32;
constexpr int WMMA_M    = 8;
constexpr int WMMA_N    = 8;
constexpr int WMMA_K    = 4;
constexpr Kokkos::Experimental::PrecisionType InputPrecision =
    Kokkos::Experimental::PrecisionType::Double;
constexpr Kokkos::Experimental::PrecisionType AccumPrecision =
    Kokkos::Experimental::PrecisionType::Double;
#endif

using Tensor     = Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>;
using Range2D    = Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>;
using Range3D    = Kokkos::MDRangePolicy<Kokkos::Rank<3>, ExecSpace>;
using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
using MemberType = TeamPolicy::member_type;
using RandPool   = Kokkos::Random_XorShift64_Pool<ExecSpace>;

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

template <class MatT>
void fill_random(MatT mat, RandPool pool) {
  Range2D policy({0, 0}, {mat.extent(0), mat.extent(1)});
  Kokkos::parallel_for(
      "fill_random_matrix", policy, KOKKOS_LAMBDA(const int i, const int j) {
        auto gen  = pool.get_state();
        mat(i, j) = gen.frand();
        pool.free_state(gen);
      });
}

void fill_random(Tensor tensor, RandPool pool) {
  Range3D policy({0, 0, 0},
                 {tensor.extent(0), tensor.extent(1), tensor.extent(2)});
  Kokkos::parallel_for(
      "fill_random_tensor", policy,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        auto gen        = pool.get_state();
        tensor(i, j, k) = gen.frand();
        pool.free_state(gen);
      });
}

#if defined(KOKKOS_ENABLE_EXPERIMENTAL_SIMD_AMX)
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

KOKKOS_INLINE_FUNCTION Scalar reference_operand_value(Scalar value) {
  return bf16_bits_to_float(float_to_bf16_bits(value));
}
#else
KOKKOS_INLINE_FUNCTION Scalar reference_operand_value(Scalar value) {
  return value;
}
#endif

template <int Size, int Axis>
KOKKOS_INLINE_FUNCTION Scalar tensor_fiber_value(Tensor tensor, const int row,
                                                 const int col) {
  if constexpr (Axis == 0) {
    const int j = col / Size;
    const int k = col % Size;
    auto fiber  = Kokkos::subview(tensor, Kokkos::ALL(), j, k);
    return fiber(row);
  } else if constexpr (Axis == 1) {
    const int i = col / Size;
    const int k = col % Size;
    auto fiber  = Kokkos::subview(tensor, i, Kokkos::ALL(), k);
    return fiber(row);
  } else {
    const int i = col / Size;
    const int j = col % Size;
    auto fiber  = Kokkos::subview(tensor, i, j, Kokkos::ALL());
    return fiber(row);
  }
}

template <int Size, int Axis, class Layout>
void matricize_tensor(Tensor tensor, Matrix<Layout> matrix) {
  Range2D policy({0, 0}, {Size, Size * Size});
  Kokkos::parallel_for(
      "matricize_tensor", policy, KOKKOS_LAMBDA(const int row, const int col) {
        matrix(row, col) = tensor_fiber_value<Size, Axis>(tensor, row, col);
      });
}

template <int Size, int Axis, class Layout>
void reference_contract(Matrix<Layout> op, Tensor tensor, Matrix<Layout> ref) {
  Range2D policy({0, 0}, {Size, Size * Size});
  Kokkos::parallel_for(
      "reference_contract", policy,
      KOKKOS_LAMBDA(const int row, const int col) {
        Scalar sum = 0.0;
        for (int q = 0; q < Size; ++q) {
          const Scalar a = reference_operand_value(op(row, q));
          const Scalar b = reference_operand_value(
              tensor_fiber_value<Size, Axis>(tensor, q, col));
          sum += a * b;
        }
        ref(row, col) = sum;
      });
}

template <int Size, class Layout, class AFragT, class BFragT, class CFragT>
struct DirectTensorCoreMatmul {
  Matrix<Layout> A;
  Matrix<Layout> B;
  Matrix<Layout> C;

  KOKKOS_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const int n_tiles_n = (Size * Size) / WMMA_N;
    const int tile_m    = member.league_rank() / n_tiles_n;
    const int tile_n    = member.league_rank() % n_tiles_n;
    const int i         = tile_m * WMMA_M;
    const int j         = tile_n * WMMA_N;

    AFragT a_frag;
    BFragT b_frag;
    CFragT c_frag;

    Kokkos::Experimental::fill_fragment(c_frag, Scalar(0.0));

    for (int k0 = 0; k0 < Size; k0 += WMMA_K) {
      auto a_tile = Kokkos::subview(A, Kokkos::pair<int, int>(i, i + WMMA_M),
                                    Kokkos::pair<int, int>(k0, k0 + WMMA_K));
      auto b_tile = Kokkos::subview(B, Kokkos::pair<int, int>(k0, k0 + WMMA_K),
                                    Kokkos::pair<int, int>(j, j + WMMA_N));

      Kokkos::Experimental::load_matrix_sync(a_frag, a_tile);
      Kokkos::Experimental::load_matrix_sync(b_frag, b_tile);
      Kokkos::Experimental::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    auto c_tile = Kokkos::subview(C, Kokkos::pair<int, int>(i, i + WMMA_M),
                                  Kokkos::pair<int, int>(j, j + WMMA_N));
    Kokkos::Experimental::store_matrix_sync(c_tile, c_frag);
  }
};

template <int Size, class Layout>
void tensor_core_contract(Matrix<Layout> op, Matrix<Layout> input,
                          Matrix<Layout> output) {
  using InputFragDType =
      Kokkos::Experimental::FragmentDType<ExecSpace, InputPrecision>::type;
  using AccumFragDType =
      Kokkos::Experimental::FragmentDType<ExecSpace, AccumPrecision>::type;
  using MMAShape = Kokkos::Experimental::mma_shape<WMMA_M, WMMA_N, WMMA_K>;
  using OperandLayoutT = typename OperandLayout<Layout>::type;

  using AFragT = Kokkos::Experimental::fragment<
      InputFragDType, Kokkos::Experimental::matrix_a_extents<WMMA_M, WMMA_K>,
      OperandLayoutT,
      Kokkos::Experimental::mma_policy<MMAShape, Kokkos::Experimental::matrix_a,
                                       OperandLayoutT>>;

  using BFragT = Kokkos::Experimental::fragment<
      InputFragDType, Kokkos::Experimental::matrix_b_extents<WMMA_K, WMMA_N>,
      OperandLayoutT,
      Kokkos::Experimental::mma_policy<MMAShape, Kokkos::Experimental::matrix_b,
                                       OperandLayoutT>>;

  using CFragT = Kokkos::Experimental::fragment<
      AccumFragDType, Kokkos::Experimental::accumulator_extents<WMMA_M, WMMA_N>,
      Kokkos::layout_right,
      Kokkos::Experimental::mma_policy<MMAShape,
                                       Kokkos::Experimental::accumulator>>;

  static_assert(Size % WMMA_M == 0);
  static_assert((Size * Size) % WMMA_N == 0);
  static_assert(Size % WMMA_K == 0);

  constexpr int team_size = 1;
  static_assert(team_size * WARP_SIZE <= 1024);

  DirectTensorCoreMatmul<Size, Layout, AFragT, BFragT, CFragT> functor{
      op, input, output};
  TeamPolicy policy((Size / WMMA_M) * ((Size * Size) / WMMA_N), team_size,
                    WARP_SIZE);

  Kokkos::parallel_for("tensor_core_contract", policy, functor);
  ExecSpace().fence();
}

template <class Layout>
double relative_error(Matrix<Layout> result, Matrix<Layout> ref) {
  double err  = 0.0;
  double norm = 0.0;
  Range2D policy({0, 0}, {result.extent(0), result.extent(1)});
  Kokkos::parallel_reduce(
      "relative_error", policy,
      KOKKOS_LAMBDA(const int i, const int j, double& err_l, double& norm_l) {
        const double e = double(result(i, j)) - double(ref(i, j));
        const double r = double(ref(i, j));
        err_l += e * e;
        norm_l += r * r;
      },
      err, norm);
  Kokkos::fence();
  return std::sqrt(err) / std::sqrt(norm);
}

template <int Size, int Axis, class Layout>
bool run_axis_layout_case() {
  RandPool pool(12345 + Size + Axis);

  Tensor tensor("tensor", Size, Size, Size);
  Matrix<Layout> op("operator", Size, Size);
  Matrix<Layout> input("input_matrix", Size, Size * Size);
  Matrix<Layout> output("tc_output", Size, Size * Size);
  Matrix<Layout> ref("ref_output", Size, Size * Size);

  fill_random(tensor, pool);
  fill_random(op, pool);
  matricize_tensor<Size, Axis>(tensor, input);
  reference_contract<Size, Axis>(op, tensor, ref);
  tensor_core_contract<Size>(op, input, output);

#if defined(KOKKOS_ENABLE_EXPERIMENTAL_SIMD_AMX)
  constexpr double tol = 1e-2;
#elif defined(KOKKOS_ENABLE_CUDA)
  constexpr double tol = 1e-15;
#elif defined(KOKKOS_ENABLE_HIP)
  constexpr double tol = 1e-7;
#endif

  const double rel_err = relative_error(output, ref);
  const bool success   = rel_err < tol;
  if (!success) {
    printf(
        "Kokkos SIMD tensor-core tensor contraction failed: axis=%d, "
        "rel_err=%.4e, "
        "tol=%.4e, size=%d\n",
        Axis, rel_err, tol, Size);
  }
  return success;
}

template <int Size, class Layout>
bool run_layout_case(int axis) {
  switch (axis) {
    case 0: return run_axis_layout_case<Size, 0, Layout>();
    case 1: return run_axis_layout_case<Size, 1, Layout>();
    case 2: return run_axis_layout_case<Size, 2, Layout>();
    default:
      printf(
          "Kokkos SIMD tensor-core tensor contraction test: unsupported axis "
          "%d\n",
          axis);
      return false;
  }
}

template <class Layout>
bool run_size_case(int size, int axis) {
  switch (size) {
#if defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_EXPERIMENTAL_SIMD_AMX)
    case 8: return run_layout_case<8, Layout>(axis);
#endif
#if (defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)) && \
    !defined(KOKKOS_ENABLE_EXPERIMENTAL_SIMD_AMX)
    case 16: return run_layout_case<16, Layout>(axis);
#endif
    case 32: return run_layout_case<32, Layout>(axis);
    default:
      printf(
          "Kokkos SIMD tensor-core tensor contraction test: unsupported size "
          "%d\n",
          size);
      return false;
  }
}

bool run_selected_case(int size, int axis, int layout_id) {
  if (layout_id == 0) return run_size_case<Kokkos::LayoutLeft>(size, axis);
  if (layout_id == 1) return run_size_case<Kokkos::LayoutRight>(size, axis);

  printf(
      "Kokkos SIMD tensor-core tensor contraction test: unsupported layout id "
      "%d\n",
      layout_id);
  return false;
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("usage: %s SIZE AXIS LAYOUT_ID\n", argv[0]);
    return 1;
  }

  const int size      = std::atoi(argv[1]);
  const int axis      = std::atoi(argv[2]);
  const int layout_id = std::atoi(argv[3]);

  Kokkos::initialize();
  {
    const bool success = run_selected_case(size, axis, layout_id);
    if (!success) {
      printf("Kokkos SIMD tensor-core tensor contraction test: FAILED\n");
      Kokkos::finalize();
      return 1;
    }

    printf("Kokkos SIMD tensor-core tensor contraction test: PASSED\n");
  }
  Kokkos::finalize();

  return 0;
}
