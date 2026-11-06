// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <cstdio>

#include "utils.hpp"

// This example contracts two rank-3 tensors over one shared index:
//
//   C(i,j,l,m) = sum_k A(i,j,k) * B(k,l,m)
//
// MMA operations do not consume that rank-3 indexing directly. They consume
// small matrix fragments, so the kernel below forms matrix-shaped tiles:
//
//   A_tile(row=i*J+j, k)
//   B_tile(k, col=l*M+m)
//   C_tile(row, col)
//
// The rank-4 result is obtained by reversing row=i*J+j and col=l*M+m.

constexpr int I_EXTENT   = 8;
constexpr int J_EXTENT   = 4;
constexpr int K_EXTENT   = 32;
constexpr int L_EXTENT   = 4;
constexpr int M_EXTENT   = 8;
constexpr int LEFT_ROWS  = I_EXTENT * J_EXTENT;
constexpr int RIGHT_COLS = L_EXTENT * M_EXTENT;

using TensorA = Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>;
using TensorB = Kokkos::View<Scalar***, Kokkos::LayoutRight, ExecSpace>;
using TensorC = Kokkos::View<Scalar****, Kokkos::LayoutRight, ExecSpace>;

using ScratchA = Kokkos::View<Scalar**, Kokkos::LayoutLeft, ScratchSpace,
                              Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using ScratchB = Kokkos::View<Scalar**, Kokkos::LayoutLeft, ScratchSpace,
                              Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

KOKKOS_INLINE_FUNCTION void row_to_left_indices(const int row, int& i, int& j) {
  i = row / J_EXTENT;
  j = row % J_EXTENT;
}

KOKKOS_INLINE_FUNCTION void col_to_right_indices(const int col, int& l,
                                                 int& m) {
  l = col / M_EXTENT;
  m = col % M_EXTENT;
}

void dematricize_result(Matrix matrix, TensorC tensor) {
  Kokkos::parallel_for(
      "dematricize_result",
      Range4D({0, 0, 0, 0}, {I_EXTENT, J_EXTENT, L_EXTENT, M_EXTENT}),
      KOKKOS_LAMBDA(const int i, const int j, const int l, const int m) {
        const int row      = i * J_EXTENT + j;
        const int col      = l * M_EXTENT + m;
        tensor(i, j, l, m) = matrix(row, col);
      });
}

void reference_contraction(TensorA a, TensorB b, TensorC c) {
  Kokkos::parallel_for(
      "reference_contraction",
      Range4D({0, 0, 0, 0}, {I_EXTENT, J_EXTENT, L_EXTENT, M_EXTENT}),
      KOKKOS_LAMBDA(const int i, const int j, const int l, const int m) {
        Scalar sum = 0.0;
        for (int k = 0; k < K_EXTENT; ++k) {
#if defined(KOKKOS_ENABLE_EXPERIMENTAL_SIMD_AMX)
          sum +=
              round_to_bf16_float(a(i, j, k)) * round_to_bf16_float(b(k, l, m));
#else
          sum += a(i, j, k) * b(k, l, m);
#endif
        }
        c(i, j, l, m) = sum;
      });
}

template <class AFragT, class BFragT, class CFragT>
struct TensorContraction {
  TensorA A;
  TensorB B;
  Matrix C_matrix;

  KOKKOS_INLINE_FUNCTION void operator()(const MemberType& member) const {
    // C_matrix has shape (I*J) x (L*M). One team computes one
    // MMA_M x MMA_N matrix tile of that matricized result.
    const int n_tiles_n = RIGHT_COLS / MMA_N;
    const int tile_m    = member.league_rank() / n_tiles_n;
    const int tile_n    = member.league_rank() % n_tiles_n;
    const int row0      = tile_m * MMA_M;
    const int col0      = tile_n * MMA_N;

    ScratchA a_matrix_tile(member.team_scratch(0), MMA_M, MMA_K);
    ScratchB b_matrix_tile(member.team_scratch(0), MMA_K, MMA_N);

    AFragT a_frag;
    BFragT b_frag;
    CFragT c_frag;
    Kokkos::Experimental::fill_fragment(c_frag, Scalar(0.0));

    for (int k0 = 0; k0 < K_EXTENT; k0 += MMA_K) {
      // Form the A operand matrix tile from the original rank-3 tensor. The
      // local row maps to global row=row0+local_row, then to tensor indices
      // (i,j) via row=i*J+j. The local column is the contracted k index.
      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, MMA_M * MMA_K), [&](const int idx) {
            const int local_row = idx / MMA_K;
            const int local_k   = idx % MMA_K;
            int i;
            int j;
            row_to_left_indices(row0 + local_row, i, j);
            a_matrix_tile(local_row, local_k) = A(i, j, k0 + local_k);
          });

      // Form the B operand matrix tile. The local row is the contracted k
      // index, while col=col0+local_col maps back to tensor indices (l,m) via
      // col=l*M+m.
      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, MMA_K * MMA_N), [&](const int idx) {
            const int local_k   = idx / MMA_N;
            const int local_col = idx % MMA_N;
            int l;
            int m;
            col_to_right_indices(col0 + local_col, l, m);
            b_matrix_tile(local_k, local_col) = B(k0 + local_k, l, m);
          });

      member.team_barrier();

      // At this point the rank-3 tensor data has been matricized into two
      // rank-2 scratch tiles. load_matrix_sync sees only those matrix-like
      // subviews, which is the representation MMAs require.
      auto a_tile =
          Kokkos::subview(a_matrix_tile, Kokkos::ALL(), Kokkos::ALL());
      auto b_tile =
          Kokkos::subview(b_matrix_tile, Kokkos::ALL(), Kokkos::ALL());

      Kokkos::Experimental::load_matrix_sync(a_frag, a_tile);
      Kokkos::Experimental::load_matrix_sync(b_frag, b_tile);
      Kokkos::Experimental::mma_sync(c_frag, a_frag, b_frag, c_frag);

      member.team_barrier();
    }

    // Store the matrix result tile to C_matrix[row0:row0+MMA_M,
    // col0:col0+MMA_N]. A separate dematricization step maps that matrix
    // result back to C(i,j,l,m).
    auto c_tile =
        Kokkos::subview(C_matrix, Kokkos::pair<int, int>(row0, row0 + MMA_M),
                        Kokkos::pair<int, int>(col0, col0 + MMA_N));
    Kokkos::Experimental::store_matrix_sync(c_tile, c_frag);
  }
};

bool run_example() {
  static_assert(LEFT_ROWS % MMA_M == 0);
  static_assert(RIGHT_COLS % MMA_N == 0);
  static_assert(K_EXTENT % MMA_K == 0);
  static_assert(WARP_SIZE <= 1024);

  using InputFragDType =
      Kokkos::Experimental::FragmentDType<ExecSpace, InputPrecision>::type;
  using AccumFragDType =
      Kokkos::Experimental::FragmentDType<ExecSpace, AccumPrecision>::type;
  using MMAShape = Kokkos::Experimental::mma_shape<MMA_M, MMA_N, MMA_K>;

  using AFragT = Kokkos::Experimental::fragment<
      InputFragDType, Kokkos::Experimental::matrix_a_extents<MMA_M, MMA_K>,
      Kokkos::layout_left,
      Kokkos::Experimental::mma_policy<MMAShape,
                                       Kokkos::Experimental::matrix_a>>;

  using BFragT = Kokkos::Experimental::fragment<
      InputFragDType, Kokkos::Experimental::matrix_b_extents<MMA_K, MMA_N>,
      Kokkos::layout_left,
      Kokkos::Experimental::mma_policy<MMAShape,
                                       Kokkos::Experimental::matrix_b>>;

  using CFragT = Kokkos::Experimental::fragment<
      AccumFragDType, Kokkos::Experimental::accumulator_extents<MMA_M, MMA_N>,
      Kokkos::layout_right,
      Kokkos::Experimental::mma_policy<MMAShape,
                                       Kokkos::Experimental::accumulator>>;

  TensorA a("A_tensor", I_EXTENT, J_EXTENT, K_EXTENT);
  TensorB b("B_tensor", K_EXTENT, L_EXTENT, M_EXTENT);
  TensorC c("C_tensor", I_EXTENT, J_EXTENT, L_EXTENT, M_EXTENT);
  TensorC c_ref("C_ref", I_EXTENT, J_EXTENT, L_EXTENT, M_EXTENT);
  Matrix c_matrix("C_matrix", LEFT_ROWS, RIGHT_COLS);

  RandPool pool(12345);
  random_rank3_view(a, pool);
  random_rank3_view(b, pool);
  fill_matrix(c_matrix, 0.0);

  // The driver gets the original tensors. It is responsible for choosing a
  // matricization and forming matrix tiles before calling Kokkos SIMD
  // MMA.
  TensorContraction<AFragT, BFragT, CFragT> contraction{a, b, c_matrix};
  TeamPolicy policy((LEFT_ROWS / MMA_M) * (RIGHT_COLS / MMA_N), 1, WARP_SIZE);
  const int scratch_size = ScratchA::shmem_size(MMA_M, MMA_K) +
                           ScratchB::shmem_size(MMA_K, MMA_N);
  policy = policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size));

  Kokkos::parallel_for("tensor_contraction", policy, contraction);
  ExecSpace().fence();

  dematricize_result(c_matrix, c);
  reference_contraction(a, b, c_ref);
  ExecSpace().fence();

#if defined(KOKKOS_ENABLE_EXPERIMENTAL_SIMD_AMX)
  constexpr double tol = 1e-2;
#elif defined(KOKKOS_ENABLE_CUDA)
  constexpr double tol = 1e-15;
#elif defined(KOKKOS_ENABLE_HIP)
  constexpr double tol = 1e-7;
#endif

  const double rel_err = relative_error_rank4(c, c_ref);
  printf(
      "Kokkos SIMD MMA tensor contraction: A(%d,%d,%d) x B(%d,%d,%d), "
      "rel_err=%.4e\n",
      I_EXTENT, J_EXTENT, K_EXTENT, K_EXTENT, L_EXTENT, M_EXTENT, rel_err);

  return rel_err < tol;
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    const bool success = run_example();
    if (!success) {
      printf("Kokkos SIMD MMA tensor contraction: FAILED\n");
      Kokkos::finalize();
      return 1;
    }

    printf("Kokkos SIMD MMA tensor contraction: PASSED\n");
  }
  Kokkos::finalize();
  return 0;
}
