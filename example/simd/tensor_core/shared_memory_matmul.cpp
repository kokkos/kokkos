// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <cstdio>

#include "utils.hpp"

// The MMA instruction shape is backend-specific. The rest of the example is
// written in terms of WMMA_M/N/K so the kernel structure is the same for CUDA
// and HIP.

template <class AFragT, class BFragT, class CFragT>
struct SharedMemoryMatmul {
  Matrix A;
  Matrix B;
  Matrix C;

  KOKKOS_INLINE_FUNCTION void operator()(const MemberType& member) const {
    // C is partitioned into BM x BN team tiles. league_rank is linearized in
    // row-major tile order: tile_m selects the block row and tile_n selects the
    // block column, so this team owns C[tile_m0:tile_m0+BM,
    // tile_n0:tile_n0+BN].
    const int tile_id   = member.league_rank();
    const int n_tiles_n = N / BN;
    const int tile_m    = tile_id / n_tiles_n;
    const int tile_n    = tile_id % n_tiles_n;
    const int tile_m0   = tile_m * BM;
    const int tile_n0   = tile_n * BN;

    // Each team contains (BM/WMMA_M) * (BN/WMMA_N) logical warp tiles.
    // team_rank is also row-major: warp_m selects the WMMA tile row inside the
    // team tile and warp_n selects the WMMA tile column. The resulting (i,j) is
    // the upper-left corner of this rank's WMMA_M x WMMA_N output tile.
    const int warp_id   = member.team_rank();
    const int n_warps_n = BN / WMMA_N;
    const int warp_m    = warp_id / n_warps_n;
    const int warp_n    = warp_id % n_warps_n;
    const int i         = tile_m0 + warp_m * WMMA_M;
    const int j         = tile_n0 + warp_n * WMMA_N;

    // The scratch views are the shared-memory staging area. The global A and
    // B tiles are copied cooperatively before fragments are loaded.
    ScratchMatrix a_shared(member.team_scratch(0), BM, BK);
    ScratchMatrix b_shared(member.team_scratch(0), BK, BN);

    // Fragments are thread-local wrappers around the backend-native MMA
    // fragment/tile type. MatrixA and MatrixB describe the operands consumed
    // by mma_sync; the accumulator fragment stores the partial C tile.
    AFragT a_frag;
    BFragT b_frag;
    CFragT c_frag;
    Kokkos::Experimental::fill_fragment(c_frag, Scalar(0.0));

    for (int k0 = 0; k0 < K; k0 += BK) {
      // Stage A[tile_m0:tile_m0+BM, k0:k0+BK] into a_shared and
      // B[k0:k0+BK, tile_n0:tile_n0+BN] into b_shared. The linear idx maps to
      // (row, col) in the corresponding scratch tile.
      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, BM * BK),
                           [&](const int idx) {
                             const int row      = idx / BK;
                             const int col      = idx % BK;
                             a_shared(row, col) = A(tile_m0 + row, k0 + col);
                           });

      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, BK * BN),
                           [&](const int idx) {
                             const int row      = idx / BN;
                             const int col      = idx % BN;
                             b_shared(row, col) = B(k0 + row, tile_n0 + col);
                           });

      member.team_barrier();

      for (int kk = 0; kk < BK; kk += WMMA_K) {
        // Extract the two operand tiles for this MMA step:
        // a_tile = a_shared[warp_m*WMMA_M:(warp_m+1)*WMMA_M, kk:kk+WMMA_K]
        // b_tile = b_shared[kk:kk+WMMA_K, warp_n*WMMA_N:(warp_n+1)*WMMA_N]
        auto a_tile = Kokkos::subview(
            a_shared,
            Kokkos::pair<int, int>(warp_m * WMMA_M, (warp_m + 1) * WMMA_M),
            Kokkos::pair<int, int>(kk, kk + WMMA_K));
        auto b_tile = Kokkos::subview(
            b_shared, Kokkos::pair<int, int>(kk, kk + WMMA_K),
            Kokkos::pair<int, int>(warp_n * WMMA_N, (warp_n + 1) * WMMA_N));

        Kokkos::Experimental::load_matrix_sync(a_frag, a_tile);
        Kokkos::Experimental::load_matrix_sync(b_frag, b_tile);

        // Accumulate C += A * B for this warp tile. The same c_frag is used
        // as both input and output so the loop over kk accumulates over K.
        Kokkos::Experimental::mma_sync(c_frag, a_frag, b_frag, c_frag);
      }

      member.team_barrier();
    }

    // Store the completed accumulator into C[i:i+WMMA_M, j:j+WMMA_N].
    // store_matrix_sync infers the memory layout and leading dimension from the
    // destination subview's strides.
    auto c_tile = Kokkos::subview(C, Kokkos::pair<int, int>(i, i + WMMA_M),
                                  Kokkos::pair<int, int>(j, j + WMMA_N));
    Kokkos::Experimental::store_matrix_sync(c_tile, c_frag);
  }
};

bool run_example() {
  static_assert(M % BM == 0 && N % BN == 0 && K % BK == 0);
  static_assert(BM % WMMA_M == 0 && BN % WMMA_N == 0 && BK % WMMA_K == 0);

  constexpr int team_size = (BM / WMMA_M) * (BN / WMMA_N);
  static_assert(team_size * WARP_SIZE <= 1024);

  using InputFragDType =
      Kokkos::Experimental::FragmentDType<ExecSpace, InputPrecision>::type;
  using AccumFragDType =
      Kokkos::Experimental::FragmentDType<ExecSpace, AccumPrecision>::type;
  using MMAShape = Kokkos::Experimental::mma_shape<WMMA_M, WMMA_N, WMMA_K>;

  using AFragT = Kokkos::Experimental::fragment<
      InputFragDType, Kokkos::Experimental::matrix_a_extents<WMMA_M, WMMA_K>,
      Kokkos::layout_left,
      Kokkos::Experimental::mma_policy<MMAShape,
                                       Kokkos::Experimental::matrix_a>>;

  using BFragT = Kokkos::Experimental::fragment<
      InputFragDType, Kokkos::Experimental::matrix_b_extents<WMMA_K, WMMA_N>,
      Kokkos::layout_left,
      Kokkos::Experimental::mma_policy<MMAShape,
                                       Kokkos::Experimental::matrix_b>>;

  using CFragT = Kokkos::Experimental::fragment<
      AccumFragDType, Kokkos::Experimental::accumulator_extents<WMMA_M, WMMA_N>,
      Kokkos::layout_right,
      Kokkos::Experimental::mma_policy<MMAShape,
                                       Kokkos::Experimental::accumulator>>;

  Matrix A("A", M, K);
  Matrix B("B", K, N);
  Matrix C("C", M, N);
  Matrix C_ref("C_ref", M, N);

  RandPool pool(12345);
  random_matrix(A, pool);
  random_matrix(B, pool);
  fill_matrix(C, 0.0);
  fill_matrix(C_ref, 0.0);

  TeamPolicy policy((M / BM) * (N / BN), team_size, WARP_SIZE);
  const int scratch_size =
      ScratchMatrix::shmem_size(BM, BK) + ScratchMatrix::shmem_size(BK, BN);
  policy = policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size));

  // Launch one team per output block. The vector length is the backend warp
  // or wavefront size so the native MMA operation sees the expected lanes.
  SharedMemoryMatmul<AFragT, BFragT, CFragT> matmul{A, B, C};
  Kokkos::parallel_for("shared_memory_matmul", policy, matmul);
  ExecSpace().fence();

  reference_matmul(A, B, C_ref);
  ExecSpace().fence();

#if defined(KOKKOS_ENABLE_HIP)
  constexpr double tol = 1e-7;
#else
  constexpr double tol = 1e-15;
#endif

  const double rel_err = relative_error(C, C_ref);
  printf(
      "Kokkos SIMD tensor-core shared-memory matmul: M,N,K=(%d,%d,%d), "
      "BM,BN,BK=(%d,%d,%d), "
      "rel_err=%.4e\n",
      M, N, K, BM, BN, BK, rel_err);

  return rel_err < tol;
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    const bool success = run_example();
    if (!success) {
      printf("Kokkos SIMD tensor-core shared-memory matmul: FAILED\n");
      Kokkos::finalize();
      return 1;
    }

    printf("Kokkos SIMD tensor-core shared-memory matmul: PASSED\n");
  }
  Kokkos::finalize();
  return 0;
}
