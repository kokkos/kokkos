// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Random.hpp>

#include <cstdlib>

#include <Kokkos_SIMD.hpp>

#if !defined(KOKKOS_ENABLE_CUDA)
#error "Kokkos SIMD tensor-core matmul tests require CUDA"
#endif

using ExecSpace = Kokkos::Cuda;

using Layout = Kokkos::LayoutLeft;
using Scalar = double;

using Matrix = Kokkos::View<Scalar**, Layout, ExecSpace>;

using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
using MemberType = TeamPolicy::member_type;

using ScratchSpace  = MemberType::scratch_memory_space;
using ScratchMatrix = Kokkos::View<Scalar**, Layout, ScratchSpace,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

using RandPool = Kokkos::Random_XorShift64_Pool<ExecSpace>;

using range2d = Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>;

constexpr int WARP_SIZE = 32;
constexpr int WMMA_M = 8;
constexpr int WMMA_N = 8;
constexpr int WMMA_K = 4;

constexpr Kokkos::Experimental::PrecisionType InputPrecision =
    Kokkos::Experimental::PrecisionType::Double;
constexpr Kokkos::Experimental::PrecisionType AccumPrecision =
    Kokkos::Experimental::PrecisionType::Double;

void fill_matrix(Matrix& mat, Scalar val) {
  range2d policy({0, 0}, {mat.extent(0), mat.extent(1)});

  Kokkos::parallel_for(
      "fill_matrix", policy,
      KOKKOS_LAMBDA(const int i, const int j) { mat(i, j) = val; });
}

void random_matrix(Matrix& mat, RandPool pool) {
  range2d policy({0, 0}, {mat.extent(0), mat.extent(1)});

  Kokkos::parallel_for(
      "random_matrix", policy, KOKKOS_LAMBDA(const int i, const int j) {
        auto gen = pool.get_state();

        mat(i, j) = gen.frand();

        pool.free_state(gen);
      });
}

void compute_relative_err(Matrix& C, Matrix& C_ref, double* rel_err) {
  double err  = 0.0f;
  double norm = 0.0f;

  range2d policy({0, 0}, {C.extent(0), C.extent(1)});
  Kokkos::parallel_reduce(
      "relative_error", policy,
      KOKKOS_LAMBDA(const int i, const int j, double& err_l, double& norm_l) {
        const double e = double(C(i, j)) - double(C_ref(i, j));
        const double r = double(C_ref(i, j));

        err_l += e * e;
        norm_l += r * r;
      },

      err, norm);
  Kokkos::fence();

  *rel_err = std::sqrt(err) / std::sqrt(norm);
}

void naive_matmul(Matrix A, Matrix B, Matrix C) {
  range2d policy({0, 0}, {C.extent(0), C.extent(1)});

  Kokkos::parallel_for(
      "NaiveMatMul", policy, KOKKOS_LAMBDA(const int i, const int j) {
        Scalar sum = 0.0;

        for (int k = 0; k < A.extent(1); ++k) sum += A(i, k) * B(k, j);

        C(i, j) = sum;
      });
}

/// Warp-level tensor core functor
template <class ABufferT, class BBufferT, class CBufferT, class AFragT,
          class BFragT, class CFragT, int BM, int BN, int BK, int M, int N,
          int K>
struct WarpLevelHardwareAcceleratedMatmul {
  ABufferT A;
  BBufferT B;
  CBufferT C;

  KOKKOS_INLINE_FUNCTION void operator()(const MemberType& member) const {
    // fragments must be thread-local
    AFragT a_frag;
    BFragT b_frag;
    CFragT c_frag;

    Kokkos::Experimental::fill_fragment(c_frag, Scalar(0.0f));

    // find output tile
    const int tile_id = member.league_rank();

    const int n_tiles_N = (N + BN - 1) / BN;
    const int tile_m    = tile_id / n_tiles_N;
    const int tile_n    = tile_id % n_tiles_N;

    const int tile_m0 = tile_m * BM;
    const int tile_n0 = tile_n * BN;

    // find output warp tile
    const int warp_id = member.team_rank();

    const int n_warps_N = BN / WMMA_N;
    const int warp_m    = warp_id / n_warps_N;
    const int warp_n    = warp_id % n_warps_N;

    const int i = tile_m0 + warp_m * WMMA_M;
    const int j = tile_n0 + warp_n * WMMA_N;

    ScratchMatrix A_tile(member.team_scratch(0), BM, BK);
    ScratchMatrix B_tile(member.team_scratch(0), BK, BN);

    // perform MMA on warp tiles
    for (int k0 = 0; k0 < K; k0 += BK) {
      // load into shared memory
      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, BM * BK), [&](const int idx) {
            const int local_i = idx / BK;
            const int local_k = idx % BK;

            A_tile(local_i, local_k) = A(tile_m0 + local_i, k0 + local_k);
          });

      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, BK * BN), [&](const int idx) {
            const int local_k = idx / BN;
            const int local_j = idx % BN;

            B_tile(local_k, local_j) = B(k0 + local_k, tile_n0 + local_j);
          });
      member.team_barrier();

      for (int wmma_k = 0; wmma_k < BK; wmma_k += WMMA_K) {
        auto a_tile = Kokkos::subview(
            A_tile,
            Kokkos::pair<int, int>(warp_m * WMMA_M, (warp_m + 1) * WMMA_M),
            Kokkos::pair<int, int>(wmma_k, wmma_k + WMMA_K));
        auto b_tile = Kokkos::subview(
            B_tile, Kokkos::pair<int, int>(wmma_k, wmma_k + WMMA_K),
            Kokkos::pair<int, int>(warp_n * WMMA_N, (warp_n + 1) * WMMA_N));

        Kokkos::Experimental::load_matrix_sync(a_frag, a_tile);
        Kokkos::Experimental::load_matrix_sync(b_frag, b_tile);

        Kokkos::Experimental::mma_sync(c_frag, a_frag, b_frag, c_frag);
      }

      member.team_barrier();
    }

    auto c_tile = Kokkos::subview(C, Kokkos::pair<int, int>(i, i + WMMA_M),
                                  Kokkos::pair<int, int>(j, j + WMMA_N));
    Kokkos::Experimental::store_matrix_sync(c_tile, c_frag);
  }
};

constexpr int DEFAULT_BM = 64;
constexpr int DEFAULT_BN = 32;
constexpr int DEFAULT_BK = 32;

template <int M, int N, int K, int BM = DEFAULT_BM, int BN = DEFAULT_BN,
          int BK = DEFAULT_BK>
bool run_matmul_case() {
  static_assert(M % BM == 0 && N % BN == 0 && K % BK == 0,
                "Problem dimensions must be divisible by block dimensions");
  static_assert(BM % WMMA_M == 0 && BN % WMMA_N == 0 && BK % WMMA_K == 0,
                "Block dimensions must be divisible by WMMA dimensions");

  constexpr int warps_m   = BM / WMMA_M;
  constexpr int warps_n   = BN / WMMA_N;
  constexpr int team_size = warps_m * warps_n;

  static_assert(team_size * WARP_SIZE <= 1024,
                "Kokkos TeamPolicy team_size * vector_length must be <= 1024");

  // initialize rand pool
  RandPool pool(12345);

  {
    // set up device views
    Matrix A("A", M, K);
    random_matrix(A, pool);

    Matrix B("B", K, N);
    random_matrix(B, pool);

    Matrix C("C", M, N);
    fill_matrix(C, 0.0);
    Kokkos::fence();

    // define fragment types
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
        AccumFragDType,
        Kokkos::Experimental::accumulator_extents<WMMA_M, WMMA_N>,
        Kokkos::layout_right,
        Kokkos::Experimental::mma_policy<MMAShape,
                                         Kokkos::Experimental::accumulator>>;

    const int n_tiles_m   = (M + BM - 1) / BM;
    const int n_tiles_n   = (N + BN - 1) / BN;
    const int league_size = n_tiles_m * n_tiles_n;

    // define functor for MMA-accelerated matmul
    WarpLevelHardwareAcceleratedMatmul<
        Matrix, Matrix, Matrix,  // global buffer types
        AFragT, BFragT, CFragT,  // thread-local fragment types
        BM, BN, BK, M, N, K      // global-level problem size
        >
        hamm{A, B, C};

    // Kokkos does not guarantee consecutive threads belong to a warp unless
    // vector_length is set to the warp size.
    TeamPolicy policy(league_size, team_size, WARP_SIZE);

    const int scratch_size =
        ScratchMatrix::shmem_size(BM, BK) + ScratchMatrix::shmem_size(BK, BN);

    policy = policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size));

    // run the functor
    Kokkos::parallel_for("naive_tc_matmul", policy, hamm);
    ExecSpace().fence();

    // check results
    Matrix C_ref("C_ref", M, N);

    // initialize ref matrix to 0
    fill_matrix(C_ref, 0.0f);
    Kokkos::fence();

    // carry out naive matmul
    naive_matmul(A, B, C_ref);
    Kokkos::fence();

    double tol = 1e-15;

    double rel_err;
    compute_relative_err(C, C_ref, &rel_err);
    const bool success = rel_err < tol;
    if (!success) {
      printf(
          "Kokkos SIMD tensor-core matmul failed: M,N,K=(%d,%d,%d), "
          "BM,BN,BK=(%d,%d,%d), "
          "WMMA_M,N,K=(%d,%d,%d), rel_err=%.4e, tol=%.4e\n",
          M, N, K, BM, BN, BK, WMMA_M, WMMA_N, WMMA_K, rel_err, tol);
    }
    return success;
  }
}

template <int BM = DEFAULT_BM, int BN = DEFAULT_BN, int BK = DEFAULT_BK>
bool run_size_case(int size) {
  switch (size) {
    case 64: return run_matmul_case<64, 64, 64, BM, BN, BK>();
    case 128: return run_matmul_case<128, 128, 128, BM, BN, BK>();
    case 256: return run_matmul_case<256, 256, 256, BM, BN, BK>();
    case 512: return run_matmul_case<512, 512, 512, BM, BN, BK>();
    case 1024: return run_matmul_case<1024, 1024, 1024, BM, BN, BK>();
    case 2048: return run_matmul_case<2048, 2048, 2048, BM, BN, BK>();
    default:
      printf("Kokkos SIMD tensor-core matmul test: unsupported size %d\n",
             size);
      return false;
  }
}

template <int BM, int BN, int BK>
bool run_block_case(int size) {
  return run_size_case<BM, BN, BK>(size);
}

bool run_selected_case(int size, int bm, int bn, int bk) {
  if (bm == 64 && bn == 32 && bk == 32) return run_block_case<64, 32, 32>(size);
  if (bm == 32 && bn == 32 && bk == 32) return run_block_case<32, 32, 32>(size);

  printf(
      "Kokkos SIMD tensor-core matmul test: unsupported block size "
      "(%d,%d,%d)\n",
      bm, bn, bk);
  return false;
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf("usage: %s SIZE BM BN BK\n", argv[0]);
    return 1;
  }

  const int size = std::atoi(argv[1]);
  const int bm   = std::atoi(argv[2]);
  const int bn   = std::atoi(argv[3]);
  const int bk   = std::atoi(argv[4]);

  Kokkos::initialize();
  {
    const bool success = run_selected_case(size, bm, bn, bk);

    if (!success) {
      printf("Kokkos SIMD tensor-core matmul test: FAILED\n");
      Kokkos::finalize();
      return 1;
    }

    printf("Kokkos SIMD tensor-core matmul test: PASSED\n");
  }
  Kokkos::finalize();

  return 0;
}
