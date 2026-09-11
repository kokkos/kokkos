// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

// Reference: Uchino, Ozaki, Imamura, "Ozaki Scheme II: A GEMM-oriented
// emulation of floating-point matrix multiplication using an integer modular
// technique", arXiv:2504.08009.

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Random.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <Kokkos_SIMD.hpp>

#if defined(KOKKOS_ENABLE_EXPERIMENTAL_SIMD_AMX)
using ExecSpace         = Kokkos::DefaultHostExecutionSpace;
constexpr int WARP_SIZE = 1;
#elif defined(KOKKOS_ENABLE_CUDA)
using ExecSpace         = Kokkos::Cuda;
constexpr int WARP_SIZE = 32;
#elif defined(KOKKOS_ENABLE_HIP)
using ExecSpace         = Kokkos::HIP;
constexpr int WARP_SIZE = 64;
#else
#error "ozaki2_dgemm requires an AMX, CUDA, or HIP Kokkos backend"
#endif

using u128 = unsigned __int128;

// One hardware INT8 MMA tile shape supported by every backend (CUDA WMMA s8,
// rocWMMA s8, and AMX _tile_dpbssd all accept 16x16x16).
constexpr int MMA_M = 16;
constexpr int MMA_N = 16;
constexpr int MMA_K = 16;

// Team (block) tile of the output that one Kokkos team computes. The problem
// dimensions must be multiples of these (and these of the warp tile below).
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;

// Warp-tile (register) blocking: each warp computes a WARP_TILE_M x WARP_TILE_N
// grid of WMMA tiles, holding that many accumulator fragments at once. Each
// loaded A operand fragment is reused across all WARP_TILE_N columns and each B
// fragment across all WARP_TILE_M rows, so the expensive shared-memory loads
// and MMA operand traffic are amortized instead of repeated per 16x16 tile.
constexpr int WARP_TILE_M = 4;
constexpr int WARP_TILE_N = 4;
constexpr int WARP_M      = WARP_TILE_M * MMA_M;
constexpr int WARP_N      = WARP_TILE_N * MMA_N;

using Layout     = Kokkos::LayoutLeft;
using DMatrix    = Kokkos::View<double**, Layout, ExecSpace>;
using I8Matrix   = Kokkos::View<std::int8_t**, Layout, ExecSpace>;
using I32Matrix  = Kokkos::View<std::int32_t**, Layout, ExecSpace>;
using I64Matrix  = Kokkos::View<long long**, Layout, ExecSpace>;
using U128Matrix = Kokkos::View<u128**, Layout, ExecSpace>;
using IVec       = Kokkos::View<int*, ExecSpace>;

using TeamPolicy   = Kokkos::TeamPolicy<ExecSpace>;
using MemberType   = TeamPolicy::member_type;
using ScratchSpace = MemberType::scratch_memory_space;
using ScratchI8    = Kokkos::View<std::int8_t**, Layout, ScratchSpace,
                                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using RandPool     = Kokkos::Random_XorShift64_Pool<ExecSpace>;
using Range2D      = Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>;
using Range1D      = Kokkos::RangePolicy<ExecSpace>;

// Moduli for the INT8 scheme. Pairwise coprime, each <= 256 so a centered
// residue fits in a signed int8. log2(prod) ~ 125.
// Coprime structure: 2^8, 3*5*17, 11*23, 13*19, 7*31, and the rest prime.
static const int kModsI8[]  = {256, 255, 253, 251, 247, 239, 233, 229,
                               227, 223, 217, 211, 199, 197, 193, 191};
static const int kNumModsI8 = sizeof(kModsI8) / sizeof(kModsI8[0]);

// Extended-Euclid modular inverse a^-1 mod m (host only).
static long long mod_inverse(long long a, long long m) {
  long long t = 0, newt = 1, r = m, newr = a % m;
  while (newr != 0) {
    long long q   = r / newr;
    long long tmp = t - q * newt;
    t             = newt;
    newt          = tmp;
    tmp           = r - q * newr;
    r             = newr;
    newr          = tmp;
  }
  if (r != 1) {
    std::fprintf(stderr, "ozaki2_dgemm: moduli not coprime\n");
    std::exit(1);
  }
  return t < 0 ? t + m : t;
}

KOKKOS_INLINE_FUNCTION int centered_mod(long long a, int p, double invp) {
  long long q = (long long)llrint((double)a * invp);
  int r       = (int)(a - q * p);
  if (2 * r >= p)
    r -= p;
  else if (2 * r < -p)
    r += p;
  return r;
}

template <class AFragT, class BFragT, class CFragT>
struct Int8TiledGemm {
  I8Matrix A8;
  I8Matrix B8;
  I32Matrix G;
  int N, K;

  KOKKOS_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const int n_tiles_n = N / BN;
    const int tile_m0   = (member.league_rank() / n_tiles_n) * BM;
    const int tile_n0   = (member.league_rank() % n_tiles_n) * BN;

    // This warp owns a WARP_M x WARP_N sub-block of the team's BM x BN tile.
    const int warp_id   = member.team_rank();
    const int n_warps_n = BN / WARP_N;
    const int warp_m0   = (warp_id / n_warps_n) * WARP_M;
    const int warp_n0   = (warp_id % n_warps_n) * WARP_N;

    ScratchI8 a_shared(member.team_scratch(0), BM, BK);
    ScratchI8 b_shared(member.team_scratch(0), BK, BN);

    // One accumulator fragment per WMMA tile in this warp's grid.
    CFragT c_frag[WARP_TILE_M][WARP_TILE_N];
    for (int wm = 0; wm < WARP_TILE_M; ++wm)
      for (int wn = 0; wn < WARP_TILE_N; ++wn)
        Kokkos::Experimental::fill_fragment(c_frag[wm][wn], std::int32_t(0));

    for (int k0 = 0; k0 < K; k0 += BK) {
      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, BM * BK),
                           [&](const int idx) {
                             const int row      = idx / BK;
                             const int col      = idx % BK;
                             a_shared(row, col) = A8(tile_m0 + row, k0 + col);
                           });

      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, BK * BN),
                           [&](const int idx) {
                             const int row      = idx / BN;
                             const int col      = idx % BN;
                             b_shared(row, col) = B8(k0 + row, tile_n0 + col);
                           });

      member.team_barrier();

      for (int kk = 0; kk < BK; kk += MMA_K) {
        // Load this warp's column of A fragments and row of B fragments once,
        // then reuse them across the whole WARP_TILE_M x WARP_TILE_N grid.
        AFragT a_frag[WARP_TILE_M];
        BFragT b_frag[WARP_TILE_N];

        for (int wm = 0; wm < WARP_TILE_M; ++wm) {
          const int r0 = warp_m0 + wm * MMA_M;
          auto a_tile =
              Kokkos::subview(a_shared, Kokkos::pair<int, int>(r0, r0 + MMA_M),
                              Kokkos::pair<int, int>(kk, kk + MMA_K));
          Kokkos::Experimental::load_matrix_sync(a_frag[wm], a_tile);
        }
        for (int wn = 0; wn < WARP_TILE_N; ++wn) {
          const int c0 = warp_n0 + wn * MMA_N;
          auto b_tile =
              Kokkos::subview(b_shared, Kokkos::pair<int, int>(kk, kk + MMA_K),
                              Kokkos::pair<int, int>(c0, c0 + MMA_N));
          Kokkos::Experimental::load_matrix_sync(b_frag[wn], b_tile);
        }

        for (int wm = 0; wm < WARP_TILE_M; ++wm)
          for (int wn = 0; wn < WARP_TILE_N; ++wn)
            Kokkos::Experimental::mma_sync(c_frag[wm][wn], a_frag[wm],
                                           b_frag[wn], c_frag[wm][wn]);
      }

      member.team_barrier();
    }

    for (int wm = 0; wm < WARP_TILE_M; ++wm) {
      for (int wn = 0; wn < WARP_TILE_N; ++wn) {
        const int i = tile_m0 + warp_m0 + wm * MMA_M;
        const int j = tile_n0 + warp_n0 + wn * MMA_N;
        auto g_tile = Kokkos::subview(G, Kokkos::pair<int, int>(i, i + MMA_M),
                                      Kokkos::pair<int, int>(j, j + MMA_N));
        Kokkos::Experimental::store_matrix_sync(g_tile, c_frag[wm][wn]);
      }
    }
  }
};

// ---------------------------------------------------------------------------
// Host-side scheme parameters derived from the chosen moduli.
// ---------------------------------------------------------------------------
struct Scheme {
  int n_mods;
  int kA, kB;
  u128 M_prod;
  std::vector<int> p;
  std::vector<double> invp;
  std::vector<long long> y;  // (M/p)^-1 mod p
  std::vector<u128> Mdivp;   // M / p
};

static Scheme make_scheme(int k) {
  Scheme s;
  s.M_prod        = 1;
  long double l2M = 0.0L;
  s.n_mods        = 0;
  while (s.n_mods < kNumModsI8) {
    int p = kModsI8[s.n_mods];
    s.M_prod *= (unsigned)p;
    l2M += std::log2l((long double)p);
    ++s.n_mods;
    if (std::floorl(l2M - std::log2l((long double)k) - 1.01L) >= 102) break;
  }

  s.p.resize(s.n_mods);
  s.invp.resize(s.n_mods);
  s.y.resize(s.n_mods);
  s.Mdivp.resize(s.n_mods);
  for (int l = 0; l < s.n_mods; ++l) {
    int p         = kModsI8[l];
    s.p[l]        = p;
    s.invp[l]     = 1.0 / p;
    s.Mdivp[l]    = s.M_prod / (unsigned)p;
    long long rem = (long long)(s.Mdivp[l] % (unsigned)p);
    s.y[l]        = mod_inverse(rem, p);
  }

  int k_sum = (int)std::floorl(l2M - std::log2l((long double)k) - 1.01L);
  s.kA = s.kB = std::min(51, k_sum / 2);
  return s;
}

// Reusable device buffers for the emulated GEMM, sized once per problem.
struct Buffers {
  I64Matrix Aq, Bq;
  I8Matrix A8, B8;
  I32Matrix G;
  U128Matrix acc;
  IVec eA, eB;
};

// ---------------------------------------------------------------------------
// The full emulated DGEMM: C = A * B in FP64 accuracy via INT8 matmuls.
// Times reported by the caller cover this whole routine (scaling, quantization,
// all per-modulus INT8 GEMMs, and CRT reconstruction) so they are comparable to
// a single FP64 GEMM call.
// ---------------------------------------------------------------------------
template <class AFragT, class BFragT, class CFragT>
static void ozaki2_dgemm(int m, int n, int k, DMatrix A, DMatrix B, DMatrix C,
                         const Scheme& scheme, Buffers buf) {
  const int kA = scheme.kA, kB = scheme.kB;
  I64Matrix Aq = buf.Aq, Bq = buf.Bq;
  I8Matrix A8 = buf.A8, B8 = buf.B8;
  I32Matrix G    = buf.G;
  U128Matrix acc = buf.acc;
  IVec eA = buf.eA, eB = buf.eB;

  // Per-row exponent of A and per-column exponent of B:
  // e = ilogb(max |.|) + 1.
  Kokkos::parallel_for(
      "row_exp", Range1D(0, m), KOKKOS_LAMBDA(const int i) {
        double mx = 0.0;
        for (int h = 0; h < k; ++h)
          mx = Kokkos::fmax(mx, Kokkos::fabs(A(i, h)));
        eA(i) = mx > 0.0 ? ilogb(mx) + 1 : 0;
      });
  Kokkos::parallel_for(
      "col_exp", Range1D(0, n), KOKKOS_LAMBDA(const int j) {
        double mx = 0.0;
        for (int h = 0; h < k; ++h)
          mx = Kokkos::fmax(mx, Kokkos::fabs(B(h, j)));
        eB(j) = mx > 0.0 ? ilogb(mx) + 1 : 0;
      });

  // Quantize to integer matrices with |entries| <= 2^kA / 2^kB.
  Kokkos::parallel_for(
      "quantize_A", Range2D({0, 0}, {m, k}),
      KOKKOS_LAMBDA(const int i, const int h) {
        Aq(i, h) = (long long)llrint(scalbn(A(i, h), kA - eA(i)));
      });
  Kokkos::parallel_for(
      "quantize_B", Range2D({0, 0}, {k, n}),
      KOKKOS_LAMBDA(const int h, const int j) {
        Bq(h, j) = (long long)llrint(scalbn(B(h, j), kB - eB(j)));
      });

  Kokkos::deep_copy(acc, u128(0));

  const int team_size   = (BM / WARP_M) * (BN / WARP_N);
  const int league_size = (m / BM) * (n / BN);
  const int scratch_size =
      ScratchI8::shmem_size(BM, BK) + ScratchI8::shmem_size(BK, BN);
  TeamPolicy gemm_policy(league_size, team_size, WARP_SIZE);
  gemm_policy = gemm_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size));

  const u128 M_prod = scheme.M_prod;

  // For each modulus: form INT8 residues, run the exact INT8 GEMM, and
  // fold the result into the CRT accumulator (kept reduced mod M).
  for (int l = 0; l < scheme.n_mods; ++l) {
    const int p       = scheme.p[l];
    const double invp = scheme.invp[l];
    const long long y = scheme.y[l];
    const u128 Mdivp  = scheme.Mdivp[l];

    Kokkos::parallel_for(
        "residue_A", Range2D({0, 0}, {m, k}),
        KOKKOS_LAMBDA(const int i, const int h) {
          A8(i, h) = (std::int8_t)centered_mod(Aq(i, h), p, invp);
        });
    Kokkos::parallel_for(
        "residue_B", Range2D({0, 0}, {k, n}),
        KOKKOS_LAMBDA(const int h, const int j) {
          B8(h, j) = (std::int8_t)centered_mod(Bq(h, j), p, invp);
        });

    Int8TiledGemm<AFragT, BFragT, CFragT> gemm{A8, B8, G, n, k};
    Kokkos::parallel_for("ozaki2_int8_gemm", gemm_policy, gemm);

    // acc = (acc + (M/p) * ((G * y) mod p)) mod M. Each added term is < M and
    // acc stays < M, so a single conditional subtract restores the invariant.
    Kokkos::parallel_for(
        "crt_accumulate", Range2D({0, 0}, {m, n}),
        KOKKOS_LAMBDA(const int i, const int j) {
          int c = centered_mod((long long)G(i, j), p, invp);
          if (c < 0) c += p;
          long long t = ((long long)c * y) % p;
          if (t < 0) t += p;
          u128 cur = acc(i, j) + Mdivp * (u128)(unsigned long long)t;
          if (cur >= M_prod) cur -= M_prod;
          acc(i, j) = cur;
        });
  }

  // 4. CRT finalize: pick the centered representative of C' and unscale.
  Kokkos::parallel_for(
      "finalize", Range2D({0, 0}, {m, n}),
      KOKKOS_LAMBDA(const int i, const int j) {
        u128 x   = acc(i, j);  // already reduced into [0, M)
        bool neg = false;
        if (x > M_prod / 2) {
          x   = M_prod - x;
          neg = true;
        }
        double v = (double)(unsigned long long)(x >> 64) * 0x1p64 +
                   (double)(unsigned long long)x;
        if (neg) v = -v;
        C(i, j) = scalbn(v, eA(i) + eB(j) - kA - kB);
      });
}

// ---------------------------------------------------------------------------
// Host data generation and the FP64 reference.
// ---------------------------------------------------------------------------
static void fill_random(DMatrix mat, RandPool pool, double spread) {
  Kokkos::parallel_for(
      "fill_random", Range2D({0, 0}, {mat.extent(0), mat.extent(1)}),
      KOKKOS_LAMBDA(const int i, const int j) {
        auto gen       = pool.get_state();
        const double u = gen.drand() * 2.0 - 1.0;  // value in [-1, 1)
        const double e = gen.drand() * 2.0 - 1.0;  // exponent spread in [-1, 1)
        mat(i, j)      = u * Kokkos::exp(spread * e);
        pool.free_state(gen);
      });
}

static void reference_matmul(int m, int n, int k, DMatrix A, DMatrix B,
                             DMatrix C) {
  Kokkos::parallel_for(
      "reference_matmul", Range2D({0, 0}, {m, n}),
      KOKKOS_LAMBDA(const int i, const int j) {
        double sum = 0.0;
        for (int kk = 0; kk < k; ++kk) sum += A(i, kk) * B(kk, j);
        C(i, j) = sum;
      });
}

static double max_componentwise_error(int m, int n, DMatrix C, DMatrix Cref,
                                      DMatrix denom) {
  double err = 0.0;
  Kokkos::parallel_reduce(
      "max_error", Range2D({0, 0}, {m, n}),
      KOKKOS_LAMBDA(const int i, const int j, double& err_l) {
        const double d = denom(i, j) > 0.0 ? denom(i, j) : 1.0;
        const double e = Kokkos::fabs(C(i, j) - Cref(i, j)) / d;
        if (e > err_l) err_l = e;
      },
      Kokkos::Max<double>(err));
  return err;
}

bool run_example(int m, int n, int k) {
  static_assert(BM % WARP_M == 0 && BN % WARP_N == 0 && BK % MMA_K == 0,
                "Block tile must be divisible by the warp tile");
  static_assert((BM / WARP_M) * (BN / WARP_N) * WARP_SIZE <= 1024,
                "team_size * vector_length must be <= 1024");

  if (m % BM || n % BN || k % BK) {
    std::fprintf(stderr,
                 "ozaki2_dgemm: m,n,k (%d,%d,%d) must be multiples of "
                 "(%d,%d,%d)\n",
                 m, n, k, BM, BN, BK);
    return false;
  }
  if (k > 131072) {
    std::fprintf(stderr,
                 "ozaki2_dgemm: k > 131072 breaks INT32 accumulation\n");
    return false;
  }
  const Scheme scheme = make_scheme(k);

  using I8FragDType =
      Kokkos::Experimental::FragmentDType<ExecSpace,
                                          Kokkos::Experimental::Int8>::type;
  using I32FragDType =
      Kokkos::Experimental::FragmentDType<ExecSpace,
                                          Kokkos::Experimental::Int32>::type;
  using MMAShape = Kokkos::Experimental::mma_shape<MMA_M, MMA_N, MMA_K>;

  using AFragT = Kokkos::Experimental::fragment<
      I8FragDType, Kokkos::Experimental::matrix_a_extents<MMA_M, MMA_K>,
      Kokkos::layout_left,
      Kokkos::Experimental::mma_policy<MMAShape,
                                       Kokkos::Experimental::matrix_a>>;
  using BFragT = Kokkos::Experimental::fragment<
      I8FragDType, Kokkos::Experimental::matrix_b_extents<MMA_K, MMA_N>,
      Kokkos::layout_left,
      Kokkos::Experimental::mma_policy<MMAShape,
                                       Kokkos::Experimental::matrix_b>>;
  using CFragT = Kokkos::Experimental::fragment<
      I32FragDType, Kokkos::Experimental::accumulator_extents<MMA_M, MMA_N>,
      Kokkos::layout_right,
      Kokkos::Experimental::mma_policy<MMAShape,
                                       Kokkos::Experimental::accumulator>>;

  DMatrix A("A", m, k), B("B", k, n), C("C", m, n);
  DMatrix C_ref("C_ref", m, n), denom("denom", m, n);
  DMatrix absA("absA", m, k), absB("absB", k, n);

  Buffers buf{I64Matrix("Aq", m, k), I64Matrix("Bq", k, n),
              I8Matrix("A8", m, k),  I8Matrix("B8", k, n),
              I32Matrix("G", m, n),  U128Matrix("acc", m, n),
              IVec("eA", m),         IVec("eB", n)};

  RandPool pool(12345);
  fill_random(A, pool, 3.0);
  fill_random(B, pool, 3.0);
  Kokkos::fence();

  // Correctness: emulated GEMM vs FP64 reference, scaled by (|A| |B|)_ij.
  ozaki2_dgemm<AFragT, BFragT, CFragT>(m, n, k, A, B, C, scheme, buf);
  reference_matmul(m, n, k, A, B, C_ref);
  Kokkos::parallel_for(
      "abs_A", Range2D({0, 0}, {m, k}),
      KOKKOS_LAMBDA(const int i, const int h) {
        absA(i, h) = Kokkos::fabs(A(i, h));
      });
  Kokkos::parallel_for(
      "abs_B", Range2D({0, 0}, {k, n}),
      KOKKOS_LAMBDA(const int h, const int j) {
        absB(h, j) = Kokkos::fabs(B(h, j));
      });
  reference_matmul(m, n, k, absA, absB, denom);
  Kokkos::fence();

  const double err     = max_componentwise_error(m, n, C, C_ref, denom);
  constexpr double tol = 1e-13;

  // Timing. The flop count is the FP64-equivalent work (2*m*n*k); the emulated
  // path actually performs n_mods times that many INT8 MACs.
  const double gflop = 2.0 * m * n * (double)k * 1e-9;
  const int iters    = 5;

  Kokkos::fence();
  Kokkos::Timer timer;
  for (int it = 0; it < iters; ++it)
    ozaki2_dgemm<AFragT, BFragT, CFragT>(m, n, k, A, B, C, scheme, buf);
  Kokkos::fence();
  const double oz_ms = timer.seconds() * 1e3 / iters;

  std::printf(
      "Kokkos SIMD MMA Ozaki-II INT8 DGEMM\n"
      "  problem        : M,N,K = (%d, %d, %d)\n"
      "  scheme         : moduli = %d, kA = kB = %d, INT8 GEMMs/call = %d\n"
      "  accuracy       : max componentwise err vs FP64 = %.3e "
      "(tol %.1e, FP64 u = %.3e)\n",
      m, n, k, scheme.n_mods, scheme.kA, scheme.n_mods, err, tol, 0x1p-53);
  std::printf(
      "  Ozaki-II INT8  : %8.3f ms   %7.2f TFLOP/s (FP64-equivalent)\n"
      "                   underlying INT8 throughput: %7.2f TOP/s\n",
      oz_ms, gflop / oz_ms, gflop * scheme.n_mods / oz_ms);

  return err < tol;
}

int main(int argc, char* argv[]) {
  // usage: ozaki2_dgemm [SIZE | M N K]   (defaults to 2048^3)
  int m = 2048, n = 2048, k = 2048;
  if (argc == 2) {
    m = n = k = std::atoi(argv[1]);
  } else if (argc >= 4) {
    m = std::atoi(argv[1]);
    n = std::atoi(argv[2]);
    k = std::atoi(argv[3]);
  }

  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    const bool success = run_example(m, n, k);
    std::printf("Kokkos SIMD MMA Ozaki-II INT8 DGEMM: %s\n",
                success ? "PASSED" : "FAILED");
    rc = success ? 0 : 1;
  }
  Kokkos::finalize();
  return rc;
}
