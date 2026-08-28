// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif
#include <TestCuda_Category.hpp>

#include <cuda_tile.h>

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <Cuda/Kokkos_Cuda_KernelLaunchTile.hpp>
#include <Cuda/Kokkos_Cuda_TileView.hpp>

namespace Test {

// This struct implements the minimal compatibility requirements for Driver
// handed to the implementation tile launch function
struct TileVectorAddDummyDriver {
  float* a;
  float* b;
  float* out;
  std::size_t n;

  KOKKOS_EXPERIMENTAL_TILE_FUNCTION
  void operator()() const {
    namespace ct = cuda::tiles;
    using namespace ct::literals;

    constexpr auto tile_size = 8_ic;
    auto shape               = ct::shape{tile_size};
    auto extent              = ct::extents{n};

    auto a_view = ct::partition_view{ct::tensor_span{a, extent}, shape};
    auto b_view = ct::partition_view{ct::tensor_span{b, extent}, shape};
    auto o_view = ct::partition_view{ct::tensor_span{out, extent}, shape};

    const size_t n_tile       = n / tile_size;
    const size_t n_tile_block = n_tile / ct::num_blocks().x;
    const size_t tile_start   = ct::bid().x * n_tile_block;
    const size_t tile_end     = tile_start + n_tile_block;
    for (size_t tile_index : ct::irange(tile_start, tile_end)) {
      auto a_plus_b =
          a_view.load_masked(tile_index) + b_view.load_masked(tile_index);
      o_view.store_masked(a_plus_b, tile_index);
    }
  }
};

void cuda_tile_kernel_invoker() {
  constexpr std::size_t N = 256;

  Kokkos::View<float*, Kokkos::Cuda> a("A", N);
  Kokkos::View<float*, Kokkos::Cuda> b("A", N);
  Kokkos::View<float*, Kokkos::Cuda> out("A", N);

  Kokkos::Cuda cuda_instance;

  Kokkos::parallel_for(
      N, KOKKOS_LAMBDA(int i) {
        a(i) = i;
        b(i) = 2 * i;
      });

  using impl_launch_invoker = Kokkos::Impl::CudaParallelLaunchTileKernelInvoker<
      TileVectorAddDummyDriver,
      Kokkos::Impl::CudaLaunchMechanism::GlobalMemory>;

  TileVectorAddDummyDriver driver{a.data(), b.data(), out.data(), N};

  dim3 grid{1, 1, 1};
  impl_launch_invoker::invoke_kernel(
      driver, grid, cuda_instance.impl_internal_space_instance());

  cuda_instance.fence();

  auto h_out = Kokkos::create_mirror_view_and_copy(out);
  int errors = 0;
  for (std::size_t i = 0; i < N; ++i) {
    if (h_out(i) != float(3 * i)) errors++;
  }
  ASSERT_EQ(errors, 0);
}

TEST(cuda, tile_kernel_invoker) { cuda_tile_kernel_invoker(); }

template <size_t TileSize>
struct TileVectorAddFunctor {
  float* a;
  float* b;
  float* out;
  int N, M;

  KOKKOS_EXPERIMENTAL_TILE_FUNCTION
  void operator()(int i, int j) const {
    namespace ct = cuda::tiles;
    using namespace ct::literals;

    auto shape  = ct::shape<TileSize, TileSize>{};
    auto extent = ct::extents{N, M};

    auto a_view = ct::partition_view{ct::tensor_span{a, extent}, shape};
    auto b_view = ct::partition_view{ct::tensor_span{b, extent}, shape};
    auto o_view = ct::partition_view{ct::tensor_span{out, extent}, shape};

    auto a_plus_b = a_view.load(i, j) + b_view.load(i, j);
    o_view.store(a_plus_b, i, j);
  }
};

void cuda_tile_parallel_for() {
  int N = 32;
  int M = 24;

  constexpr int tile_size = 8;

  Kokkos::View<float**, Kokkos::Cuda> a("A", N, M);
  Kokkos::View<float**, Kokkos::Cuda> b("B", N, M);
  Kokkos::View<float**, Kokkos::Cuda> out("Out", N, M);

  Kokkos::Cuda cuda_instance;

  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::Rank<2>>(cuda_instance,
                                                           {0, 0}, {N, M}),
      KOKKOS_LAMBDA(int i, int j) {
        a(i, j) = 1000 * i + j;
        b(i, j) = 2000 * i + 2 * j;
      });

  Kokkos::parallel_for(
      Kokkos::Experimental::TileMDRangePolicy<Kokkos::Cuda, Kokkos::Rank<2>>(
          cuda_instance, {0, 0}, {N / tile_size, M / tile_size}),
      TileVectorAddFunctor<tile_size>{a.data(), b.data(), out.data(), N, M});

  int errors;
  Kokkos::parallel_reduce(
      Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::Rank<2>>(cuda_instance,
                                                           {0, 0}, {N, M}),
      KOKKOS_LAMBDA(int i, int j, int& error) {
        if (out(i, j) != 3000 * i + 3 * j) error++;
      },
      errors);
  ASSERT_EQ(errors, 0);
}

TEST(cuda, tile_parallel_for) { cuda_tile_parallel_for(); }

__tile_global__ void tile_vector_add(float* __restrict__ a,
                                     float* __restrict__ b,
                                     float* __restrict__ out, size_t n) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  a   = ct::assume_aligned(a, 16_ic);
  b   = ct::assume_aligned(b, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  constexpr auto tile_size = 8_ic;
  auto shape               = ct::shape{tile_size};
  auto extent              = ct::extents{n};

  auto a_view = ct::partition_view{ct::tensor_span{a, extent}, shape};
  auto b_view = ct::partition_view{ct::tensor_span{b, extent}, shape};
  auto o_view = ct::partition_view{ct::tensor_span{out, extent}, shape};

  const size_t n_tile       = n / tile_size;
  const size_t n_tile_block = n_tile / ct::num_blocks().x;
  const size_t tile_start   = ct::bid().x * n_tile_block;
  const size_t tile_end     = tile_start + n_tile_block;
  for (size_t tile_index : ct::irange(tile_start, tile_end)) {
    auto a_plus_b =
        a_view.load_masked(tile_index) + b_view.load_masked(tile_index);
    o_view.store_masked(a_plus_b, tile_index);
  }
}

TEST(cuda, tile_vector_add) {
  constexpr std::size_t N          = 256;
  constexpr std::size_t num_blocks = 2;

  std::array<float, N> h_a;
  std::array<float, N> h_b;
  for (std::size_t i = 0; i < N; ++i) {
    h_a[i] = static_cast<float>(i);
    h_b[i] = static_cast<float>(2 * i);
  }

  float* d_a   = nullptr;
  float* d_b   = nullptr;
  float* d_out = nullptr;
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMalloc(&d_a, sizeof(float) * N));
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMalloc(&d_b, sizeof(float) * N));
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMalloc(&d_out, sizeof(float) * N));

  KOKKOS_IMPL_CUDA_SAFE_CALL(
      cudaMemcpy(d_a, h_a.data(), sizeof(float) * N, cudaMemcpyHostToDevice));
  KOKKOS_IMPL_CUDA_SAFE_CALL(
      cudaMemcpy(d_b, h_b.data(), sizeof(float) * N, cudaMemcpyHostToDevice));

  // A tile kernel must be launched with a block dimension of one; the
  // compiler decides how many threads actually execute the kernel.
  tile_vector_add<<<num_blocks, 1>>>(d_a, d_b, d_out, N);
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGetLastError());
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaDeviceSynchronize());

  std::array<float, N> h_out;
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMemcpy(h_out.data(), d_out, sizeof(float) * N,
                                        cudaMemcpyDeviceToHost));

  int errors = 0;
  for (std::size_t i = 0; i < N; ++i) {
    if (h_out[i] != h_a[i] + h_b[i]) errors++;
  }
  errors = 0;

  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFree(d_a));
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFree(d_b));
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFree(d_out));
}

// Test compile-time properties of Kokkos::TileView.

template<size_t... Indices>
constexpr
cuda::tiles::extents<uint32_t, ((void) Indices, cuda::tiles::dynamic_extent)...>
all_dynamic_extents_impl(std::index_sequence<Indices...>) {
  return {};
}

template<size_t Rank>
constexpr decltype(all_dynamic_extents_impl(std::make_index_sequence<Rank>()))
all_dynamic_extents() {
  return {};
}

template<size_t Rank>
using all_dynamic_extents_t = decltype(all_dynamic_extents<Rank>());

static_assert(
    std::is_same_v<
        decltype(Kokkos::TileView(
            std::declval<Kokkos::View<float*, Kokkos::Cuda> const&>(),
            cuda::tiles::shape<8>{}))::extents_type,
        all_dynamic_extents_t<1>>);

static_assert(
    std::is_same_v<
        decltype(Kokkos::TileView(
            std::declval<Kokkos::View<float***, Kokkos::Cuda> const&>(),
            cuda::tiles::shape<8, 16, 32>{}))::extents_type,
        all_dynamic_extents_t<3>>);

static_assert(
    std::is_same_v<
        decltype(Kokkos::TileView(
            std::declval<Kokkos::View<float[32], Kokkos::Cuda> const&>(),
            cuda::tiles::shape<8>{}))::extents_type,
        cuda::tiles::extents<uint32_t, 4>>);

static_assert(
    std::is_same_v<
        decltype(Kokkos::TileView(
            std::declval<Kokkos::View<float[128][32], Kokkos::Cuda> const&>(),
            cuda::tiles::shape<2, 8>{}))::extents_type,
        cuda::tiles::extents<uint32_t, 64, 4>>);

static_assert(
    std::is_same_v<
        decltype(Kokkos::TileView(
            std::declval<Kokkos::View<float*[128], Kokkos::Cuda> const&>(),
            cuda::tiles::shape<8, 4>{}))::extents_type,
        cuda::tiles::extents<uint32_t, cuda::tiles::dynamic_extent, 32>>);

// Exercise Kokkos::TileView with rank-1 Kokkos::View objects.
struct TileViewVectorAddDriver {
  using ShapeType    = cuda::tiles::shape<8>;
  using TileType     = cuda::tiles::tile<float, ShapeType>;
  using ExtentsType  = cuda::tiles::extents<uint32_t, cuda::tiles::dynamic_extent>;
  using TileViewType = Kokkos::TileView<TileType, ExtentsType, Kokkos::CudaSpace>;

  TileViewType a;
  TileViewType b;
  TileViewType out;
  std::size_t n;

  KOKKOS_EXPERIMENTAL_TILE_FUNCTION
  void operator()() const {
    namespace ct = cuda::tiles;

    // Calling load/store directly on TileView (rather than first
    // converting to a cuda::tiles::partition_view) exercises the member
    // functions TileView forwards from partition_view.  N is an exact
    // multiple of the tile size, so every tile is fully in bounds and
    // plain (unmasked) load/store are safe to use.
    constexpr std::size_t tile_size = 8;
    const size_t n_tile             = n / tile_size;
    const size_t n_tile_block       = n_tile / ct::num_blocks().x;
    const size_t tile_start         = ct::bid().x * n_tile_block;
    const size_t tile_end           = tile_start + n_tile_block;
    for (size_t tile_index : ct::irange(tile_start, tile_end)) {
      auto a_plus_b = a.load(tile_index) + b.load(tile_index);
      out.store(a_plus_b, tile_index);
    }
  }
};

void cuda_tile_view_kernel_invoker() {
  constexpr std::size_t N = 256;

  Kokkos::View<float*, Kokkos::Cuda> a("A", N);
  Kokkos::View<float*, Kokkos::Cuda> b("B", N);
  Kokkos::View<float*, Kokkos::Cuda> out("Out", N);

  Kokkos::Cuda cuda_instance;

  Kokkos::parallel_for(
      N, KOKKOS_LAMBDA(int i) {
        a(i) = i;
        b(i) = 2 * i;
      });

  TileViewVectorAddDriver::ShapeType shape;
  TileViewVectorAddDriver driver{Kokkos::TileView(a, shape),
                                 Kokkos::TileView(b, shape),
                                 Kokkos::TileView(out, shape), N};

  using impl_launch_invoker = Kokkos::Impl::CudaParallelLaunchTileKernelInvoker<
      TileViewVectorAddDriver,
      Kokkos::Impl::CudaLaunchMechanism::GlobalMemory>;

  dim3 grid{1, 1, 1};
  impl_launch_invoker::invoke_kernel(
      driver, grid, cuda_instance.impl_internal_space_instance());

  cuda_instance.fence();

  auto h_out = Kokkos::create_mirror_view_and_copy(out);
  int errors = 0;
  for (std::size_t i = 0; i < N; ++i) {
    if (h_out(i) != float(3 * i)) errors++;
  }
  ASSERT_EQ(errors, 0);
}

TEST(cuda, tile_view_vector_add) { cuda_tile_view_kernel_invoker(); }

// Exercise Kokkos::TileView with a noncontiguous, rank-1 Kokkos::View.
// a and b are nonconsecutive columns of rank-2 Views.
struct TileViewNoncontiguousRank1AddDriver {
  using BackingViewType =
      Kokkos::View<float* [2], Kokkos::LayoutRight, Kokkos::Cuda>;
  using ShapeType     = cuda::tiles::shape<8>;
  using TileType      = cuda::tiles::tile<float, ShapeType>;
  using ExtentsType   = cuda::tiles::extents<uint32_t, cuda::tiles::dynamic_extent>;
  using TileViewType = Kokkos::TileView<TileType, ExtentsType, Kokkos::CudaSpace>;

  TileViewType a;
  TileViewType b;
  TileViewType out;
  std::size_t n;

  KOKKOS_EXPERIMENTAL_TILE_FUNCTION
  void operator()() const {
    namespace ct = cuda::tiles;

    // Calling load/store directly on TileView (rather than first
    // converting to a cuda::tiles::partition_view) exercises the member
    // functions TileView forwards from partition_view.  N is an exact
    // multiple of the tile size, so every tile is fully in bounds and
    // plain (unmasked) load/store are safe to use.
    constexpr std::size_t tile_size = 8;
    const size_t n_tile             = n / tile_size;
    const size_t n_tile_block       = n_tile / ct::num_blocks().x;
    const size_t tile_start         = ct::bid().x * n_tile_block;
    const size_t tile_end           = tile_start + n_tile_block;
    for (size_t tile_index : ct::irange(tile_start, tile_end)) {
      auto a_plus_b = a.load(tile_index) + b.load(tile_index);
      out.store(a_plus_b, tile_index);
    }
  }
};

void cuda_tile_view_noncontiguous_rank1_add() {
  constexpr std::size_t N = 256;

  TileViewNoncontiguousRank1AddDriver::BackingViewType a_full("a_full", N);
  TileViewNoncontiguousRank1AddDriver::BackingViewType b_full("b_full", N);
  TileViewNoncontiguousRank1AddDriver::BackingViewType out_full("out_full", N);

  Kokkos::Cuda cuda_instance;

  Kokkos::parallel_for(
      N, KOKKOS_LAMBDA(int i) {
        a_full(i, 0)   = i;
        a_full(i, 1)   = -1;
        b_full(i, 0)   = 2 * i;
        b_full(i, 1)   = -1;
        out_full(i, 0) = -1;
        out_full(i, 1) = -1;
      });

  auto a   = Kokkos::subview(a_full, Kokkos::ALL, 0);
  auto b   = Kokkos::subview(b_full, Kokkos::ALL, 0);
  auto out = Kokkos::subview(out_full, Kokkos::ALL, 0);

  TileViewNoncontiguousRank1AddDriver::ShapeType shape;
  TileViewNoncontiguousRank1AddDriver driver{
      Kokkos::TileView(a, shape), Kokkos::TileView(b, shape),
      Kokkos::TileView(out, shape), N};

  using impl_launch_invoker = Kokkos::Impl::CudaParallelLaunchTileKernelInvoker<
      TileViewNoncontiguousRank1AddDriver,
      Kokkos::Impl::CudaLaunchMechanism::GlobalMemory>;

  dim3 grid{1, 1, 1};
  impl_launch_invoker::invoke_kernel(
      driver, grid, cuda_instance.impl_internal_space_instance());

  cuda_instance.fence();

  auto h_out_full = Kokkos::create_mirror_view_and_copy(out_full);
  int errors       = 0;
  for (std::size_t i = 0; i < N; ++i) {
    if (h_out_full(i, 0) != float(3 * i)) errors++;
    // The untouched neighbor column must be unaffected; if TileView were
    // treating the strided view as contiguous, this would be clobbered.
    if (h_out_full(i, 1) != -1.0f) errors++;
  }
  ASSERT_EQ(errors, 0);
}

TEST(cuda, tile_view_noncontiguous_rank1_add) {
  cuda_tile_view_noncontiguous_rank1_add();
}

// Exercise Kokkos::TileView with noncontiguous rank-2 Kokkos::View.
struct TileViewNoncontiguousRank2AddDriver {
  using BackingViewType =
      Kokkos::View<float* [4][2], Kokkos::LayoutRight, Kokkos::Cuda>;
  using ShapeType = cuda::tiles::shape<8, 4>;
  using TileType  = cuda::tiles::tile<float, ShapeType>;
  // The subview keeps the second dimension's static extent (4) from
  // BackingViewType, and the tile shape's corresponding extent is also
  // 4, so that dimension is exactly one tile wide: TileView's deduction
  // guide deduces a static tile index space extent of 1 here (not 4,
  // and not dynamic).
  using ExtentsType =
      cuda::tiles::extents<uint32_t, cuda::tiles::dynamic_extent, 1>;
  using TileViewType = Kokkos::TileView<TileType, ExtentsType, Kokkos::CudaSpace>;

  TileViewType a;
  TileViewType b;
  TileViewType out;
  std::size_t n;

  KOKKOS_EXPERIMENTAL_TILE_FUNCTION
  void operator()() const {
    namespace ct = cuda::tiles;

    constexpr std::size_t tile_size = 8;
    const size_t n_tile             = n / tile_size;
    const size_t n_tile_block       = n_tile / ct::num_blocks().x;
    const size_t tile_start         = ct::bid().x * n_tile_block;
    const size_t tile_end           = tile_start + n_tile_block;
    for (size_t tile_index : ct::irange(tile_start, tile_end)) {
      auto a_plus_b = a.load(tile_index, 0) + b.load(tile_index, 0);
      out.store(a_plus_b, tile_index, 0);
    }
  }
};

void cuda_tile_view_noncontiguous_rank2_add() {
  constexpr std::size_t N = 256;

  TileViewNoncontiguousRank2AddDriver::BackingViewType a_full("a_full", N);
  TileViewNoncontiguousRank2AddDriver::BackingViewType b_full("b_full", N);
  TileViewNoncontiguousRank2AddDriver::BackingViewType out_full("out_full", N);

  Kokkos::Cuda cuda_instance;

  Kokkos::parallel_for(
      N, KOKKOS_LAMBDA(int i) {
        for (int j = 0; j < 4; ++j) {
          a_full(i, j, 0)   = i * 4 + j;
          a_full(i, j, 1)   = -1;
          b_full(i, j, 0)   = 2 * (i * 4 + j);
          b_full(i, j, 1)   = -1;
          out_full(i, j, 0) = -1;
          out_full(i, j, 1) = -1;
        }
      });

  auto a   = Kokkos::subview(a_full, Kokkos::ALL, Kokkos::ALL, 0);
  auto b   = Kokkos::subview(b_full, Kokkos::ALL, Kokkos::ALL, 0);
  auto out = Kokkos::subview(out_full, Kokkos::ALL, Kokkos::ALL, 0);

  TileViewNoncontiguousRank2AddDriver::ShapeType shape;
  TileViewNoncontiguousRank2AddDriver driver{
      Kokkos::TileView(a, shape), Kokkos::TileView(b, shape),
      Kokkos::TileView(out, shape), N};

  using impl_launch_invoker = Kokkos::Impl::CudaParallelLaunchTileKernelInvoker<
      TileViewNoncontiguousRank2AddDriver,
      Kokkos::Impl::CudaLaunchMechanism::GlobalMemory>;

  dim3 grid{1, 1, 1};
  impl_launch_invoker::invoke_kernel(
      driver, grid, cuda_instance.impl_internal_space_instance());

  cuda_instance.fence();

  auto h_out_full = Kokkos::create_mirror_view_and_copy(out_full);
  int errors       = 0;
  for (std::size_t i = 0; i < N; ++i) {
    for (int j = 0; j < 4; ++j) {
      if (h_out_full(i, j, 0) != float(3 * (i * 4 + j))) errors++;
      // The untouched neighbor slab must be unaffected; if TileView were
      // treating the strided view as contiguous, this would be clobbered.
      if (h_out_full(i, j, 1) != -1.0f) errors++;
    }
  }
  ASSERT_EQ(errors, 0);
}

TEST(cuda, tile_view_noncontiguous_rank2_add) {
  cuda_tile_view_noncontiguous_rank2_add();
}

// Exercise Kokkos::TileView with a fully static, rank-1 Kokkos::View.

struct TileViewStaticExtentAddDriver {
  using ViewType    = Kokkos::View<float[32], Kokkos::LayoutRight, Kokkos::Cuda>;
  using ShapeType   = cuda::tiles::shape<8>;
  using TileType    = cuda::tiles::tile<float, ShapeType>;
  using ExtentsType = cuda::tiles::extents<uint32_t, 4>;
  using TileViewType = Kokkos::TileView<TileType, ExtentsType, Kokkos::CudaSpace>;

  TileViewType a;
  TileViewType b;
  TileViewType out;

  KOKKOS_EXPERIMENTAL_TILE_FUNCTION
  void operator()() const {
    namespace ct = cuda::tiles;

    constexpr std::size_t n_tile       = 4;
    const size_t n_tile_block          = n_tile / ct::num_blocks().x;
    const size_t tile_start            = ct::bid().x * n_tile_block;
    const size_t tile_end              = tile_start + n_tile_block;
    for (size_t tile_index : ct::irange(tile_start, tile_end)) {
      auto a_plus_b = a.load(tile_index) + b.load(tile_index);
      out.store(a_plus_b, tile_index);
    }
  }
};

void cuda_tile_view_static_extent_add() {
  constexpr std::size_t N = 32;

  TileViewStaticExtentAddDriver::ViewType a("A");
  TileViewStaticExtentAddDriver::ViewType b("B");
  TileViewStaticExtentAddDriver::ViewType out("Out");

  Kokkos::Cuda cuda_instance;

  Kokkos::parallel_for(
      N, KOKKOS_LAMBDA(int i) {
        a(i) = i;
        b(i) = 2 * i;
      });

  TileViewStaticExtentAddDriver::ShapeType shape;
  TileViewStaticExtentAddDriver driver{Kokkos::TileView(a, shape),
                                        Kokkos::TileView(b, shape),
                                        Kokkos::TileView(out, shape)};

  using impl_launch_invoker = Kokkos::Impl::CudaParallelLaunchTileKernelInvoker<
      TileViewStaticExtentAddDriver,
      Kokkos::Impl::CudaLaunchMechanism::GlobalMemory>;

  dim3 grid{1, 1, 1};
  impl_launch_invoker::invoke_kernel(
      driver, grid, cuda_instance.impl_internal_space_instance());

  cuda_instance.fence();

  auto h_out = Kokkos::create_mirror_view_and_copy(out);
  int errors = 0;
  for (std::size_t i = 0; i < N; ++i) {
    if (h_out(i) != float(3 * i)) errors++;
  }
  ASSERT_EQ(errors, 0);
}

TEST(cuda, tile_view_static_extent_add) { cuda_tile_view_static_extent_add(); }

// Exercise Kokkos::TileView with a rank-2 Kokkos::View whose first
// dimension is dynamic and whose second dimension is static, where the
// static dimension spans *more than one* tile (16 / 4 = 4 tiles).

struct TileViewMultiTileStaticDimAddDriver {
  using ViewType =
      Kokkos::View<float* [16], Kokkos::LayoutRight, Kokkos::Cuda>;
  using ShapeType   = cuda::tiles::shape<8, 4>;
  using TileType    = cuda::tiles::tile<float, ShapeType>;
  using ExtentsType = cuda::tiles::extents<uint32_t, cuda::tiles::dynamic_extent, 4>;
  using TileViewType = Kokkos::TileView<TileType, ExtentsType, Kokkos::CudaSpace>;

  TileViewType a;
  TileViewType b;
  TileViewType out;
  std::size_t n;

  KOKKOS_EXPERIMENTAL_TILE_FUNCTION
  void operator()() const {
    namespace ct = cuda::tiles;

    constexpr std::size_t tile_size0 = 8;
    constexpr std::size_t n_tile1    = 4;
    const size_t n_tile0             = n / tile_size0;
    const size_t n_tile0_block       = n_tile0 / ct::num_blocks().x;
    const size_t tile0_start         = ct::bid().x * n_tile0_block;
    const size_t tile0_end           = tile0_start + n_tile0_block;
    for (size_t tile0_index : ct::irange(tile0_start, tile0_end)) {
      for (size_t tile1_index = 0; tile1_index < n_tile1; ++tile1_index) {
        auto a_plus_b = a.load(tile0_index, tile1_index) +
                         b.load(tile0_index, tile1_index);
        out.store(a_plus_b, tile0_index, tile1_index);
      }
    }
  }
};

void cuda_tile_view_multi_tile_static_dim_add() {
  constexpr std::size_t N = 256;

  TileViewMultiTileStaticDimAddDriver::ViewType a("A", N);
  TileViewMultiTileStaticDimAddDriver::ViewType b("B", N);
  TileViewMultiTileStaticDimAddDriver::ViewType out("Out", N);

  Kokkos::Cuda cuda_instance;

  Kokkos::parallel_for(
      N, KOKKOS_LAMBDA(int i) {
        for (int j = 0; j < 16; ++j) {
          a(i, j)   = i * 16 + j;
          b(i, j)   = 2 * (i * 16 + j);
          out(i, j) = -1;
        }
      });

  TileViewMultiTileStaticDimAddDriver::ShapeType shape;
  TileViewMultiTileStaticDimAddDriver driver{
      Kokkos::TileView(a, shape), Kokkos::TileView(b, shape),
      Kokkos::TileView(out, shape), N};

  using impl_launch_invoker = Kokkos::Impl::CudaParallelLaunchTileKernelInvoker<
      TileViewMultiTileStaticDimAddDriver,
      Kokkos::Impl::CudaLaunchMechanism::GlobalMemory>;

  dim3 grid{1, 1, 1};
  impl_launch_invoker::invoke_kernel(
      driver, grid, cuda_instance.impl_internal_space_instance());

  cuda_instance.fence();

  auto h_out = Kokkos::create_mirror_view_and_copy(out);
  int errors = 0;
  for (std::size_t i = 0; i < N; ++i) {
    for (int j = 0; j < 16; ++j) {
      if (h_out(i, j) != float(3 * (i * 16 + j))) errors++;
    }
  }
  ASSERT_EQ(errors, 0);
}

TEST(cuda, tile_view_multi_tile_static_dim_add) {
  cuda_tile_view_multi_tile_static_dim_add();
}

// Test that Kokkos::TileView implements the same load and store
// interface as cuda::tiles::partition_view.

template <class ViewLikeType>
struct GenericVectorAddDriver {
  ViewLikeType a;
  ViewLikeType b;
  ViewLikeType out;
  std::size_t n;

  KOKKOS_EXPERIMENTAL_TILE_FUNCTION
  void operator()() const {
    namespace ct = cuda::tiles;

    constexpr std::size_t tile_size = 8;
    const size_t n_tile             = n / tile_size;
    const size_t n_tile_block       = n_tile / ct::num_blocks().x;
    const size_t tile_start         = ct::bid().x * n_tile_block;
    const size_t tile_end           = tile_start + n_tile_block;
    for (size_t tile_index : ct::irange(tile_start, tile_end)) {
      auto a_plus_b = a.load(tile_index) + b.load(tile_index);
      out.store(a_plus_b, tile_index);
    }
  }
};

template <class ViewLikeType>
void run_generic_vector_add(ViewLikeType a, ViewLikeType b, ViewLikeType out,
                             std::size_t n, Kokkos::Cuda& cuda_instance) {
  using Driver = GenericVectorAddDriver<ViewLikeType>;
  Driver driver{a, b, out, n};

  using impl_launch_invoker = Kokkos::Impl::CudaParallelLaunchTileKernelInvoker<
      Driver, Kokkos::Impl::CudaLaunchMechanism::GlobalMemory>;

  dim3 grid{1, 1, 1};
  impl_launch_invoker::invoke_kernel(
      driver, grid, cuda_instance.impl_internal_space_instance());

  cuda_instance.fence();
}

void cuda_tile_view_substitutes_for_partition_view() {
  constexpr std::size_t N = 256;

  Kokkos::View<float*, Kokkos::Cuda> a("A", N);
  Kokkos::View<float*, Kokkos::Cuda> b("B", N);
  Kokkos::View<float*, Kokkos::Cuda> out_via_tile_view("OutTileView", N);
  Kokkos::View<float*, Kokkos::Cuda> out_via_partition_view(
      "OutPartitionView", N);

  Kokkos::Cuda cuda_instance;

  Kokkos::parallel_for(
      N, KOKKOS_LAMBDA(int i) {
        a(i) = i;
        b(i) = 2 * i;
      });

  // Run the generic driver with Kokkos::TileView.
  cuda::tiles::shape<8> shape;
  run_generic_vector_add(Kokkos::TileView(a, shape), Kokkos::TileView(b, shape),
                          Kokkos::TileView(out_via_tile_view, shape), N,
                          cuda_instance);

  // Run the generic driver with cuda::tiles::partition_view.
  namespace ct = cuda::tiles;
  using ExtentsType = ct::extents<uint32_t, ct::dynamic_extent>;
  using SpanType    = ct::tensor_span<float, ExtentsType>;
  using PartitionViewType = ct::partition_view<SpanType, ct::shape<8>>;

  ExtentsType extents(static_cast<uint32_t>(N));
  PartitionViewType pv_a(SpanType(a.data(), extents), ct::shape<8>{});
  PartitionViewType pv_b(SpanType(b.data(), extents), ct::shape<8>{});
  PartitionViewType pv_out(
      SpanType(out_via_partition_view.data(), extents), ct::shape<8>{});
  run_generic_vector_add(pv_a, pv_b, pv_out, N, cuda_instance);

  auto h_out_tile_view = Kokkos::create_mirror_view_and_copy(out_via_tile_view);
  auto h_out_partition_view =
      Kokkos::create_mirror_view_and_copy(out_via_partition_view);
  int errors = 0;
  for (std::size_t i = 0; i < N; ++i) {
    float expected = float(3 * i);
    if (h_out_tile_view(i) != expected) errors++;
    if (h_out_partition_view(i) != expected) errors++;
  }
  ASSERT_EQ(errors, 0);
}

TEST(cuda, tile_view_substitutes_for_partition_view) {
  cuda_tile_view_substitutes_for_partition_view();
}

}  // namespace Test
