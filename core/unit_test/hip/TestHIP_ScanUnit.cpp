// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif
#include <TestHIP_Category.hpp>

struct DummyFunctor {
  using value_type = int;
  void operator()(const int, value_type&, bool) const {}
};

struct DummyScalarReductionFunctor {
  using value_type   = int;
  using pointer_type = int*;

  KOKKOS_INLINE_FUNCTION
  void join(value_type* dst, value_type const* src) const { *dst += *src; }
};

template <int N>
__global__ void start_intra_block_scan()
    __attribute__((amdgpu_flat_work_group_size(1, 1024))) {
  __shared__ DummyFunctor::value_type values[N];
  const int i = threadIdx.y;
  values[i]   = i + 1;
  __syncthreads();

  DummyFunctor f;
  typename Kokkos::Impl::FunctorAnalysis<
      Kokkos::Impl::FunctorPatternInterface::SCAN,
      Kokkos::RangePolicy<Kokkos::HIP>, DummyFunctor,
      DummyFunctor::value_type>::Reducer reducer(f);
  Kokkos::Impl::hip_intra_block_reduce_scan<true>(reducer, values);

  __syncthreads();
  if (values[i] != ((i + 2) * (i + 1)) / 2) {
    printf("Value for %d should be %d but is %d\n", i, ((i + 2) * (i + 1)) / 2,
           values[i]);
    Kokkos::abort("Test for intra_block_reduce_scan failed!");
  }
}

template <int N>
void test_intra_block_scan() {
  dim3 grid(1, 1, 1);
  dim3 block(1, N, 1);
  start_intra_block_scan<N><<<grid, block, 0, nullptr>>>();
}

template <int N>
__global__ void start_scalar_intra_block_reduction_test(int* out)
    __attribute__((amdgpu_flat_work_group_size(1, 1024))) {
  constexpr int warp_size = Kokkos::Impl::HIPTraits::WarpSize;
  static_assert((N % warp_size) == 0);
  static_assert((N / warp_size) != warp_size);

  __shared__ int values[N];
  const int i = threadIdx.y * blockDim.x + threadIdx.x;

  values[i] = 1;
  __syncthreads();

  DummyScalarReductionFunctor f;
  int result = 0;
  Kokkos::Impl::HIPReductionsFunctor<DummyScalarReductionFunctor, false>::
      scalar_intra_block_reduction(f, values[i], false, &result, N / warp_size,
                                   values);
  __syncthreads();

  if (i == 0) out[0] = result;
}

template <int N>
void test_scalar_intra_block_reduction_partial_mask() {
  Kokkos::View<int, TEST_EXECSPACE> out("out");
  dim3 grid(1, 1, 1);
  dim3 block(1, N, 1);
  start_scalar_intra_block_reduction_test<N>
      <<<grid, block, 0, nullptr>>>(out.data());
  Kokkos::fence();
  auto out_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
  ASSERT_EQ(out_h(), N);
}

TEST(TEST_CATEGORY, scan_unit) {
  if (std::is_same_v<TEST_EXECSPACE,
                     typename Kokkos::HIPSpace::execution_space>) {
    test_intra_block_scan<1>();
    test_intra_block_scan<2>();
    test_intra_block_scan<4>();
    test_intra_block_scan<8>();
    test_intra_block_scan<16>();
    test_intra_block_scan<32>();
    test_intra_block_scan<64>();
    test_intra_block_scan<128>();
    test_intra_block_scan<256>();
    test_intra_block_scan<512>();
    test_intra_block_scan<1024>();
  }
}

TEST(TEST_CATEGORY, reduce_scan_scalar_intra_warp_partial_mask) {
  if (std::is_same_v<TEST_EXECSPACE,
                     typename Kokkos::HIPSpace::execution_space>) {
    // N > warp_size forces the second warp-level reduction pass to use
    // width=N/warp_size, i.e. width != warp_size.
    test_scalar_intra_block_reduction_partial_mask<128>();
    test_scalar_intra_block_reduction_partial_mask<256>();
  }
}
