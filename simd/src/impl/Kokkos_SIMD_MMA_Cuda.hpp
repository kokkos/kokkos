// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_SIMD_MMA_IMPL_CUDA_HPP
#define KOKKOS_SIMD_MMA_IMPL_CUDA_HPP

#ifndef KOKKOS_ENABLE_CUDA
#error "Kokkos_SIMD_MMA_Cuda.hpp requires KOKKOS_ENABLE_CUDA"
#endif

#include <cuda.h>
#include <mma.h>

namespace Kokkos {
namespace Experimental {
namespace Impl {

template <PrecisionType P>
struct FragmentDTypeImpl<Kokkos::Cuda, P> {
  using type = std::conditional_t<
      P == PrecisionType::Double, double,
      std::conditional_t<
          P == PrecisionType::Float, float,
          std::conditional_t<
              P == PrecisionType::TF32, nvcuda::wmma::precision::tf32,
              std::conditional_t<
                  P == PrecisionType::Int8, signed char,
                  std::conditional_t<P == PrecisionType::Int32, int,
                                     NotImplementedError<void>>>>>>;
};

template <FragmentUse U, int MMA_M, int MMA_N, int MMA_K, class DType,
          class Layout>
struct NativeFragmentTImpl<Kokkos::Cuda, U, MMA_M, MMA_N, MMA_K, DType,
                           Layout> {
  static_assert(is_supported_operand_layout_v<Layout>,
                "CUDA native fragments support only Kokkos::layout_left and "
                "Kokkos::layout_right operand layouts");

  using type = std::conditional_t<
      U == FragmentUse::Accumulator,
      nvcuda::wmma::fragment<nvcuda::wmma::accumulator, MMA_M, MMA_N, MMA_K,
                             DType>,
      std::conditional_t<
          U == FragmentUse::MatrixA &&
              std::is_same_v<Layout, Kokkos::layout_right>,
          nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, MMA_M, MMA_N, MMA_K,
                                 DType, nvcuda::wmma::row_major>,
          std::conditional_t<
              U == FragmentUse::MatrixB &&
                  std::is_same_v<Layout, Kokkos::layout_right>,
              nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, MMA_M, MMA_N,
                                     MMA_K, DType, nvcuda::wmma::row_major>,
              std::conditional_t<
                  U == FragmentUse::MatrixA &&
                      std::is_same_v<Layout, Kokkos::layout_left>,
                  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, MMA_M, MMA_N,
                                         MMA_K, DType, nvcuda::wmma::col_major>,
                  std::conditional_t<
                      U == FragmentUse::MatrixB &&
                          std::is_same_v<Layout, Kokkos::layout_left>,
                      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, MMA_M,
                                             MMA_N, MMA_K, DType,
                                             nvcuda::wmma::col_major>,
                      NotImplementedError<void>>>>>>;
};

template <class FragmentT, class ValueT>
KOKKOS_INLINE_FUNCTION void load_matrix_sync(ExecutionSpaceTag<Kokkos::Cuda>,
                                             FragmentT& destination,
                                             const ValueT* source,
                                             const int stride) {
  nvcuda::wmma::load_matrix_sync(destination, source, stride);
}

template <class FragmentT, class ValueT>
KOKKOS_INLINE_FUNCTION void fill_fragment(ExecutionSpaceTag<Kokkos::Cuda>,
                                          FragmentT& fragment, ValueT value) {
  nvcuda::wmma::fill_fragment(fragment, value);
}

template <class FragmentT, class ValueT>
KOKKOS_INLINE_FUNCTION void store_matrix_sync(ExecutionSpaceTag<Kokkos::Cuda>,
                                              ValueT* destination,
                                              FragmentT& source,
                                              const int stride,
                                              const bool row_major) {
  const nvcuda::wmma::layout_t mem_layout =
      row_major ? nvcuda::wmma::mem_row_major : nvcuda::wmma::mem_col_major;

  nvcuda::wmma::store_matrix_sync(destination, source, stride, mem_layout);
}

template <class DFragT, class AFragT, class BFragT, class CFragT>
KOKKOS_INLINE_FUNCTION void mma_sync(ExecutionSpaceTag<Kokkos::Cuda>,
                                     DFragT& d_frag, AFragT& a_frag,
                                     BFragT& b_frag, CFragT& c_frag) {
  nvcuda::wmma::mma_sync(d_frag, a_frag, b_frag, c_frag);
}

}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#endif
