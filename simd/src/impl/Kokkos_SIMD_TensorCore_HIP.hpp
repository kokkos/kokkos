// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_SIMD_TENSORCORE_IMPL_HIP_HPP
#define KOKKOS_SIMD_TENSORCORE_IMPL_HIP_HPP

#ifndef KOKKOS_ENABLE_HIP
#error "Kokkos_SIMD_TensorCore_HIP.hpp requires KOKKOS_ENABLE_HIP"
#endif

#include <rocwmma/rocwmma.hpp>

namespace Kokkos {
namespace Experimental {
namespace Impl {

template <PrecisionType P>
struct FragmentDTypeImpl<Kokkos::HIP, P> {
  using type = std::conditional_t<
      P == PrecisionType::Double, rocwmma::float64_t,
      std::conditional_t<
          P == PrecisionType::Float, rocwmma::float32_t,
          std::conditional_t<
              P == PrecisionType::Int8, rocwmma::int8_t,
              std::conditional_t<P == PrecisionType::Int32, rocwmma::int32_t,
                                 NotImplementedError<void>>>>>;
};

template <FragmentUse U, int MMA_M, int MMA_N, int MMA_K, class DType,
          class Layout>
struct NativeFragmentTImpl<Kokkos::HIP, U, MMA_M, MMA_N, MMA_K, DType, Layout> {
  static_assert(is_supported_operand_layout_v<Layout>,
                "HIP native fragments support only Kokkos::layout_left and "
                "Kokkos::layout_right operand layouts");

  using type = std::conditional_t<
      U == FragmentUse::Accumulator,
      rocwmma::fragment<rocwmma::accumulator, MMA_M, MMA_N, MMA_K, DType>,
      std::conditional_t<
          U == FragmentUse::MatrixA &&
              std::is_same_v<Layout, Kokkos::layout_right>,
          rocwmma::fragment<rocwmma::matrix_a, MMA_M, MMA_N, MMA_K, DType,
                            rocwmma::row_major>,
          std::conditional_t<
              U == FragmentUse::MatrixB &&
                  std::is_same_v<Layout, Kokkos::layout_right>,
              rocwmma::fragment<rocwmma::matrix_b, MMA_M, MMA_N, MMA_K, DType,
                                rocwmma::row_major>,
              std::conditional_t<
                  U == FragmentUse::MatrixA &&
                      std::is_same_v<Layout, Kokkos::layout_left>,
                  rocwmma::fragment<rocwmma::matrix_a, MMA_M, MMA_N, MMA_K,
                                    DType, rocwmma::col_major>,
                  std::conditional_t<
                      U == FragmentUse::MatrixB &&
                          std::is_same_v<Layout, Kokkos::layout_left>,
                      rocwmma::fragment<rocwmma::matrix_b, MMA_M, MMA_N, MMA_K,
                                        DType, rocwmma::col_major>,
                      NotImplementedError<void>>>>>>;
};

template <class FragmentT, class ValueT>
KOKKOS_INLINE_FUNCTION void load_matrix_sync(ExecutionSpaceTag<Kokkos::HIP>,
                                             FragmentT& destination,
                                             const ValueT* source,
                                             const int stride) {
  rocwmma::load_matrix_sync(destination, source, stride);
}

template <class FragmentT, class ValueT>
KOKKOS_INLINE_FUNCTION void fill_fragment(ExecutionSpaceTag<Kokkos::HIP>,
                                          FragmentT& fragment, ValueT value) {
  rocwmma::fill_fragment(fragment, value);
}

template <class FragmentT, class ValueT>
KOKKOS_INLINE_FUNCTION void store_matrix_sync(ExecutionSpaceTag<Kokkos::HIP>,
                                              ValueT* destination,
                                              FragmentT& source,
                                              const int stride,
                                              const bool row_major) {
  const rocwmma::layout_t mem_layout =
      row_major ? rocwmma::mem_row_major : rocwmma::mem_col_major;

  rocwmma::store_matrix_sync(destination, source, stride, mem_layout);
}

template <class DFragT, class AFragT, class BFragT, class CFragT>
KOKKOS_INLINE_FUNCTION void mma_sync(ExecutionSpaceTag<Kokkos::HIP>,
                                     DFragT& d_frag, AFragT& a_frag,
                                     BFragT& b_frag, CFragT& c_frag) {
  rocwmma::mma_sync(d_frag, a_frag, b_frag, c_frag);
}

}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#endif
