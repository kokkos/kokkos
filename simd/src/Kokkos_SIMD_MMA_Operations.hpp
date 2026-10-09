// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_SIMD_MMA_OPERATIONS_HPP
#define KOKKOS_SIMD_MMA_OPERATIONS_HPP

#include <Kokkos_SIMD_MMA_Fragment.hpp>

namespace Kokkos {
namespace Experimental {

namespace Impl {

template <class FragmentT>
KOKKOS_INLINE_FUNCTION typename FragmentT::data_handle_type& native_fragment(
    FragmentT& fragment) {
  return fragment.data_handle();
}

template <class MatrixT, class = void>
struct matrix_rank : std::integral_constant<int, -1> {};

template <class MatrixT>
struct matrix_rank<MatrixT, std::void_t<decltype(MatrixT::rank)>>
    : std::integral_constant<int, static_cast<int>(MatrixT::rank)> {};

template <class MatrixT>
inline constexpr int matrix_rank_v = matrix_rank<MatrixT>::value;

// MMA operations target tiny matrices for hardware-accelerated matrix
// multiplication. Thus, it is implicitly assumed that the inputs/output of MMA
// operations are matrices. All tensors with rank > 2 must be matricized using
// alternate Views/Subviews before using MMA operations.
template <class MatrixT>
KOKKOS_INLINE_FUNCTION void check_rank2_matrix_tile(const MatrixT&) {
  static_assert(
      matrix_rank_v<MatrixT> == 2,
      "Kokkos SIMD MMA matrix operations expect a rank-2 Kokkos "
      "view/subview representing a matricized MMA tile");
}

template <class FragmentT>
KOKKOS_INLINE_FUNCTION constexpr int fragment_extent_0() {
  if constexpr (FragmentT::use == FragmentUse::MatrixA) {
    return FragmentT::mma_m;
  } else if constexpr (FragmentT::use == FragmentUse::MatrixB) {
    return FragmentT::mma_k;
  } else {
    return FragmentT::mma_m;
  }
}

template <class FragmentT>
KOKKOS_INLINE_FUNCTION constexpr int fragment_extent_1() {
  if constexpr (FragmentT::use == FragmentUse::MatrixA) {
    return FragmentT::mma_k;
  } else if constexpr (FragmentT::use == FragmentUse::MatrixB) {
    return FragmentT::mma_n;
  } else {
    return FragmentT::mma_n;
  }
}

// Enforces that Views/Subviews passed to load_matrix_sync have extents that
// match the fragment extents
template <class FragmentT, class MatrixT>
KOKKOS_INLINE_FUNCTION void check_load_tile_extents(const MatrixT& tile) {
  if (tile.extent(0) != fragment_extent_0<FragmentT>() ||
      tile.extent(1) != fragment_extent_1<FragmentT>()) {
    Kokkos::abort(
        "Kokkos::Experimental::load_matrix_sync source tile extents do not "
        "match the "
        "destination fragment extents");
  }
}

// See check_load_tile_extents(...)
template <class FragmentT, class MatrixT>
KOKKOS_INLINE_FUNCTION void check_store_tile_extents(const MatrixT& tile) {
  static_assert(FragmentT::use == FragmentUse::Accumulator,
                "Kokkos::Experimental::store_matrix_sync expects an "
                "accumulator fragment");

  if (tile.extent(0) != fragment_extent_0<FragmentT>() ||
      tile.extent(1) != fragment_extent_1<FragmentT>()) {
    Kokkos::abort(
        "Kokkos::Experimental::store_matrix_sync destination tile extents do "
        "not match "
        "the source accumulator fragment extents");
  }
}

template <class Layout, class MatrixT>
KOKKOS_INLINE_FUNCTION int load_leading_dimension(const MatrixT& matrix) {
  if constexpr (std::is_same_v<Layout, Kokkos::layout_right> ||
                std::is_same_v<Layout, Kokkos::LayoutRight>) {
    if (matrix.stride(1) != 1) {
      Kokkos::abort(
          "Kokkos::Experimental::load_matrix_sync requires a compact row-major "
          "source "
          "tile for Kokkos::layout_right operands");
    }
    return static_cast<int>(matrix.stride(0));
  } else if constexpr (std::is_same_v<Layout, Kokkos::layout_left> ||
                       std::is_same_v<Layout, Kokkos::LayoutLeft>) {
    if (matrix.stride(0) != 1) {
      Kokkos::abort(
          "Kokkos::Experimental::load_matrix_sync requires a compact "
          "column-major source "
          "tile for Kokkos::layout_left operands");
    }
    return static_cast<int>(matrix.stride(1));
  } else {
    NotImplementedError<Layout>();
  }
}

struct StoreLayoutAndLeadingDimension {
  int stride;
  bool row_major;
};

template <class MatrixT>
KOKKOS_INLINE_FUNCTION StoreLayoutAndLeadingDimension
store_layout_and_leading_dimension(const MatrixT& matrix) {
  if (matrix.stride(1) == 1) {
    return {static_cast<int>(matrix.stride(0)), true};
  }

  if (matrix.stride(0) == 1) {
    return {static_cast<int>(matrix.stride(1)), false};
  }

  Kokkos::abort(
      "Kokkos::Experimental::store_matrix_sync requires a compact row-major or "
      "column-major destination tile");
  return {0, true};
}

template <class DFragT, class AFragT, class BFragT, class CFragT>
KOKKOS_INLINE_FUNCTION constexpr void check_mma_sync_contract() {
  static_assert(DFragT::use == FragmentUse::Accumulator,
                "Kokkos::Experimental::mma_sync expects d_frag to be an "
                "accumulator fragment");
  static_assert(AFragT::use == FragmentUse::MatrixA,
                "Kokkos::Experimental::mma_sync expects a_frag to be a "
                "matrix_a fragment");
  static_assert(BFragT::use == FragmentUse::MatrixB,
                "Kokkos::Experimental::mma_sync expects b_frag to be a "
                "matrix_b fragment");
  static_assert(CFragT::use == FragmentUse::Accumulator,
                "Kokkos::Experimental::mma_sync expects c_frag to be an "
                "accumulator fragment");

  using execution_space = typename DFragT::policy_type::execution_space;
  static_assert(
      std::is_same_v<execution_space,
                     typename AFragT::policy_type::execution_space> &&
          std::is_same_v<execution_space,
                         typename BFragT::policy_type::execution_space> &&
          std::is_same_v<execution_space,
                         typename CFragT::policy_type::execution_space>,
      "Kokkos::Experimental::mma_sync fragments must target the same execution "
      "space");

  static_assert(
      DFragT::mma_m == AFragT::mma_m && DFragT::mma_m == CFragT::mma_m,
      "Kokkos::Experimental::mma_sync fragments must agree on M");
  static_assert(
      DFragT::mma_n == BFragT::mma_n && DFragT::mma_n == CFragT::mma_n,
      "Kokkos::Experimental::mma_sync fragments must agree on N");
  static_assert(DFragT::mma_k == AFragT::mma_k &&
                    DFragT::mma_k == BFragT::mma_k &&
                    DFragT::mma_k == CFragT::mma_k,
                "Kokkos::Experimental::mma_sync fragments must agree on K");

  static_assert(std::is_same_v<typename AFragT::element_type,
                               typename BFragT::element_type>,
                "Kokkos::Experimental::mma_sync matrix_a and matrix_b "
                "fragments must use the same element type");
  static_assert(std::is_same_v<typename DFragT::element_type,
                               typename CFragT::element_type>,
                "Kokkos::Experimental::mma_sync d and c accumulator fragments "
                "must use the same element type");
}

}  // namespace Impl

/// Load a rank-2 Kokkos tile into a matrix_a or matrix_b fragment.
///
/// The source view/subview must have the logical extents required by the
/// fragment role: M x K for matrix_a and K x N for matrix_b. Its memory layout
/// must be compact in the direction selected by the fragment's mma_policy:
/// Kokkos::layout_left operands require stride(0) == 1, and
/// Kokkos::layout_right operands require stride(1) == 1.
///
/// This function performs public contract checks, computes the backend leading
/// dimension, and then dispatches to the execution-space implementation stored
/// in the fragment policy.
template <class FragmentT, class MatrixT>
KOKKOS_INLINE_FUNCTION void load_matrix_sync(FragmentT& destination,
                                             const MatrixT& source) {
  using operand_layout  = typename FragmentT::policy_type::operand_layout;
  using execution_space = typename FragmentT::policy_type::execution_space;

  Impl::check_rank2_matrix_tile(source);
  Impl::check_load_tile_extents<FragmentT>(source);
  const int stride = Impl::load_leading_dimension<operand_layout>(source);

  Impl::load_matrix_sync(Impl::ExecutionSpaceTag<execution_space>{},
                         Impl::native_fragment(destination), source.data(),
                         stride);
}

/// Fill a fragment with a scalar value.
///
/// This is primarily intended for accumulator fragments before the first
/// mma_sync call in a reduction over K. Backend implementations decide which
/// fragment roles support filling; unsupported roles fail during compilation or
/// through the backend contract.
template <class FragmentT, class ValueT>
KOKKOS_INLINE_FUNCTION void fill_fragment(FragmentT& fragment, ValueT value) {
  using execution_space = typename FragmentT::policy_type::execution_space;

  Impl::fill_fragment(Impl::ExecutionSpaceTag<execution_space>{},
                      Impl::native_fragment(fragment), value);
}

/// Store an accumulator fragment into a rank-2 Kokkos tile.
///
/// The destination view/subview must have M x N extents matching the source
/// accumulator fragment. The destination must be compact row-major or compact
/// column-major because backend stores take one leading dimension and a memory
/// layout flag.
template <class MatrixT, class FragmentT>
KOKKOS_INLINE_FUNCTION void store_matrix_sync(const MatrixT& destination,
                                              FragmentT& source) {
  using execution_space = typename FragmentT::policy_type::execution_space;

  Impl::check_rank2_matrix_tile(destination);
  Impl::check_store_tile_extents<FragmentT>(destination);

  const auto layout = Impl::store_layout_and_leading_dimension(destination);

  Impl::store_matrix_sync(Impl::ExecutionSpaceTag<execution_space>{},
                          destination.data(), Impl::native_fragment(source),
                          layout.stride, layout.row_major);
}

/// Perform one matrix-multiply-accumulate operation.
///
/// The public semantics are:
///
///   d_frag = a_frag * b_frag + c_frag
///
/// for one mma_shape<M, N, K> tile. The fragments should use compatible
/// policies: a matrix_a fragment with M x K extents, a matrix_b fragment with
/// K x N extents, and accumulator fragments with M x N extents, all targeting
/// the same execution-space backend.
///
/// This wrapper dispatches to the backend native MMA primitive.
/// Backend-specific limitations may apply, including supported shapes and
/// operand precisions.
template <class DFragT, class AFragT, class BFragT, class CFragT>
KOKKOS_INLINE_FUNCTION void mma_sync(DFragT& d_frag, AFragT& a_frag,
                                     BFragT& b_frag, CFragT& c_frag) {
  using execution_space = typename DFragT::policy_type::execution_space;

  Impl::check_mma_sync_contract<DFragT, AFragT, BFragT, CFragT>();

  Impl::mma_sync(Impl::ExecutionSpaceTag<execution_space>{},
                 Impl::native_fragment(d_frag), Impl::native_fragment(a_frag),
                 Impl::native_fragment(b_frag), Impl::native_fragment(c_frag));
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
