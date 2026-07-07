// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_SIMD_TENSORCORE_FRAGMENT_HPP
#define KOKKOS_SIMD_TENSORCORE_FRAGMENT_HPP

#include <type_traits>

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <mdspan/mdspan.hpp>

namespace Kokkos {
namespace Experimental {

/// Backend-independent role of a fragment in one MMA operation.
enum FragmentUse { MatrixA, MatrixB, Accumulator };
enum PrecisionType { Float, Double, TF32, BF16, Int8, Int32 };

template <class>
inline constexpr bool always_false_v = false;

template <class T>
struct NotImplementedError {
  static_assert(always_false_v<T>, "Not implemented");
};

namespace Impl {

template <class ExecSpace, PrecisionType P>
struct FragmentDTypeImpl {
  using type = NotImplementedError<void>;
};

template <class ExecSpace>
struct ExecutionSpaceTag {};

template <class ExecSpace, FragmentUse U, int MMA_M, int MMA_N, int MMA_K,
          class DType, class Layout>
struct NativeFragmentTImpl {
  using type = NotImplementedError<void>;
};

template <class Layout>
inline constexpr bool is_supported_operand_layout_v =
    std::is_same_v<Layout, Kokkos::layout_left> ||
    std::is_same_v<Layout, Kokkos::layout_right>;

template <class ExecSpace, class FragmentT, class ValueT>
KOKKOS_INLINE_FUNCTION void load_matrix_sync(ExecutionSpaceTag<ExecSpace>,
                                             FragmentT&, const ValueT*,
                                             const int) {
  static_assert(always_false_v<ExecSpace>,
                "Kokkos::Experimental::load_matrix_sync is not implemented for "
                "this execution space");
}

template <class ExecSpace, class FragmentT, class ValueT>
KOKKOS_INLINE_FUNCTION void fill_fragment(ExecutionSpaceTag<ExecSpace>,
                                          FragmentT&, ValueT) {
  static_assert(always_false_v<ExecSpace>,
                "Kokkos::Experimental::fill_fragment is not implemented for "
                "this execution space");
}

template <class ExecSpace>
KOKKOS_INLINE_FUNCTION void store_matrix_sync(ExecutionSpaceTag<ExecSpace>,
                                              ...) {
  static_assert(
      always_false_v<ExecSpace>,
      "Kokkos::Experimental::store_matrix_sync is not implemented for this "
      "execution space");
}

template <class ExecSpace, class DFragT, class AFragT, class BFragT,
          class CFragT>
KOKKOS_INLINE_FUNCTION void mma_sync(ExecutionSpaceTag<ExecSpace>, DFragT&,
                                     AFragT&, BFragT&, CFragT&) {
  static_assert(
      always_false_v<ExecSpace>,
      "Kokkos::Experimental::mma_sync is not implemented for this execution "
      "space");
}

}  // namespace Impl

}  // namespace Experimental
}  // namespace Kokkos

#if defined(KOKKOS_ENABLE_CUDA)
#include <impl/Kokkos_SIMD_TensorCore_Cuda.hpp>
#endif

#if defined(KOKKOS_ENABLE_HIP)
#include <impl/Kokkos_SIMD_TensorCore_HIP.hpp>
#endif

#if defined(KOKKOS_ENABLE_EXPERIMENTAL_SIMD_AMX)
#include <impl/Kokkos_SIMD_TensorCore_AMX.hpp>
#endif

namespace Kokkos {
namespace Experimental {

namespace Impl {

template <FragmentUse U, int MMA_M, int MMA_N, int MMA_K>
struct FragmentExtents;

template <int MMA_M, int MMA_N, int MMA_K>
struct FragmentExtents<FragmentUse::MatrixA, MMA_M, MMA_N, MMA_K> {
  using type = Kokkos::extents<int, MMA_M, MMA_K>;
};

template <int MMA_M, int MMA_N, int MMA_K>
struct FragmentExtents<FragmentUse::MatrixB, MMA_M, MMA_N, MMA_K> {
  using type = Kokkos::extents<int, MMA_K, MMA_N>;
};

template <int MMA_M, int MMA_N, int MMA_K>
struct FragmentExtents<FragmentUse::Accumulator, MMA_M, MMA_N, MMA_K> {
  using type = Kokkos::extents<int, MMA_M, MMA_N>;
};

template <FragmentUse U, class Extents, int MMA_M, int MMA_N, int MMA_K>
inline constexpr bool fragment_extents_match_v =
    Extents::rank() == 2 &&
    Extents::static_extent(0) ==
        FragmentExtents<U, MMA_M, MMA_N, MMA_K>::type::static_extent(0) &&
    Extents::static_extent(1) ==
        FragmentExtents<U, MMA_M, MMA_N, MMA_K>::type::static_extent(1);

template <class MatrixType>
struct FragmentUseFromMatrixType;

#if defined(KOKKOS_ENABLE_EXPERIMENTAL_SIMD_AMX)
using DefaultMMAExecutionSpace = Kokkos::DefaultHostExecutionSpace;
#elif defined(KOKKOS_ENABLE_CUDA)
using DefaultMMAExecutionSpace = Kokkos::Cuda;
#elif defined(KOKKOS_ENABLE_HIP)
using DefaultMMAExecutionSpace = Kokkos::HIP;
#else
using DefaultMMAExecutionSpace = Kokkos::DefaultExecutionSpace;
#endif

}  // namespace Impl

/// Role tag for the left-hand input operand of an MMA operation.
///
/// For mma_shape<M, N, K>, a matrix_a fragment has logical extents M x K.
struct matrix_a {};

/// Role tag for the right-hand input operand of an MMA operation.
///
/// For mma_shape<M, N, K>, a matrix_b fragment has logical extents K x N.
struct matrix_b {};

/// Role tag for the accumulator/output operand of an MMA operation.
///
/// For mma_shape<M, N, K>, an accumulator fragment has logical extents M x N.
struct accumulator {};

namespace Impl {

template <>
struct FragmentUseFromMatrixType<matrix_a> {
  static constexpr FragmentUse value = FragmentUse::MatrixA;
};

template <>
struct FragmentUseFromMatrixType<matrix_b> {
  static constexpr FragmentUse value = FragmentUse::MatrixB;
};

template <>
struct FragmentUseFromMatrixType<accumulator> {
  static constexpr FragmentUse value = FragmentUse::Accumulator;
};

}  // namespace Impl

/// Compile-time description of one hardware MMA tile shape.
///
/// mma_shape<M, N, K> describes the operation
///
///   D[M, N] = A[M, K] * B[K, N] + C[M, N]
///
/// for a single tensor-core instruction-level tile. The supported values
/// depend on the selected backend and element types; unsupported combinations
/// fail when the fragment's native backend storage type is instantiated.
template <int M, int N, int K>
struct mma_shape {
  static constexpr int m = M;
  static constexpr int n = N;
  static constexpr int k = K;
};

/// Policy that binds a fragment role to a shape, operand memory layout, and
/// execution space.
///
/// Shape is an mma_shape<M, N, K>. MatrixType is one of matrix_a, matrix_b, or
/// accumulator and determines the fragment role and expected logical extents.
/// OperandLayout describes the compact source layout used by load_matrix_sync
/// for matrix_a/matrix_b operands: Kokkos::layout_left means column-major and
/// Kokkos::layout_right means row-major. ExecSpace chooses the backend native
/// fragment implementation, defaulting to the enabled MMA backend.
template <class Shape, class MatrixType,
          class OperandLayout = Kokkos::layout_left,
          class ExecSpace     = Impl::DefaultMMAExecutionSpace>
struct mma_policy {
  using execution_space = ExecSpace;
  using shape_type      = Shape;
  using matrix_type     = MatrixType;
  using operand_layout  = OperandLayout;

  static_assert(Impl::is_supported_operand_layout_v<OperandLayout>,
                "mma_policy supports only Kokkos::layout_left and "
                "Kokkos::layout_right operand layouts");

  static constexpr FragmentUse use =
      Impl::FragmentUseFromMatrixType<MatrixType>::value;

  static constexpr int mma_m = Shape::m;
  static constexpr int mma_n = Shape::n;
  static constexpr int mma_k = Shape::k;
};

/// A fragment is the per-thread/per-vector-lane object that holds one native
/// backend tile. It is not a Kokkos::View and should normally be declared as a
/// local variable inside the Kokkos kernel that uses it.
///
/// Element is the backend element type, usually obtained from
/// FragmentDType<ExecSpace, Precision>. Extents must match the policy role:
/// matrix_a uses M x K, matrix_b uses K x N, and accumulator uses M x N. Layout
/// is the fragment's mdspan-style layout metadata; the operand memory layout
/// used by load_matrix_sync is selected by mma_policy::operand_layout.
///
/// The fragment owns only the backend-native storage handle. Portable code
/// should manipulate it through load_matrix_sync, fill_fragment, mma_sync, and
/// store_matrix_sync rather than by inspecting data_handle().
template <class Element, class Extents, class Layout, class Policy>
class fragment {
 public:
  using element_type = Element;
  using extents_type = Extents;
  using layout_type  = Layout;
  using policy_type  = Policy;
  using mapping_type = typename Layout::template mapping<Extents>;
  using data_handle_type =
      typename Impl::NativeFragmentTImpl<typename Policy::execution_space,
                                         Policy::use, Policy::mma_m,
                                         Policy::mma_n, Policy::mma_k, Element,
                                         typename Policy::operand_layout>::type;

  static constexpr FragmentUse use = Policy::use;
  static constexpr int mma_m       = Policy::mma_m;
  static constexpr int mma_n       = Policy::mma_n;
  static constexpr int mma_k       = Policy::mma_k;

  static_assert(
      Impl::fragment_extents_match_v<use, Extents, mma_m, mma_n, mma_k>,
      "fragment extents must match the mma_policy matrix role and "
      "shape");

  KOKKOS_DEFAULTED_FUNCTION fragment() = default;

  KOKKOS_INLINE_FUNCTION data_handle_type& data_handle() { return handle_; }

  KOKKOS_INLINE_FUNCTION const data_handle_type& data_handle() const {
    return handle_;
  }

  KOKKOS_INLINE_FUNCTION constexpr const mapping_type& mapping() const {
    return mapping_;
  }

  KOKKOS_INLINE_FUNCTION constexpr const extents_type& extents() const {
    return mapping_.extents();
  }

 private:
  data_handle_type handle_;
  mapping_type mapping_{extents_type{}};
};

/// Map a portable precision selector to the concrete backend element type.
///
/// Use this helper when declaring public fragments so the same source can build
/// against different execution spaces.
template <class ExecSpace, PrecisionType P>
struct FragmentDType {
  using type = typename Impl::FragmentDTypeImpl<ExecSpace, P>::type;
};

/// Convenience extents for a matrix_a fragment with logical shape MMA_M x
/// MMA_K.
template <int MMA_M, int MMA_K>
using matrix_a_extents = Kokkos::extents<int, MMA_M, MMA_K>;

/// Convenience extents for a matrix_b fragment with logical shape MMA_K x
/// MMA_N.
template <int MMA_K, int MMA_N>
using matrix_b_extents = Kokkos::extents<int, MMA_K, MMA_N>;

/// Convenience extents for an accumulator fragment with logical shape MMA_M x
/// MMA_N.
template <int MMA_M, int MMA_N>
using accumulator_extents = Kokkos::extents<int, MMA_M, MMA_N>;

}  // namespace Experimental
}  // namespace Kokkos

#endif
