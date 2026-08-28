// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_CUDA_TILEVIEW_HPP
#define KOKKOS_CUDA_TILEVIEW_HPP

#include <Kokkos_Macros.hpp>
#include <Kokkos_Assert.hpp>

#ifdef KOKKOS_ENABLE_CUDA_TILE

#include <cuda_tile.h>

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_TILEVIEW
#endif

#include <Kokkos_Concepts.hpp>
#include <Kokkos_Layout.hpp>
#include <Kokkos_View.hpp>

#ifdef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_TILEVIEW
#undef KOKKOS_IMPL_PUBLIC_INCLUDE
#undef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_TILEVIEW
#endif

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace Kokkos {
namespace Impl {

namespace ct = cuda::tiles;

// Returns the TargetExtents (cuda::tiles::extents) object with the
// same extents as the input Kokkos::View's extents.
template <class TargetExtents, class ViewType, size_t... Ranks>
KOKKOS_INLINE_FUNCTION TargetExtents cuda_tile_extents_from_view(
    ViewType const& view, std::index_sequence<Ranks...>) {
  return TargetExtents(
      static_cast<typename TargetExtents::index_type>(view.extent(Ranks))...);
}

// Returns the TargetExtents (cuda::tiles::extents) object with the
// same extents as the input Kokkos::View's strides.
template <class TargetExtents, class ViewType, size_t... Ranks>
KOKKOS_INLINE_FUNCTION TargetExtents cuda_tile_strides_from_view(
    ViewType const& view, std::index_sequence<Ranks...>) {
  return TargetExtents(
      static_cast<typename TargetExtents::index_type>(view.stride(Ranks))...);
}


template <size_t... Ranks>
constexpr auto cuda_tile_dextents_impl(std::index_sequence<Ranks...>) ->
  ct::extents<uint32_t, ((void)Ranks, ct::dynamic_extent)...>;

// cuda::tiles::extents specialization with rank Rank,
// all of whose extents are dynamic.
template <size_t Rank>
using cuda_tile_dextents =
  decltype(cuda_tile_dextents_impl(std::make_index_sequence<Rank>()));

// Declared, but never defined -- only usable in an unevaluated context
// (e.g., decltype) -- so that its trailing return type can be extracted
// as a cuda::tiles::extents specialization via CudaTileExtentsProduct.
// Requires that, in every extent where both Extents1 and Extents2 are
// static, their product doesn't need to be (and isn't) computed with any
// remainder -- multiplication is always exact, unlike the division in
// cuda_tile_index_space_extents_from_view below.
template <class Extents1, class Extents2, size_t... Ranks>
constexpr auto cuda_tile_extents_product(std::index_sequence<Ranks...>) -> ct::extents<
    uint32_t,
    (Extents1::static_extent(Ranks) == ct::dynamic_extent ||
             Extents2::static_extent(Ranks) == ct::dynamic_extent
         ? ct::dynamic_extent
         : Extents1::static_extent(Ranks) * Extents2::static_extent(Ranks))...>;

// The cuda::tiles::extents specialization for the elementwise product of
// Extents1 and Extents2 (which must have the same rank), static wherever
// both are static.
//
// TileView needs this because it discards the input Kokkos::View's
// extents type, and only keeps the tile index space type and the tile
// shape.  If TileView ever needs to support input extents that aren't
// elementwise divisible by the tile shape (which it would do via
// masked loads and stores), it needs some way to hang on to the input
// Kokkos::View's extents type.
template <class Extents1, class Extents2>
using CudaTileExtentsProduct =
    decltype(cuda_tile_extents_product<Extents1, Extents2>(
        std::make_index_sequence<Extents1::rank()>{}));

// Declared, but never defined -- only usable in an unevaluated context
// (e.g., decltype) -- so that its trailing return type can be extracted
// as a cuda::tiles::extents specialization via
// CudaTileIndexSpaceExtentsFromView.  The requires clause rejects
// ViewType/TileShape combinations where a static View extent isn't an
// exact multiple of the corresponding tile shape extent, rather than
// silently truncating; a dynamic View extent is checked at run time
// instead, in TileView's constructor.
template <class ViewType, class TileShape, size_t... Ranks>
  requires ((ViewType::static_extent(Ranks) == 0 ||
             ViewType::static_extent(Ranks) % TileShape::static_extent(Ranks) ==
                 0) &&
            ...)
constexpr auto cuda_tile_index_space_extents_from_view(std::index_sequence<Ranks...>)
    -> ct::extents<uint32_t,
                    (ViewType::static_extent(Ranks) == 0
                         ? ct::dynamic_extent
                         : ViewType::static_extent(Ranks) /
                               TileShape::static_extent(Ranks))...>;

// The cuda::tiles::extents specialization for the tile index space of a
// View partitioned according to TileShape: static in each dimension
// where ViewType has a static extent (in which case that extent must be
// an exact multiple of TileShape's corresponding extent), and dynamic
// elsewhere.  TileView's CTAD deduction guide uses this to deduce
// Extents from the View and tile shape passed to TileView's constructor.
template <class ViewType, class TileShape>
using CudaTileIndexSpaceExtentsFromView =
    decltype(cuda_tile_index_space_extents_from_view<ViewType, TileShape>(
        std::make_index_sequence<ViewType::rank()>{}));

// Returns whether each of the View's runtime extents is evenly
// divisible by the corresponding (static) extent of TileShapeType.
// Tile loads and stores indices in units of whole tiles, so a View
// whose extents aren't multiples of the tile shape would leave a
// partial tile at the boundary.  partition_view can handle that with
// load_masked and store_masked, but we don't want to require that
// TileView be able to handle that case for now.
template <class TileShapeType, class ViewType, size_t... Ranks>
KOKKOS_INLINE_FUNCTION bool view_extents_evenly_divisible_by_tile_shape(
    ViewType const& view, std::index_sequence<Ranks...>) {
  return ((view.extent(Ranks) % TileShapeType::static_extent(Ranks) == 0) &&
          ...);
}

// Returns whether all of TileShapeType's (static) extents are nonzero.
// A tile shape with a zero extent can't represent any tiles.
template <class TileShapeType, size_t... Ranks>
constexpr bool tile_shape_extents_all_nonzero(std::index_sequence<Ranks...>) {
  return ((TileShapeType::static_extent(Ranks) != 0) && ...);
}

}  // namespace Impl

/// \brief Applies a tile partitioning to a Kokkos::View
///
/// \tparam TileType cuda::tiles::tile specialization describing the
///   element type and shape of the tiles into which the View will be
///   partitioned
///
/// \tparam Extents cuda::tiles::extents specialization describing the
///    "tile index space type," that is, the type of the index space
///    over which tile loads and stores iterate.  This is the space of
///    tile indices (how many tiles fit along each dimension), not the
///    space of element indices into the underlying Kokkos::View; e.g.,
///    for a View with 256 elements partitioned into tiles of shape
///    <8>, Extents' (runtime) extent is 32, not 256.
///
/// \tparam MemorySpace Kokkos memory space in which the data live
///
/// Construct this on host from a Kokkos::View (the array to
/// partition) and a cuda::tiles::shape (the shape of each tile in the
/// partition).  The resulting object is a nonowning view of the
/// Kokkos::View's data.
///
/// The TileView object can be bytewise copied to device for use in
/// Tile kernels.  There, it supports load and store operations, just
/// like a cuda::tiles::partition_view.
///
/// TileView does not currently support partial out-of-bounds access
/// with masked loads and stores.  The cuda::tiles::tensor_span that
/// it exposes may thus have extents reduced to be elementwise
/// multiples of the tile extents.
template <class TileType, class Extents, Kokkos::MemorySpace MemorySpace>
class TileView {
  static_assert(Impl::ct::tile_shape<typename TileType::shape_type>,
                "Kokkos::TileView: TileType::shape_type must be a "
                "cuda::tiles::tile_shape");
  static_assert(TileType::shape_type::rank() == Extents::rank(),
                "Kokkos::TileView: TileType's shape rank must match "
                "Extents rank");

public:
  using tile_type       = TileType;
  using element_type    = typename TileType::element_type;
  using value_type      = std::remove_cv_t<element_type>;
  using tile_shape_type = typename TileType::shape_type;
  using extents_type    = Extents;
  using memory_space    = MemorySpace;

private:
  using strides_type = Impl::cuda_tile_dextents<extents_type::rank()>;
  using layout_type  = Impl::ct::layout_strided<strides_type>;
  using mapping_type = typename layout_type::template mapping<
      Impl::CudaTileExtentsProduct<extents_type, tile_shape_type>>;
  
public:
  using span_type = Impl::ct::tensor_span<
      element_type, typename mapping_type::extents_type,
      layout_type>;
  using partition_view_type =
      Impl::ct::partition_view<span_type, tile_shape_type>;

  using index_type      = typename partition_view_type::index_type;
  using rank_type       = typename partition_view_type::rank_type;
  using view_shape_type = typename partition_view_type::view_shape_type;
  using view_tile_type  = typename partition_view_type::view_tile_type;

  /// \brief Construct from a Kokkos::View and a tile shape on the host.
  ///
  /// \param view Kokkos::View such that each extent is a nonzero
  ///   multiple of the corresponding extent of \c ts
  ///
  /// \param ts cuda::tiles::shape whose extents are all nonzero
  ///
  /// For now, Kokkos requires that each extent of \c view is a
  /// nonzero multiple of each extent of \c ts.  This lets Kokkos
  /// defer providing masked variants of loads and stores.
  template <class ViewType>
    requires(Kokkos::is_view_v<ViewType>)
  TileView(ViewType const& view, [[maybe_unused]] tile_shape_type ts) noexcept
      : span_(view.data(),
              mapping_type(
                  Impl::cuda_tile_extents_from_view<
                      Impl::CudaTileExtentsProduct<extents_type, tile_shape_type>>(
                      view, std::make_index_sequence<ViewType::rank()>{}),
                  Impl::cuda_tile_strides_from_view<strides_type>(
                      view, std::make_index_sequence<ViewType::rank()>{}))) {
    static_assert(static_cast<size_t>(ViewType::rank()) ==
                      extents_type::rank(),
                  "Kokkos::TileView: View rank must match Extents rank");
    static_assert(
        std::is_same_v<typename ViewType::non_const_value_type, value_type>,
        "Kokkos::TileView: View's value_type must match TileType's "
        "element_type");
    static_assert(
        std::is_same_v<typename ViewType::memory_space, memory_space>,
        "Kokkos::TileView: View's memory_space must match TileView's "
        "MemorySpace");
    static_assert(
        Impl::tile_shape_extents_all_nonzero<tile_shape_type>(
            std::make_index_sequence<tile_shape_type::rank()>{}),
        "Kokkos::TileView: TileShape's extents must all be nonzero");
    KOKKOS_ASSERT(
        (Impl::view_extents_evenly_divisible_by_tile_shape<tile_shape_type>(
            view, std::make_index_sequence<ViewType::rank()>{})));
  }

  /// \brief Underlying cuda::tiles::tensor_span that this object views
  ///
  /// TileView does not currently support partial out-of-bounds access
  /// with masked loads and stores.  The tensor_span that it exposes
  /// may thus have extents reduced to be elementwise multiples of the
  /// tile extents.
  KOKKOS_EXPERIMENTAL_TILE_FUNCTION
  span_type span() const noexcept { return span_; }

  /// \brief The tile shape
  KOKKOS_EXPERIMENTAL_TILE_FUNCTION
  tile_shape_type shape() const noexcept { return tile_shape_type{}; }

  /// \brief Convert this object into a partition_view.
  ///
  /// Use this only if you need to call an interface that expects a
  /// partition_view.  Otherwise, just call load and store directly on
  /// this object.
  KOKKOS_EXPERIMENTAL_TILE_FUNCTION
  operator partition_view_type() const noexcept {
    return partition_view_type(span_, tile_shape_type{});
  }

  // The remaining member functions mirror cuda::tiles::partition_view's
  // load/store interface (same names, template parameters, and function
  // parameters), each forwarding to the partition_view built by the
  // conversion operator above.  This lets a TileView be used directly in
  // a tile kernel body wherever a partition_view would be.
  //
  // TileView doesn't yet implement masked loads or stores.  For now,
  // we would like to explore an interface that only exposes in-bounds
  // loads and stores.  On the other hand, masking is useful for
  // reasons other than simplifying boundary handling.
  //
  // TileView doesn't yet implement atomic loads or stores.

  template <class... Idx>
    requires (sizeof...(Idx) == extents_type::rank())
  KOKKOS_EXPERIMENTAL_TILE_FUNCTION view_tile_type
  load(Idx... idx) const noexcept {
    return static_cast<partition_view_type>(*this).load(idx...);
  }

  template <Impl::ct::tile_like Value, class... Idx>
    requires (
      sizeof...(Idx) == extents_type::rank()
      // && is_convertible_v<Value, view_tile_type>
    )
  KOKKOS_EXPERIMENTAL_TILE_FUNCTION void
  store(Value tile_value, Idx... idx) const noexcept {
    static_cast<partition_view_type>(*this).store(tile_value, idx...);
  }

 private:
  span_type span_;
};

// TileView needs a deduction guide because the constructor's
// parameters don't have the same types as its template parameters.
template <class ViewType, class TileShape>
TileView(ViewType const&, TileShape) -> TileView<
    Impl::ct::tile<typename ViewType::non_const_value_type, TileShape>,
    Impl::CudaTileIndexSpaceExtentsFromView<ViewType, TileShape>,
    typename ViewType::memory_space>;

}  // namespace Kokkos

#endif  // KOKKOS_ENABLE_CUDA_TILE
#endif  // KOKKOS_CUDA_TILEVIEW_HPP
