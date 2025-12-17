//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER
#include <mdspan/mdspan.hpp>

#include <cassert>
#include <iostream>
#include <type_traits>


// Simple tiled layout.
// Hard-coded for 2D, column-major across tiles
// and row-major within each tile
struct SimpleTileLayout2D {
  template <class Extents>
  struct mapping {

    // for simplicity
    static_assert(Extents::rank() == 2, "SimpleTileLayout2D is hard-coded for 2D layout");

    using extents_type = Extents;
    using rank_type = typename Extents::rank_type;
    using size_type = typename Extents::size_type;
    using layout_type = SimpleTileLayout2D;

    mapping() noexcept = default;
    mapping(const mapping&) noexcept = default;
    mapping& operator=(const mapping&) noexcept = default;

    mapping(
      Extents const& exts,
      size_type row_tile,
      size_type col_tile
    ) noexcept
      : extents_(exts),
        row_tile_size_(row_tile),
        col_tile_size_(col_tile)
    {
      // For simplicity, don't worry about negatives/zeros/etc.
      assert(row_tile > 0);
      assert(col_tile > 0);
      assert(exts.extent(0) > 0);
      assert(exts.extent(1) > 0);
    }

    MDSPAN_TEMPLATE_REQUIRES(
      class OtherExtents,
      /* requires */ (::std::is_constructible<extents_type, OtherExtents>::value)
    )
    MDSPAN_CONDITIONAL_EXPLICIT(
      (!::std::is_convertible<OtherExtents, extents_type>::value)
    )
    constexpr mapping(const mapping<OtherExtents>& input_mapping) noexcept :
      extents_(input_mapping.extents()),
      row_tile_size_(input_mapping.row_tile_size_),
      col_tile_size_(input_mapping.col_tile_size_)
    {}

    //------------------------------------------------------------
    // Helper members (not part of the layout concept)

    constexpr size_type
    n_row_tiles() const noexcept {
      return extents_.extent(0) / row_tile_size_ + size_type((extents_.extent(0) % row_tile_size_) != 0);
    }

    constexpr size_type
    n_column_tiles() const noexcept {
      return extents_.extent(1) / col_tile_size_ + size_type((extents_.extent(1) % col_tile_size_) != 0);
    }

    constexpr size_type
    tile_size() const noexcept {
      return row_tile_size_ * col_tile_size_;
    }

    size_type
    tile_offset(size_type row, size_type col) const noexcept {
      // This could probably be more efficient, but for example purposes...
      auto col_tile = col / col_tile_size_;
      auto row_tile = row / row_tile_size_;
      // We're hard-coding this to *column-major* layout across tiles
      return (col_tile * n_row_tiles() + row_tile) * tile_size();
    }

    size_type
    offset_in_tile(size_type row, size_type col) const noexcept {
      auto t_row = row % row_tile_size_;
      auto t_col = col % col_tile_size_;
      // We're hard-coding this to *row-major* within tiles
      return t_row * col_tile_size_ + t_col;
    }

    //------------------------------------------------------------
    // Required members

    constexpr const extents_type& extents() const {
      return extents_;
    }

    constexpr size_type
    required_span_size() const noexcept {
      return n_row_tiles() * n_column_tiles() * tile_size();
    }

    template<class RowIndex, class ColIndex>
    // requires(is_convertible_v<RowIndex, size_type> &&
    //   is_convertible_v<ColIndex, size_type> &&
    //   is_nothrow_constructible_v<size_type, RowIndex> &&
    //   is_nothrow_constructible_v<size_type, ColIndex>)
    constexpr size_type
    operator()(RowIndex row, ColIndex col) const noexcept {
      // TODO (mfh 2022/08/04 check precondition that
      // extents_type::index-cast(row, col)
      // is a multidimensional index in extents_.
      return tile_offset(row, col) + offset_in_tile(row, col);
    }


    // Mapping is always unique
    static constexpr bool is_always_unique() noexcept { return true; }
    // Only exhaustive if extents_.extent(0) % column_tile_size == 0, so not always
    static constexpr bool is_always_exhaustive() noexcept { return false; }
    // There is not always a regular stride between elements in a given dimension
    static constexpr bool is_always_strided() noexcept { return false; }

    static constexpr bool is_unique() noexcept { return true; }
    constexpr bool is_exhaustive() const noexcept {
      // Only exhaustive if extents fit exactly into tile sizes...
      return (extents_.extent(0) % row_tile_size_ == 0) && (extents_.extent(1) % col_tile_size_ == 0);
    }
    // There are some circumstances where this is strided, but we're not
    // concerned about that optimization, so we're allowed to just return false here
    constexpr bool is_strided() const noexcept { return false; }

  private:
    Extents extents_;
    size_type row_tile_size_ = 1;
    size_type col_tile_size_ = 1;
  };
};



int main() {
  //----------------------------------------
  // 5x2 example, tiled as 3x3
  constexpr int n_rows = 2;
  constexpr int n_cols = 5;
  int data_row_major[] = {
       1,   2,   3,  /*|*/  4,   5,  /* X | */
       6,   7,   8,  /*|*/  9,  10   /* X | */
     /*X,   X,   X,    |    X    X      X |
      *------------------------------------ */
  };
  // manually tiling the above, using -1 as filler
  static constexpr int X = -1;
  int data_tiled[] = {
    /* tile 0, 0 */  1,  2,  3,  6,  7,  8,  X,  X,  X,
    /* tile 0, 1 */  4,  5,  X,  9, 10,  X,  X,  X,  X,
  };
  //----------------------------------------
  // Just use dynamic extents for the purposes of demonstration
  using extents_type = Kokkos::dextents<size_t, 2>;
  using tiled_mdspan = Kokkos::mdspan<int, extents_type, SimpleTileLayout2D>;
  using tiled_layout_type = typename SimpleTileLayout2D::template mapping<extents_type>;
  using row_major_mdspan = Kokkos::mdspan<int, extents_type, Kokkos::layout_right>;
  //----------------------------------------
  auto tiled = tiled_mdspan(data_tiled, tiled_layout_type(extents_type(n_rows, n_cols), 3, 3));
  auto row_major = row_major_mdspan(data_row_major, n_rows, n_cols);
  //----------------------------------------
  // Check that we did things correctly
  int failures = 0;
  for (int irow = 0; irow < n_rows; ++irow) {
    for (int icol = 0; icol < n_cols; ++icol) {
#if MDSPAN_USE_BRACKET_OPERATOR
      if(tiled[irow, icol] != row_major[irow, icol]) {
        std::cout << "Mismatch for entry " << irow << ", " << icol << ":" << std::endl;
        std::cout << "  tiled(" << irow << ", " << icol << ") = "
                  << tiled[irow, icol] << std::endl;
        std::cout << "  row_major(" << irow << ", " << icol << ") = "
                  << row_major[irow, icol] << std::endl;
#else
      if(tiled(irow, icol) != row_major(irow, icol)) {
        std::cout << "Mismatch for entry " << irow << ", " << icol << ":" << std::endl;
        std::cout << "  tiled(" << irow << ", " << icol << ") = "
                  << tiled(irow, icol) << std::endl;
        std::cout << "  row_major(" << irow << ", " << icol << ") = "
                  << row_major(irow, icol) << std::endl;
#endif
        ++failures;

      }

    }
  }
  if(failures == 0) {
    std::cout << "Success! SimpleTiledLayout2D works as expected." << std::endl;
  }
}
