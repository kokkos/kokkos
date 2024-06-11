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

#ifndef KOKKOS_BIN_OPS_PUBLIC_API_HPP_
#define KOKKOS_BIN_OPS_PUBLIC_API_HPP_

#include <Kokkos_Macros.hpp>
#include <type_traits>

namespace Kokkos {

template <class KeyViewType>
struct BinOp1D {
  int max_bins_ = {};
  double mul_   = {};
  double min_   = {};

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4
  KOKKOS_DEPRECATED BinOp1D() = default;
#else
  BinOp1D() = delete;
#endif

  // Construct BinOp with number of bins, minimum value and maximum value
  BinOp1D(int max_bins__, typename KeyViewType::const_value_type min,
          typename KeyViewType::const_value_type max)
      : max_bins_(max_bins__ + 1),
        // Cast to double to avoid possible overflow when using integer
        mul_(static_cast<double>(max_bins__) /
             (static_cast<double>(max) - static_cast<double>(min))),
        min_(static_cast<double>(min)) {
    // For integral types the number of bins may be larger than the range
    // in which case we can exactly have one unique value per bin
    // and then don't need to sort bins.
    if (std::is_integral<typename KeyViewType::const_value_type>::value &&
        (static_cast<double>(max) - static_cast<double>(min)) <=
            static_cast<double>(max_bins__)) {
      mul_ = 1.;
    }
  }

  // Determine bin index from key value
  template <class ViewType>
  KOKKOS_INLINE_FUNCTION int bin(ViewType& keys, const int& i) const {
    return static_cast<int>(mul_ * (static_cast<double>(keys(i)) - min_));
  }

  // Return maximum bin index + 1
  KOKKOS_INLINE_FUNCTION
  int max_bins() const { return max_bins_; }

  // Compare to keys within a bin if true new_val will be put before old_val
  template <class ViewType, typename iType1, typename iType2>
  KOKKOS_INLINE_FUNCTION bool operator()(ViewType& keys, iType1& i1,
                                         iType2& i2) const {
    return keys(i1) < keys(i2);
  }
};

namespace Experimental {

template <int DIM, class KeyViewType>
struct BinOpND {
  static_assert(DIM > 1);

  int max_bins_[DIM] = {};
  double mul_[DIM]   = {};
  double min_[DIM]   = {};

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4
  KOKKOS_DEPRECATED BinOpND() = default;
#else
  BinOpND() = delete;
#endif

  BinOpND(int max_bins__[], typename KeyViewType::const_value_type min[],
          typename KeyViewType::const_value_type max[]) {
    for (int d = 0; d < DIM; ++d) {
      max_bins_[d] = max_bins__[d];
      mul_[d]      = static_cast<double>(max_bins__[d]) /
                (static_cast<double>(max[d]) - static_cast<double>(min[d]));
      min_[d] = static_cast<double>(min[d]);
    }
  }

  template <class ViewType>
  KOKKOS_INLINE_FUNCTION int bin(ViewType& keys, const int& i) const {
    int n = static_cast<int>(mul_[0] * (keys(i, 0) - min_[0]));
    for (int d = 1; d < DIM; ++d)
      n = n * max_bins_[d] + static_cast<int>(mul_[d] * (keys(i, d) - min_[d]));
    return n;
  }

  KOKKOS_INLINE_FUNCTION
  int max_bins() const {
    int n = max_bins_[0];
    for (int d = 1; d < DIM; ++d) n *= max_bins_[d];
    return n;
  }

  template <class ViewType, typename iType1, typename iType2>
  KOKKOS_INLINE_FUNCTION bool operator()(ViewType& keys, iType1& i1,
                                         iType2& i2) const {
    for (int d = 0; d < DIM; ++d) {
      if (keys(i1, d) > keys(i2, d)) return true;
      if (!(keys(i1, d) == keys(i2, d))) break;
    }
    return false;
  }
};

}  // namespace Experimental

template <class KeyViewType>
using BinOp2D = Experimental::BinOpND<2, KeyViewType>;

template <class KeyViewType>
using BinOp3D = Experimental::BinOpND<3, KeyViewType>;

}  // namespace Kokkos
#endif
