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

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#include <Kokkos_Macros.hpp>
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif

#ifndef KOKKOS_VIEW_FWD_HPP
#define KOKKOS_VIEW_FWD_HPP

namespace Kokkos {
namespace Impl {

template <class DataType>
struct ViewArrayAnalysis;

template <class DataType, class ArrayLayout,
          typename ValueType =
              typename ViewArrayAnalysis<DataType>::non_const_value_type>
struct ViewDataAnalysis;

template <class, class...>
class ViewMapping {
 public:
  enum : bool { is_assignable_data_type = false };
  enum : bool { is_assignable = false };
};

template <typename IntType>
constexpr KOKKOS_INLINE_FUNCTION std::size_t count_valid_integers(
    const IntType i0, const IntType i1, const IntType i2, const IntType i3,
    const IntType i4, const IntType i5, const IntType i6, const IntType i7) {
  static_assert(std::is_integral<IntType>::value,
                "count_valid_integers() must have integer arguments.");

  return (i0 != KOKKOS_INVALID_INDEX) + (i1 != KOKKOS_INVALID_INDEX) +
         (i2 != KOKKOS_INVALID_INDEX) + (i3 != KOKKOS_INVALID_INDEX) +
         (i4 != KOKKOS_INVALID_INDEX) + (i5 != KOKKOS_INVALID_INDEX) +
         (i6 != KOKKOS_INVALID_INDEX) + (i7 != KOKKOS_INVALID_INDEX);
}

KOKKOS_INLINE_FUNCTION
void runtime_check_rank(const size_t rank, const size_t dyn_rank,
                        const bool is_void_spec, const size_t i0,
                        const size_t i1, const size_t i2, const size_t i3,
                        const size_t i4, const size_t i5, const size_t i6,
                        const size_t i7, const std::string& label) {
  (void)(label);

  if (is_void_spec) {
    const size_t num_passed_args =
        count_valid_integers(i0, i1, i2, i3, i4, i5, i6, i7);

    if (num_passed_args != dyn_rank && num_passed_args != rank) {
      KOKKOS_IF_ON_HOST(
          const std::string message =
              "Constructor for Kokkos View '" + label +
              "' has mismatched number of arguments. Number of arguments = " +
              std::to_string(num_passed_args) +
              " but dynamic rank = " + std::to_string(dyn_rank) + " \n";
          Kokkos::abort(message.c_str());)
      KOKKOS_IF_ON_DEVICE(Kokkos::abort("Constructor for Kokkos View has "
                                        "mismatched number of arguments.");)
    }
  }
}

} /* namespace Impl */
} /* namespace Kokkos */

#endif  /* #ifndef KOKKOS_VIEW_FWD_HPP */
