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

template <class ViewType, int Traits = 0>
struct ViewUniformType;

} /* namespace Impl */

template <class DataType, class... Properties>
class View;

template <class>
struct is_view : public std::false_type {};

template <class D, class... P>
struct is_view<View<D, P...>> : public std::true_type {};

template <class D, class... P>
struct is_view<const View<D, P...>> : public std::true_type {};

template <class T>
inline constexpr bool is_view_v = is_view<T>::value;

} /* namespace Kokkos */

#endif  /* #ifndef KOKKOS_VIEW_FWD_HPP */
