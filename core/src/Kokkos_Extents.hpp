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
#ifndef KOKKOS_KOKKOS_EXTENTS_HPP
#define KOKKOS_KOKKOS_EXTENTS_HPP

#include <cstddef>
#include <type_traits>
#include <Kokkos_Macros.hpp>

namespace Kokkos {
namespace Experimental {

constexpr ptrdiff_t dynamic_extent = -1;

template <ptrdiff_t... ExtentSpecs>
struct Extents {
  /* TODO @enhancement flesh this out more */
};

template <class Exts, ptrdiff_t NewExtent>
struct PrependExtent;

template <ptrdiff_t... Exts, ptrdiff_t NewExtent>
struct PrependExtent<Extents<Exts...>, NewExtent> {
  using type = Extents<NewExtent, Exts...>;
};

template <class Exts, ptrdiff_t NewExtent>
struct AppendExtent;

template <ptrdiff_t... Exts, ptrdiff_t NewExtent>
struct AppendExtent<Extents<Exts...>, NewExtent> {
  using type = Extents<Exts..., NewExtent>;
};

}  // end namespace Experimental

namespace Impl {

namespace _parse_view_extents_impl {

template <class T>
struct _all_remaining_extents_dynamic : std::true_type {};

template <class T>
struct _all_remaining_extents_dynamic<T*> : _all_remaining_extents_dynamic<T> {
};

template <class T, unsigned N>
struct _all_remaining_extents_dynamic<T[N]> : std::false_type {};

template <class T, class Result, class = void>
struct _parse_impl {
  using type = Result;
};

// We have to treat the case of int**[x] specially, since it *doesn't* go
// backwards
template <class T, ptrdiff_t... ExtentSpec>
struct _parse_impl<T*, Kokkos::Experimental::Extents<ExtentSpec...>,
                   std::enable_if_t<_all_remaining_extents_dynamic<T>::value>>
    : _parse_impl<T, Kokkos::Experimental::Extents<
                         Kokkos::Experimental::dynamic_extent, ExtentSpec...>> {
};

// int*(*[x])[y] should still work also (meaning int[][x][][y])
template <class T, ptrdiff_t... ExtentSpec>
struct _parse_impl<
    T*, Kokkos::Experimental::Extents<ExtentSpec...>,
    std::enable_if_t<!_all_remaining_extents_dynamic<T>::value>> {
  using _next = Kokkos::Experimental::AppendExtent<
      typename _parse_impl<T, Kokkos::Experimental::Extents<ExtentSpec...>,
                           void>::type,
      Kokkos::Experimental::dynamic_extent>;
  using type = typename _next::type;
};

template <class T, ptrdiff_t... ExtentSpec, unsigned N>
struct _parse_impl<T[N], Kokkos::Experimental::Extents<ExtentSpec...>, void>
    : _parse_impl<
          T, Kokkos::Experimental::Extents<ExtentSpec...,
                                           ptrdiff_t(N)>  // TODO @pedantic this
                                                          // could be a
                                                          // narrowing cast
          > {};

}  // end namespace _parse_view_extents_impl

template <class DataType>
struct ParseViewExtents {
  using type = typename _parse_view_extents_impl ::_parse_impl<
      DataType, Kokkos::Experimental::Extents<>>::type;
};

template <class ValueType, ptrdiff_t Ext>
struct ApplyExtent {
  using type = ValueType[Ext];
};

template <class ValueType>
struct ApplyExtent<ValueType, Kokkos::Experimental::dynamic_extent> {
  using type = ValueType*;
};

template <class ValueType, unsigned N, ptrdiff_t Ext>
struct ApplyExtent<ValueType[N], Ext> {
  using type = typename ApplyExtent<ValueType, Ext>::type[N];
};

template <class ValueType, ptrdiff_t Ext>
struct ApplyExtent<ValueType*, Ext> {
  using type = ValueType * [Ext];
};

template <class ValueType>
struct ApplyExtent<ValueType*, Kokkos::Experimental::dynamic_extent> {
  using type =
      typename ApplyExtent<ValueType,
                           Kokkos::Experimental::dynamic_extent>::type*;
};

template <class ValueType, unsigned N>
struct ApplyExtent<ValueType[N], Kokkos::Experimental::dynamic_extent> {
  using type =
      typename ApplyExtent<ValueType,
                           Kokkos::Experimental::dynamic_extent>::type[N];
};

}  // end namespace Impl

}  // end namespace Kokkos

#endif  // KOKKOS_KOKKOS_EXTENTS_HPP
