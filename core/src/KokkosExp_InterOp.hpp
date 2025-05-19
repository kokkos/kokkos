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

#ifndef KOKKOS_CORE_EXP_INTEROP_HPP
#define KOKKOS_CORE_EXP_INTEROP_HPP
#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_INTEROP
#endif

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Layout.hpp>
#include <Kokkos_MemoryTraits.hpp>
#include <Kokkos_View.hpp>
#include <impl/Kokkos_Utilities.hpp>
#include <type_traits>

namespace Kokkos {
namespace Impl {

// ------------------------------------------------------------------ //
//  this is used to convert
//      Kokkos::Device<ExecSpace, MemSpace> to MemSpace
//
template <typename Tp>
struct device_memory_space {
  using type = Tp;
};

template <typename ExecT, typename MemT>
struct device_memory_space<Kokkos::Device<ExecT, MemT>> {
  using type = MemT;
};

template <typename Tp>
using device_memory_space_t = typename device_memory_space<Tp>::type;

// ------------------------------------------------------------------ //
//  this is the impl version which takes a view and converts to python
//  view type
//
template <typename, typename...>
struct python_view_type_impl;

template <template <typename...> class ViewT, typename ValueT,
          typename... Types>
struct python_view_type_impl<ViewT<ValueT>, type_list<Types...>> {
  using type = ViewT<ValueT, device_memory_space_t<Types>...>;
};

template <template <typename...> class ViewT, typename ValueT,
          typename... Types>
struct python_view_type_impl<ViewT<ValueT, Types...>>
    : python_view_type_impl<ViewT<ValueT>,
                            filter_type_list_t<is_default_memory_trait,
                                               type_list<Types...>, false>> {};

template <typename... T>
using python_view_type_impl_t = typename python_view_type_impl<T...>::type;

}  // namespace Impl
}  // namespace Kokkos

namespace Kokkos {

template <typename DataType, class... Properties>
class DynRankView;

namespace Impl {

// Duplicate from the header file for DynRankView to avoid core depending on
// containers.
template <class>
struct is_dyn_rank_view_dup : public std::false_type {};

template <class D, class... P>
struct is_dyn_rank_view_dup<Kokkos::DynRankView<D, P...>>
    : public std::true_type {};

}  // namespace Impl

namespace Experimental {

// ------------------------------------------------------------------ //
//  this is used to extract the uniform type of a view
//
template <typename ViewT>
struct python_view_type {
  static_assert(
      Kokkos::is_view<std::decay_t<ViewT>>::value ||
          Kokkos::Impl::is_dyn_rank_view_dup<std::decay_t<ViewT>>::value,
      "Error! python_view_type only supports Kokkos::View and "
      "Kokkos::DynRankView");

  using type =
      Kokkos::Impl::python_view_type_impl_t<typename ViewT::array_type>;
};

template <typename ViewT>
using python_view_type_t = typename python_view_type<ViewT>::type;

template <typename Tp>
auto as_python_type(Tp&& _v) {
  using cast_type = python_view_type_t<Tp>;
  return static_cast<cast_type>(std::forward<Tp>(_v));
}
}  // namespace Experimental
}  // namespace Kokkos

#ifdef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_INTEROP
#undef KOKKOS_IMPL_PUBLIC_INCLUDE
#undef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_INTEROP
#endif
#endif
