/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_CORE_EXP_INTEROP_HPP
#define KOKKOS_CORE_EXP_INTEROP_HPP

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Layout.hpp>
#include <Kokkos_MemoryTraits.hpp>
#include <Kokkos_View.hpp>
#include <Kokkos_DynRankView.hpp>
#include <impl/Kokkos_Utilities.hpp>
#include <type_traits>

namespace Kokkos {
namespace Experimental {
namespace Impl {

template <typename... T>
using type_list = Kokkos::Impl::type_list<T...>;

// ------------------------------------------------------------------ //
//  concat_type_list combines types in multiple type_lists

// forward declaration
template <typename... T>
struct concat_type_list;

// alias
template <typename... T>
using concat_type_list_t = typename concat_type_list<T...>::type;

// final instantiation
template <typename... T>
struct concat_type_list<type_list<T...>> {
  using type = type_list<T...>;
};

// combine consecutive type_lists
template <typename... T, typename... U, typename... Tail>
struct concat_type_list<type_list<T...>, type_list<U...>, Tail...>
    : concat_type_list<type_list<T..., U...>, Tail...> {};

// ------------------------------------------------------------------ //
//  gather generates type-list of types which satisfy
//  PredicateT<T>::value == ValueT

template <template <typename> class PredicateT, typename TypeListT,
          bool ValueT = false>
struct gather;

template <template <typename> class PredicateT, typename... T, bool ValueT>
struct gather<PredicateT, type_list<T...>, ValueT> {
  using type =
      concat_type_list_t<std::conditional_t<PredicateT<T>::value == ValueT,
                                            type_list<T>, type_list<>>...>;
};

template <template <typename> class PredicateT, typename T, bool ValueT = false>
using gather_t = typename gather<PredicateT, T, ValueT>::type;

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
//  this identifies the default memory trait
//
template <typename Tp>
struct is_default_memory_trait : std::false_type {};

template <>
struct is_default_memory_trait<Kokkos::MemoryTraits<0>> : std::true_type {};

// ------------------------------------------------------------------ //
//  this is the impl version which takes a view and converts to python
//  view type
//
template <typename, typename...>
struct python_view_type;

template <template <typename...> class ViewT, typename ValueT,
          typename... Types>
struct python_view_type<ViewT<ValueT>, type_list<Types...>> {
  using type = ViewT<ValueT, device_memory_space_t<Types>...>;
};

template <template <typename...> class ViewT, typename ValueT,
          typename... Types>
struct python_view_type<ViewT<ValueT, Types...>>
    : python_view_type<ViewT<ValueT>,
                       gather_t<is_default_memory_trait, type_list<Types...>>> {
};

template <typename... T>
using python_view_type_t = typename python_view_type<T...>::type;

}  // namespace Impl

//--------------------------------------------------------------------------------------//
//  this is used to extract the uniform type of a view
//
template <typename ViewT>
struct python_view_type {
  static_assert(Kokkos::is_view<std::decay_t<ViewT>>::value ||
                    Kokkos::is_dyn_rank_view<std::decay_t<ViewT>>::value,
                "Error! python_view_type only supports Kokkos::View and "
                "Kokkos::DynRankView");

  using type = Impl::python_view_type_t<typename ViewT::array_type>;
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

#endif
