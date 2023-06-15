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

#ifndef KOKKOS_VIEW_TRAITS_HPP
#define KOKKOS_VIEW_TRAITS_HPP

#include "Kokkos_ViewFwd.hpp"

namespace Kokkos {
/** \class ViewTraits
 *  \brief Traits class for accessing attributes of a View.
 *
 * This is an implementation detail of View.  It is only of interest
 * to developers implementing a new specialization of View.
 *
 * Template argument options:
 *   - View< DataType >
 *   - View< DataType , Space >
 *   - View< DataType , Space , MemoryTraits >
 *   - View< DataType , ArrayLayout >
 *   - View< DataType , ArrayLayout , Space >
 *   - View< DataType , ArrayLayout , MemoryTraits >
 *   - View< DataType , ArrayLayout , Space , MemoryTraits >
 *   - View< DataType , MemoryTraits >
 */

template <class DataType, class... Properties>
struct ViewTraits;

template <>
struct ViewTraits<void> {
  using execution_space = void;
  using memory_space    = void;
  using HostMirrorSpace = void;
  using array_layout    = void;
  using memory_traits   = void;
  using specialize      = void;
  using hooks_policy    = void;
};

template <class... Prop>
struct ViewTraits<void, void, Prop...> {
  // Ignore an extraneous 'void'
  using execution_space = typename ViewTraits<void, Prop...>::execution_space;
  using memory_space    = typename ViewTraits<void, Prop...>::memory_space;
  using HostMirrorSpace = typename ViewTraits<void, Prop...>::HostMirrorSpace;
  using array_layout    = typename ViewTraits<void, Prop...>::array_layout;
  using memory_traits   = typename ViewTraits<void, Prop...>::memory_traits;
  using specialize      = typename ViewTraits<void, Prop...>::specialize;
  using hooks_policy    = typename ViewTraits<void, Prop...>::hooks_policy;
};

template <class HooksPolicy, class... Prop>
struct ViewTraits<
    std::enable_if_t<Kokkos::Experimental::is_hooks_policy<HooksPolicy>::value>,
    HooksPolicy, Prop...> {
  using execution_space = typename ViewTraits<void, Prop...>::execution_space;
  using memory_space    = typename ViewTraits<void, Prop...>::memory_space;
  using HostMirrorSpace = typename ViewTraits<void, Prop...>::HostMirrorSpace;
  using array_layout    = typename ViewTraits<void, Prop...>::array_layout;
  using memory_traits   = typename ViewTraits<void, Prop...>::memory_traits;
  using specialize      = typename ViewTraits<void, Prop...>::specialize;
  using hooks_policy    = HooksPolicy;
};

template <class ArrayLayout, class... Prop>
struct ViewTraits<std::enable_if_t<Kokkos::is_array_layout<ArrayLayout>::value>,
                  ArrayLayout, Prop...> {
  // Specify layout, keep subsequent space and memory traits arguments

  using execution_space = typename ViewTraits<void, Prop...>::execution_space;
  using memory_space    = typename ViewTraits<void, Prop...>::memory_space;
  using HostMirrorSpace = typename ViewTraits<void, Prop...>::HostMirrorSpace;
  using array_layout    = ArrayLayout;
  using memory_traits   = typename ViewTraits<void, Prop...>::memory_traits;
  using specialize      = typename ViewTraits<void, Prop...>::specialize;
  using hooks_policy    = typename ViewTraits<void, Prop...>::hooks_policy;
};

template <class Space, class... Prop>
struct ViewTraits<std::enable_if_t<Kokkos::is_space<Space>::value>, Space,
                  Prop...> {
  // Specify Space, memory traits should be the only subsequent argument.

  static_assert(
      std::is_same<typename ViewTraits<void, Prop...>::execution_space,
                   void>::value &&
          std::is_same<typename ViewTraits<void, Prop...>::memory_space,
                       void>::value &&
          std::is_same<typename ViewTraits<void, Prop...>::HostMirrorSpace,
                       void>::value &&
          std::is_same<typename ViewTraits<void, Prop...>::array_layout,
                       void>::value,
      "Only one View Execution or Memory Space template argument");

  using execution_space = typename Space::execution_space;
  using memory_space    = typename Space::memory_space;
  using HostMirrorSpace =
      typename Kokkos::Impl::HostMirror<Space>::Space::memory_space;
  using array_layout  = typename execution_space::array_layout;
  using memory_traits = typename ViewTraits<void, Prop...>::memory_traits;
  using specialize    = typename ViewTraits<void, Prop...>::specialize;
  using hooks_policy  = typename ViewTraits<void, Prop...>::hooks_policy;
};

template <class MemoryTraits, class... Prop>
struct ViewTraits<
    std::enable_if_t<Kokkos::is_memory_traits<MemoryTraits>::value>,
    MemoryTraits, Prop...> {
  // Specify memory trait, should not be any subsequent arguments

  static_assert(
      std::is_same<typename ViewTraits<void, Prop...>::execution_space,
                   void>::value &&
          std::is_same<typename ViewTraits<void, Prop...>::memory_space,
                       void>::value &&
          std::is_same<typename ViewTraits<void, Prop...>::array_layout,
                       void>::value &&
          std::is_same<typename ViewTraits<void, Prop...>::memory_traits,
                       void>::value &&
          std::is_same<typename ViewTraits<void, Prop...>::hooks_policy,
                       void>::value,
      "MemoryTrait is the final optional template argument for a View");

  using execution_space = void;
  using memory_space    = void;
  using HostMirrorSpace = void;
  using array_layout    = void;
  using memory_traits   = MemoryTraits;
  using specialize      = void;
  using hooks_policy    = void;
};

template <class DataType, class... Properties>
struct ViewTraits {
 private:
  // Unpack the properties arguments
  using prop = ViewTraits<void, Properties...>;

  using ExecutionSpace =
      std::conditional_t<!std::is_void<typename prop::execution_space>::value,
                         typename prop::execution_space,
                         Kokkos::DefaultExecutionSpace>;

  using MemorySpace =
      std::conditional_t<!std::is_void<typename prop::memory_space>::value,
                         typename prop::memory_space,
                         typename ExecutionSpace::memory_space>;

  using ArrayLayout =
      std::conditional_t<!std::is_void<typename prop::array_layout>::value,
                         typename prop::array_layout,
                         typename ExecutionSpace::array_layout>;

  using HostMirrorSpace = std::conditional_t<
      !std::is_void<typename prop::HostMirrorSpace>::value,
      typename prop::HostMirrorSpace,
      typename Kokkos::Impl::HostMirror<ExecutionSpace>::Space>;

  using MemoryTraits =
      std::conditional_t<!std::is_void<typename prop::memory_traits>::value,
                         typename prop::memory_traits,
                         typename Kokkos::MemoryManaged>;

  using HooksPolicy =
      std::conditional_t<!std::is_void<typename prop::hooks_policy>::value,
                         typename prop::hooks_policy,
                         Kokkos::Experimental::DefaultViewHooks>;

  // Analyze data type's properties,
  // May be specialized based upon the layout and value type
  using data_analysis = Kokkos::Impl::ViewDataAnalysis<DataType, ArrayLayout>;

 public:
  //------------------------------------
  // Data type traits:

  using data_type           = typename data_analysis::type;
  using const_data_type     = typename data_analysis::const_type;
  using non_const_data_type = typename data_analysis::non_const_type;

  //------------------------------------
  // Compatible array of trivial type traits:

  using scalar_array_type = typename data_analysis::scalar_array_type;
  using const_scalar_array_type =
      typename data_analysis::const_scalar_array_type;
  using non_const_scalar_array_type =
      typename data_analysis::non_const_scalar_array_type;

  //------------------------------------
  // Value type traits:

  using value_type           = typename data_analysis::value_type;
  using const_value_type     = typename data_analysis::const_value_type;
  using non_const_value_type = typename data_analysis::non_const_value_type;

  //------------------------------------
  // Mapping traits:

  using array_layout = ArrayLayout;
  using dimension    = typename data_analysis::dimension;

  using specialize = std::conditional_t<
      std::is_void<typename data_analysis::specialize>::value,
      typename prop::specialize,
      typename data_analysis::specialize>; /* mapping specialization tag */

  static constexpr unsigned rank         = dimension::rank;
  static constexpr unsigned rank_dynamic = dimension::rank_dynamic;

  //------------------------------------
  // Execution space, memory space, memory access traits, and host mirror space.

  using execution_space   = ExecutionSpace;
  using memory_space      = MemorySpace;
  using device_type       = Kokkos::Device<ExecutionSpace, MemorySpace>;
  using memory_traits     = MemoryTraits;
  using host_mirror_space = HostMirrorSpace;
  using hooks_policy      = HooksPolicy;

  using size_type = typename MemorySpace::size_type;

  enum { is_hostspace = std::is_same<MemorySpace, HostSpace>::value };
  enum { is_managed = MemoryTraits::is_unmanaged == 0 };
  enum { is_random_access = MemoryTraits::is_random_access == 1 };

  //------------------------------------
};

template <class T1, class T2>
struct is_always_assignable_impl;

template <class... ViewTDst, class... ViewTSrc>
struct is_always_assignable_impl<Kokkos::View<ViewTDst...>,
                                 Kokkos::View<ViewTSrc...>> {
  using mapping_type = Kokkos::Impl::ViewMapping<
      typename Kokkos::View<ViewTDst...>::traits,
      typename Kokkos::View<ViewTSrc...>::traits,
      typename Kokkos::View<ViewTDst...>::traits::specialize>;

  constexpr static bool value =
      mapping_type::is_assignable &&
      static_cast<int>(Kokkos::View<ViewTDst...>::rank_dynamic) >=
          static_cast<int>(Kokkos::View<ViewTSrc...>::rank_dynamic);
};

template <class View1, class View2>
using is_always_assignable = is_always_assignable_impl<
    std::remove_reference_t<View1>,
    std::remove_const_t<std::remove_reference_t<View2>>>;

template <class T1, class T2>
inline constexpr bool is_always_assignable_v =
    is_always_assignable<T1, T2>::value;

template <class... ViewTDst, class... ViewTSrc>
constexpr bool is_assignable(const Kokkos::View<ViewTDst...>& dst,
                             const Kokkos::View<ViewTSrc...>& src) {
  using DstTraits = typename Kokkos::View<ViewTDst...>::traits;
  using SrcTraits = typename Kokkos::View<ViewTSrc...>::traits;
  using mapping_type =
      Kokkos::Impl::ViewMapping<DstTraits, SrcTraits,
                                typename DstTraits::specialize>;

  return is_always_assignable_v<Kokkos::View<ViewTDst...>,
                                Kokkos::View<ViewTSrc...>> ||
         (mapping_type::is_assignable &&
          ((DstTraits::dimension::rank_dynamic >= 1) ||
           (dst.static_extent(0) == src.extent(0))) &&
          ((DstTraits::dimension::rank_dynamic >= 2) ||
           (dst.static_extent(1) == src.extent(1))) &&
          ((DstTraits::dimension::rank_dynamic >= 3) ||
           (dst.static_extent(2) == src.extent(2))) &&
          ((DstTraits::dimension::rank_dynamic >= 4) ||
           (dst.static_extent(3) == src.extent(3))) &&
          ((DstTraits::dimension::rank_dynamic >= 5) ||
           (dst.static_extent(4) == src.extent(4))) &&
          ((DstTraits::dimension::rank_dynamic >= 6) ||
           (dst.static_extent(5) == src.extent(5))) &&
          ((DstTraits::dimension::rank_dynamic >= 7) ||
           (dst.static_extent(6) == src.extent(6))) &&
          ((DstTraits::dimension::rank_dynamic >= 8) ||
           (dst.static_extent(7) == src.extent(7))));
}

} /* namespace Kokkos */

#endif  /* #ifndef KOKKOS_VIEW_TRAITS_HPP */
