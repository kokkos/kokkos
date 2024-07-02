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
#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_IMPL_VIEW_LEGACY
#include <View/Kokkos_View_legacy.hpp>
#else
#ifndef KOKKOS_VIEW_HPP
#define KOKKOS_VIEW_HPP

#include <View/Kokkos_BasicView.hpp>

// Class to provide a uniform type
namespace Kokkos {
namespace Impl {
template <class ViewType, int Traits = 0>
struct ViewUniformType;
} /* namespace Impl */

template <class T1, class T2>
struct is_always_assignable_impl;

template <class... ViewTDst, class... ViewTSrc>
struct is_always_assignable_impl<Kokkos::View<ViewTDst...>,
                                 Kokkos::View<ViewTSrc...> > {
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
    std::remove_const_t<std::remove_reference_t<View2> > >;

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
                                Kokkos::View<ViewTSrc...> > ||
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

namespace Kokkos {

template <class DataType, class... Properties>
class View;

template <class>
struct is_view : public std::false_type {};

template <class D, class... P>
struct is_view<View<D, P...> > : public std::true_type {};

template <class D, class... P>
struct is_view<const View<D, P...> > : public std::true_type {};

template <class T>
inline constexpr bool is_view_v = is_view<T>::value;

template <class DataType, class... Properties>
class View : public BasicView<DataType, Properties...> {
 private:
  template <class, class...>
  friend class View;
  template <class, class...>
  friend class BasicView;
  template <class, class...>
  friend class Kokkos::Impl::ViewMapping;
  template <typename V>
  friend struct Kokkos::Impl::ViewTracker;

  using base_t = BasicView<DataType, Properties...>;

 public:
  using traits            = ViewTraits<DataType, Properties...>;
  using mdspan_type       = typename base_t::mdspan_type;
  using view_tracker_type = typename base_t::view_tracker_type;
  using pointer_type      = typename base_t::pointer_type;
  using reference_type    = typename base_t::reference_type;

  //----------------------------------------
  /** \brief  Compatible view of array of scalar types */
  using array_type =
      View<typename traits::scalar_array_type, typename traits::array_layout,
           typename traits::device_type, typename traits::hooks_policy,
           typename traits::memory_traits>;

  /** \brief  Compatible view of const data type */
  using const_type =
      View<typename traits::const_data_type, typename traits::array_layout,
           typename traits::device_type, typename traits::hooks_policy,
           typename traits::memory_traits>;

  /** \brief  Compatible view of non-const data type */
  using non_const_type =
      View<typename traits::non_const_data_type, typename traits::array_layout,
           typename traits::device_type, typename traits::hooks_policy,
           typename traits::memory_traits>;

  /** \brief  Compatible HostMirror view */
  using HostMirror =
      View<typename traits::non_const_data_type, typename traits::array_layout,
           Device<DefaultHostExecutionSpace,
                  typename traits::host_mirror_space::memory_space>,
           typename traits::hooks_policy>;

  /** \brief  Compatible HostMirror view */
  using host_mirror_type =
      View<typename traits::non_const_data_type, typename traits::array_layout,
           typename traits::host_mirror_space, typename traits::hooks_policy>;

  /** \brief Unified types */
  using uniform_type = typename Impl::ViewUniformType<View, 0>::type;
  using uniform_const_type =
      typename Impl::ViewUniformType<View, 0>::const_type;
  using uniform_runtime_type =
      typename Impl::ViewUniformType<View, 0>::runtime_type;
  using uniform_runtime_const_type =
      typename Impl::ViewUniformType<View, 0>::runtime_const_type;
  using uniform_nomemspace_type =
      typename Impl::ViewUniformType<View, 0>::nomemspace_type;
  using uniform_const_nomemspace_type =
      typename Impl::ViewUniformType<View, 0>::const_nomemspace_type;
  using uniform_runtime_nomemspace_type =
      typename Impl::ViewUniformType<View, 0>::runtime_nomemspace_type;
  using uniform_runtime_const_nomemspace_type =
      typename Impl::ViewUniformType<View, 0>::runtime_const_nomemspace_type;

  //----------------------------------------
  // Domain rank and extents

  static constexpr Impl::integral_constant<size_t, base_t::rank()> rank = {};
  static constexpr Impl::integral_constant<size_t, base_t::rank_dynamic()>
      rank_dynamic = {};
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4
  enum {Rank KOKKOS_DEPRECATED_WITH_COMMENT("Use rank instead.") = rank()};
#endif

  KOKKOS_INLINE_FUNCTION constexpr typename traits::array_layout layout()
      const {
    return base_t::m_map.layout();
  }

  KOKKOS_INLINE_FUNCTION constexpr size_t stride_0() const { return stride(0); }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_1() const { return stride(1); }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_2() const { return stride(2); }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_3() const { return stride(3); }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_4() const { return stride(4); }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_5() const { return stride(5); }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_6() const { return stride(6); }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_7() const { return stride(7); }

  template <typename iType>
  KOKKOS_INLINE_FUNCTION constexpr std::enable_if_t<
      std::is_integral<iType>::value, size_t>
  stride(iType r) const {
    // base class doesn't have constraint
    return base_t::stride(r);
  }

  template <typename iType>
  KOKKOS_INLINE_FUNCTION void stride(iType* const s) const {
    base_t::m_map.stride(s);
  }

  //----------------------------------------
  // Range span is the span which contains all members.

  enum {
    reference_type_is_lvalue_reference =
        std::is_lvalue_reference<reference_type>::value
  };

  //  KOKKOS_INLINE_FUNCTION constexpr size_t span() const { return
  //  required_span_size(); } KOKKOS_INLINE_FUNCTION bool span_is_contiguous()
  //  const {
  //    return is_exhaustive();
  //  }
  KOKKOS_INLINE_FUNCTION constexpr bool is_allocated() const {
    return data() != nullptr;
  }
  KOKKOS_INLINE_FUNCTION constexpr pointer_type data() const {
    return base_t::m_map.data();
  }

  //----------------------------------------
  // Allow specializations to query their specialized map

  KOKKOS_INLINE_FUNCTION
  const Kokkos::Impl::ViewMapping<traits, typename traits::specialize>&
  impl_map() const {
    return base_t::m_map;
  }
  KOKKOS_INLINE_FUNCTION
  const Kokkos::Impl::SharedAllocationTracker& impl_track() const {
    return base_t::m_track.m_tracker;
  }
  //----------------------------------------

 private:

#ifdef KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK
  template <typename... Is>
  static KOKKOS_FUNCTION void check_access_member_function_valid_args(
      Is... is) {
    static_assert(sizeof...(Is) <= 8 - rank);
    static_assert(Kokkos::Impl::are_integral<Is...>::value);
    if (!((is == static_cast<IS>(0)) && ... && true))
      Kokkos::abort("Extra arguments to Kokkos::access must be zero");
  }
#else
  template <typename... Is>
  static KOKKOS_FUNCTION void check_access_member_function_valid_args(Is...) {
    static_assert(sizeof...(Is) <= 8 - rank);
    static_assert(Kokkos::Impl::are_integral<Is...>::value);
  }
#endif

 public:
  //------------------------------
  // Rank 0

  template <typename... Is>
  KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<
      (Kokkos::Impl::always_true<Is...>::value && (0 == rank)), reference_type>
  access(Is... extra) const {
    check_access_member_function_valid_args(extra...);
    return base_t::operator()();
  }

  //------------------------------
  // Rank 1

  template <typename I0, typename... Is>
  KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<
      (Kokkos::Impl::always_true<I0, Is...>::value && (1 == rank)),
      reference_type>
  access(I0 i0, Is... extra) const {
    check_access_member_function_valid_args(extra...);
    return base_t::operator()(i0);
  }

  //------------------------------
  // Rank 2

  template <typename I0, typename I1, typename... Is>
  KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<
      (Kokkos::Impl::always_true<I0, I1, Is...>::value && (2 == rank)),
      reference_type>
  access(I0 i0, I1 i1, Is... extra) const {
    check_access_member_function_valid_args(extra...);
    return base_t::operator()(i0, i1);
  }

  //------------------------------
  // Rank 3

  template <typename I0, typename I1, typename I2, typename... Is>
  KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<
      (Kokkos::Impl::always_true<I0, I1, I2, Is...>::value && (3 == rank)),
      reference_type>
  access(I0 i0, I1 i1, I2 i2, Is... extra) const {
    check_access_member_function_valid_args(extra...);
    return base_t::operator()(i0, i1, i2);
  }

  //------------------------------
  // Rank 4

  template <typename I0, typename I1, typename I2, typename I3, typename... Is>
  KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<
      (Kokkos::Impl::always_true<I0, I1, I2, I3, Is...>::value && (4 == rank)),
      reference_type>
  access(I0 i0, I1 i1, I2 i2, I3 i3, Is... extra) const {
    check_access_member_function_valid_args(extra...);
    return base_t::operator()(i0, i1, i2, i3);
  }

  //------------------------------
  // Rank 5

  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename... Is>
  KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<
      (Kokkos::Impl::always_true<I0, I1, I2, I3, I4, Is...>::value &&
       (5 == rank)),
      reference_type>
  access(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4, Is... extra) const {
    check_access_member_function_valid_args(extra...);
    return base_t::operator()(i0, i1, i2, i3, i4);
  }

  //------------------------------
  // Rank 6

  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5, typename... Is>
  KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<
      (Kokkos::Impl::always_true<I0, I1, I2, I3, I4, I5, Is...>::value &&
       (6 == rank)),
      reference_type>
  access(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4, I5 i5, Is... extra) const {
    check_access_member_function_valid_args(extra...);
    return base_t::operator()(i0, i1, i2, i3, i4, i5);
  }

  //------------------------------
  // Rank 7

  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5, typename I6, typename... Is>
  KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<
      (Kokkos::Impl::always_true<I0, I1, I2, I3, I4, I5, I6, Is...>::value &&
       (7 == rank)),
      reference_type>
  access(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4, I5 i5, I6 i6, Is... extra) const {
    check_access_member_function_valid_args(extra...);
    return base_t::operator()(i0, i1, i2, i3, i4, i5, i6);
  }

  //------------------------------
  // Rank 8

  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5, typename I6, typename I7, typename... Is>
  KOKKOS_FORCEINLINE_FUNCTION
      std::enable_if_t<(Kokkos::Impl::always_true<I0, I1, I2, I3, I4, I5, I6,
                                                  I7, Is...>::value &&
                        (8 == rank)),
                       reference_type>
      access(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4, I5 i5, I6 i6, I7 i7,
             Is... extra) const {
    check_access_member_function_valid_args(extra...);
    return base_t::operator()(i0, i1, i2, i3, i4, i5, i6, i7);
  }

#undef KOKKOS_IMPL_VIEW_OPERATOR_VERIFY

  //----------------------------------------
  // Standard destructor, constructors, and assignment operators

  KOKKOS_DEFAULTED_FUNCTION
  ~View() = default;

  KOKKOS_DEFAULTED_FUNCTION
  View() = default;

  KOKKOS_FUNCTION
  View(const View& other) : base_t(other) {}

  KOKKOS_FUNCTION
  View(View&& other) : base_t(other) {}

  KOKKOS_FUNCTION
  View& operator=(const View& other) {
    base_t::operator=(other);
    return *this;
  }

  KOKKOS_FUNCTION
  View& operator=(View&& other) {
    base_t::operator=(other);
    return *this;
  }

  //----------------------------------------
  // Compatible view copy constructor and assignment
  // may assign unmanaged from managed.

  template <class RT, class... RP>
  KOKKOS_INLINE_FUNCTION View(
      const View<RT, RP...>& rhs,
      std::enable_if_t<Kokkos::Impl::ViewMapping<
          traits, typename View<RT, RP...>::traits,
          typename traits::specialize>::is_assignable_data_type>* = nullptr)
      : base_t(rhs) {}

  template <class RT, class... RP>
  KOKKOS_INLINE_FUNCTION std::enable_if_t<
      Kokkos::Impl::ViewMapping<
          traits, typename View<RT, RP...>::traits,
          typename traits::specialize>::is_assignable_data_type,
      View>&
  operator=(const View<RT, RP...>& rhs) {
    base_t::operator=(rhs);
    return *this;
  }

  //----------------------------------------
  // Compatible subview constructor
  // may assign unmanaged from managed.

  template <class RT, class... RP, class Arg0, class... Args>
  KOKKOS_INLINE_FUNCTION View(const View<RT, RP...>& src_view, const Arg0 arg0,
                              Args... args)
      : base_t(src_view, arg0, args...) {}

 public:
  //----------------------------------------
  // Allocation according to allocation properties and array layout

  template <class... P>
  explicit inline View(
      const Impl::ViewCtorProp<P...>& arg_prop,
      std::enable_if_t<!Impl::ViewCtorProp<P...>::has_pointer,
                       typename traits::array_layout> const& arg_layout)
      : base_t(arg_prop, arg_layout) {}

  // Wrap memory according to properties and array layout
  template <class... P>
  explicit KOKKOS_INLINE_FUNCTION View(
      const Impl::ViewCtorProp<P...>& arg_prop,
      std::enable_if_t<Impl::ViewCtorProp<P...>::has_pointer,
                       typename traits::array_layout> const& arg_layout)
      : base_t(arg_prop, arg_layout) {}

  // Simple dimension-only layout
  template <class... P>
  explicit inline View(
      const Impl::ViewCtorProp<P...>& arg_prop,
      std::enable_if_t<!Impl::ViewCtorProp<P...>::has_pointer, size_t> const
          arg_N0          = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG)
      : View(arg_prop,
             typename traits::array_layout(arg_N0, arg_N1, arg_N2, arg_N3,
                                           arg_N4, arg_N5, arg_N6, arg_N7)) {
    static_assert(traits::array_layout::is_extent_constructible,
                  "Layout is not constructible from extent arguments. Use "
                  "overload taking a layout object instead.");
  }

  template <class... P>
  explicit KOKKOS_INLINE_FUNCTION View(
      const Impl::ViewCtorProp<P...>& arg_prop,
      std::enable_if_t<Impl::ViewCtorProp<P...>::has_pointer, size_t> const
          arg_N0          = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG)
      : View(arg_prop,
             typename traits::array_layout(arg_N0, arg_N1, arg_N2, arg_N3,
                                           arg_N4, arg_N5, arg_N6, arg_N7)) {
    static_assert(traits::array_layout::is_extent_constructible,
                  "Layout is not constructible from extent arguments. Use "
                  "overload taking a layout object instead.");
  }

  // Allocate with label and layout
  template <typename Label>
  explicit inline View(
      const Label& arg_label,
      std::enable_if_t<Kokkos::Impl::is_view_label<Label>::value,
                       typename traits::array_layout> const& arg_layout)
      : View(Impl::ViewCtorProp<std::string>(arg_label), arg_layout) {}

  // Allocate label and layout, must disambiguate from subview constructor.
  template <typename Label>
  explicit inline View(
      const Label& arg_label,
      std::enable_if_t<Kokkos::Impl::is_view_label<Label>::value, const size_t>
          arg_N0          = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG)
      : View(Impl::ViewCtorProp<std::string>(arg_label),
             typename traits::array_layout(arg_N0, arg_N1, arg_N2, arg_N3,
                                           arg_N4, arg_N5, arg_N6, arg_N7)) {
    static_assert(traits::array_layout::is_extent_constructible,
                  "Layout is not constructible from extent arguments. Use "
                  "overload taking a layout object instead.");
  }

  // Construct view from ViewTracker and map
  // This should be the preferred method because future extensions may need to
  // use the ViewTracker class.
  template <class Traits>
  KOKKOS_INLINE_FUNCTION View(
      const view_tracker_type& track,
      const Kokkos::Impl::ViewMapping<Traits, typename Traits::specialize>& map)
      : base_t(track, map) {}

  // Construct View from internal shared allocation tracker object and map
  // This is here for backwards compatibility for classes that derive from
  // Kokkos::View
  template <class Traits>
  KOKKOS_INLINE_FUNCTION View(
      const typename view_tracker_type::track_type& track,
      const Kokkos::Impl::ViewMapping<Traits, typename Traits::specialize>& map)
      : base_t(track, map) {}

  //----------------------------------------
  // Memory span required to wrap these dimensions.
  static constexpr size_t required_allocation_size(
      typename traits::array_layout const& layout) {
    return base_t::map_type::memory_span(layout);
  }

  static constexpr size_t required_allocation_size(
      const size_t arg_N0 = 0, const size_t arg_N1 = 0, const size_t arg_N2 = 0,
      const size_t arg_N3 = 0, const size_t arg_N4 = 0, const size_t arg_N5 = 0,
      const size_t arg_N6 = 0, const size_t arg_N7 = 0) {
    static_assert(traits::array_layout::is_extent_constructible,
                  "Layout is not constructible from extent arguments. Use "
                  "overload taking a layout object instead.");
    return base_t::map_type::memory_span(typename traits::array_layout(
        arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5, arg_N6, arg_N7));
  }

  explicit KOKKOS_INLINE_FUNCTION View(
      pointer_type arg_ptr, const size_t arg_N0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG)
      : View(Impl::ViewCtorProp<pointer_type>(arg_ptr),
             typename traits::array_layout(arg_N0, arg_N1, arg_N2, arg_N3,
                                           arg_N4, arg_N5, arg_N6, arg_N7)) {
    static_assert(traits::array_layout::is_extent_constructible,
                  "Layout is not constructible from extent arguments. Use "
                  "overload taking a layout object instead.");
  }

  explicit KOKKOS_INLINE_FUNCTION View(
      pointer_type arg_ptr, const typename traits::array_layout& arg_layout)
      : View(Impl::ViewCtorProp<pointer_type>(arg_ptr), arg_layout) {}

  //----------------------------------------
  // Shared scratch memory constructor

  static KOKKOS_INLINE_FUNCTION size_t
  shmem_size(const size_t arg_N0 = KOKKOS_INVALID_INDEX,
             const size_t arg_N1 = KOKKOS_INVALID_INDEX,
             const size_t arg_N2 = KOKKOS_INVALID_INDEX,
             const size_t arg_N3 = KOKKOS_INVALID_INDEX,
             const size_t arg_N4 = KOKKOS_INVALID_INDEX,
             const size_t arg_N5 = KOKKOS_INVALID_INDEX,
             const size_t arg_N6 = KOKKOS_INVALID_INDEX,
             const size_t arg_N7 = KOKKOS_INVALID_INDEX) {
    static_assert(traits::array_layout::is_extent_constructible,
                  "Layout is not constructible from extent arguments. Use "
                  "overload taking a layout object instead.");
    const size_t num_passed_args = Impl::count_valid_integers(
        arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5, arg_N6, arg_N7);

    if (std::is_void<typename traits::specialize>::value &&
        num_passed_args != rank_dynamic) {
      Kokkos::abort(
          "Kokkos::View::shmem_size() rank_dynamic != number of arguments.\n");
    }

    return View::shmem_size(typename traits::array_layout(
        arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5, arg_N6, arg_N7));
  }

 private:
  // Want to be able to align to minimum scratch alignment or sizeof or alignof
  // elements
  static constexpr size_t scratch_value_alignment =
      max({sizeof(typename traits::value_type),
           alignof(typename traits::value_type),
           static_cast<size_t>(
               traits::execution_space::scratch_memory_space::ALIGN)});

 public:
  static KOKKOS_INLINE_FUNCTION size_t
  shmem_size(typename traits::array_layout const& arg_layout) {
    return base_t::map_type::memory_span(arg_layout) + scratch_value_alignment;
  }

  explicit KOKKOS_INLINE_FUNCTION View(
      const typename traits::execution_space::scratch_memory_space& arg_space,
      const typename traits::array_layout& arg_layout)
      : View(Impl::ViewCtorProp<pointer_type>(
                 reinterpret_cast<pointer_type>(arg_space.get_shmem_aligned(
                     base_t::map_type::memory_span(arg_layout),
                     scratch_value_alignment))),
             arg_layout) {}

  explicit KOKKOS_INLINE_FUNCTION View(
      const typename traits::execution_space::scratch_memory_space& arg_space,
      const size_t arg_N0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG)
      : View(
            Impl::ViewCtorProp<pointer_type>(
                reinterpret_cast<pointer_type>(arg_space.get_shmem_aligned(
                    base_t::map_type::memory_span(typename traits::array_layout(
                        arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5, arg_N6,
                        arg_N7)),
                    scratch_value_alignment))),
            typename traits::array_layout(arg_N0, arg_N1, arg_N2, arg_N3,
                                          arg_N4, arg_N5, arg_N6, arg_N7)) {
    static_assert(traits::array_layout::is_extent_constructible,
                  "Layout is not constructible from extent arguments. Use "
                  "overload taking a layout object instead.");
  }

  //----------------------------------------
  // MDSpan converting constructors
  template <typename U = typename Impl::MDSpanViewTraits<traits>::mdspan_type>
  KOKKOS_INLINE_FUNCTION
#if defined(__cpp_conditional_explicit) && \
    (__cpp_conditional_explicit >= 201806L)
      // FIXME C++20 reevaluate after determining minium compiler versions
      explicit(traits::is_managed)
#endif
          View(const typename Impl::MDSpanViewTraits<traits>::mdspan_type& mds,
               std::enable_if_t<
                   !std::is_same_v<Impl::UnsupportedKokkosArrayLayout, U> >* =
                   nullptr)
      : View(mds.data_handle(),
             Impl::array_layout_from_mapping<
                 typename traits::array_layout,
                 typename Impl::MDSpanViewTraits<traits>::mdspan_type>(
                 mds.mapping())) {
  }

  template <class ElementType, class ExtentsType, class LayoutType,
            class AccessorType>
  KOKKOS_INLINE_FUNCTION
#if defined(__cpp_conditional_explicit) && \
    (__cpp_conditional_explicit >= 201806L)
      // FIXME C++20 reevaluate after determining minium compiler versions
      explicit(!std::is_convertible_v<
               Kokkos::mdspan<ElementType, ExtentsType, LayoutType,
                              AccessorType>,
               typename Impl::MDSpanViewTraits<traits>::mdspan_type>)
#endif
          View(const Kokkos::mdspan<ElementType, ExtentsType, LayoutType,
                                    AccessorType>& mds)
      : View(typename Impl::MDSpanViewTraits<traits>::mdspan_type(mds)) {
  }
};

template <typename D, class... P>
KOKKOS_INLINE_FUNCTION constexpr unsigned rank(const View<D, P...>&) {
  return View<D, P...>::rank();
}

namespace Impl {

template <typename ValueType, unsigned int Rank>
struct RankDataType {
  using type = typename RankDataType<ValueType, Rank - 1>::type*;
};

template <typename ValueType>
struct RankDataType<ValueType, 0> {
  using type = ValueType;
};

template <unsigned N, typename... Args>
KOKKOS_FUNCTION std::enable_if_t<
    N == View<Args...>::rank() &&
        std::is_same<typename ViewTraits<Args...>::specialize, void>::value,
    View<Args...> >
as_view_of_rank_n(View<Args...> v) {
  return v;
}

// Placeholder implementation to compile generic code for DynRankView; should
// never be called
template <unsigned N, typename T, typename... Args>
KOKKOS_FUNCTION std::enable_if_t<
    N != View<T, Args...>::rank() &&
        std::is_same<typename ViewTraits<T, Args...>::specialize, void>::value,
    View<typename RankDataType<typename View<T, Args...>::value_type, N>::type,
         Args...> >
as_view_of_rank_n(View<T, Args...>) {
  Kokkos::abort("Trying to get at a View of the wrong rank");
  return {};
}

template <typename Function, typename... Args>
void apply_to_view_of_static_rank(Function&& f, View<Args...> a) {
  f(a);
}

}  // namespace Impl
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

template <class D, class... P, class... Args>
KOKKOS_INLINE_FUNCTION auto subview(const View<D, P...>& src, Args... args) {
  static_assert(View<D, P...>::rank == sizeof...(Args),
                "subview requires one argument for each source View rank");

  return typename Kokkos::Impl::ViewMapping<
      void /* deduce subview type from source view traits */
      ,
      typename Impl::RemoveAlignedMemoryTrait<D, P...>::type,
      Args...>::type(src, args...);
}

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4
template <class MemoryTraits, class D, class... P, class... Args>
KOKKOS_DEPRECATED KOKKOS_INLINE_FUNCTION auto subview(const View<D, P...>& src,
                                                      Args... args) {
  static_assert(View<D, P...>::rank == sizeof...(Args),
                "subview requires one argument for each source View rank");
  static_assert(Kokkos::is_memory_traits<MemoryTraits>::value);

  return typename Kokkos::Impl::ViewMapping<
      void /* deduce subview type from source view traits */
      ,
      typename Impl::RemoveAlignedMemoryTrait<D, P..., MemoryTraits>::type,
      Args...>::type(src, args...);
}
#endif

template <class V, class... Args>
using Subview = decltype(subview(std::declval<V>(), std::declval<Args>()...));

} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

template <class LT, class... LP, class RT, class... RP>
KOKKOS_INLINE_FUNCTION bool operator==(const View<LT, LP...>& lhs,
                                       const View<RT, RP...>& rhs) {
  // Same data, layout, dimensions
  using lhs_traits = ViewTraits<LT, LP...>;
  using rhs_traits = ViewTraits<RT, RP...>;

  return std::is_same<typename lhs_traits::const_value_type,
                      typename rhs_traits::const_value_type>::value &&
         std::is_same<typename lhs_traits::array_layout,
                      typename rhs_traits::array_layout>::value &&
         std::is_same<typename lhs_traits::memory_space,
                      typename rhs_traits::memory_space>::value &&
         View<LT, LP...>::rank() == View<RT, RP...>::rank() &&
         lhs.data() == rhs.data() && lhs.span() == rhs.span() &&
         lhs.extent(0) == rhs.extent(0) && lhs.extent(1) == rhs.extent(1) &&
         lhs.extent(2) == rhs.extent(2) && lhs.extent(3) == rhs.extent(3) &&
         lhs.extent(4) == rhs.extent(4) && lhs.extent(5) == rhs.extent(5) &&
         lhs.extent(6) == rhs.extent(6) && lhs.extent(7) == rhs.extent(7);
}

template <class LT, class... LP, class RT, class... RP>
KOKKOS_INLINE_FUNCTION bool operator!=(const View<LT, LP...>& lhs,
                                       const View<RT, RP...>& rhs) {
  return !(operator==(lhs, rhs));
}

} /* namespace Kokkos */

#include <impl/Kokkos_ViewUniformType.hpp>
#include <impl/Kokkos_Atomic_View.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #ifndef KOKKOS_VIEW_HPP */
#endif /* KOKKOS_ENABLE_IMPL_VIEW_LEGACY */
