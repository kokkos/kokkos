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

#ifndef KOKKOS_VIEW_HPP
#define KOKKOS_VIEW_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_IMPL_MDSPAN) && !defined(KOKKOS_COMPILER_INTEL)
#include <View/Kokkos_BasicView.hpp>
#endif
#ifdef KOKKOS_ENABLE_IMPL_VIEW_LEGACY
#include <View/Kokkos_ViewLegacy.hpp>
#else

#include <View/Kokkos_ViewTraits.hpp>
#include <Kokkos_MemoryTraits.hpp>

// This will eventually be removed
namespace Kokkos::Impl {
template <class, class...>
class ViewMapping;
}
#include <View/Kokkos_ViewMapping.hpp>
#include <Kokkos_MinMax.hpp>

// Class to provide a uniform type
namespace Kokkos {
namespace Impl {
template <class ViewType, int Traits>
struct ViewUniformType;

template <class ParentView>
struct ViewTracker;
} /* namespace Impl */

template <class T1, class T2>
struct is_always_assignable_impl;

template <class... ViewTDst, class... ViewTSrc>
struct is_always_assignable_impl<Kokkos::View<ViewTDst...>,
                                 Kokkos::View<ViewTSrc...> > {
  using dst_mdspan = typename Kokkos::View<ViewTDst...>::mdspan_type;
  using src_mdspan = typename Kokkos::View<ViewTSrc...>::mdspan_type;

  constexpr static bool value =
      std::is_constructible_v<dst_mdspan, src_mdspan> &&
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
  using dst_mdspan = typename Kokkos::View<ViewTDst...>::mdspan_type;
  using src_mdspan = typename Kokkos::View<ViewTSrc...>::mdspan_type;

  return is_always_assignable_v<Kokkos::View<ViewTDst...>,
                                Kokkos::View<ViewTSrc...> > ||
         (std::is_constructible_v<dst_mdspan, src_mdspan> &&
          ((dst_mdspan::rank_dynamic() >= 1) ||
           (dst.static_extent(0) == src.extent(0))) &&
          ((dst_mdspan::rank_dynamic() >= 2) ||
           (dst.static_extent(1) == src.extent(1))) &&
          ((dst_mdspan::rank_dynamic() >= 3) ||
           (dst.static_extent(2) == src.extent(2))) &&
          ((dst_mdspan::rank_dynamic() >= 4) ||
           (dst.static_extent(3) == src.extent(3))) &&
          ((dst_mdspan::rank_dynamic() >= 5) ||
           (dst.static_extent(4) == src.extent(4))) &&
          ((dst_mdspan::rank_dynamic() >= 6) ||
           (dst.static_extent(5) == src.extent(5))) &&
          ((dst_mdspan::rank_dynamic() >= 7) ||
           (dst.static_extent(6) == src.extent(6))) &&
          ((dst_mdspan::rank_dynamic() == 8) ||
           (dst.static_extent(7) == src.extent(7))));
}

namespace Impl {
template <class... Properties>
struct BasicViewFromTraits {
  using view_traits        = ViewTraits<Properties...>;
  using mdspan_view_traits = MDSpanViewTraits<view_traits>;
  using element_type       = typename view_traits::value_type;
  using extents_type       = typename mdspan_view_traits::extents_type;
  using layout_type        = typename mdspan_view_traits::mdspan_layout_type;
  using accessor_type      = typename mdspan_view_traits::accessor_type;

  using type =
      BV::BasicView<element_type, extents_type, layout_type, accessor_type>;
};
}  // namespace Impl

template <class DataType, class... Properties>
struct ViewTraits;

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
class View : public Impl::BasicViewFromTraits<DataType, Properties...>::type {
 private:
  template <class, class...>
  friend class View;
  template <typename V>
  friend struct Kokkos::Impl::ViewTracker;

  using base_t =
      typename Impl::BasicViewFromTraits<DataType, Properties...>::type;

 public:
  using base_t::base_t;

  // typedefs originally from ViewTraits
  using traits               = ViewTraits<DataType, Properties...>;
  using const_value_type     = typename traits::const_value_type;
  using non_const_value_type = typename traits::non_const_value_type;
  using data_type            = DataType;
  using const_data_type      = typename traits::const_data_type;
  using non_const_data_type  = typename traits::non_const_data_type;
  using view_tracker_type    = Impl::ViewTracker<View>;
  using array_layout         = typename traits::array_layout;
  using device_type          = typename traits::device_type;
  using execution_space      = typename traits::execution_space;
  using memory_space         = typename traits::memory_space;
  using memory_traits        = typename traits::memory_traits;
  using host_mirror_space    = typename traits::host_mirror_space;
  using index_type           = typename traits::memory_space::size_type;

  // aliases from BasicView

  // FIXME: Should be unsigned
  // FIXME: these are overriden so that their types are identical when using
  // BasicView or Legacy we will need to obtain these from base_t in the future
  // and deprecate old behavior
  using size_type    = typename memory_space::size_type;
  using value_type   = typename traits::value_type;
  using pointer_type = typename traits::value_type*;

  using scalar_array_type       = typename traits::scalar_array_type;
  using const_scalar_array_type = typename traits::const_scalar_array_type;
  using non_const_scalar_array_type =
      typename traits::non_const_scalar_array_type;

  // typedefs from BasicView
  using mdspan_type    = typename base_t::mdspan_type;
  using reference_type = typename base_t::reference;

  //----------------------------------------
  // Compatible view of array of scalar types
  using array_type =
      View<typename traits::scalar_array_type, typename traits::array_layout,
           typename traits::device_type, typename traits::hooks_policy,
           typename traits::memory_traits>;

  // Compatible view of const data type
  using const_type =
      View<typename traits::const_data_type, typename traits::array_layout,
           typename traits::device_type, typename traits::hooks_policy,
           typename traits::memory_traits>;

  // Compatible view of non-const data type
  using non_const_type =
      View<typename traits::non_const_data_type, typename traits::array_layout,
           typename traits::device_type, typename traits::hooks_policy,
           typename traits::memory_traits>;

  // Compatible HostMirror view
  using host_mirror_type =
      View<typename traits::non_const_data_type, typename traits::array_layout,
           Device<DefaultHostExecutionSpace,
                  typename traits::host_mirror_space::memory_space>,
           typename traits::hooks_policy>;

  // Compatible HostMirror view
  using HostMirror = host_mirror_type;

  // Unified types
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

  KOKKOS_INLINE_FUNCTION constexpr array_layout layout() const {
    return Impl::array_layout_from_mapping<array_layout, mdspan_type>(
        base_t::mapping());
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
  KOKKOS_INLINE_FUNCTION constexpr std::enable_if_t<std::is_integral_v<iType>,
                                                    size_t>
  stride(iType r) const {
    // base class doesn't have constraint
    // FIXME: Eventually we need to deprecate this behavior and just use
    // BasicView implementation
    return base_t::stride(r);
  }

  template <typename iType>
  KOKKOS_INLINE_FUNCTION void stride(iType* const s) const {
    if constexpr (rank() > 0) {
      for (size_t r = 0; r < rank(); r++) s[r] = base_t::stride(r);
      s[rank()] = s[rank() - 1] * base_t::extent(rank() - 1);
    }
  }

  //----------------------------------------
  // Range span is the span which contains all members.

  static constexpr auto reference_type_is_lvalue_reference =
      std::is_lvalue_reference_v<reference_type>;

  KOKKOS_INLINE_FUNCTION constexpr size_t span() const {
    return base_t::mapping().required_span_size();
  }
  KOKKOS_INLINE_FUNCTION bool span_is_contiguous() const {
    return base_t::is_exhaustive();
  }
  KOKKOS_INLINE_FUNCTION constexpr bool is_allocated() const {
    return data() != nullptr;
  }
  KOKKOS_INLINE_FUNCTION constexpr pointer_type data() const {
    return static_cast<pointer_type>(base_t::data_handle());
  }

  KOKKOS_INLINE_FUNCTION constexpr int extent_int(size_t r) const {
    return static_cast<int>(base_t::extent(r));
  }
  //----------------------------------------
  // Allow specializations to query their specialized map

  KOKKOS_INLINE_FUNCTION
  auto impl_map() const {
    using map_type =
        Kokkos::Impl::ViewMapping<traits, typename traits::specialize>;
    using offset_type = typename map_type::offset_type;
    return map_type(
        data(), offset_type(std::integral_constant<unsigned, 0>(), layout()));
  }

  KOKKOS_INLINE_FUNCTION
  const Kokkos::Impl::SharedAllocationTracker& impl_track() const {
    if constexpr (traits::is_managed) {
      return base_t::data_handle().tracker();
    } else {
      static const Kokkos::Impl::SharedAllocationTracker empty_tracker = {};
      return empty_tracker;
    }
  }
  //----------------------------------------
  // Operators always provided by View

  template <class OtherIndexType>
  KOKKOS_FUNCTION constexpr reference_type operator[](
      const OtherIndexType& idx) const {
    return base_t::operator()(idx);
  }

 private:

#ifdef KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK
  template <typename... Is>
  static KOKKOS_FUNCTION void check_access_member_function_valid_args(
      Is... is) {
    static_assert(sizeof...(Is) <= 8 - rank);
    static_assert(Kokkos::Impl::are_integral<Is...>::value);
    if (!((is == static_cast<Is>(0)) && ... && true))
      Kokkos::abort("Extra arguments to Kokkos::access must be zero");
  }
#else
  template <typename... Is>
  static KOKKOS_FUNCTION void check_access_member_function_valid_args(Is...) {
    // cast to int to work around pointless comparison of unsigned to 0 warning
    static_assert(static_cast<int>(sizeof...(Is)) <=
                  static_cast<int>(8 - rank));
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

  KOKKOS_FUNCTION
  View(typename base_t::data_handle_type p,
       const typename base_t::mapping_type& m)
      : base_t(p, m){};

  //----------------------------------------
  // Compatible view copy constructor and assignment
  // may assign unmanaged from managed.

  template <class OtherT, class... OtherArgs>
  //    requires(std::is_constructible_v<
  //             mdspan_type, typename View<OtherT, OtherArgs...>::mdspan_type>)
  KOKKOS_INLINE_FUNCTION View(
      const View<OtherT, OtherArgs...>& other,
      std::enable_if_t<
          std::is_constructible_v<
              mdspan_type, typename View<OtherT, OtherArgs...>::mdspan_type>,
          void*> = nullptr)
      : base_t(static_cast<typename mdspan_type::data_handle_type>(
                   other.data_handle()),
               static_cast<typename mdspan_type::mapping_type>(other.mapping()),
               static_cast<typename mdspan_type::accessor_type>(
                   other.accessor())) {
    base_t::check_basic_view_constructibility(other.mapping());
  }

  //----------------------------------------
  // Compatible subview constructor
  // may assign unmanaged from managed.

  template <class RT, class... RP, class Arg0, class... Args>
  KOKKOS_INLINE_FUNCTION View(const View<RT, RP...>& src_view, const Arg0 arg0,
                              Args... args)
      : base_t(Impl::subview_ctor_tag, src_view, arg0, args...) {}

 public:
  //----------------------------------------
  // Allocation according to allocation properties and array layout

  template <class... P>
  explicit inline View(const Impl::ViewCtorProp<P...>& arg_prop,
                       std::enable_if_t<!Impl::ViewCtorProp<P...>::has_pointer,
                                        const typename traits::array_layout&>
                           arg_layout)
      : base_t(
            arg_prop,
            Impl::mapping_from_array_layout<typename mdspan_type::mapping_type>(
                arg_layout)) {}

  template <class... P>
  KOKKOS_FUNCTION explicit inline View(
      const Impl::ViewCtorProp<P...>& arg_prop,
      std::enable_if_t<Impl::ViewCtorProp<P...>::has_pointer,
                       const typename traits::array_layout&>
          arg_layout)
      : base_t(
            arg_prop,
            Impl::mapping_from_array_layout<typename mdspan_type::mapping_type>(
                arg_layout)) {}

  template <class... P>
  KOKKOS_FUNCTION explicit inline View(
      const typename base_t::data_handle_type& handle,
      typename traits::array_layout const& arg_layout)
      : base_t(
            handle,
            Impl::mapping_from_array_layout<typename mdspan_type::mapping_type>(
                arg_layout)) {}

#ifdef KOKKOS_ENABLE_CXX17
  template <class Layout>
  KOKKOS_FUNCTION explicit inline View(
      const typename base_t::data_handle_type& handle, const Layout& arg_layout,
      std::enable_if_t<
          (std::is_same_v<Layout, LayoutStride> &&
           std::is_same_v<typename base_t::layout_type, layout_stride>) ||
              (std::is_same_v<Layout, LayoutLeft> &&
               std::is_same_v<typename base_t::layout_type, layout_left>) ||
              (std::is_same_v<Layout, LayoutLeft> &&
               std::is_same_v<typename base_t::layout_type,
                              Experimental::layout_left_padded<> >) ||
              (std::is_same_v<Layout, LayoutRight> &&
               std::is_same_v<typename base_t::layout_type, layout_right>) ||
              (std::is_same_v<Layout, LayoutRight> &&
               std::is_same_v<typename base_t::layout_type,
                              Experimental::layout_right_padded<> >),
          void*> = nullptr)
      : base_t(
            handle,
            Impl::mapping_from_array_layout<typename mdspan_type::mapping_type>(
                arg_layout)) {}
#else
  // Constructors from legacy layouts when using Views of the new layouts
  // LayoutLeft -> layout_left, layout_left_padded
  // LayoutRight -> layout_right, layout_right_padded
  // LayoutStride -> layout_stride
  KOKKOS_FUNCTION
  explicit inline View(const base_t::data_handle_type& handle,
                       const LayoutStride& arg_layout)
    requires(std::is_same_v<typename base_t::layout_type, layout_stride>)
      : base_t(
            handle,
            Impl::mapping_from_array_layout<typename mdspan_type::mapping_type>(
                arg_layout)) {}

  KOKKOS_FUNCTION
  explicit inline View(const base_t::data_handle_type& handle,
                       const LayoutLeft& arg_layout)
    requires(std::is_same_v<typename base_t::layout_type,
                            Experimental::layout_left_padded<> >)
      : base_t(
            handle,
            Impl::mapping_from_array_layout<typename mdspan_type::mapping_type>(
                arg_layout)) {}

  KOKKOS_FUNCTION
  explicit inline View(const base_t::data_handle_type& handle,
                       const LayoutRight& arg_layout)
    requires(std::is_same_v<typename base_t::layout_type,
                            Experimental::layout_right_padded<> >)
      : base_t(
            handle,
            Impl::mapping_from_array_layout<typename mdspan_type::mapping_type>(
                arg_layout)) {}

  KOKKOS_FUNCTION
  explicit inline View(const base_t::data_handle_type& handle,
                       const LayoutLeft& arg_layout)
    requires(std::is_same_v<typename base_t::layout_type, layout_left>)
      : base_t(
            handle,
            Impl::mapping_from_array_layout<typename mdspan_type::mapping_type>(
                arg_layout)) {}

  KOKKOS_FUNCTION
  explicit inline View(const base_t::data_handle_type& handle,
                       const LayoutRight& arg_layout)
    requires(std::is_same_v<typename base_t::layout_type, layout_right>)
      : base_t(
            handle,
            Impl::mapping_from_array_layout<typename mdspan_type::mapping_type>(
                arg_layout)) {}
#endif

#ifndef KOKKOS_ENABLE_CXX17
  template <class P, class... Args>
    requires(std::is_convertible_v<P, pointer_type>)
  KOKKOS_FUNCTION View(P ptr_, Args... args)
      : View(Kokkos::view_wrap(static_cast<pointer_type>(ptr_)), args...) {}

  // Special function to be preferred over the above for string literals
  // when pointer type is char*
  template <class L, class... Args>
    requires(std::is_same_v<pointer_type, char*> &&
             std::is_same_v<const char*, L>)
  explicit View(L label, Args... args)
      : View(Kokkos::view_alloc(std::string(label)), args...) {}

  // Special function to be preferred over the above for passing in 0, NULL or
  // nullptr when pointer type is char*
  template <class... Args>
  explicit View(decltype(nullptr), Args... args)
      : View(Kokkos::view_wrap(pointer_type(nullptr)), args...) {}
#else
  template <
      class P, class... Args,
      std::enable_if_t<std::is_convertible_v<P, pointer_type>, size_t> = 0ul>
  KOKKOS_FUNCTION View(P ptr_, Args... args)
      : View(Kokkos::view_wrap(static_cast<pointer_type>(ptr_)), args...) {}

  // Special function to be preferred over the above for string literals
  // when pointer type is char*
  template <class L, class... Args,
            std::enable_if_t<(std::is_same_v<pointer_type, char*> &&
                              std::is_same_v<const char*, L>),
                             size_t> = 0ul>
  explicit View(L label, Args... args)
      : View(Kokkos::view_alloc(std::string(label)), args...) {}

  // Special function to be preferred over the above for passing in 0, NULL or
  // nullptr when pointer type is char*
  template <class... Args>
  explicit View(decltype(nullptr), Args... args)
      : View(Kokkos::view_wrap(pointer_type(nullptr)), args...) {}
#endif

  // FIXME: Constructor which allows always 8 sizes should be deprecated
  template <class... P>
  explicit inline View(
      const Impl::ViewCtorProp<P...>& arg_prop,
      std::enable_if_t<!Impl::ViewCtorProp<P...>::has_pointer, const size_t>
          arg_N0          = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG)
      : base_t(arg_prop,
               Impl::mapping_from_ctor_and_8sizes<
                   typename mdspan_type::mapping_type, sizeof(value_type)>(
                   arg_prop, arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5,
                   arg_N6, arg_N7)) {
    static_assert(traits::array_layout::is_extent_constructible,
                  "Layout is not constructible from extent arguments. Use "
                  "overload taking a layout object instead.");
  }

  template <class... P>
  KOKKOS_FUNCTION explicit inline View(
      const Impl::ViewCtorProp<P...>& arg_prop,
      std::enable_if_t<Impl::ViewCtorProp<P...>::has_pointer, const size_t>
          arg_N0          = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG)
      : base_t(arg_prop,
               Impl::mapping_from_ctor_and_8sizes<
                   typename mdspan_type::mapping_type, sizeof(value_type)>(
                   arg_prop, arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5,
                   arg_N6, arg_N7)) {
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
  explicit inline View(const std::string& arg_label,
                       const size_t arg_N0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
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

  //----------------------------------------
  // Memory span required to wrap these dimensions.
  KOKKOS_FUNCTION
  static constexpr size_t required_allocation_size(
      typename traits::array_layout const& layout) {
    return Impl::mapping_from_array_layout<typename base_t::mapping_type>(
               layout)
               .required_span_size() *
           sizeof(value_type);
  }

  KOKKOS_FUNCTION
  static constexpr size_t required_allocation_size(
      const size_t arg_N0 = 0, const size_t arg_N1 = 0, const size_t arg_N2 = 0,
      const size_t arg_N3 = 0, const size_t arg_N4 = 0, const size_t arg_N5 = 0,
      const size_t arg_N6 = 0, const size_t arg_N7 = 0) {
    static_assert(traits::array_layout::is_extent_constructible,
                  "Layout is not constructible from extent arguments. Use "
                  "overload taking a layout object instead.");
    return required_allocation_size(typename traits::array_layout(
        arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5, arg_N6, arg_N7));
  }

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

    if (std::is_void_v<typename traits::specialize> &&
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
    return Impl::mapping_from_array_layout<typename base_t::mapping_type>(
               arg_layout)
                   .required_span_size() *
               sizeof(value_type) +
           scratch_value_alignment;
  }

  explicit KOKKOS_INLINE_FUNCTION View(
      const typename traits::execution_space::scratch_memory_space& arg_space,
      const typename traits::array_layout& arg_layout)
      : View(Impl::ViewCtorProp<pointer_type>(
                 static_cast<pointer_type>(arg_space.get_shmem_aligned(
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
      : View(Impl::ViewCtorProp<pointer_type>(
                 static_cast<pointer_type>(arg_space.get_shmem_aligned(
                     required_allocation_size(typename traits::array_layout(
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
#ifndef KOKKOS_ENABLE_CXX17
      explicit(traits::is_managed)
#endif
          View(const typename Impl::MDSpanViewTraits<traits>::mdspan_type& mds,
               std::enable_if_t<
                   !std::is_same_v<Impl::UnsupportedKokkosArrayLayout, U> >* =
                   nullptr)
      : base_t(mds) {
  }

  template <class ElementType, class ExtentsType, class LayoutType,
            class AccessorType>
  KOKKOS_INLINE_FUNCTION
#ifndef KOKKOS_ENABLE_CXX17
      explicit(!std::is_convertible_v<
               Kokkos::mdspan<ElementType, ExtentsType, LayoutType,
                              AccessorType>,
               typename Impl::MDSpanViewTraits<traits>::mdspan_type>)
#endif
          View(const Kokkos::mdspan<ElementType, ExtentsType, LayoutType,
                                    AccessorType>& mds,
               std::enable_if_t<
                   std::is_constructible_v<
                       base_t, Kokkos::mdspan<ElementType, ExtentsType,
                                              LayoutType, AccessorType> >,
                   void*> = nullptr)
      : base_t(mds) {
  }

 public:
  //----------------------------------------
  // Allocation tracking properties
  std::string label() const {
    if constexpr (traits::is_managed) {
      return this->data_handle().get_label();
    } else {
      return "";
    }
  }

  int use_count() const {
    if constexpr (traits::is_managed) {
      return this->data_handle().use_count();
    } else {
      return 0;
    }
  }

  KOKKOS_FUNCTION
  constexpr typename base_t::index_type extent(size_t r) const noexcept {
    // casting to int to avoid warning for pointless comparison of unsigned
    // with 0
    if (static_cast<int>(r) >= static_cast<int>(base_t::extents_type::rank()))
      return 1;
    return base_t::extent(r);
  }
  KOKKOS_FUNCTION
  static constexpr size_t static_extent(size_t r) noexcept {
    // casting to int to avoid warning for pointless comparison of unsigned
    // with 0
    if (static_cast<int>(r) >= static_cast<int>(base_t::extents_type::rank()))
      return 1;
    size_t value = base_t::extents_type::static_extent(r);
    return value == Kokkos::dynamic_extent ? 0 : value;
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
        std::is_same_v<typename ViewTraits<Args...>::specialize, void>,
    View<Args...> >
as_view_of_rank_n(View<Args...> v) {
  return v;
}

// Placeholder implementation to compile generic code for DynRankView; should
// never be called
template <unsigned N, typename T, typename... Args>
KOKKOS_FUNCTION std::enable_if_t<
    N != View<T, Args...>::rank() &&
        std::is_same_v<typename ViewTraits<T, Args...>::specialize, void>,
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

  return std::is_same_v<typename lhs_traits::const_value_type,
                        typename rhs_traits::const_value_type> &&
         std::is_same_v<typename lhs_traits::array_layout,
                        typename rhs_traits::array_layout> &&
         std::is_same_v<typename lhs_traits::memory_space,
                        typename rhs_traits::memory_space> &&
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

#include <View/Kokkos_ViewUniformType.hpp>
#include <View/Kokkos_ViewAtomic.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* KOKKOS_ENABLE_IMPL_VIEW_LEGACY */
#endif /* #ifndef KOKKOS_VIEW_HPP */
