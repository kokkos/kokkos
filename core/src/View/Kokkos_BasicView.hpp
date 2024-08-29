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
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif

#ifndef KOKKOS_BASIC_VIEW_HPP
#define KOKKOS_BASIC_VIEW_HPP
#include <type_traits>
#include <Kokkos_Macros.hpp>
#include <impl/Kokkos_Utilities.hpp>
#include <impl/Kokkos_SharedAlloc.hpp>
#include "Kokkos_ViewAlloc.hpp"
#include "MDSpan/Kokkos_MDSpan_Accessor.hpp"
#include "MDSpan/Kokkos_MDSpan_Header.hpp"
#include <View/Kokkos_ViewCtor.hpp>
#include <View/Kokkos_ViewTraits.hpp>

namespace Kokkos {

namespace Impl {
constexpr inline struct subview_ctor_tag_t {
} subview_ctor_tag;

template <class T>
struct KokkosSliceToMDSpanSliceImpl {
  using type = T;
  static constexpr decltype(auto) transform(const T &s) { return s; }
};

template <>
struct KokkosSliceToMDSpanSliceImpl<Kokkos::ALL_t> {
  using type = full_extent_t;
  static constexpr decltype(auto) transform(Kokkos::ALL_t) {
    return full_extent;
  }
};

template <class T>
using kokkos_slice_to_mdspan_slice =
    typename KokkosSliceToMDSpanSliceImpl<T>::type;

template <class T>
constexpr decltype(auto) transform_kokkos_slice_to_mdspan_slice(const T &s) {
  return KokkosSliceToMDSpanSliceImpl<T>::transform(s);
}
}  // namespace Impl

template <class ElementType, class Extents, class LayoutPolicy,
          class AccessorPolicy>
class BasicView
    : public mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy> {
 public:
  using mdspan_type =
      mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>;
  using extents_type     = typename mdspan_type::extents_type;
  using layout_type      = typename mdspan_type::layout_type;
  using accessor_type    = typename mdspan_type::accessor_type;
  using mapping_type     = typename mdspan_type::mapping_type;
  using element_type     = typename mdspan_type::element_type;
  using value_type       = typename mdspan_type::value_type;
  using index_type       = typename mdspan_type::index_type;
  using size_type        = typename mdspan_type::size_type;
  using rank_type        = typename mdspan_type::rank_type;
  using data_handle_type = typename mdspan_type::data_handle_type;
  using reference        = typename mdspan_type::reference;
  using memory_space     = typename accessor_type::memory_space;
  using execution_space  = typename memory_space::execution_space;

  ///
  /// \name Constructors
  ///
  ///@{
  using mdspan_type::mdspan;

  ///
  /// Construct from a given mapping
  ///
  KOKKOS_INLINE_FUNCTION explicit constexpr BasicView(
      const std::string &label, const mapping_type &mapping)
      : BasicView(mapping, label, memory_space{}, execution_space{},
                  std::false_type{}, std::false_type{}, std::false_type{}) {}

  ///
  /// Construct from a given extents
  ///
  KOKKOS_INLINE_FUNCTION explicit constexpr BasicView(const std::string &label,
                                                      const extents_type &ext)
      : BasicView(label, mapping_type{ext}) {}

  template <typename... OtherIndexTypes>
  KOKKOS_INLINE_FUNCTION explicit constexpr BasicView(
      const std::enable_if_t<
          std::is_constructible_v<extents_type, OtherIndexTypes...>,
          std::string> &label,
      OtherIndexTypes... indices)
      : BasicView(label, extents_type{indices...}) {}
  ///@}

 private:
  template <class... P>
  data_handle_type create_data_handle(
      const Impl::ViewCtorProp<P...> &arg_prop,
      const typename mdspan_type::mapping_type &arg_mapping) {
    // Copy the input allocation properties with possibly defaulted properties
    // We need to split it in two to avoid MSVC compiler errors
    auto prop_copy_tmp =
        Impl::with_properties_if_unset(arg_prop, std::string{});
    auto prop_copy = Impl::with_properties_if_unset(
        prop_copy_tmp, memory_space{}, execution_space{});
    using alloc_prop = decltype(prop_copy);

    if (alloc_prop::initialize &&
        !alloc_prop::execution_space::impl_is_initialized()) {
      // If initializing view data then
      // the execution space must be initialized.
      Kokkos::Impl::throw_runtime_exception(
          "Constructing View and initializing data with uninitialized "
          "execution space");
    }
    return data_handle_type(Impl::make_shared_allocation_record<ElementType>(
        arg_mapping, Impl::get_property<Impl::LabelTag>(prop_copy),
        Impl::get_property<Impl::MemorySpaceTag>(prop_copy),
        &Impl::get_property<Impl::ExecutionSpaceTag>(prop_copy),
        std::integral_constant<bool, alloc_prop::allow_padding>(),
        std::integral_constant<bool, alloc_prop::initialize>(),
        std::integral_constant<bool, alloc_prop::sequential_host_init>()));
  }

 public:
  template <class... P>
  explicit inline BasicView(
      const Impl::ViewCtorProp<P...> &arg_prop,
      std::enable_if_t<!Impl::ViewCtorProp<P...>::has_pointer,
                       typename mdspan_type::mapping_type> const &arg_mapping)
      : mdspan_type(create_data_handle(arg_prop, arg_mapping), arg_mapping) {}

  template <class... P>
  explicit inline BasicView(
      const Impl::ViewCtorProp<P...> &arg_prop,
      std::enable_if_t<Impl::ViewCtorProp<P...>::has_pointer,
                       typename mdspan_type::mapping_type> const &arg_mapping)
      : mdspan_type(
            data_handle_type(Impl::get_property<Impl::PointerTag>(arg_prop)),
            arg_mapping) {}

 protected:
  template <class OtherElementType, class OtherExtents, class OtherLayoutPolicy,
            class OtherAccessorPolicy, class... SliceSpecifiers>
  KOKKOS_INLINE_FUNCTION BasicView(
      Impl::subview_ctor_tag_t,
      const BasicView<OtherElementType, OtherExtents, OtherLayoutPolicy,
                      OtherAccessorPolicy> &src_view,
      SliceSpecifiers... slices)
      : mdspan_type(submdspan(
            src_view,
            Impl::transform_kokkos_slice_to_mdspan_slice(slices)...)) {}

 public:
  //----------------------------------------
  // Conversion to MDSpan
  template <class OtherElementType, class OtherExtents, class OtherLayoutPolicy,
            class OtherAccessor,
            typename = std::enable_if_t<
                std::is_assignable_v<mdspan<OtherElementType, OtherExtents,
                                            OtherLayoutPolicy, OtherAccessor>,
                                     mdspan_type>>>
  KOKKOS_INLINE_FUNCTION constexpr operator mdspan<
      OtherElementType, OtherExtents, OtherLayoutPolicy, OtherAccessor>() {
    return mdspan_type(*this);
  }

  // Here we use an overload instead of a default parameter as a workaround
  // to a potential compiler bug with clang 17. It may be present in other
  // compilers
  template <class OtherAccessorType = AccessorPolicy,
            typename                = std::enable_if_t<std::is_assignable_v<
                typename mdspan_type::data_handle_type,
                typename OtherAccessorType::data_handle_type>>>
  KOKKOS_INLINE_FUNCTION constexpr auto to_mdspan() {
    using ret_mdspan_type =
        mdspan<typename mdspan_type::element_type,
               typename mdspan_type::extents_type,
               typename mdspan_type::layout_type, OtherAccessorType>;
    return ret_mdspan_type(
        static_cast<typename OtherAccessorType::data_handle_type>(
            mdspan_type::data_handle()),
        mdspan_type::mapping(),
        static_cast<OtherAccessorType>(mdspan_type::accessor()));
  }

  template <class OtherAccessorType = AccessorPolicy,
            typename                = std::enable_if_t<std::is_assignable_v<
                typename mdspan_type::data_handle_type,
                typename OtherAccessorType::data_handle_type>>>
  KOKKOS_INLINE_FUNCTION constexpr auto to_mdspan(
      const OtherAccessorType &other_accessor) {
    using ret_mdspan_type =
        mdspan<typename mdspan_type::element_type,
               typename mdspan_type::extents_type,
               typename mdspan_type::layout_type, OtherAccessorType>;
    return ret_mdspan_type(
        static_cast<typename OtherAccessorType::data_handle_type>(
            mdspan_type::data_handle()),
        mdspan_type::mapping(), other_accessor);
  }

  void assign_data(element_type *ptr) {
    mdspan_type::operator=(
        mdspan_type{typename mdspan_type::data_handle_type(ptr),
                    mdspan_type::mapping(), mdspan_type::accessor()});
  }

 private:
  template <typename E, bool AllowPadding, bool Initialize,
            bool SequentialHostInit>
  KOKKOS_INLINE_FUNCTION constexpr BasicView(
      const mapping_type &layout_mapping, std::string_view label,
      const memory_space &mem_space_instance, const E &exec_space_instance,
      std::integral_constant<bool, AllowPadding> allow_padding,
      std::integral_constant<bool, Initialize> initialize,
      std::integral_constant<bool, SequentialHostInit> sequential_host_init)
      : mdspan_type(
            data_handle_type(Impl::make_shared_allocation_record<ElementType>(
                layout_mapping, label, mem_space_instance, &exec_space_instance,
                allow_padding, initialize, sequential_host_init)),
            layout_mapping) {}
};
}  // namespace Kokkos

#endif
