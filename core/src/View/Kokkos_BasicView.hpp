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

namespace Kokkos {
namespace Impl {}  // namespace Impl

template <class ElementType, class Extents, class LayoutPolicy,
          class AccessorPolicy>
class BasicView {
 public:
  using extents_type  = Extents;
  using layout_type   = LayoutPolicy;
  using accessor_type = AccessorPolicy;
  using mdspan_type =
      mdspan<ElementType, extents_type, layout_type, accessor_type>;
  using index_type       = typename extents_type::index_type;
  using rank_type        = typename extents_type::rank_type;
  using data_handle_type = typename mdspan_type::data_handle_type;
  using reference        = typename mdspan_type::reference;
  using mapping_type     = typename mdspan_type::mapping_type;
  using execution_space  = typename accessor_type::execution_space;
  using memory_space     = typename accessor_type::memory_space;

 private:
  using default_alloc_params =
      Impl::alloc_params<memory_space, execution_space>;

 public:
  static constexpr Impl::integral_constant<rank_type, extents_type::rank()>
      rank = {};
  static constexpr Impl::integral_constant<rank_type,
                                           extents_type::rank_dynamic()>
      rank_dynamic = {};

  ///
  /// \name Constructors
  ///
  ///@{
  KOKKOS_INLINE_FUNCTION constexpr BasicView()                      = default;
  KOKKOS_INLINE_FUNCTION constexpr BasicView(const BasicView &)     = default;
  KOKKOS_INLINE_FUNCTION constexpr BasicView(BasicView &&) noexcept = default;

  KOKKOS_INLINE_FUNCTION constexpr BasicView &operator=(const BasicView &) =
      default;

  KOKKOS_INLINE_FUNCTION constexpr BasicView &operator=(BasicView &&) = default;

  ///
  /// Converting constructor
  ///
  template <class OtherElementType, class OtherExtents, class OtherLayoutPolicy,
            class OtherAccessor,
            typename = std::enable_if_t<std::is_constructible_v<
                mdspan_type, const mdspan<OtherElementType, OtherExtents,
                                          OtherLayoutPolicy, OtherAccessor> &>>>
  KOKKOS_INLINE_FUNCTION constexpr BasicView(
      const BasicView<OtherElementType, OtherExtents, OtherLayoutPolicy,
                      OtherAccessor> &other)
      : m_tracker(other.m_tracker), m_data(other.m_data) {}

  // TODO: subview constructor

  ///
  /// Construct from a given mapping
  ///
  KOKKOS_INLINE_FUNCTION explicit constexpr BasicView(
      const std::string &label, const mapping_type &mapping)
      : BasicView(mapping, label, default_alloc_params{}) {}

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

  KOKKOS_INLINE_FUNCTION constexpr std::string label() const noexcept {
    return m_tracker.get_label<memory_space>();
  }

  KOKKOS_INLINE_FUNCTION constexpr extents_type extents() const noexcept {
    return m_data.extents();
  }

  KOKKOS_INLINE_FUNCTION constexpr index_type extent(
      rank_type r) const noexcept {
    return m_data.extent(r);
  }

  KOKKOS_INLINE_FUNCTION constexpr index_type static_extent(
      rank_type r) const noexcept {
    return m_data.static_extent(r);
  }

  KOKKOS_INLINE_FUNCTION constexpr index_type size() const noexcept {
    return m_data.size();
  }

  KOKKOS_INLINE_FUNCTION constexpr index_type stride(
      rank_type r) const noexcept {
    return m_data.stride(r);
  }

  KOKKOS_INLINE_FUNCTION constexpr index_type span() const noexcept {
    return m_data.mapping().required_span_size();
  }

  KOKKOS_INLINE_FUNCTION constexpr bool span_is_contiguous() const noexcept {
    return m_data.is_exhaustive();
  }

  KOKKOS_INLINE_FUNCTION constexpr const data_handle_type &data()
      const noexcept {
    return m_data.data_handle();
  }

  KOKKOS_INLINE_FUNCTION constexpr bool is_allocated() const noexcept {
    return m_tracker.has_record();
  }

  template <class... OtherIndexTypes,
            class = std::enable_if_t<(sizeof...(OtherIndexTypes) == rank())>>
  KOKKOS_INLINE_FUNCTION constexpr reference operator()(
      OtherIndexTypes... indices) const {
#if KOKKOS_IMPL_USE_MDSPAN_BRACKET
    return m_data[indices...];
#else
    return m_data(indices...);
#endif
  }

#if (defined(__cpp_multidimensional_subscript) && \
     (__cpp_multidimensional_subscript >= 202110L))
  template <class... OtherIndexType,
            class = std::enable_if_t<(sizeof...(OtherIndexType) == rank())>>
  KOKKOS_INLINE_FUNCTION constexpr reference operator[](
      OtherIndexType... Indices) const {
#if KOKKOS_IMPL_USE_MDSPAN_BRACKET
    return m_data[indices...];
#else
    return m_data(indices...);
#endif
  }
#endif

  KOKKOS_INLINE_FUNCTION int use_count() const noexcept {
    return m_tracker.use_count();
  }

 private:
  using tracker_type = Impl::SharedAllocationTracker;

  template <typename M, typename E, bool AllowPadding, bool Initialize>
  KOKKOS_INLINE_FUNCTION constexpr BasicView(
      const mapping_type &layout_mapping, std::string_view label,
      const Impl::alloc_params<M, E, AllowPadding, Initialize> &params)
      : BasicView(layout_mapping, label, params.memory_space,
                  &params.execution_space, params.allow_padding,
                  params.initialize) {}

  template <typename E, bool AllowPadding, bool Initialize>
  KOKKOS_INLINE_FUNCTION constexpr BasicView(
      const mapping_type &layout_mapping, std::string_view label,
      const memory_space &mem_space_instance, const E *exec_space_instance,
      std::integral_constant<bool, AllowPadding> allow_padding,
      std::integral_constant<bool, Initialize> initialize)
      : m_tracker(allocate_record(layout_mapping, label, mem_space_instance,
                                  exec_space_instance, allow_padding,
                                  initialize)),
        m_data(static_cast<data_handle_type>(
                   m_tracker.get_record<memory_space>()->data()),
               layout_mapping) {}

  template <typename E, bool AllowPadding, bool Initialize>
  static KOKKOS_INLINE_FUNCTION constexpr tracker_type allocate_record(
      const mapping_type &layout_mapping, std::string_view label,
      const memory_space &mem_space_instance, const E *exec_space_instance,
      std::integral_constant<bool, AllowPadding> allow_padding,
      std::integral_constant<bool, Initialize> initialize) {
    static_assert(SpaceAccessibility<execution_space, memory_space>::accessible);
    auto *rec = Impl::make_shared_allocation_record<ElementType>(
        layout_mapping, label, mem_space_instance, exec_space_instance,
        allow_padding, initialize);
    tracker_type track;
    track.assign_allocated_record_to_uninitialized(rec);

    return track;
  }

  Impl::SharedAllocationTracker
      m_tracker;  // Must be listed before m_data for allocate_record to work
  mdspan_type m_data;
};
}  // namespace Kokkos

#endif
