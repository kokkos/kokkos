// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include "submdspan_extents.hpp"
#include "submdspan_canonicalize_slices.hpp"
#include "submdspan_mapping.hpp"

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
template <class ElementType, class Extents, class LayoutPolicy,
          class AccessorPolicy, class... SliceSpecifiers>
MDSPAN_INLINE_FUNCTION
constexpr auto
submdspan(const mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy> &src,
          SliceSpecifiers... slices) {
  detail::check_submdspan_slice_mandates<Extents>(std::make_index_sequence<Extents::rank()>(), slices...);

  const auto sub_submdspan_mapping_result = submdspan_mapping(src.mapping(),
        detail::canonical_slice<typename Extents::index_type>(slices)...);
  // NVCC has a problem with the deduction so lets figure out the type
  using sub_mapping_t = std::remove_cv_t<decltype(sub_submdspan_mapping_result.mapping)>;
  using sub_extents_t = typename sub_mapping_t::extents_type;
  using sub_layout_t = typename sub_mapping_t::layout_type;
  using sub_accessor_t = typename AccessorPolicy::offset_policy;
  return mdspan<ElementType, sub_extents_t, sub_layout_t, sub_accessor_t>(
      src.accessor().offset(src.data_handle(), sub_submdspan_mapping_result.offset),
      sub_submdspan_mapping_result.mapping,
      sub_accessor_t(src.accessor()));
}
} // namespace MDSPAN_IMPL_STANDARD_NAMESPACE
