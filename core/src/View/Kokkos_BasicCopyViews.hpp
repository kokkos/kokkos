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

#ifndef KOKKOS_BASIC_COPYVIEWS_HPP
#define KOKKOS_BASIC_COPYVIEWS_HPP

#include <type_traits>
#include <sstream>
#include "Kokkos_BasicView.hpp"
#include <Kokkos_CopyViews.hpp>

namespace Kokkos {
namespace Impl {

  template <class IndexType, std::size_t... Extents>
  class print_extents
  {
  public:

    using extents_type = Kokkos::extents<IndexType, Extents...>;
    print_extents(const extents_type &extents)
      : m_extents(extents) {}

    friend std::ostream &operator<<(std::ostream &strm, const print_extents &pe) {
      return print_impl(strm, std::make_index_sequence<decltype(extents)::rank());
    }

  private:

    template <std::size_t... Indices>
    std::ostream &print_impl(std::ostream &strm, std::index_sequence<Indices...>) {
      return _strm << '(' << (print_extent<Indices>(m_extents.extent(Indices)) << ...) << ')';
    }

    template <std::size_t Index>
    class print_extent {
    public:

      print_extent(IndexType ext)
        : m_ext(ext)
      {}

      std::ostream &operator<<(std::ostream &strm, const print_extent &pe) {
        if constexpr (IndexType == 0) {
          return strm << m_ext;
        } else {
          return strm << ", " << m_ext;
        }
      }
    private:

      IndexType m_ext;
    };

    const extents_type &m_extents;
  };

  template <class L, class R>
  constexpr bool strides_equal(const L &lhs, const R &rhs) {
    static_assert(L::rank() == R::rank());
    using rank_type = typename L::rank_type;
    for (rank_type r = 0; r < L::rank(); ++r) {
      if (lhs.stride(r) != rhs.stride(r))
        return false;
    }
    return true;
  }
  }
}  // namespace Impl

///
/// A deep copy between views of the default specialization, compatible
/// type, same non-zero rank, same contiguous layout.
///
template <class DstElementType, class DstExtents, class DstExecutionSpace, class DstMemorySpace, class DstLayoutPolicy,
          class SrcElementType, class SrcExtents, class SrcExecutionSpace, class SrcMemorySpace, class SrcLayoutPolicy>
inline std::enable_if_t<(DstExtents::rank() != 0) && (SrcExtents::rank() != 0)> deep_copy(
    const BasicView<DstElementType, DstExtents, DstExecutionSpace, DstMemorySpace, DstLayoutPolicy> &dst,
    const BasicView<SrcElementType, SrcExtents, SrcExecutionSpace, SrcMemorySpace, SrcLayoutPolicy>& src) {
  using dst_type            = BasicView<DstElementType, DstExtents, DstExecutionSpace, DstMemorySpace, DstLayoutPolicy>;
  using src_type            = BasicView<SrcElementType, SrcExtents, SrcExecutionSpace, SrcMemorySpace, SrcLayoutPolicy>;
  using dst_execution_space = DstExecutionSpace;
  using src_execution_space = SrcExecutionSpace;
  using dst_memory_space    = DstMemorySpace;
  using src_memory_space    = SrcMemorySpace;
  using dst_element_type    = DstElementType;
  using src_element_type    = SrcElementType;
  using src_extents         = SrcExtents;
  using dst_extents         = DstExtents;

  static_assert(!std::is_const_v<dst_element_type>, "deep_copy requires non-const destination type");

  static_assert(src_extents::rank() == dst_extents::rank(), "deep_copy requires Views of equal rank");

  if (Kokkos::Tools::Experimental::get_callbacks().begin_deep_copy != nullptr) {
    Kokkos::Profiling::beginDeepCopy(
        Kokkos::Profiling::make_space_handle(dst_memory_space::name()),
        dst.label(), dst.data(),
        Kokkos::Profiling::make_space_handle(src_memory_space::name()),
        src.label(), src.data(),
        src.span() * sizeof(typename dst_type::value_type));
  }

  // throw if dimension mismatch
  if (dst.extents() != src.extents()) {
    std::stringstream message;
    message << "Deprecation Error: Kokkos::deep_copy extents of views don't match: ";
    message << "dst view: " << dst.label() << Impl::print_extents(dst.extents());
    message << " and src view: " << src.label() << Impl::print_extents(src.extents());
    Kokkos::Impl::throw_runtime_exception(message.str());
  }

  if (!dst.is_allocated() || !src.is_allocated()) {
    Kokkos::fence(
        "Kokkos::deep_copy: copy between contiguous views, fence due to null "
        "argument");
    if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
      Kokkos::Profiling::endDeepCopy();
    }
    return;
  }

  constexpr bool dst_can_access_src =
      Kokkos::SpaceAccessibility<dst_execution_space,
                                 src_memory_space>::accessible;

  constexpr bool src_can_access_dst =
      Kokkos::SpaceAccessibility<src_execution_space,
                                 dst_memory_space>::accessible;

  // Checking for Overlapping Views.
  dst_value_type* dst_start = dst.data();
  dst_value_type* dst_end   = dst.data() + dst.span();
  src_value_type* src_start = src.data();
  src_value_type* src_end   = src.data() + src.span();
  if (((std::ptrdiff_t)dst_start == (std::ptrdiff_t)src_start) &&
      ((std::ptrdiff_t)dst_end == (std::ptrdiff_t)src_end) &&
      (dst.span_is_contiguous() && src.span_is_contiguous())) {
    Kokkos::fence(
        "Kokkos::deep_copy: copy between contiguous views, fence due to same "
        "spans");
    if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
      Kokkos::Profiling::endDeepCopy();
    }
    return;
  }

  if ((((std::ptrdiff_t)dst_start < (std::ptrdiff_t)src_end) &&
       ((std::ptrdiff_t)dst_end > (std::ptrdiff_t)src_start)) &&
      ((dst.span_is_contiguous() && src.span_is_contiguous()))) {
    std::stringstream message;
    message << "Error: Kokkos::deep_copy of overlapping views: ";
    message << dst.label() << '(' << std::to_string(static_cast<std::ptrdiff_t>(dst_start))
            << ", " << std::to_string(static_cast<std::ptrdiff_t>(dst_end_) << ") ";
    message << src.label() << '(' << std::to_string(static_cast<std::ptrdiff_t>(src_start))
            << ", " << std::to_string(static_cast<std::ptrdiff_t>(src_end)) << ')';
    Kokkos::Impl::throw_runtime_exception(message.str());
  }

  // If same type, equal layout, equal dimensions, equal span, and contiguous
  // memory then can byte-wise copy

  if (std::is_same<dst_element_type, std::remove_const_t<src_element_type>>
      && (std::is_same<typename dst_type::array_layout,
                    typename src_type::array_layout>::value ||
       (dst_type::rank() == 1)) &&
      dst.span_is_contiguous() && src.span_is_contiguous() &&
      ((dst_type::rank < 1) || (dst.stride_0() == src.stride_0())) &&
      ((dst_type::rank < 2) || (dst.stride_1() == src.stride_1())) &&
      ((dst_type::rank < 3) || (dst.stride_2() == src.stride_2())) &&
      ((dst_type::rank < 4) || (dst.stride_3() == src.stride_3())) &&
      ((dst_type::rank < 5) || (dst.stride_4() == src.stride_4())) &&
      ((dst_type::rank < 6) || (dst.stride_5() == src.stride_5())) &&
      ((dst_type::rank < 7) || (dst.stride_6() == src.stride_6())) &&
      ((dst_type::rank < 8) || (dst.stride_7() == src.stride_7()))) {
    const size_t nbytes = sizeof(typename dst_type::value_type) * dst.span();
    Kokkos::fence(
        "Kokkos::deep_copy: copy between contiguous views, pre view equality "
        "check");
    if ((void*)dst.data() != (void*)src.data()) {
      Kokkos::Impl::DeepCopy<dst_memory_space, src_memory_space>(
          dst.data(), src.data(), nbytes);
      Kokkos::fence(
          "Kokkos::deep_copy: copy between contiguous views, post deep copy "
          "fence");
    }
  } else {
    Kokkos::fence(
        "Kokkos::deep_copy: copy between contiguous views, pre copy fence");
    Impl::view_copy(dst, src);
    Kokkos::fence(
        "Kokkos::deep_copy: copy between contiguous views, post copy fence");
  }
  if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
    Kokkos::Profiling::endDeepCopy();
  }
}
}  // namespace Kokkos

#endif  // KOKKOS_BASIC_COPYVIEWS_HPP