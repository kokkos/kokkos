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

#define KOKKOS_NO_UNIQUE_ADDRESS _MDSPAN_NO_UNIQUE_ADDRESS
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
class BasicView {
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

  KOKKOS_FUNCTION static constexpr rank_type rank() noexcept {
    return extents_type::rank();
  }
  KOKKOS_FUNCTION static constexpr rank_type rank_dynamic() noexcept {
    return extents_type::rank_dynamic();
  }
  KOKKOS_FUNCTION static constexpr size_t static_extent(rank_type r) noexcept {
    return extents_type::static_extent(r);
  }
  KOKKOS_FUNCTION constexpr index_type extent(rank_type r) const noexcept {
    return map.extents().extent(r);
  };

 protected:
  // These are pre-condition checks which are in Kokkos::View 4.4 unconditional
  // - i.e. in release mode
  template <class OtherMapping>
  KOKKOS_FUNCTION static constexpr void check_basic_view_constructibility(
      const OtherMapping &rhs) {
    using src_t          = typename OtherMapping::layout_type;
    using dst_t          = layout_type;
    constexpr size_t rnk = mdspan_type::rank();
    if constexpr (!std::is_same_v<src_t, dst_t>) {
      if constexpr (Impl::is_layout_left_padded<dst_t>::value) {
        if constexpr (std::is_same_v<src_t, layout_stride>) {
          index_type stride = 1;
          for (size_t r = 0; r < rnk; r++) {
            if (rhs.stride(r) != stride)
              Kokkos::abort("View assignment must have compatible layouts");
            if constexpr (rnk > 1)
              stride *= (r == 0 ? rhs.stride(1) : rhs.extents().extent(r));
          }
        }
      }
      if constexpr (Impl::is_layout_right_padded<dst_t>::value) {
        if constexpr (std::is_same_v<src_t, layout_stride>) {
          index_type stride = 1;
          if constexpr (rnk > 0) {
            for (size_t r = rnk; r > 0; r--) {
              if (rhs.stride(r - 1) != stride)
                Kokkos::abort("View assignment must have compatible layouts");
              if constexpr (rnk > 1)
                stride *= (r == rnk ? rhs.stride(r - 2)
                                    : rhs.extents().extent(r - 1));
            }
          }
        }
      }
      if constexpr (std::is_same_v<dst_t, layout_left>) {
        if constexpr (std::is_same_v<src_t, layout_stride>) {
          index_type stride = 1;
          for (size_t r = 0; r < rnk; r++) {
            if (rhs.stride(r) != stride)
              Kokkos::abort("View assignment must have compatible layouts");
            stride *= rhs.extents().extent(r);
          }
        } else if constexpr (Impl::is_layout_left_padded<src_t>::value &&
                             rnk > 1) {
          if (rhs.stride(1) != rhs.extents().extent(0))
            Kokkos::abort("View assignment must have compatible layouts");
        }
      }
      if constexpr (std::is_same_v<dst_t, layout_right>) {
        if constexpr (std::is_same_v<src_t, layout_stride>) {
          index_type stride = 1;
          if constexpr (rnk > 0) {
            for (size_t r = rnk; r > 0; r--) {
              if (rhs.stride(r - 1) != stride)
                Kokkos::abort("View assignment must have compatible layouts");
              stride *= rhs.extents().extent(r - 1);
            }
          }
        } else if constexpr (Impl::is_layout_right_padded<src_t>::value &&
                             rnk > 1) {
          if (rhs.stride(rnk - 2) != rhs.extents().extent(rnk - 1))
            Kokkos::abort("View assignment must have compatible layouts");
        }
      }
    }
  }

 public:
  KOKKOS_FUNCTION constexpr BasicView() = default;

  // Constructors from mdspan
  KOKKOS_FUNCTION constexpr BasicView(const BasicView &) = default;
  KOKKOS_FUNCTION constexpr BasicView(BasicView &&)      = default;

  KOKKOS_FUNCTION constexpr BasicView(const mdspan_type &other)
      : ptr(other.data_handle()), map(other.mapping()), acc(other.accessor()){};
  KOKKOS_FUNCTION constexpr BasicView(mdspan_type &&other)
      : ptr(std::move(other.data_handle())),
        map(std::move(other.mapping())),
        acc(std::move(other.accessor())){};

  template <class... OtherIndexTypes>
  //    requires((std::is_convertible_v<OtherIndexTypes, index_type> && ...) &&
  //             (std::is_nothrow_constructible_v<index_type, OtherIndexTypes>
  //             && ...) &&
  //             ((sizeof...(OtherIndexTypes) == rank()) ||
  //             (sizeof...(OtherIndexTypes) == rank_dynamic())) &&
  //             std::is_constructible_v<mapping_type, extents_type> &&
  //             std::is_default_constructible_v<accessor_type>)
    requires(std::is_constructible_v<mdspan_type, data_handle_type,
                                     OtherIndexTypes...>)
  KOKKOS_FUNCTION explicit constexpr BasicView(data_handle_type p,
                                               OtherIndexTypes... exts)
      : ptr(std::move(p)),
        map(extents_type(static_cast<index_type>(std::move(exts))...)),
        acc{} {}

  template <class OtherIndexType, size_t Size>
  //    requires(is_convertible_v<const _OtherIndexType&, index_type> &&
  //             is_nothrow_constructible_v<index_type, const _OtherIndexType&>
  //             &&
  //             ((_Size == rank()) || (_Size == rank_dynamic())) &&
  //             is_constructible_v<mapping_type, extents_type> &&
  //             is_default_constructible_v<accessor_type>)
    requires(std::is_constructible_v<mdspan_type, data_handle_type,
                                     std::array<OtherIndexType, Size>>)
  explicit(Size != rank_dynamic()) KOKKOS_FUNCTION
      constexpr BasicView(data_handle_type p,
                          const Array<OtherIndexType, Size> &exts)
      : ptr(std::move(p)), map(extents_type(exts)), acc{} {}

  KOKKOS_FUNCTION constexpr BasicView(data_handle_type p,
                                      const extents_type &exts)
    requires(std::is_default_constructible_v<accessor_type> &&
             std::is_constructible_v<mapping_type, const extents_type &>)
      : ptr(std::move(p)), map(exts), acc{} {}

  KOKKOS_FUNCTION constexpr BasicView(data_handle_type p, const mapping_type &m)
    requires(std::is_default_constructible_v<accessor_type>)
      : ptr(std::move(p)), map(m), acc{} {}

  KOKKOS_FUNCTION constexpr BasicView(data_handle_type p, const mapping_type &m,
                                      const accessor_type &a)
      : ptr(std::move(p)), map(m), acc(a) {}

  template <class OtherT, class OtherE, class OtherL, class OtherA>
    requires(std::is_constructible_v<mdspan_type,
                                     typename BasicView<OtherT, OtherE, OtherL,
                                                        OtherA>::mdspan_type>)
  explicit(
      !std::is_convertible_v<const typename OtherL::template mapping<OtherE> &,
                             mapping_type> ||
      !std::is_convertible_v<const OtherA &, accessor_type>)
      KOKKOS_INLINE_FUNCTION
      BasicView(const BasicView<OtherT, OtherE, OtherL, OtherA> &other)
      : ptr(other.ptr), map(other.map), acc(other.acc) {
    // Kokkos View precondition checks happen in release builds
    check_basic_view_constructibility(other.mapping());

    static_assert(
        std::is_constructible_v<data_handle_type,
                                const typename OtherA::data_handle_type &>,
        "Kokkos::View: incompatible data_handle_type for View construction");
    static_assert(std::is_constructible_v<extents_type, OtherE>,
                  "Kokkos::View: incompatible extents for View construction");
  }

  template <class OtherT, class OtherE, class OtherL, class OtherA>
    requires(std::is_constructible_v<mdspan_type,
                                     mdspan<OtherT, OtherE, OtherL, OtherA>>)
  explicit(
      !std::is_convertible_v<const typename OtherL::template mapping<OtherE> &,
                             mapping_type> ||
      !std::is_convertible_v<const OtherA &, accessor_type>)
      KOKKOS_INLINE_FUNCTION
      BasicView(const mdspan<OtherT, OtherE, OtherL, OtherA> &other)
      : ptr(other.data_handle()), map(other.mapping()), acc(other.accessor()) {
    // Kokkos View precondition checks happen in release builds
    check_basic_view_constructibility(other.mapping());

    static_assert(
        std::is_constructible_v<data_handle_type,
                                const typename OtherA::data_handle_type &>,
        "Kokkos::View: incompatible data_handle_type for View construction");
    static_assert(std::is_constructible_v<extents_type, OtherE>,
                  "Kokkos::View: incompatible extents for View construction");
  }

  KOKKOS_FUNCTION constexpr BasicView &operator=(const BasicView &) = default;
  KOKKOS_FUNCTION constexpr BasicView &operator=(BasicView &&)      = default;

  // Allocating constructors specific to BasicView
  ///
  /// Construct from a given mapping
  ///
  KOKKOS_INLINE_FUNCTION explicit constexpr BasicView(
      const std::string &label, const mapping_type &mapping)
      : BasicView(view_alloc(label), mapping) {}

  ///
  /// Construct from a given extents
  ///
  KOKKOS_INLINE_FUNCTION explicit constexpr BasicView(const std::string &label,
                                                      const extents_type &ext)
      : BasicView(view_alloc(label), mapping_type{ext}) {}

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
    constexpr bool has_exec = Impl::ViewCtorProp<P...>::has_execution_space;
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
        has_exec ? &Impl::get_property<Impl::ExecutionSpaceTag>(prop_copy)
                 : nullptr,
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
      : BasicView(create_data_handle(arg_prop, arg_mapping), arg_mapping) {}

  template <class... P>
  explicit inline BasicView(
      const Impl::ViewCtorProp<P...> &arg_prop,
      std::enable_if_t<Impl::ViewCtorProp<P...>::has_pointer,
                       typename mdspan_type::mapping_type> const &arg_mapping)
      : BasicView(
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
      : BasicView(submdspan(
            src_view.to_mdspan(),
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
  KOKKOS_INLINE_FUNCTION constexpr
  operator mdspan<OtherElementType, OtherExtents, OtherLayoutPolicy,
                  OtherAccessor>() const {
    return mdspan_type(ptr, map, acc);
  }

  // Here we use an overload instead of a default parameter as a workaround
  // to a potential compiler bug with clang 17. It may be present in other
  // compilers
  template <class OtherAccessorType = AccessorPolicy,
            typename                = std::enable_if_t<std::is_constructible_v<
                typename mdspan_type::data_handle_type,
                typename OtherAccessorType::data_handle_type>>>
  KOKKOS_INLINE_FUNCTION constexpr auto to_mdspan() const {
    using ret_mdspan_type =
        mdspan<typename mdspan_type::element_type,
               typename mdspan_type::extents_type,
               typename mdspan_type::layout_type, OtherAccessorType>;
    return ret_mdspan_type(
        static_cast<typename OtherAccessorType::data_handle_type>(
            data_handle()),
        mapping(), static_cast<OtherAccessorType>(accessor()));
  }

  template <
      class OtherAccessorType = AccessorPolicy,
      typename                = std::enable_if_t<std::is_assignable_v<
          data_handle_type, typename OtherAccessorType::data_handle_type>>>
  KOKKOS_INLINE_FUNCTION constexpr auto to_mdspan(
      const OtherAccessorType &other_accessor) const {
    using ret_mdspan_type =
        mdspan<element_type, extents_type, layout_type, OtherAccessorType>;
    return ret_mdspan_type(
        static_cast<typename OtherAccessorType::data_handle_type>(
            data_handle()),
        mapping(), other_accessor);
  }

  KOKKOS_FUNCTION void assign_data(element_type *ptr_) { ptr = ptr_; }

  // ========================= mdspan =================================

  // [mdspan.mdspan.members], members

// Introducing the C++20 and C++23 variants of the operators already
#ifndef KOKKOS_ENABLE_CXX17
#ifndef KOKKOS_ENABLE_CXX20
  // C++23 only operator[]
  template <class... OtherIndexTypes>
    requires((std::is_convertible_v<OtherIndexTypes, index_type> && ...) &&
             (std::is_nothrow_constructible_v<index_type, OtherIndexTypes> &&
              ...) &&
             (sizeof...(OtherIndexTypes) == rank()))
  KOKKOS_FUNCTION constexpr reference operator[](
      OtherIndexTypes... indices) const {
    return acc.access(ptr, map(static_cast<index_type>(std::move(indices))...));
  }

  template <class OtherIndexType>
    requires(
        std::is_convertible_v<const OtherIndexType &, index_type> &&
        std::is_nothrow_constructible_v<index_type, const OtherIndexType &>)
  KOKKOS_FUNCTION constexpr reference operator[](
      const Array<OtherIndexType, rank()> &indices) const {
    return acc.access(ptr, [&]<size_t... Idxs>(std::index_sequence<Idxs...>) {
      return map(indices[Idxs]...);
    }(std::make_index_sequence<rank()>()));
  }

  template <class _OtherIndexType>
    requires(
        std::is_convertible_v<const OtherIndexType &, index_type> &&
        std::is_nothrow_constructible_v<index_type, const OtherIndexType &>)
  KOKKOS_FUNCTION constexpr reference operator[](
      std::span<OtherIndexType, rank()> indices) const {
    return acc.access(ptr, [&]<size_t... Idxs>(std::index_sequence<Idxs...>) {
      return map(indices[Idxs]...);
    }(std::make_index_sequence<rank()>()));
  }
#endif
  // C++20 operator()
  template <class... OtherIndexTypes>
    requires((std::is_convertible_v<OtherIndexTypes, index_type> && ...) &&
             (std::is_nothrow_constructible_v<index_type, OtherIndexTypes> &&
              ...) &&
             (sizeof...(OtherIndexTypes) == rank()))
  KOKKOS_FUNCTION constexpr reference operator()(
      OtherIndexTypes... indices) const {
    return acc.access(ptr, map(static_cast<index_type>(std::move(indices))...));
  }

  template <class OtherIndexType>
    requires(
        std::is_convertible_v<const OtherIndexType &, index_type> &&
        std::is_nothrow_constructible_v<index_type, const OtherIndexType &>)
  KOKKOS_FUNCTION constexpr reference operator()(
      const Array<OtherIndexType, rank()> &indices) const {
    return acc.access(ptr, [&]<size_t... Idxs>(std::index_sequence<Idxs...>) {
      return map(indices[Idxs]...);
    }(std::make_index_sequence<rank()>()));
  }

  template <class OtherIndexType>
    requires(
        std::is_convertible_v<const OtherIndexType &, index_type> &&
        std::is_nothrow_constructible_v<index_type, const OtherIndexType &>)
  KOKKOS_FUNCTION constexpr reference operator()(
      std::span<OtherIndexType, rank()> indices) const {
    return acc.access(ptr, [&]<size_t... Idxs>(std::index_sequence<Idxs...>) {
      return map(indices[Idxs]...);
    }(std::make_index_sequence<rank()>()));
  }
#else
// C++17 variant of operator()
#endif

  KOKKOS_FUNCTION constexpr size_type size() const noexcept {
    return [&]<size_t... Idxs>(std::index_sequence<Idxs...>) {
      return ((static_cast<size_type>(map.extents().extent(Idxs))) * ... *
              size_type(1));
    }(std::make_index_sequence<rank()>());
  }

  [[nodiscard]] KOKKOS_FUNCTION constexpr bool empty() const noexcept {
    return [&]<size_t... Idxs>(std::index_sequence<Idxs...>) {
      return (rank() > 0) &&
             ((map.extents().extent(Idxs) == index_type(0)) || ... || false);
    }(std::make_index_sequence<rank()>());
  }

  KOKKOS_FUNCTION friend constexpr void swap(BasicView &x,
                                             BasicView &y) noexcept {
    kokkos_swap(x.ptr, y.ptr);
    kokkos_swap(x.map, y.map);
    kokkos_swap(x.acc, y.acc);
  }

  KOKKOS_FUNCTION constexpr const extents_type &extents() const noexcept {
    return map.extents();
  };
  KOKKOS_FUNCTION constexpr const data_handle_type &data_handle()
      const noexcept {
    return ptr;
  };
  KOKKOS_FUNCTION constexpr const mapping_type &mapping() const noexcept {
    return map;
  };
  KOKKOS_FUNCTION constexpr const accessor_type &accessor() const noexcept {
    return acc;
  };

  KOKKOS_FUNCTION static constexpr bool is_always_unique() noexcept {
    return mapping_type::is_always_unique();
  };
  KOKKOS_FUNCTION static constexpr bool is_always_exhaustive() noexcept {
    return mapping_type::is_always_exhaustive();
  };
  KOKKOS_FUNCTION static constexpr bool is_always_strided() noexcept {
    return mapping_type::is_always_strided();
  };

  KOKKOS_FUNCTION constexpr bool is_unique() const { return map.is_unique(); };
  KOKKOS_FUNCTION constexpr bool is_exhaustive() const {
    return map.is_exhaustive();
  };
  KOKKOS_FUNCTION constexpr bool is_strided() const {
    return map.is_strided();
  };
  KOKKOS_FUNCTION constexpr index_type stride(rank_type r) const {
    return map.stride(r);
  };

 protected:
  KOKKOS_NO_UNIQUE_ADDRESS data_handle_type ptr{};
  KOKKOS_NO_UNIQUE_ADDRESS mapping_type map{};
  KOKKOS_NO_UNIQUE_ADDRESS accessor_type acc{};

  template <class, class, class, class>
  friend class BasicView;
};
}  // namespace Kokkos

#endif
