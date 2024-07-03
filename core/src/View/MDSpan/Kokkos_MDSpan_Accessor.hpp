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

#ifndef KOKKOS_MDSPAN_ACCESSOR_HPP
#define KOKKOS_MDSPAN_ACCESSOR_HPP

#include <Kokkos_Macros.hpp>
#include <Kokkos_Concepts.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <impl/Kokkos_SharedAlloc.hpp>

namespace Kokkos {

// For now use the accessors in Impl namespace, as an
// implementation detail for rebasing View on mdspan
namespace Impl {

template <class MemorySpace, class NestedAccessor>
struct SpaceAwareAccessor {
  // Part of Accessor Requirements
  using element_type     = typename NestedAccessor::element_type;
  using reference        = typename NestedAccessor::reference;
  using data_handle_type = typename NestedAccessor::data_handle_type;
  using offset_policy =
      SpaceAwareAccessor<MemorySpace, typename NestedAccessor::offset_policy>;

  // Specific to SpaceAwareAccessor
  using memory_space         = MemorySpace;
  using nested_accessor_type = NestedAccessor;

  static_assert(is_memory_space_v<memory_space>);

  KOKKOS_DEFAULTED_FUNCTION
  constexpr SpaceAwareAccessor() = default;

  template <
      class OtherMemorySpace, class OtherNestedAccessorType,
      std::enable_if_t<
          MemorySpaceAccess<MemorySpace, OtherMemorySpace>::assignable &&
              std::is_constructible_v<NestedAccessor, OtherNestedAccessorType>,
          int> = 0>
  KOKKOS_FUNCTION constexpr SpaceAwareAccessor(
      const SpaceAwareAccessor<OtherMemorySpace, OtherNestedAccessorType>&
          other) noexcept
      : nested_acc(other.nested_acc) {}

  KOKKOS_FUNCTION
  SpaceAwareAccessor(const NestedAccessor& acc) : nested_acc(acc) {}

  KOKKOS_FUNCTION
  explicit operator NestedAccessor() const { return nested_acc; }

  KOKKOS_FUNCTION
  constexpr reference access(data_handle_type p, size_t i) const noexcept {
    Kokkos::Impl::runtime_check_memory_access_violation<memory_space>(
        "Kokkos::SpaceAwareAccessor ERROR: attempt to access inaccessible "
        "memory space");
    return nested_acc.access(p, i);
  }

  KOKKOS_FUNCTION
  constexpr typename offset_policy::data_handle_type offset(
      data_handle_type p, size_t i) const noexcept {
    return nested_acc.offset(p, i);
  }

  // Canonical way for accessing nested accessor see ISO C++
  // [linalg.scaled.scaledaccessor]
  KOKKOS_FUNCTION
  constexpr const NestedAccessor& nested_accessor() const noexcept {
    return nested_acc;
  }

 private:
// We either compile with our custom mdspan impl
// in which case we discover inside it whether no_unique_address
// works, or we use C++23 in which case it better be available
#ifdef _MDSPAN_NO_UNIQUE_ADDRESS
  _MDSPAN_NO_UNIQUE_ADDRESS
#else
  [[no_unique_address]]
#endif
  NestedAccessor nested_acc;
  template <class, class>
  friend struct SpaceAwareAccessor;
};

template <class NestedAccessor>
struct SpaceAwareAccessor<AnonymousSpace, NestedAccessor> {
  // Part of Accessor Requirements
  using element_type     = typename NestedAccessor::element_type;
  using reference        = typename NestedAccessor::reference;
  using data_handle_type = typename NestedAccessor::data_handle_type;

  using offset_policy =
      SpaceAwareAccessor<AnonymousSpace,
                         typename NestedAccessor::offset_policy>;

  // Specific to SpaceAwareAccessor
  using memory_space         = AnonymousSpace;
  using nested_accessor_type = NestedAccessor;

  KOKKOS_DEFAULTED_FUNCTION
  constexpr SpaceAwareAccessor() = default;

  template <class OtherMemorySpace, class OtherNestedAccessorType,
            std::enable_if_t<std::is_constructible_v<NestedAccessor,
                                                     OtherNestedAccessorType>,
                             int> = 0>
  KOKKOS_FUNCTION constexpr SpaceAwareAccessor(
      const SpaceAwareAccessor<OtherMemorySpace, OtherNestedAccessorType>&
          other) noexcept
      : nested_acc(other.nested_acc) {}

  KOKKOS_FUNCTION
  SpaceAwareAccessor(const NestedAccessor& acc) : nested_acc(acc) {}

  KOKKOS_FUNCTION
  explicit operator NestedAccessor() const { return nested_acc; }

  KOKKOS_FUNCTION
  constexpr reference access(data_handle_type p, size_t i) const noexcept {
    return nested_acc.access(p, i);
  }

  KOKKOS_FUNCTION
  constexpr typename offset_policy::data_handle_type offset(
      data_handle_type p, size_t i) const noexcept {
    return nested_acc.offset(p, i);
  }

  // Canonical way for accessing nested accessor see ISO C++
  // [linalg.scaled.scaledaccessor]
  KOKKOS_FUNCTION
  constexpr const NestedAccessor& nested_accessor() const noexcept {
    return nested_acc;
  }

 private:
// We either compile with our custom mdspan impl
// in which case we discover inside it whether no_unique_address
// works, or we use C++23 in which case it better be available
#ifdef _MDSPAN_NO_UNIQUE_ADDRESS
  _MDSPAN_NO_UNIQUE_ADDRESS
#else
  [[no_unique_address]]
#endif
  NestedAccessor nested_acc;
  template <class, class>
  friend struct SpaceAwareAccessor;
};

template <class ElementType, class MemorySpace>
class ReferenceCountedDataHandle {
 public:
  using value_type = ElementType;
  using pointer    = value_type*;
  using reference  = value_type&;

  ReferenceCountedDataHandle() = default;
  explicit ReferenceCountedDataHandle(SharedAllocationRecord<void, void>* rec) {
    m_tracker.assign_allocated_record_to_uninitialized(rec);
    m_handle = static_cast<pointer>(get_record()->data());
  }

  ReferenceCountedDataHandle(const ReferenceCountedDataHandle&)     = default;
  ReferenceCountedDataHandle(ReferenceCountedDataHandle&&) noexcept = default;
  ReferenceCountedDataHandle& operator=(const ReferenceCountedDataHandle&) =
      default;
  ReferenceCountedDataHandle& operator=(ReferenceCountedDataHandle&&) = default;

  ReferenceCountedDataHandle with_offset(size_t offset) const {
    auto ret     = *this;
    ret.m_handle += offset;
    return ret;
  }

  pointer get() const noexcept { return m_handle; }

  bool has_record() const { return m_tracker.has_record(); }
  auto* get_record() const { return m_tracker.get_record<MemorySpace>(); }
  size_t use_count() const noexcept { return m_tracker.use_count(); }

  std::string get_label() const { return m_tracker.get_label<MemorySpace>(); }

 private:
  SharedAllocationTracker m_tracker;
  pointer m_handle = nullptr;
};

template <class ElementType, class MemorySpace>
class ReferenceCountedAccessor {
 public:
  using element_type     = ElementType;
  using data_handle_type = ReferenceCountedDataHandle<ElementType, MemorySpace>;
  using reference        = typename data_handle_type::reference;
  using offset_policy    = ReferenceCountedAccessor;

  constexpr ReferenceCountedAccessor() noexcept = default;

  constexpr reference access(data_handle_type p, size_t i) const {
    return p.get()[i];
  }

  constexpr data_handle_type offset(data_handle_type p, size_t i) const {
    return p.with_offset(i);
  }
};

template <class ElementType, class MemorySpace>
using checked_reference_counted_accessor =
    SpaceAwareAccessor<MemorySpace,
                       ReferenceCountedAccessor<ElementType, MemorySpace>>;

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_MDSPAN_ACCESSOR_HPP
