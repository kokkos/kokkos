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

#ifndef KOKKOS_EXPERIMENTAL_SYCL_VIEW_HPP
#define KOKKOS_EXPERIMENTAL_SYCL_VIEW_HPP

#include <View/Kokkos_ViewMapping.hpp>

// Prior to oneAPI 2023.1.0, this gives
//
// InvalidModule: Invalid SPIR-V module: Casts from private/local/global address
// space are allowed only to generic
#if defined(__INTEL_LLVM_COMPILER) && __INTEL_LLVM_COMPILER >= 20230100

namespace Kokkos {
namespace Impl {

template <typename ValueType, typename MemorySpace>
struct SYCLUSMHandle {
  mutable std::conditional_t<
      std::is_same_v<MemorySpace, Kokkos::SYCL::scratch_memory_space_l0>,
      sycl::local_ptr<ValueType>, Impl::sycl_device_ptr<ValueType>>
      m_ptr;
  static constexpr sycl::access::address_space address_space =
      std::is_same_v<MemorySpace, Kokkos::SYCL::scratch_memory_space_l0>
          ? sycl::access::address_space::local_space
          : sycl::access::address_space::global_space;

  template <typename iType>
  KOKKOS_FORCEINLINE_FUNCTION ValueType& operator()(const iType& i) const {
    return m_ptr[i];
  }

  template <typename iType>
  KOKKOS_FORCEINLINE_FUNCTION ValueType& operator[](const iType& i) const {
    return m_ptr[i];
  }

  KOKKOS_FUNCTION
  operator ValueType*() const { return static_cast<ValueType*>(m_ptr.get()); }

  KOKKOS_DEFAULTED_FUNCTION
  SYCLUSMHandle() = default;

  KOKKOS_FUNCTION
  explicit SYCLUSMHandle(ValueType* const arg_ptr) : m_ptr(arg_ptr) {}

  KOKKOS_FUNCTION
  SYCLUSMHandle(const SYCLUSMHandle& arg_handle, size_t offset)
      : m_ptr(
#ifdef SYCL_EXT_ONEAPI_ADDRESS_CAST
            sycl::ext::oneapi::experimental::static_address_cast<
                AddressSpace, ValueType, decorated::yes>(arg_handle.m_ptr +
                                                         offset)
#else
            arg_handle.m_ptr + offset
#endif
        ) {
  }

  SYCLUSMHandle& operator=(ValueType* const arg_ptr) {
    return *this = SYCLUSMHandle(arg_ptr);
  }
};

template <class Traits>
struct ViewDataHandle<
    Traits,
    std::enable_if_t<std::is_void<typename Traits::specialize>::value &&
                     !Traits::memory_traits::is_aligned &&
                     !Traits::memory_traits::is_restrict &&
                     !Traits::memory_traits::is_atomic &&
                     (std::is_same_v<typename Traits::memory_space,
                                     Kokkos::SYCL::scratch_memory_space_l0> ||
                      std::is_same_v<typename Traits::memory_space,
                                     Kokkos::SYCL::scratch_memory_space_l1>)>> {
  // private:
  using value_type   = typename Traits::value_type;
  using memory_space = typename Traits::memory_space;
  using handle_type  = SYCLUSMHandle<value_type, memory_space>;
  using return_type  = typename Traits::value_type&;
  using track_type   = Kokkos::Impl::SharedAllocationTracker;

 public:
  KOKKOS_INLINE_FUNCTION
  static handle_type assign(value_type* const arg_data_ptr,
                            track_type const& /*arg_tracker*/) {
    return handle_type(arg_data_ptr);
  }

  KOKKOS_INLINE_FUNCTION
  static handle_type assign(value_type* const arg_data_ptr, size_t offset) {
    return handle_type(arg_data_ptr + offset);
  }

  KOKKOS_INLINE_FUNCTION
  static handle_type assign(const handle_type arg_data_ptr, size_t offset) {
    return handle_type(arg_data_ptr, offset);
  }
};

}  // namespace Impl
}  // namespace Kokkos

#endif  // #if defined(__INTEL_LLVM_COMPILER) && __INTEL_LLVM_COMPILER >=
        // 20230100

#endif /* #ifndef KOKKOS_SYCL_VIEW_HPP */
