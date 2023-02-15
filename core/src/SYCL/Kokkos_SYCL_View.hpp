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

#include <impl/Kokkos_ViewMapping.hpp>

// Prior to oneAPI 2023.1.0, this gives
//
// InvalidModule: Invalid SPIR-V module: Casts from private/local/global address
// space are allowed only to generic
#if defined(__INTEL_LLVM_COMPILER) && __INTEL_LLVM_COMPILER >= 20230100

namespace Kokkos {
namespace Impl {

template <class Traits>
struct ViewDataHandle<
    Traits,
    std::enable_if_t<
        std::is_void<typename Traits::specialize>::value &&
        !Traits::memory_traits::is_aligned &&
        !Traits::memory_traits::is_restrict &&
        !Traits::memory_traits::is_atomic &&
        (is_sycl_type_space<typename Traits::memory_space>::value ||
         std::is_same_v<typename Traits::memory_space,
                        ScratchMemorySpace<Kokkos::Experimental::SYCL>>)>> {
  using value_type   = typename Traits::value_type;
  using memory_space = typename Traits::memory_space;
  using handle_type =
      std::conditional_t<is_sycl_type_space<memory_space>::value,
                         sycl::global_ptr<value_type>,
                         sycl::device_ptr<value_type>>;
  using return_type = typename Traits::value_type&;
  using track_type  = Kokkos::Impl::SharedAllocationTracker;

  KOKKOS_INLINE_FUNCTION
  static value_type* assign(value_type* arg_data_ptr,
                            track_type const& /*arg_tracker*/) {
    return (value_type*)(arg_data_ptr);
  }

  KOKKOS_INLINE_FUNCTION
  static value_type* assign(handle_type const arg_data_ptr, size_t offset) {
    return (value_type*)(arg_data_ptr + offset);
  }
};

}  // namespace Impl
}  // namespace Kokkos

#endif  // #if defined(__INTEL_LLVM_COMPILER) && __INTEL_LLVM_COMPILER >=
        // 20230100

#endif /* #ifndef KOKKOS_SYCL_VIEW_HPP */
