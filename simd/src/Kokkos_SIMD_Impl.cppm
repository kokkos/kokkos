// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

module;

#include <Kokkos_Macros.hpp>

// FIXME_CLANG: Including the same function using intrinsics in both the
// `kokkos.simd` and `kokkos.simd_impl` modules results in "definition with same
// mangled name" errors when compiling with clang, most likely because of this
// bug https://github.com/llvm/llvm-project/issues/195661. Making sure that the
// `basic_simd` and `basic_simd_mask` implementations are only included in the
// `kokkos.simd` module and not in `kokkos.simd_impl` fixes the issue.
#if defined(KOKKOS_COMPILER_CLANG)
#define KOKKOS_IMPL_SIMD_MODULES_DONT_INCLUDE_IMPLEMENTATION
#endif

#include <Kokkos_SIMD.hpp>

export module kokkos.simd_impl;

export {
  namespace Kokkos::Experimental {
  namespace simd_abi::Impl {
  using ::Kokkos::Experimental::simd_abi::Impl::host_fixed_native;
  using ::Kokkos::Experimental::simd_abi::Impl::native_abi;
  using ::Kokkos::Experimental::simd_abi::Impl::native_fixed_abi;
  }  // namespace simd_abi::Impl

  namespace Impl {
  using ::Kokkos::Experimental::Impl::abi_set;
  using ::Kokkos::Experimental::Impl::data_type_set;
  using ::Kokkos::Experimental::Impl::data_types;
  using ::Kokkos::Experimental::Impl::device_abi_set;
  using ::Kokkos::Experimental::Impl::host_abi_set;
  using ::Kokkos::Experimental::Impl::Identity;
  using ::Kokkos::Experimental::Impl::simd_size_t;
  }  // namespace Impl
  }  // namespace Kokkos::Experimental
}
