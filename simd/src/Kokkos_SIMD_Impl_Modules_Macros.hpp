// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#define KOKKOS_SIMD_MODULES_IMPL_SYMBOLS                           \
  namespace Kokkos::Experimental {                                 \
  namespace simd_abi::Impl {                                       \
  using ::Kokkos::Experimental::simd_abi::Impl::host_fixed_native; \
  using ::Kokkos::Experimental::simd_abi::Impl::native_abi;        \
  using ::Kokkos::Experimental::simd_abi::Impl::native_fixed_abi;  \
  }                                                                \
  namespace Impl {                                                 \
  using ::Kokkos::Experimental::Impl::abi_set;                     \
  using ::Kokkos::Experimental::Impl::data_type_set;               \
  using ::Kokkos::Experimental::Impl::data_types;                  \
  using ::Kokkos::Experimental::Impl::device_abi_set;              \
  using ::Kokkos::Experimental::Impl::host_abi_set;                \
  using ::Kokkos::Experimental::Impl::Identity;                    \
  using ::Kokkos::Experimental::Impl::simd_size_t;                 \
  }                                                                \
  }
