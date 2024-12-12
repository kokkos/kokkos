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

#ifndef KOKKOS_SIMD_HPP
#define KOKKOS_SIMD_HPP

#include <Kokkos_SIMD_Common.hpp>

// suppress NVCC warnings with the [[nodiscard]] attribute on overloaded
// operators implemented as hidden friends
#if defined(KOKKOS_COMPILER_NVCC) && KOKKOS_COMPILER_NVCC < 1130
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
#endif

#include <Kokkos_SIMD_Scalar.hpp>

#include <Kokkos_Macros.hpp>

// FIXME_OPENMPTARGET The device pass disables all compiler macros checked
#ifdef KOKKOS_ENABLE_OPENMPTARGET
#if defined(KOKKOS_ARCH_AVX2)
#include <Kokkos_SIMD_AVX2.hpp>
#endif

#if defined(KOKKOS_ARCH_AVX512XEON)
#include <Kokkos_SIMD_AVX512.hpp>
#endif

#if defined(KOKKOS_ARCH_ARM_NEON)
#include <Kokkos_SIMD_NEON.hpp>
#endif
#else  // KOKKOS_ENABLE_OPENMPTARGET
#if defined(KOKKOS_ARCH_AVX) && !defined(__AVX__)
#error "__AVX__ must be defined for KOKKOS_ARCH_AVX"
#endif

#if defined(KOKKOS_ARCH_AVX2)
#if !defined(__AVX2__)
#error "__AVX2__ must be defined for KOKKOS_ARCH_AVX2"
#endif
#include <Kokkos_SIMD_AVX2.hpp>
#endif

#if defined(KOKKOS_ARCH_AVX512XEON)
#if !defined(__AVX512F__)
#error "__AVX512F__ must be defined for KOKKOS_ARCH_AVX512XEON"
#endif
#include <Kokkos_SIMD_AVX512.hpp>
#endif

#if defined(KOKKOS_ARCH_ARM_NEON)
#if !defined(__ARM_NEON)
#error "__ARM_NEON must be definded for KOKKOS_ARCH_ARM_NEON"
#endif
#include <Kokkos_SIMD_NEON.hpp>
#endif
#endif

#if defined(KOKKOS_COMPILER_NVCC) && KOKKOS_COMPILER_NVCC < 1130
#pragma GCC diagnostic pop
#endif

#include <Kokkos_SIMD_Common_Math.hpp>

namespace Kokkos {
namespace Experimental {

namespace simd_abi {

namespace Impl {

#if defined(KOKKOS_ARCH_AVX512XEON)
using host_fixed_native = avx512_fixed_size<8>;
template <int N>
using host_native_abi = avx512_fixed_size<N>;

#elif defined(KOKKOS_ARCH_AVX2)
using host_fixed_native = avx2_fixed_size<4>;
template <int N>
using host_native_abi = avx2_fixed_size<N>;

#elif defined(KOKKOS_ARCH_ARM_NEON)
using host_fixed_native = neon_fixed_size<2>;
template <int N>
using host_native_abi = neon_fixed_size<N>;

#else
using host_fixed_native = scalar;
template <int N>
using host_native_abi = scalar;
#endif

template <class T>
struct ForSpace;

#ifdef KOKKOS_ENABLE_SERIAL
template <>
struct ForSpace<Kokkos::Serial> {
  using type = host_fixed_native;

  template <int N>
  using simd_abi = host_native_abi<N>;
};
#endif

#ifdef KOKKOS_ENABLE_CUDA
template <>
struct ForSpace<Kokkos::Cuda> {
  using type = scalar;

  template <int N>
  using simd_abi = scalar;
};
#endif

#ifdef KOKKOS_ENABLE_THREADS
template <>
struct ForSpace<Kokkos::Threads> {
  using type = host_fixed_native;

  template <int N>
  using simd_abi = host_native_abi<N>;
};
#endif

#ifdef KOKKOS_ENABLE_HPX
template <>
struct ForSpace<Kokkos::Experimental::HPX> {
  using type = scalar;

  template <int N>
  using simd_abi = scalar;
};
#endif

#ifdef KOKKOS_ENABLE_OPENMP
template <>
struct ForSpace<Kokkos::OpenMP> {
  using type = host_fixed_native;

  template <int N>
  using simd_abi = host_native_abi<N>;
};
#endif

#ifdef KOKKOS_ENABLE_OPENMPTARGET
template <>
struct ForSpace<Kokkos::Experimental::OpenMPTarget> {
  using type = scalar;

  template <int N>
  using simd_abi = scalar;
};
#endif

#ifdef KOKKOS_ENABLE_OPENACC
template <>
struct ForSpace<Kokkos::Experimental::OpenACC> {
  using type = scalar;

  template <int N>
  using simd_abi = scalar;
};
#endif

#ifdef KOKKOS_ENABLE_HIP
template <>
struct ForSpace<Kokkos::HIP> {
  using type = scalar;

  template <int N>
  using simd_abi = scalar;
};
#endif

#ifdef KOKKOS_ENABLE_SYCL
template <>
struct ForSpace<Kokkos::SYCL> {
  using type = scalar;

  template <int N>
  using simd_abi = scalar;
};
#endif

template <class Space = Kokkos::DefaultExecutionSpace>
using native_fixed_abi = typename ForSpace<Space>::type;

template <int N, class Space = Kokkos::DefaultExecutionSpace>
using native_abi = typename ForSpace<Space>::template simd_abi<N>;

}  // namespace Impl

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4
template <class Space>
using ForSpace KOKKOS_DEPRECATED =
    typename Impl::ForSpace<typename Space::execution_space>::type;

template <class T>
using native KOKKOS_DEPRECATED = ForSpace<Kokkos::DefaultExecutionSpace>;
#endif

}  // namespace simd_abi

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4
template <class T>
using native_simd KOKKOS_DEPRECATED = basic_simd<T, simd_abi::native<T>>;
template <class T>
using native_simd_mask KOKKOS_DEPRECATED =
    basic_simd_mask<T, simd_abi::native<T>>;

template <class T, class Abi>
using simd KOKKOS_DEPRECATED_WITH_COMMENT(
    "Temporarily use Impl::simd<T> instead") = basic_simd<T, Abi>;

template <class T, class Abi>
using simd_mask KOKKOS_DEPRECATED_WITH_COMMENT(
    "Temporarily use Impl::simd_mask<T> instead") = basic_simd_mask<T, Abi>;
#endif

namespace Impl {

template <class T, int N = 0>
using simd =
    basic_simd<T,
               std::conditional_t<(N == 0), simd_abi::Impl::native_fixed_abi<>,
                                  simd_abi::Impl::native_abi<N>>>;

template <class T, int N = 0>
using simd_mask = basic_simd_mask<
    T, std::conditional_t<(N == 0), simd_abi::Impl::native_fixed_abi<>,
                          simd_abi::Impl::native_abi<N>>>;

template <class... Abis>
class abi_set {};

template <typename... Ts>
class data_types {};

#if defined(KOKKOS_ARCH_AVX512XEON)
using host_abi_set  = abi_set<simd_abi::scalar, simd_abi::avx512_fixed_size<8>,
                             simd_abi::avx512_fixed_size<16>>;
using data_type_set = data_types<std::int32_t, std::uint32_t, std::int64_t,
                                 std::uint64_t, double, float>;
#elif defined(KOKKOS_ARCH_AVX2)
using host_abi_set    = abi_set<simd_abi::scalar, simd_abi::avx2_fixed_size<4>,
                             simd_abi::avx2_fixed_size<8>>;
using data_type_set =
    data_types<std::int32_t, std::int64_t, std::uint64_t, double, float>;
#elif defined(KOKKOS_ARCH_ARM_NEON)
using host_abi_set    = abi_set<simd_abi::scalar, simd_abi::neon_fixed_size<2>,
                             simd_abi::neon_fixed_size<4>>;
using data_type_set =
    data_types<std::int32_t, std::int64_t, std::uint64_t, double, float>;
#else
using host_abi_set    = abi_set<simd_abi::scalar>;
using data_type_set   = data_types<std::int32_t, std::uint32_t, std::int64_t,
                                 std::uint64_t, double, float>;
#endif

using device_abi_set = abi_set<simd_abi::scalar>;

}  // namespace Impl

}  // namespace Experimental
}  // namespace Kokkos

#endif
