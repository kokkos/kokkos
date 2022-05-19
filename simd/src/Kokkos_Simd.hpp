#ifndef KOKKOS_SIMD_HPP
#define KOKKOS_SIMD_HPP

#include <Kokkos_Simd_Common.hpp>

#include <Kokkos_Simd_Scalar.hpp>

#ifdef KOKKOS_ARCH_AVX512XEON
#include <Kokkos_Simd_AVX512.hpp>
#endif

namespace Kokkos {
namespace Experimental {

namespace simd_abi {

template <class ... Abis>
class abi_set {};

using host_abi_set = abi_set<scalar
#ifdef KOKKOS_ARCH_AVX512XEON
      ,avx512_fixed_size<8>
#endif
      >;

using device_abi_set = abi_set<scalar
      >;

#if defined(KOKKOS_ARCH_AVX512XEON)
using host_native = avx512_fixed_size<8>;
#else
using host_native = scalar;
#endif

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
using device_native = scalar;
#else
using device_native = host_native;
#endif

using native = host_native;

}

template <class T>
using device_simd = simd<T, simd_abi::device_native>;
template <class T>
using device_simd_mask = simd_mask<T, simd_abi::device_native>;

}
}

#endif
