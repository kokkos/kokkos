/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

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

template <class... Abis>
class abi_set {};

#ifdef KOKKOS_ARCH_AVX512XEON
using host_abi_set = abi_set<scalar, avx512_fixed_size<8>>;
#else
using host_abi_set  = abi_set<scalar>;
#endif

using device_abi_set = abi_set<scalar>;

#if defined(KOKKOS_ARCH_AVX512XEON)
using host_native = avx512_fixed_size<8>;
#else
using host_native   = scalar;
#endif

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
using device_native = scalar;
#else
using device_native = host_native;
#endif

using native = host_native;

}  // namespace simd_abi

template <class T>
using device_simd = simd<T, simd_abi::device_native>;
template <class T>
using device_simd_mask = simd_mask<T, simd_abi::device_native>;

}  // namespace Experimental
}  // namespace Kokkos

#endif
