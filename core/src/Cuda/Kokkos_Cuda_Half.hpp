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

#ifndef KOKKOS_CUDA_HALF_HPP_
#define KOKKOS_CUDA_HALF_HPP_

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

#ifndef KOKKOS_IMPL_HALF_TYPE_DEFINED
// Make sure no one else tries to define half_t
#define KOKKOS_IMPL_HALF_TYPE_DEFINED

namespace Kokkos {
namespace Experimental {

using half_device_type = __half;

// Forward declarations
class half_t;

KOKKOS_INLINE_FUNCTION
half_t cast_to_half(float val);
KOKKOS_INLINE_FUNCTION
half_t cast_to_half(double val);
KOKKOS_INLINE_FUNCTION
half_t cast_to_half(short val);
KOKKOS_INLINE_FUNCTION
half_t cast_to_half(int val);
KOKKOS_INLINE_FUNCTION
half_t cast_to_half(long val);
KOKKOS_INLINE_FUNCTION
half_t cast_to_half(long long val);
KOKKOS_INLINE_FUNCTION
half_t cast_to_half(unsigned short val);
KOKKOS_INLINE_FUNCTION
half_t cast_to_half(unsigned int val);
KOKKOS_INLINE_FUNCTION
half_t cast_to_half(unsigned long val);
KOKKOS_INLINE_FUNCTION
half_t cast_to_half(unsigned long long val);

class half_t {
  half_device_type val;

 public:
  // Conversion operator for __half(bar) = half_t(foo)
  KOKKOS_FUNCTION
  operator half_device_type() const { return val; }

// NOTE: Changing below to 1 produces constructor overload error
#if 0
  // Conversion operator for bool(bar) = half_t(foo)
  KOKKOS_FUNCTION
  operator bool() const {
    return static_cast<bool>(__half2float(val) == 0.0F);
  }
#endif

  KOKKOS_FUNCTION
  half_t(half_device_type rhs = 0) : val(rhs) {}

  // Cast rhs to half for assignment to lhs of type half_t
  template <class T>
  KOKKOS_FUNCTION half_t(T rhs) : half_t(cast_to_half(rhs)) {}

  // Unary operators
  KOKKOS_FUNCTION
  half_t operator+() const {
    half_t tmp = *this;
#ifdef __CUDA_ARCH__
    // printf("half_t unary operator+\n");
    tmp.val = +tmp.val;
#else
    // printf("float unary operator+\n");
    tmp.val = __float2half(+__half2float(tmp.val));
#endif
    return tmp;
  }

  KOKKOS_FUNCTION
  half_t operator-() const {
    half_t tmp = *this;
#ifdef __CUDA_ARCH__
    // printf("half_t unary operator-\n");
    tmp.val = -tmp.val;
#else
    // printf("float unary operator-\n");
    tmp.val = __float2half(-__half2float(tmp.val));
#endif
    return tmp;
  }

  // Prefix operators
  KOKKOS_FUNCTION
  half_t& operator++() {
#ifdef __CUDA_ARCH__
    // printf("half_t prefix operator++\n");
    ++val;
#else
    // printf("float prefix operator++\n");
    float tmp = __half2float(val);
    ++tmp;
    val = __float2half(tmp);
#endif
    return *this;
  }

  KOKKOS_FUNCTION
  half_t& operator--() {
#ifdef __CUDA_ARCH__
    // printf("half_t prefix operator--\n");
    --val;
#else
    // printf("float prefix operator--\n");
    float tmp = __half2float(val);
    --tmp;
    val = __float2half(tmp);
#endif
    return *this;
  }

  // Postfix operators
  KOKKOS_FUNCTION
  half_t operator++(int) {
    half_t tmp = *this;
    operator++();
    return tmp;
  }

  KOKKOS_FUNCTION
  half_t operator--(int) {
    half_t tmp = *this;
    operator--();
    return tmp;
  }

  // Binary operators
  KOKKOS_FUNCTION
  half_t& operator=(half_device_type rhs) {
    // printf("half_device_type operator=\n");
    val = rhs;
    return *this;
  }

  template <class T>
  KOKKOS_FUNCTION half_t operator=(T rhs) {
    // printf("T operator=\n");
    val = cast_to_half(rhs).val;
    return *this;
  }

  // Compound operators
  KOKKOS_FUNCTION
  half_t& operator+=(half_t rhs) {
#ifdef __CUDA_ARCH__
    // printf("half_t operator+=\n");
    val += rhs.val;
#else
    // printf("float operator+=\n");
    val = __float2half(__half2float(val) + __half2float(rhs.val));
#endif
    return *this;
  }

  KOKKOS_FUNCTION
  half_t& operator-=(half_t rhs) {
#ifdef __CUDA_ARCH__
    // printf("half_t operator-=\n");
    val -= rhs.val;
#else
    // printf("float operator-=\n");
    val = __float2half(__half2float(val) - __half2float(rhs.val));
#endif
    return *this;
  }

  KOKKOS_FUNCTION
  half_t& operator*=(half_t rhs) {
#ifdef __CUDA_ARCH__
    // printf("half_t operator*=\n");
    val *= rhs.val;
#else
    // printf("float operator*=\n");
    val = __float2half(__half2float(val) * __half2float(rhs.val));
#endif
    return *this;
  }

  KOKKOS_FUNCTION
  half_t& operator/=(half_t rhs) {
#ifdef __CUDA_ARCH__
    // printf("half_t operator/=\n");
    val /= rhs.val;
#else
    // printf("float operator/=\n");
    val = __float2half(__half2float(val) / __half2float(rhs.val));
#endif
    return *this;
  }

  // Binary Arithmetic
  KOKKOS_FUNCTION
  half_t operator+(half_t rhs) const {
    half_t tmp = *this;
#ifdef __CUDA_ARCH__
    // printf("half_t operator+\n");
    tmp.val += rhs.val;
#else
    // printf("float operator+\n");
    tmp.val = __float2half(__half2float(tmp.val) + __half2float(rhs.val));
#endif
    return tmp;
  }

  KOKKOS_FUNCTION
  half_t operator-(half_t rhs) const {
    half_t tmp = *this;
#ifdef __CUDA_ARCH__
    // printf("half_t operator-\n");
    tmp.val -= rhs.val;
#else
    // printf("float operator-\n");
    tmp.val = __float2half(__half2float(tmp.val) - __half2float(rhs.val));
#endif
    return tmp;
  }

  KOKKOS_FUNCTION
  half_t operator*(half_t rhs) const {
    half_t tmp = *this;
#ifdef __CUDA_ARCH__
    // printf("half_t operator*\n");
    tmp.val *= rhs.val;
#else
    // printf("float operator*\n");
    tmp.val = __float2half(__half2float(tmp.val) * __half2float(rhs.val));
#endif
    return tmp;
  }

  KOKKOS_FUNCTION
  half_t operator/(half_t rhs) const {
    half_t tmp = *this;
#ifdef __CUDA_ARCH__
    // printf("half_t operator/\n");
    tmp.val /= rhs.val;
#else
    // printf("float operator/\n");
    tmp.val = __float2half(__half2float(tmp.val) / __half2float(rhs.val));
#endif
    return tmp;
  }

  // Logical operators
  KOKKOS_FUNCTION
  half_t operator!() const {
    half_t tmp = *this;
#ifdef __CUDA_ARCH__
    // printf("half_t operator!\n");
    tmp.val = !tmp.val;
#else
    // printf("float operator!\n");
    tmp.val = __float2half(!__half2float(tmp.val));
#endif
    return tmp;
  }

#if 1
  // NOTE: Loses short-circuit evaluation
  KOKKOS_FUNCTION
  bool operator&&(half_t rhs) const {
    half_t tmp = *this;
#ifdef __CUDA_ARCH__
    // printf("half_t operator&&\n");
    return tmp.val && rhs.val;
#else
    // printf("float operator&&\n");
    return __half2float(tmp.val) && __half2float(rhs.val);
#endif
  }

  // NOTE: Loses short-circuit evaluation
  KOKKOS_FUNCTION
  bool operator||(half_t rhs) const {
    half_t tmp = *this;
#ifdef __CUDA_ARCH__
    // printf("half_t operator||\n");
    return tmp.val || rhs.val;
#else
    // printf("float operator||\n");
    return __half2float(tmp.val) || __half2float(rhs.val);
#endif
  }
#endif

  // Comparison operators
  KOKKOS_FUNCTION
  bool operator==(half_t rhs) const {
    half_t tmp = *this;
#ifdef __CUDA_ARCH__
    // printf("half_t operator==\n");
    return tmp.val == rhs.val;
#else
    // printf("float operator==\n");
    return __half2float(tmp.val) == __half2float(rhs.val);
#endif
  }

  KOKKOS_FUNCTION
  bool operator!=(half_t rhs) const {
    half_t tmp = *this;
#ifdef __CUDA_ARCH__
    // printf("half_t operator!=\n");
    return tmp.val != rhs.val;
#else
    // printf("float operator!=\n");
    return __half2float(tmp.val) != __half2float(rhs.val);
#endif
  }

  KOKKOS_FUNCTION
  bool operator<(half_t rhs) const {
    half_t tmp = *this;
#ifdef __CUDA_ARCH__
    // printf("half_t operator<\n");
    return tmp.val < rhs.val;
#else
    // printf("float operator<\n");
    return __half2float(tmp.val) < __half2float(rhs.val);
#endif
  }

  KOKKOS_FUNCTION
  bool operator>(half_t rhs) const {
    half_t tmp = *this;
#ifdef __CUDA_ARCH__
    // printf("half_t operator>\n");
    return tmp.val > rhs.val;
#else
    // printf("float operator>\n");
    return __half2float(tmp.val) > __half2float(rhs.val);
#endif
  }

  KOKKOS_FUNCTION
  bool operator<=(half_t rhs) const {
    half_t tmp = *this;
#ifdef __CUDA_ARCH__
    // printf("half_t operator<=\n");
    return tmp.val <= rhs.val;
#else
    // printf("float operator<=\n");
    return __half2float(tmp.val) <= __half2float(rhs.val);
#endif
  }

  KOKKOS_FUNCTION
  bool operator>=(half_t rhs) const {
    half_t tmp = *this;
#ifdef __CUDA_ARCH__
    // printf("half_t operator>=\n");
    return tmp.val >= rhs.val;
#else
    // printf("float operator>=\n");
    return __half2float(tmp.val) >= __half2float(rhs.val);
#endif
  }
};

constexpr const bool half_is_float = false;

// CUDA before 11.1 only has the half <-> float conversions marked host device
// So we will largely convert to float on the host for conversion
// But still call the correct functions on the device
#if (CUDA_VERSION < 11100)

KOKKOS_INLINE_FUNCTION
half_t cast_to_half(half_t val) { return val; }

KOKKOS_INLINE_FUNCTION
half_t cast_to_half(float val) { return __float2half(val); }

KOKKOS_INLINE_FUNCTION
half_t cast_to_half(double val) {
  // double2half was only introduced in CUDA 11 too
  return __float2half(static_cast<float>(val));
}

KOKKOS_INLINE_FUNCTION
half_t cast_to_half(short val) {
#ifdef __CUDA_ARCH__
  return __short2half_rn(val);
#else
  return __float2half(static_cast<float>(val));
#endif
}

KOKKOS_INLINE_FUNCTION
half_t cast_to_half(unsigned short val) {
#ifdef __CUDA_ARCH__
  return __ushort2half_rn(val);
#else
  return __float2half(static_cast<float>(val));
#endif
}

KOKKOS_INLINE_FUNCTION
half_t cast_to_half(int val) {
#ifdef __CUDA_ARCH__
  return __int2half_rn(val);
#else
  return __float2half(static_cast<float>(val));
#endif
}

KOKKOS_INLINE_FUNCTION
half_t cast_to_half(unsigned int val) {
#ifdef __CUDA_ARCH__
  return __uint2half_rn(val);
#else
  return __float2half(static_cast<float>(val));
#endif
}

KOKKOS_INLINE_FUNCTION
half_t cast_to_half(long long val) {
#ifdef __CUDA_ARCH__
  return __ll2half_rn(val);
#else
  return __float2half(static_cast<float>(val));
#endif
}

KOKKOS_INLINE_FUNCTION
half_t cast_to_half(unsigned long long val) {
#ifdef __CUDA_ARCH__
  return __ull2half_rn(val);
#else
  return __float2half(static_cast<float>(val));
#endif
}

KOKKOS_INLINE_FUNCTION
half_t cast_to_half(long val) {
  return cast_to_half(static_cast<long long>(val));
}

KOKKOS_INLINE_FUNCTION
half_t cast_to_half(unsigned long val) {
  return cast_to_half(static_cast<unsigned long long>(val));
}

template <class T>
KOKKOS_INLINE_FUNCTION
    typename std::enable_if<std::is_same<T, float>::value, T>::type
    cast_from_half(half_t val) {
  return __half2float(val);
}

template <class T>
KOKKOS_INLINE_FUNCTION
    typename std::enable_if<std::is_same<T, double>::value, T>::type
    cast_from_half(half_t val) {
  return static_cast<T>(__half2float(val));
}

template <class T>
KOKKOS_INLINE_FUNCTION
    typename std::enable_if<std::is_same<T, short>::value, T>::type
    cast_from_half(half_t val) {
#ifdef __CUDA_ARCH__
  return __half2short_rz(val);
#else
  return static_cast<T>(__half2float(val));
#endif
}

template <class T>
KOKKOS_INLINE_FUNCTION
    typename std::enable_if<std::is_same<T, unsigned short>::value, T>::type
    cast_from_half(half_t val) {
#ifdef __CUDA_ARCH__
  return __half2ushort_rz(val);
#else
  return static_cast<T>(__half2float(val));
#endif
}
template <class T>
KOKKOS_INLINE_FUNCTION
    typename std::enable_if<std::is_same<T, int>::value, T>::type
    cast_from_half(half_t val) {
#ifdef __CUDA_ARCH__
  return __half2int_rz(val);
#else
  return static_cast<T>(__half2float(val));
#endif
}

template <class T>
KOKKOS_INLINE_FUNCTION
    typename std::enable_if<std::is_same<T, unsigned>::value, T>::type
    cast_from_half(half_t val) {
#ifdef __CUDA_ARCH__
  return __half2uint_rz(val);
#else
  return static_cast<T>(__half2float(val));
#endif
}

template <class T>
KOKKOS_INLINE_FUNCTION
    typename std::enable_if<std::is_same<T, long long>::value, T>::type
    cast_from_half(half_t val) {
#ifdef __CUDA_ARCH__
  return __half2ll_rz(val);
#else
  return static_cast<T>(__half2float(val));
#endif
}

template <class T>
KOKKOS_INLINE_FUNCTION
    typename std::enable_if<std::is_same<T, unsigned long long>::value, T>::type
    cast_from_half(half_t val) {
#ifdef __CUDA_ARCH__
  return __half2ull_rz(val);
#else
  return static_cast<T>(__half2float(val));
#endif
}

template <class T>
KOKKOS_INLINE_FUNCTION
    typename std::enable_if<std::is_same<T, long>::value, T>::type
    cast_from_half(half_t val) {
  return static_cast<T>(cast_from_half<long long>(val));
}

template <class T>
KOKKOS_INLINE_FUNCTION
    typename std::enable_if<std::is_same<T, unsigned long>::value, T>::type
    cast_from_half(half_t val) {
  return static_cast<T>(cast_from_half<unsigned long long>(val));
}

#else  // CUDA 11.1 versions follow

KOKKOS_INLINE_FUNCTION
half_t cast_to_half(float val) { return __float2half(val); }
KOKKOS_INLINE_FUNCTION
half_t cast_to_half(double val) { return __double2half(val); }
KOKKOS_INLINE_FUNCTION
half_t cast_to_half(short val) { return __short2half_rn(val); }
KOKKOS_INLINE_FUNCTION
half_t cast_to_half(unsigned short val) { return __ushort2half_rn(val); }
KOKKOS_INLINE_FUNCTION
half_t cast_to_half(int val) { return __int2half_rn(val); }
KOKKOS_INLINE_FUNCTION
half_t cast_to_half(unsigned int val) { return __uint2half_rn(val); }
KOKKOS_INLINE_FUNCTION
half_t cast_to_half(long long val) { return __ll2half_rn(val); }
KOKKOS_INLINE_FUNCTION
half_t cast_to_half(unsigned long long val) { return __ull2half_rn(val); }
KOKKOS_INLINE_FUNCTION
half_t cast_to_half(long val) {
  return cast_to_half(static_cast<long long>(val));
}
KOKKOS_INLINE_FUNCTION
half_t cast_to_half(unsigned long val) {
  return cast_to_half(static_cast<unsigned long long>(val));
}

template <class T>
KOKKOS_INLINE_FUNCTION
    typename std::enable_if<std::is_same<T, float>::value, T>::type
    cast_from_half(half_t val) {
  return __half2float(val);
}
template <class T>
KOKKOS_INLINE_FUNCTION
    typename std::enable_if<std::is_same<T, double>::value, T>::type
    cast_from_half(half_t val) {
  return __half2double(val);
}
template <class T>
KOKKOS_INLINE_FUNCTION
    typename std::enable_if<std::is_same<T, short>::value, T>::type
    cast_from_half(half_t val) {
  return __half2short_rz(val);
}
template <class T>
KOKKOS_INLINE_FUNCTION
    typename std::enable_if<std::is_same<T, unsigned short>::value, T>::type
    cast_from_half(half_t val) {
  return __half2ushort_rz(val);
}
template <class T>
KOKKOS_INLINE_FUNCTION
    typename std::enable_if<std::is_same<T, int>::value, T>::type
    cast_from_half(half_t val) {
  return __half2int_rz(val);
}
template <class T>
KOKKOS_INLINE_FUNCTION
    typename std::enable_if<std::is_same<T, unsigned int>::value, T>::type
    cast_from_half(half_t val) {
  return __half2uint_rz(val);
}
template <class T>
KOKKOS_INLINE_FUNCTION
    typename std::enable_if<std::is_same<T, long long>::value, T>::type
    cast_from_half(half_t val) {
  return __half2ll_rz(val);
}
template <class T>
KOKKOS_INLINE_FUNCTION
    typename std::enable_if<std::is_same<T, unsigned long long>::value, T>::type
    cast_from_half(half_t val) {
  return __half2ull_rz(val);
}
template <class T>
KOKKOS_INLINE_FUNCTION
    typename std::enable_if<std::is_same<T, long>::value, T>::type
    cast_from_half(half_t val) {
  return static_cast<T>(cast_from_half<long long>(val));
}
template <class T>
KOKKOS_INLINE_FUNCTION
    typename std::enable_if<std::is_same<T, unsigned long>::value, T>::type
    cast_from_half(half_t val) {
  return static_cast<T>(cast_from_half<unsigned long long>(val));
}
#endif
}  // namespace Experimental
}  // namespace Kokkos
#endif  // KOKKOS_IMPL_HALF_TYPE_DEFINED
#endif  // KOKKOS_ENABLE_CUDA
#endif
