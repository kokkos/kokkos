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
#if !(defined(KOKKOS_COMPILER_CLANG) && KOKKOS_COMPILER_CLANG < 900) && \
    !(defined(KOKKOS_ARCH_KEPLER) || defined(KOKKOS_ARCH_MAXWELL50) ||  \
      defined(KOKKOS_ARCH_MAXWELL52))
#include <cuda_fp16.h>
#include <iosfwd>  // istream & ostream for extraction and insertion ops
#include <string>
#include <Kokkos_NumericTraits.hpp>  // reduction_identity

#ifndef KOKKOS_IMPL_HALF_TYPE_DEFINED
// Make sure no one else tries to define half_t
#define KOKKOS_IMPL_HALF_TYPE_DEFINED

namespace Kokkos {
namespace Impl {
struct half_impl_t {
  using type = __half;
};
}  // namespace Impl
namespace Experimental {

// Forward declarations
class half_t;

KOKKOS_INLINE_FUNCTION
half_t cast_to_half(float val);
KOKKOS_INLINE_FUNCTION
half_t cast_to_half(bool val);
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

template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, float>::value, T>
    cast_from_half(half_t);
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, bool>::value, T>
    cast_from_half(half_t);
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, double>::value, T>
    cast_from_half(half_t);
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, short>::value, T>
    cast_from_half(half_t);
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, int>::value, T>
    cast_from_half(half_t);
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, long>::value, T>
    cast_from_half(half_t);
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, long long>::value, T>
    cast_from_half(half_t);
template <class T>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<std::is_same<T, unsigned short>::value, T>
        cast_from_half(half_t);
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, unsigned int>::value, T>
    cast_from_half(half_t);
template <class T>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<std::is_same<T, unsigned long>::value, T>
        cast_from_half(half_t);
template <class T>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<std::is_same<T, unsigned long long>::value, T>
        cast_from_half(half_t);

class alignas(2) half_t {
 public:
  using impl_type = Kokkos::Impl::half_impl_t::type;

 private:
  impl_type val;

 public:
  KOKKOS_FUNCTION
  half_t() : val(0.0F) {}

  // Copy constructors
  KOKKOS_DEFAULTED_FUNCTION
  half_t(const half_t&) noexcept = default;

  KOKKOS_INLINE_FUNCTION
  half_t(const volatile half_t& rhs) {
#ifdef __CUDA_ARCH__
    val = rhs.val;
#else
    const volatile uint16_t* rv_ptr =
        reinterpret_cast<const volatile uint16_t*>(&rhs.val);
    const uint16_t rv_val = *rv_ptr;
    val                   = reinterpret_cast<const impl_type&>(rv_val);
#endif  // __CUDA_ARCH__
  }

  // Don't support implicit conversion back to impl_type.
  // impl_type is a storage only type on host.
  KOKKOS_FUNCTION
  explicit operator impl_type() const { return val; }
  KOKKOS_FUNCTION
  explicit operator float() const { return cast_from_half<float>(*this); }
  KOKKOS_FUNCTION
  explicit operator bool() const { return cast_from_half<bool>(*this); }
  KOKKOS_FUNCTION
  explicit operator double() const { return cast_from_half<double>(*this); }
  KOKKOS_FUNCTION
  explicit operator short() const { return cast_from_half<short>(*this); }
  KOKKOS_FUNCTION
  explicit operator int() const { return cast_from_half<int>(*this); }
  KOKKOS_FUNCTION
  explicit operator long() const { return cast_from_half<long>(*this); }
  KOKKOS_FUNCTION
  explicit operator long long() const {
    return cast_from_half<long long>(*this);
  }
  KOKKOS_FUNCTION
  explicit operator unsigned short() const {
    return cast_from_half<unsigned short>(*this);
  }
  KOKKOS_FUNCTION
  explicit operator unsigned int() const {
    return cast_from_half<unsigned int>(*this);
  }
  KOKKOS_FUNCTION
  explicit operator unsigned long() const {
    return cast_from_half<unsigned long>(*this);
  }
  KOKKOS_FUNCTION
  explicit operator unsigned long long() const {
    return cast_from_half<unsigned long long>(*this);
  }

  /**
   * Conversion constructors.
   *
   * Support implicit conversions from impl_type, float, double -> half_t
   * Mixed precision expressions require upcasting which is done in the
   * "// Binary Arithmetic" operator overloads below.
   *
   * Support implicit conversions from integral types -> half_t.
   * Expressions involving half_t with integral types require downcasting
   * the integral types to half_t. Existing operator overloads can handle this
   * with the addition of the below implicit conversion constructors.
   */
  KOKKOS_FUNCTION
  half_t(impl_type rhs) : val(rhs) {}
  KOKKOS_FUNCTION
  half_t(float rhs) : val(cast_to_half(rhs).val) {}
  KOKKOS_FUNCTION
  half_t(double rhs) : val(cast_to_half(rhs).val) {}
  KOKKOS_FUNCTION
  explicit half_t(bool rhs) : val(cast_to_half(rhs).val) {}
  KOKKOS_FUNCTION
  half_t(short rhs) : val(cast_to_half(rhs).val) {}
  KOKKOS_FUNCTION
  half_t(int rhs) : val(cast_to_half(rhs).val) {}
  KOKKOS_FUNCTION
  half_t(long rhs) : val(cast_to_half(rhs).val) {}
  KOKKOS_FUNCTION
  half_t(long long rhs) : val(cast_to_half(rhs).val) {}
  KOKKOS_FUNCTION
  half_t(unsigned short rhs) : val(cast_to_half(rhs).val) {}
  KOKKOS_FUNCTION
  half_t(unsigned int rhs) : val(cast_to_half(rhs).val) {}
  KOKKOS_FUNCTION
  half_t(unsigned long rhs) : val(cast_to_half(rhs).val) {}
  KOKKOS_FUNCTION
  half_t(unsigned long long rhs) : val(cast_to_half(rhs).val) {}

  // Unary operators
  KOKKOS_FUNCTION
  half_t operator+() const {
    half_t tmp = *this;
#ifdef __CUDA_ARCH__
    tmp.val = +tmp.val;
#else
    tmp.val               = __float2half(+__half2float(tmp.val));
#endif
    return tmp;
  }

  KOKKOS_FUNCTION
  half_t operator-() const {
    half_t tmp = *this;
#ifdef __CUDA_ARCH__
    tmp.val = -tmp.val;
#else
    tmp.val               = __float2half(-__half2float(tmp.val));
#endif
    return tmp;
  }

  // Prefix operators
  KOKKOS_FUNCTION
  half_t& operator++() {
#ifdef __CUDA_ARCH__
    ++val;
#else
    float tmp             = __half2float(val);
    ++tmp;
    val       = __float2half(tmp);
#endif
    return *this;
  }

  KOKKOS_FUNCTION
  half_t& operator--() {
#ifdef __CUDA_ARCH__
    --val;
#else
    float tmp = __half2float(val);
    --tmp;
    val     = __float2half(tmp);
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
  half_t& operator=(impl_type rhs) {
    val = rhs;
    return *this;
  }

  template <class T>
  KOKKOS_FUNCTION half_t& operator=(T rhs) {
    val = cast_to_half(rhs).val;
    return *this;
  }

  template <class T>
  KOKKOS_FUNCTION void operator=(T rhs) volatile {
    impl_type new_val = cast_to_half(rhs).val;
    volatile uint16_t* val_ptr =
        reinterpret_cast<volatile uint16_t*>(const_cast<impl_type*>(&val));
    *val_ptr = reinterpret_cast<uint16_t&>(new_val);
  }

  // Compound operators
  KOKKOS_FUNCTION
  half_t& operator+=(half_t rhs) {
#ifdef __CUDA_ARCH__
    val += rhs.val;
#else
    val     = __float2half(__half2float(val) + __half2float(rhs.val));
#endif
    return *this;
  }

  KOKKOS_FUNCTION
  void operator+=(const volatile half_t& rhs) volatile {
    half_t tmp_rhs = rhs;
    half_t tmp_lhs = *this;

    tmp_lhs += tmp_rhs;
    *this = tmp_lhs;
  }

  // Compound operators: upcast overloads for +=
  template <class T>
  KOKKOS_FUNCTION std::enable_if_t<
      std::is_same<T, float>::value || std::is_same<T, double>::value, T> friend
  operator+=(T& lhs, half_t rhs) {
    lhs += static_cast<T>(rhs);
    return lhs;
  }

  KOKKOS_FUNCTION
  half_t& operator+=(float rhs) {
    float result = static_cast<float>(val) + rhs;
    val          = static_cast<impl_type>(result);
    return *this;
  }

  KOKKOS_FUNCTION
  half_t& operator+=(double rhs) {
    double result = static_cast<double>(val) + rhs;
    val           = static_cast<impl_type>(result);
    return *this;
  }

  KOKKOS_FUNCTION
  half_t& operator-=(half_t rhs) {
#ifdef __CUDA_ARCH__
    val -= rhs.val;
#else
    val     = __float2half(__half2float(val) - __half2float(rhs.val));
#endif
    return *this;
  }

  KOKKOS_FUNCTION
  void operator-=(const volatile half_t& rhs) volatile {
    half_t tmp_rhs = rhs;
    half_t tmp_lhs = *this;

    tmp_lhs -= tmp_rhs;
    *this = tmp_lhs;
  }

  // Compund operators: upcast overloads for -=
  template <class T>
  KOKKOS_FUNCTION std::enable_if_t<
      std::is_same<T, float>::value || std::is_same<T, double>::value, T> friend
  operator-=(T& lhs, half_t rhs) {
    lhs -= static_cast<T>(rhs);
    return lhs;
  }

  KOKKOS_FUNCTION
  half_t& operator-=(float rhs) {
    float result = static_cast<float>(val) - rhs;
    val          = static_cast<impl_type>(result);
    return *this;
  }

  KOKKOS_FUNCTION
  half_t& operator-=(double rhs) {
    double result = static_cast<double>(val) - rhs;
    val           = static_cast<impl_type>(result);
    return *this;
  }

  KOKKOS_FUNCTION
  half_t& operator*=(half_t rhs) {
#ifdef __CUDA_ARCH__
    val *= rhs.val;
#else
    val     = __float2half(__half2float(val) * __half2float(rhs.val));
#endif
    return *this;
  }

  KOKKOS_FUNCTION
  void operator*=(const volatile half_t& rhs) volatile {
    half_t tmp_rhs = rhs;
    half_t tmp_lhs = *this;

    tmp_lhs *= tmp_rhs;
    *this = tmp_lhs;
  }

  // Compund operators: upcast overloads for *=
  template <class T>
  KOKKOS_FUNCTION std::enable_if_t<
      std::is_same<T, float>::value || std::is_same<T, double>::value, T> friend
  operator*=(T& lhs, half_t rhs) {
    lhs *= static_cast<T>(rhs);
    return lhs;
  }

  KOKKOS_FUNCTION
  half_t& operator*=(float rhs) {
    float result = static_cast<float>(val) * rhs;
    val          = static_cast<impl_type>(result);
    return *this;
  }

  KOKKOS_FUNCTION
  half_t& operator*=(double rhs) {
    double result = static_cast<double>(val) * rhs;
    val           = static_cast<impl_type>(result);
    return *this;
  }

  KOKKOS_FUNCTION
  half_t& operator/=(half_t rhs) {
#ifdef __CUDA_ARCH__
    val /= rhs.val;
#else
    val     = __float2half(__half2float(val) / __half2float(rhs.val));
#endif
    return *this;
  }

  KOKKOS_FUNCTION
  void operator/=(const volatile half_t& rhs) volatile {
    half_t tmp_rhs = rhs;
    half_t tmp_lhs = *this;

    tmp_lhs /= tmp_rhs;
    *this = tmp_lhs;
  }

  // Compund operators: upcast overloads for /=
  template <class T>
  KOKKOS_FUNCTION std::enable_if_t<
      std::is_same<T, float>::value || std::is_same<T, double>::value, T> friend
  operator/=(T& lhs, half_t rhs) {
    lhs /= static_cast<T>(rhs);
    return lhs;
  }

  KOKKOS_FUNCTION
  half_t& operator/=(float rhs) {
    float result = static_cast<float>(val) / rhs;
    val          = static_cast<impl_type>(result);
    return *this;
  }

  KOKKOS_FUNCTION
  half_t& operator/=(double rhs) {
    double result = static_cast<double>(val) / rhs;
    val           = static_cast<impl_type>(result);
    return *this;
  }

  // Binary Arithmetic
  KOKKOS_FUNCTION
  half_t friend operator+(half_t lhs, half_t rhs) {
#ifdef __CUDA_ARCH__
    lhs.val += rhs.val;
#else
    lhs.val = __float2half(__half2float(lhs.val) + __half2float(rhs.val));
#endif
    return lhs;
  }

  // Binary Arithmetic upcast operators for +
  template <class T>
  KOKKOS_FUNCTION std::enable_if_t<
      std::is_same<T, float>::value || std::is_same<T, double>::value, T> friend
  operator+(half_t lhs, T rhs) {
    return T(lhs) + rhs;
  }

  template <class T>
  KOKKOS_FUNCTION std::enable_if_t<
      std::is_same<T, float>::value || std::is_same<T, double>::value, T> friend
  operator+(T lhs, half_t rhs) {
    return lhs + T(rhs);
  }

  KOKKOS_FUNCTION
  half_t friend operator-(half_t lhs, half_t rhs) {
#ifdef __CUDA_ARCH__
    lhs.val -= rhs.val;
#else
    lhs.val = __float2half(__half2float(lhs.val) - __half2float(rhs.val));
#endif
    return lhs;
  }

  // Binary Arithmetic upcast operators for -
  template <class T>
  KOKKOS_FUNCTION std::enable_if_t<
      std::is_same<T, float>::value || std::is_same<T, double>::value, T> friend
  operator-(half_t lhs, T rhs) {
    return T(lhs) - rhs;
  }

  template <class T>
  KOKKOS_FUNCTION std::enable_if_t<
      std::is_same<T, float>::value || std::is_same<T, double>::value, T> friend
  operator-(T lhs, half_t rhs) {
    return lhs - T(rhs);
  }

  KOKKOS_FUNCTION
  half_t friend operator*(half_t lhs, half_t rhs) {
#ifdef __CUDA_ARCH__
    lhs.val *= rhs.val;
#else
    lhs.val = __float2half(__half2float(lhs.val) * __half2float(rhs.val));
#endif
    return lhs;
  }

  // Binary Arithmetic upcast operators for *
  template <class T>
  KOKKOS_FUNCTION std::enable_if_t<
      std::is_same<T, float>::value || std::is_same<T, double>::value, T> friend
  operator*(half_t lhs, T rhs) {
    return T(lhs) * rhs;
  }

  template <class T>
  KOKKOS_FUNCTION std::enable_if_t<
      std::is_same<T, float>::value || std::is_same<T, double>::value, T> friend
  operator*(T lhs, half_t rhs) {
    return lhs * T(rhs);
  }

  KOKKOS_FUNCTION
  half_t friend operator/(half_t lhs, half_t rhs) {
#ifdef __CUDA_ARCH__
    lhs.val /= rhs.val;
#else
    lhs.val = __float2half(__half2float(lhs.val) / __half2float(rhs.val));
#endif
    return lhs;
  }

  // Binary Arithmetic upcast operators for /
  template <class T>
  KOKKOS_FUNCTION std::enable_if_t<
      std::is_same<T, float>::value || std::is_same<T, double>::value, T> friend
  operator/(half_t lhs, T rhs) {
    return T(lhs) / rhs;
  }

  template <class T>
  KOKKOS_FUNCTION std::enable_if_t<
      std::is_same<T, float>::value || std::is_same<T, double>::value, T> friend
  operator/(T lhs, half_t rhs) {
    return lhs / T(rhs);
  }

  // Logical operators
  KOKKOS_FUNCTION
  bool operator!() const {
#ifdef __CUDA_ARCH__
    return static_cast<bool>(!val);
#else
    return !__half2float(val);
#endif
  }

  // NOTE: Loses short-circuit evaluation
  KOKKOS_FUNCTION
  bool operator&&(half_t rhs) const {
#ifdef __CUDA_ARCH__
    return static_cast<bool>(val && rhs.val);
#else
    return __half2float(val) && __half2float(rhs.val);
#endif
  }

  // NOTE: Loses short-circuit evaluation
  KOKKOS_FUNCTION
  bool operator||(half_t rhs) const {
#ifdef __CUDA_ARCH__
    return static_cast<bool>(val || rhs.val);
#else
    return __half2float(val) || __half2float(rhs.val);
#endif
  }

  // Comparison operators
  KOKKOS_FUNCTION
  bool operator==(half_t rhs) const {
#ifdef __CUDA_ARCH__
    return static_cast<bool>(val == rhs.val);
#else
    return __half2float(val) == __half2float(rhs.val);
#endif
  }

  KOKKOS_FUNCTION
  bool operator!=(half_t rhs) const {
#ifdef __CUDA_ARCH__
    return static_cast<bool>(val != rhs.val);
#else
    return __half2float(val) != __half2float(rhs.val);
#endif
  }

  KOKKOS_FUNCTION
  bool operator<(half_t rhs) const {
#ifdef __CUDA_ARCH__
    return static_cast<bool>(val < rhs.val);
#else
    return __half2float(val) < __half2float(rhs.val);
#endif
  }

  KOKKOS_FUNCTION
  bool operator>(half_t rhs) const {
#ifdef __CUDA_ARCH__
    return static_cast<bool>(val > rhs.val);
#else
    return __half2float(val) > __half2float(rhs.val);
#endif
  }

  KOKKOS_FUNCTION
  bool operator<=(half_t rhs) const {
#ifdef __CUDA_ARCH__
    return static_cast<bool>(val <= rhs.val);
#else
    return __half2float(val) <= __half2float(rhs.val);
#endif
  }

  KOKKOS_FUNCTION
  bool operator>=(half_t rhs) const {
#ifdef __CUDA_ARCH__
    return static_cast<bool>(val >= rhs.val);
#else
    return __half2float(val) >= __half2float(rhs.val);
#endif
  }

  KOKKOS_FUNCTION
  friend bool operator==(const volatile half_t& lhs,
                         const volatile half_t& rhs) {
    half_t tmp_lhs = lhs, tmp_rhs = rhs;
    return tmp_lhs == tmp_rhs;
  }

  KOKKOS_FUNCTION
  friend bool operator!=(const volatile half_t& lhs,
                         const volatile half_t& rhs) {
    half_t tmp_lhs = lhs, tmp_rhs = rhs;
    return tmp_lhs != tmp_rhs;
  }

  KOKKOS_FUNCTION
  friend bool operator<(const volatile half_t& lhs,
                        const volatile half_t& rhs) {
    half_t tmp_lhs = lhs, tmp_rhs = rhs;
    return tmp_lhs < tmp_rhs;
  }

  KOKKOS_FUNCTION
  friend bool operator>(const volatile half_t& lhs,
                        const volatile half_t& rhs) {
    half_t tmp_lhs = lhs, tmp_rhs = rhs;
    return tmp_lhs > tmp_rhs;
  }

  KOKKOS_FUNCTION
  friend bool operator<=(const volatile half_t& lhs,
                         const volatile half_t& rhs) {
    half_t tmp_lhs = lhs, tmp_rhs = rhs;
    return tmp_lhs <= tmp_rhs;
  }

  KOKKOS_FUNCTION
  friend bool operator>=(const volatile half_t& lhs,
                         const volatile half_t& rhs) {
    half_t tmp_lhs = lhs, tmp_rhs = rhs;
    return tmp_lhs >= tmp_rhs;
  }

  // Insertion and extraction operators
  friend std::ostream& operator<<(std::ostream& os, const half_t& x) {
    const std::string out = std::to_string(static_cast<double>(x));
    os << out;
    return os;
  }

  friend std::istream& operator>>(std::istream& is, half_t& x) {
    std::string in;
    is >> in;
    x = std::stod(in);
    return is;
  }
};

// CUDA before 11.1 only has the half <-> float conversions marked host device
// So we will largely convert to float on the host for conversion
// But still call the correct functions on the device
#if (CUDA_VERSION < 11100)

KOKKOS_INLINE_FUNCTION
half_t cast_to_half(half_t val) { return val; }

KOKKOS_INLINE_FUNCTION
half_t cast_to_half(float val) { return half_t(__float2half(val)); }

KOKKOS_INLINE_FUNCTION
half_t cast_to_half(bool val) { return cast_to_half(static_cast<float>(val)); }

KOKKOS_INLINE_FUNCTION
half_t cast_to_half(double val) {
  // double2half was only introduced in CUDA 11 too
  return half_t(__float2half(static_cast<float>(val)));
}

KOKKOS_INLINE_FUNCTION
half_t cast_to_half(short val) {
#ifdef __CUDA_ARCH__
  return half_t(__short2half_rn(val));
#else
  return half_t(__float2half(static_cast<float>(val)));
#endif
}

KOKKOS_INLINE_FUNCTION
half_t cast_to_half(unsigned short val) {
#ifdef __CUDA_ARCH__
  return half_t(__ushort2half_rn(val));
#else
  return half_t(__float2half(static_cast<float>(val)));
#endif
}

KOKKOS_INLINE_FUNCTION
half_t cast_to_half(int val) {
#ifdef __CUDA_ARCH__
  return half_t(__int2half_rn(val));
#else
  return half_t(__float2half(static_cast<float>(val)));
#endif
}

KOKKOS_INLINE_FUNCTION
half_t cast_to_half(unsigned int val) {
#ifdef __CUDA_ARCH__
  return half_t(__uint2half_rn(val));
#else
  return half_t(__float2half(static_cast<float>(val)));
#endif
}

KOKKOS_INLINE_FUNCTION
half_t cast_to_half(long long val) {
#ifdef __CUDA_ARCH__
  return half_t(__ll2half_rn(val));
#else
  return half_t(__float2half(static_cast<float>(val)));
#endif
}

KOKKOS_INLINE_FUNCTION
half_t cast_to_half(unsigned long long val) {
#ifdef __CUDA_ARCH__
  return half_t(__ull2half_rn(val));
#else
  return half_t(__float2half(static_cast<float>(val)));
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
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, float>::value, T>
cast_from_half(half_t val) {
  return __half2float(half_t::impl_type(val));
}

template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, bool>::value, T>
cast_from_half(half_t val) {
  return static_cast<T>(cast_from_half<float>(val));
}

template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, double>::value, T>
cast_from_half(half_t val) {
  return static_cast<T>(__half2float(half_t::impl_type(val)));
}

template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, short>::value, T>
cast_from_half(half_t val) {
#ifdef __CUDA_ARCH__
  return __half2short_rz(half_t::impl_type(val));
#else
  return static_cast<T>(__half2float(half_t::impl_type(val)));
#endif
}

template <class T>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<std::is_same<T, unsigned short>::value, T>
    cast_from_half(half_t val) {
#ifdef __CUDA_ARCH__
  return __half2ushort_rz(half_t::impl_type(val));
#else
  return static_cast<T>(__half2float(half_t::impl_type(val)));
#endif
}
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, int>::value, T>
cast_from_half(half_t val) {
#ifdef __CUDA_ARCH__
  return __half2int_rz(half_t::impl_type(val));
#else
  return static_cast<T>(__half2float(half_t::impl_type(val)));
#endif
}

template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, unsigned>::value, T>
cast_from_half(half_t val) {
#ifdef __CUDA_ARCH__
  return __half2uint_rz(half_t::impl_type(val));
#else
  return static_cast<T>(__half2float(half_t::impl_type(val)));
#endif
}

template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, long long>::value, T>
cast_from_half(half_t val) {
#ifdef __CUDA_ARCH__
  return __half2ll_rz(half_t::impl_type(val));
#else
  return static_cast<T>(__half2float(half_t::impl_type(val)));
#endif
}

template <class T>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<std::is_same<T, unsigned long long>::value, T>
    cast_from_half(half_t val) {
#ifdef __CUDA_ARCH__
  return __half2ull_rz(half_t::impl_type(val));
#else
  return static_cast<T>(__half2float(half_t::impl_type(val)));
#endif
}

template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, long>::value, T>
cast_from_half(half_t val) {
  return static_cast<T>(cast_from_half<long long>(val));
}

template <class T>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<std::is_same<T, unsigned long>::value, T>
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
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, float>::value, T>
cast_from_half(half_t val) {
  return __half2float(val);
}
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, double>::value, T>
cast_from_half(half_t val) {
  return __half2double(val);
}
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, short>::value, T>
cast_from_half(half_t val) {
  return __half2short_rz(val);
}
template <class T>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<std::is_same<T, unsigned short>::value, T>
    cast_from_half(half_t val) {
  return __half2ushort_rz(val);
}
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, int>::value, T>
cast_from_half(half_t val) {
  return __half2int_rz(val);
}
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, unsigned int>::value, T>
cast_from_half(half_t val) {
  return __half2uint_rz(val);
}
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, long long>::value, T>
cast_from_half(half_t val) {
  return __half2ll_rz(val);
}
template <class T>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<std::is_same<T, unsigned long long>::value, T>
    cast_from_half(half_t val) {
  return __half2ull_rz(val);
}
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, long>::value, T>
cast_from_half(half_t val) {
  return static_cast<T>(cast_from_half<long long>(val));
}
template <class T>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<std::is_same<T, unsigned long>::value, T>
    cast_from_half(half_t val) {
  return static_cast<T>(cast_from_half<unsigned long long>(val));
}
#endif
}  // namespace Experimental

// use float as the return type for sum and prod since cuda_fp16.h
// has no constexpr functions for casting to __half
template <>
struct reduction_identity<Kokkos::Experimental::half_t> {
  KOKKOS_FORCEINLINE_FUNCTION constexpr static float sum() noexcept {
    return 0.0F;
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static float prod() noexcept {
    return 1.0F;
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static float max() noexcept {
    return -65504.0F;
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static float min() noexcept {
    return 65504.0F;
  }
};

}  // namespace Kokkos
#endif  // KOKKOS_IMPL_HALF_TYPE_DEFINED
#endif  // KOKKOS_ENABLE_CUDA
#endif  // Disables for half_t on cuda:
        // Clang/8||KEPLER30||KEPLER32||KEPLER37||MAXWELL50||MAXWELL52
#endif
