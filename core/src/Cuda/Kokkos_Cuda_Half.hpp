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
#if (CUDA_VERSION >= 11000)
#include <cuda_bf16.h>
#endif             // CUDA_VERSION >= 11000
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
#if (CUDA_VERSION >= 11000)
struct bhalf_impl_t {
  using type = __nv_bfloat16;
};
#endif  // CUDA_VERSION >= 11000
}  // namespace Impl

namespace Experimental {
namespace Impl {
template <class FloatType>
class floating_point_wrapper;
}

/********************** BEGIN half forward declarations  **********************/
// Declare half_t (binary16)
using half_t = Kokkos::Experimental::Impl::floating_point_wrapper<
    Kokkos::Impl::half_impl_t ::type>;
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
KOKKOS_INLINE_FUNCTION
half_t cast_to_half(half_t);

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

// declare bhalf_t
#if (CUDA_VERSION >= 11000)
#define KOKKOS_IMPL_BHALF_TYPE_DEFINED
using bhalf_t = Kokkos::Experimental::Impl::floating_point_wrapper<
    Kokkos::Impl ::bhalf_impl_t ::type>;

KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(float val);
KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(bool val);
KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(double val);
KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(short val);
KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(int val);
KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(long val);
KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(long long val);
KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(unsigned short val);
KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(unsigned int val);
KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(unsigned long val);
KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(unsigned long long val);
KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(bhalf_t val);

template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, float>::value, T>
    cast_from_bhalf(bhalf_t);
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, bool>::value, T>
    cast_from_bhalf(bhalf_t);
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, double>::value, T>
    cast_from_bhalf(bhalf_t);
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, short>::value, T>
    cast_from_bhalf(bhalf_t);
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, int>::value, T>
    cast_from_bhalf(bhalf_t);
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, long>::value, T>
    cast_from_bhalf(bhalf_t);
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, long long>::value, T>
    cast_from_bhalf(bhalf_t);
template <class T>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<std::is_same<T, unsigned short>::value, T>
        cast_from_bhalf(bhalf_t);
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, unsigned int>::value, T>
    cast_from_bhalf(bhalf_t);
template <class T>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<std::is_same<T, unsigned long>::value, T>
        cast_from_bhalf(bhalf_t);
template <class T>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<std::is_same<T, unsigned long long>::value, T>
        cast_from_bhalf(bhalf_t);
#endif  // CUDA_VERSION < 11000
/*********************** END half forward declarations  ***********************/

namespace Impl {
template <class FloatType>
class alignas(FloatType) floating_point_wrapper {
 public:
  using impl_type = FloatType;

 private:
  impl_type val;
  using fixed_width_integer_type = std::conditional_t<
      sizeof(impl_type) == 2, uint16_t,
      std::conditional_t<
          sizeof(impl_type) == 4, uint32_t,
          std::conditional_t<sizeof(impl_type) == 8, uint64_t, void>>>;
  static_assert(!std::is_void<fixed_width_integer_type>::value,
                "Invalid impl_type");

  // BEGIN: Casting wrappers for supporting multiple impl types
  static KOKKOS_INLINE_FUNCTION Kokkos::Impl::half_impl_t::type float2impl(
      float x, Kokkos::Impl::half_impl_t::type&) {
    return __float2half(x);
  }

#if CUDA_VERSION >= 11000
  static KOKKOS_INLINE_FUNCTION Kokkos::Impl::bhalf_impl_t::type float2impl(
      float x, Kokkos::Impl::bhalf_impl_t::type&) {
    return __float2bfloat16(x);
  }
#endif  // CUDA_VERSION >= 11000

  static KOKKOS_INLINE_FUNCTION float impl2float(
      Kokkos::Impl::half_impl_t::type x) {
    return __half2float(x);
  }
#if CUDA_VERSION >= 11000
  static KOKKOS_INLINE_FUNCTION float impl2float(
      Kokkos::Impl::bhalf_impl_t::type x) {
    return __bfloat162float(x);
  }
#endif  // CUDA_VERSION >= 11000

  template <class T>
  static KOKKOS_INLINE_FUNCTION Kokkos::Experimental::half_t cast_to_wrapper(
      T x, volatile Kokkos::Impl::half_impl_t::type&) {
    return Kokkos::Experimental::cast_to_half(x);
  }

#if CUDA_VERSION >= 11000
  template <class T>
  static KOKKOS_INLINE_FUNCTION Kokkos::Experimental::bhalf_t cast_to_wrapper(
      T x, volatile Kokkos::Impl::bhalf_impl_t::type&) {
    return Kokkos::Experimental::cast_to_bhalf(x);
  }
#endif  // CUDA_VERSION >= 11000

  template <class T>
  static KOKKOS_INLINE_FUNCTION T
  cast_from_wrapper(const Kokkos::Experimental::half_t& x) {
    return Kokkos::Experimental::cast_from_half<T>(x);
  }

#if CUDA_VERSION >= 11000
  template <class T>
  static KOKKOS_INLINE_FUNCTION T
  cast_from_wrapper(const Kokkos::Experimental::bhalf_t& x) {
    return Kokkos::Experimental::cast_from_bhalf<T>(x);
  }
#endif  // CUDA_VERSION >= 11000
        // END: Casting wrappers for supporting multiple impl types

 public:
  KOKKOS_FUNCTION
  floating_point_wrapper() : val(0.0F) {}

// Copy constructors
// Getting "C2580: multiple versions of a defaulted special
// member function are not allowed" with VS 16.11.3 and CUDA 11.4.2
#if defined(_WIN32) && defined(KOKKOS_ENABLE_CUDA)
  KOKKOS_FUNCTION
  floating_point_wrapper(const floating_point_wrapper& rhs) : val(rhs.val) {}
#else
  KOKKOS_DEFAULTED_FUNCTION
  floating_point_wrapper(const floating_point_wrapper&) noexcept = default;
#endif

  KOKKOS_INLINE_FUNCTION
  floating_point_wrapper(const volatile floating_point_wrapper& rhs) {
#ifdef __CUDA_ARCH__
    val = rhs.val;
#else
    const volatile fixed_width_integer_type* rv_ptr =
        reinterpret_cast<const volatile fixed_width_integer_type*>(&rhs.val);
    const fixed_width_integer_type rv_val = *rv_ptr;
    val       = reinterpret_cast<const impl_type&>(rv_val);
#endif  // __CUDA_ARCH__
  }

  // Don't support implicit conversion back to impl_type.
  // impl_type is a storage only type on host.
  KOKKOS_FUNCTION
  explicit operator impl_type() const { return val; }
  KOKKOS_FUNCTION
  explicit operator float() const { return cast_from_wrapper<float>(*this); }
  KOKKOS_FUNCTION
  explicit operator bool() const { return cast_from_wrapper<bool>(*this); }
  KOKKOS_FUNCTION
  explicit operator double() const { return cast_from_wrapper<double>(*this); }
  KOKKOS_FUNCTION
  explicit operator short() const { return cast_from_wrapper<short>(*this); }
  KOKKOS_FUNCTION
  explicit operator int() const { return cast_from_wrapper<int>(*this); }
  KOKKOS_FUNCTION
  explicit operator long() const { return cast_from_wrapper<long>(*this); }
  KOKKOS_FUNCTION
  explicit operator long long() const {
    return cast_from_wrapper<long long>(*this);
  }
  KOKKOS_FUNCTION
  explicit operator unsigned short() const {
    return cast_from_wrapper<unsigned short>(*this);
  }
  KOKKOS_FUNCTION
  explicit operator unsigned int() const {
    return cast_from_wrapper<unsigned int>(*this);
  }
  KOKKOS_FUNCTION
  explicit operator unsigned long() const {
    return cast_from_wrapper<unsigned long>(*this);
  }
  KOKKOS_FUNCTION
  explicit operator unsigned long long() const {
    return cast_from_wrapper<unsigned long long>(*this);
  }

  /**
   * Conversion constructors.
   *
   * Support implicit conversions from impl_type, float, double ->
   * floating_point_wrapper. Mixed precision expressions require upcasting which
   * is done in the
   * "// Binary Arithmetic" operator overloads below.
   *
   * Support implicit conversions from integral types -> floating_point_wrapper.
   * Expressions involving floating_point_wrapper with integral types require
   * downcasting the integral types to floating_point_wrapper. Existing operator
   * overloads can handle this with the addition of the below implicit
   * conversion constructors.
   */
  KOKKOS_FUNCTION
  floating_point_wrapper(impl_type rhs) : val(rhs) {}
  KOKKOS_FUNCTION
  floating_point_wrapper(float rhs) : val(cast_to_wrapper(rhs, val).val) {}
  KOKKOS_FUNCTION
  floating_point_wrapper(double rhs) : val(cast_to_wrapper(rhs, val).val) {}
  KOKKOS_FUNCTION
  explicit floating_point_wrapper(bool rhs)
      : val(cast_to_wrapper(rhs, val).val) {}
  KOKKOS_FUNCTION
  floating_point_wrapper(short rhs) : val(cast_to_wrapper(rhs, val).val) {}
  KOKKOS_FUNCTION
  floating_point_wrapper(int rhs) : val(cast_to_wrapper(rhs, val).val) {}
  KOKKOS_FUNCTION
  floating_point_wrapper(long rhs) : val(cast_to_wrapper(rhs, val).val) {}
  KOKKOS_FUNCTION
  floating_point_wrapper(long long rhs) : val(cast_to_wrapper(rhs, val).val) {}
  KOKKOS_FUNCTION
  floating_point_wrapper(unsigned short rhs)
      : val(cast_to_wrapper(rhs, val).val) {}
  KOKKOS_FUNCTION
  floating_point_wrapper(unsigned int rhs)
      : val(cast_to_wrapper(rhs, val).val) {}
  KOKKOS_FUNCTION
  floating_point_wrapper(unsigned long rhs)
      : val(cast_to_wrapper(rhs, val).val) {}
  KOKKOS_FUNCTION
  floating_point_wrapper(unsigned long long rhs)
      : val(cast_to_wrapper(rhs, val).val) {}

  // Unary operators
  KOKKOS_FUNCTION
  floating_point_wrapper operator+() const {
    floating_point_wrapper tmp = *this;
#ifdef __CUDA_ARCH__
    tmp.val = +tmp.val;
#else
    tmp.val   = float2impl(+impl2float(tmp.val), tmp.val);
#endif
    return tmp;
  }

  KOKKOS_FUNCTION
  floating_point_wrapper operator-() const {
    floating_point_wrapper tmp = *this;
#ifdef __CUDA_ARCH__
    tmp.val = -tmp.val;
#else
    tmp.val   = float2impl(-impl2float(tmp.val), tmp.val);
#endif
    return tmp;
  }

  // Prefix operators
  KOKKOS_FUNCTION
  floating_point_wrapper& operator++() {
#ifdef __CUDA_ARCH__
    val = val + impl_type(1.0F);  // cuda has no operator++ for __nv_bfloat
#else
    float tmp = impl2float(val);
    ++tmp;
    val       = float2impl(tmp, val);
#endif
    return *this;
  }

  KOKKOS_FUNCTION
  floating_point_wrapper& operator--() {
#ifdef __CUDA_ARCH__
    val = val - impl_type(1.0F);  // cuda has no operator-- for __nv_bfloat
#else
    float tmp = impl2float(val);
    --tmp;
    val     = float2impl(tmp, val);
#endif
    return *this;
  }

  // Postfix operators
  KOKKOS_FUNCTION
  floating_point_wrapper operator++(int) {
    floating_point_wrapper tmp = *this;
    operator++();
    return tmp;
  }

  KOKKOS_FUNCTION
  floating_point_wrapper operator--(int) {
    floating_point_wrapper tmp = *this;
    operator--();
    return tmp;
  }

  // Binary operators
  KOKKOS_FUNCTION
  floating_point_wrapper& operator=(impl_type rhs) {
    val = rhs;
    return *this;
  }

  template <class T>
  KOKKOS_FUNCTION floating_point_wrapper& operator=(T rhs) {
    val = cast_to_wrapper(rhs, val).val;
    return *this;
  }

  template <class T>
  KOKKOS_FUNCTION void operator=(T rhs) volatile {
    impl_type new_val = cast_to_wrapper(rhs, val).val;
    volatile fixed_width_integer_type* val_ptr =
        reinterpret_cast<volatile fixed_width_integer_type*>(
            const_cast<impl_type*>(&val));
    *val_ptr = reinterpret_cast<fixed_width_integer_type&>(new_val);
  }

  // Compound operators
  KOKKOS_FUNCTION
  floating_point_wrapper& operator+=(floating_point_wrapper rhs) {
#ifdef __CUDA_ARCH__
    val = val + rhs.val;  // cuda has no operator+= for __nv_bfloat
#else
    val     = float2impl(impl2float(val) + impl2float(rhs.val), val);
#endif
    return *this;
  }

  KOKKOS_FUNCTION
  void operator+=(const volatile floating_point_wrapper& rhs) volatile {
    floating_point_wrapper tmp_rhs = rhs;
    floating_point_wrapper tmp_lhs = *this;

    tmp_lhs += tmp_rhs;
    *this = tmp_lhs;
  }

  // Compound operators: upcast overloads for +=
  template <class T>
  KOKKOS_FUNCTION std::enable_if_t<
      std::is_same<T, float>::value || std::is_same<T, double>::value, T> friend
  operator+=(T& lhs, floating_point_wrapper rhs) {
    lhs += static_cast<T>(rhs);
    return lhs;
  }

  KOKKOS_FUNCTION
  floating_point_wrapper& operator+=(float rhs) {
    float result = static_cast<float>(val) + rhs;
    val          = static_cast<impl_type>(result);
    return *this;
  }

  KOKKOS_FUNCTION
  floating_point_wrapper& operator+=(double rhs) {
    double result = static_cast<double>(val) + rhs;
    val           = static_cast<impl_type>(result);
    return *this;
  }

  KOKKOS_FUNCTION
  floating_point_wrapper& operator-=(floating_point_wrapper rhs) {
#ifdef __CUDA_ARCH__
    val = val - rhs.val;  // cuda has no operator-= for __nv_bfloat
#else
    val     = float2impl(impl2float(val) - impl2float(rhs.val), val);
#endif
    return *this;
  }

  KOKKOS_FUNCTION
  void operator-=(const volatile floating_point_wrapper& rhs) volatile {
    floating_point_wrapper tmp_rhs = rhs;
    floating_point_wrapper tmp_lhs = *this;

    tmp_lhs -= tmp_rhs;
    *this = tmp_lhs;
  }

  // Compund operators: upcast overloads for -=
  template <class T>
  KOKKOS_FUNCTION std::enable_if_t<
      std::is_same<T, float>::value || std::is_same<T, double>::value, T> friend
  operator-=(T& lhs, floating_point_wrapper rhs) {
    lhs -= static_cast<T>(rhs);
    return lhs;
  }

  KOKKOS_FUNCTION
  floating_point_wrapper& operator-=(float rhs) {
    float result = static_cast<float>(val) - rhs;
    val          = static_cast<impl_type>(result);
    return *this;
  }

  KOKKOS_FUNCTION
  floating_point_wrapper& operator-=(double rhs) {
    double result = static_cast<double>(val) - rhs;
    val           = static_cast<impl_type>(result);
    return *this;
  }

  KOKKOS_FUNCTION
  floating_point_wrapper& operator*=(floating_point_wrapper rhs) {
#ifdef __CUDA_ARCH__
    val = val * rhs.val;  // cuda has no operator*= for __nv_bfloat
#else
    val     = float2impl(impl2float(val) * impl2float(rhs.val), val);
#endif
    return *this;
  }

  KOKKOS_FUNCTION
  void operator*=(const volatile floating_point_wrapper& rhs) volatile {
    floating_point_wrapper tmp_rhs = rhs;
    floating_point_wrapper tmp_lhs = *this;

    tmp_lhs *= tmp_rhs;
    *this = tmp_lhs;
  }

  // Compund operators: upcast overloads for *=
  template <class T>
  KOKKOS_FUNCTION std::enable_if_t<
      std::is_same<T, float>::value || std::is_same<T, double>::value, T> friend
  operator*=(T& lhs, floating_point_wrapper rhs) {
    lhs *= static_cast<T>(rhs);
    return lhs;
  }

  KOKKOS_FUNCTION
  floating_point_wrapper& operator*=(float rhs) {
    float result = static_cast<float>(val) * rhs;
    val          = static_cast<impl_type>(result);
    return *this;
  }

  KOKKOS_FUNCTION
  floating_point_wrapper& operator*=(double rhs) {
    double result = static_cast<double>(val) * rhs;
    val           = static_cast<impl_type>(result);
    return *this;
  }

  KOKKOS_FUNCTION
  floating_point_wrapper& operator/=(floating_point_wrapper rhs) {
#ifdef __CUDA_ARCH__
    val = val / rhs.val;  // cuda has no operator/= for __nv_bfloat
#else
    val     = float2impl(impl2float(val) / impl2float(rhs.val), val);
#endif
    return *this;
  }

  KOKKOS_FUNCTION
  void operator/=(const volatile floating_point_wrapper& rhs) volatile {
    floating_point_wrapper tmp_rhs = rhs;
    floating_point_wrapper tmp_lhs = *this;

    tmp_lhs /= tmp_rhs;
    *this = tmp_lhs;
  }

  // Compund operators: upcast overloads for /=
  template <class T>
  KOKKOS_FUNCTION std::enable_if_t<
      std::is_same<T, float>::value || std::is_same<T, double>::value, T> friend
  operator/=(T& lhs, floating_point_wrapper rhs) {
    lhs /= static_cast<T>(rhs);
    return lhs;
  }

  KOKKOS_FUNCTION
  floating_point_wrapper& operator/=(float rhs) {
    float result = static_cast<float>(val) / rhs;
    val          = static_cast<impl_type>(result);
    return *this;
  }

  KOKKOS_FUNCTION
  floating_point_wrapper& operator/=(double rhs) {
    double result = static_cast<double>(val) / rhs;
    val           = static_cast<impl_type>(result);
    return *this;
  }

  // Binary Arithmetic
  KOKKOS_FUNCTION
  floating_point_wrapper friend operator+(floating_point_wrapper lhs,
                                          floating_point_wrapper rhs) {
#ifdef __CUDA_ARCH__
    lhs += rhs;
#else
    lhs.val = float2impl(impl2float(lhs.val) + impl2float(rhs.val), lhs.val);
#endif
    return lhs;
  }

  // Binary Arithmetic upcast operators for +
  template <class T>
  KOKKOS_FUNCTION std::enable_if_t<
      std::is_same<T, float>::value || std::is_same<T, double>::value, T> friend
  operator+(floating_point_wrapper lhs, T rhs) {
    return T(lhs) + rhs;
  }

  template <class T>
  KOKKOS_FUNCTION std::enable_if_t<
      std::is_same<T, float>::value || std::is_same<T, double>::value, T> friend
  operator+(T lhs, floating_point_wrapper rhs) {
    return lhs + T(rhs);
  }

  KOKKOS_FUNCTION
  floating_point_wrapper friend operator-(floating_point_wrapper lhs,
                                          floating_point_wrapper rhs) {
#ifdef __CUDA_ARCH__
    lhs -= rhs;
#else
    lhs.val = float2impl(impl2float(lhs.val) - impl2float(rhs.val), lhs.val);
#endif
    return lhs;
  }

  // Binary Arithmetic upcast operators for -
  template <class T>
  KOKKOS_FUNCTION std::enable_if_t<
      std::is_same<T, float>::value || std::is_same<T, double>::value, T> friend
  operator-(floating_point_wrapper lhs, T rhs) {
    return T(lhs) - rhs;
  }

  template <class T>
  KOKKOS_FUNCTION std::enable_if_t<
      std::is_same<T, float>::value || std::is_same<T, double>::value, T> friend
  operator-(T lhs, floating_point_wrapper rhs) {
    return lhs - T(rhs);
  }

  KOKKOS_FUNCTION
  floating_point_wrapper friend operator*(floating_point_wrapper lhs,
                                          floating_point_wrapper rhs) {
#ifdef __CUDA_ARCH__
    lhs *= rhs;
#else
    lhs.val = float2impl(impl2float(lhs.val) * impl2float(rhs.val), lhs.val);
#endif
    return lhs;
  }

  // Binary Arithmetic upcast operators for *
  template <class T>
  KOKKOS_FUNCTION std::enable_if_t<
      std::is_same<T, float>::value || std::is_same<T, double>::value, T> friend
  operator*(floating_point_wrapper lhs, T rhs) {
    return T(lhs) * rhs;
  }

  template <class T>
  KOKKOS_FUNCTION std::enable_if_t<
      std::is_same<T, float>::value || std::is_same<T, double>::value, T> friend
  operator*(T lhs, floating_point_wrapper rhs) {
    return lhs * T(rhs);
  }

  KOKKOS_FUNCTION
  floating_point_wrapper friend operator/(floating_point_wrapper lhs,
                                          floating_point_wrapper rhs) {
#ifdef __CUDA_ARCH__
    lhs /= rhs;
#else
    lhs.val = float2impl(impl2float(lhs.val) / impl2float(rhs.val), lhs.val);
#endif
    return lhs;
  }

  // Binary Arithmetic upcast operators for /
  template <class T>
  KOKKOS_FUNCTION std::enable_if_t<
      std::is_same<T, float>::value || std::is_same<T, double>::value, T> friend
  operator/(floating_point_wrapper lhs, T rhs) {
    return T(lhs) / rhs;
  }

  template <class T>
  KOKKOS_FUNCTION std::enable_if_t<
      std::is_same<T, float>::value || std::is_same<T, double>::value, T> friend
  operator/(T lhs, floating_point_wrapper rhs) {
    return lhs / T(rhs);
  }

  // Logical operators
  KOKKOS_FUNCTION
  bool operator!() const {
#ifdef __CUDA_ARCH__
    return static_cast<bool>(!val);
#else
    return !impl2float(val);
#endif
  }

  // NOTE: Loses short-circuit evaluation
  KOKKOS_FUNCTION
  bool operator&&(floating_point_wrapper rhs) const {
#ifdef __CUDA_ARCH__
    return static_cast<bool>(val && rhs.val);
#else
    return impl2float(val) && impl2float(rhs.val);
#endif
  }

  // NOTE: Loses short-circuit evaluation
  KOKKOS_FUNCTION
  bool operator||(floating_point_wrapper rhs) const {
#ifdef __CUDA_ARCH__
    return static_cast<bool>(val || rhs.val);
#else
    return impl2float(val) || impl2float(rhs.val);
#endif
  }

  // Comparison operators
  KOKKOS_FUNCTION
  bool operator==(floating_point_wrapper rhs) const {
#ifdef __CUDA_ARCH__
    return static_cast<bool>(val == rhs.val);
#else
    return impl2float(val) == impl2float(rhs.val);
#endif
  }

  KOKKOS_FUNCTION
  bool operator!=(floating_point_wrapper rhs) const {
#ifdef __CUDA_ARCH__
    return static_cast<bool>(val != rhs.val);
#else
    return impl2float(val) != impl2float(rhs.val);
#endif
  }

  KOKKOS_FUNCTION
  bool operator<(floating_point_wrapper rhs) const {
#ifdef __CUDA_ARCH__
    return static_cast<bool>(val < rhs.val);
#else
    return impl2float(val) < impl2float(rhs.val);
#endif
  }

  KOKKOS_FUNCTION
  bool operator>(floating_point_wrapper rhs) const {
#ifdef __CUDA_ARCH__
    return static_cast<bool>(val > rhs.val);
#else
    return impl2float(val) > impl2float(rhs.val);
#endif
  }

  KOKKOS_FUNCTION
  bool operator<=(floating_point_wrapper rhs) const {
#ifdef __CUDA_ARCH__
    return static_cast<bool>(val <= rhs.val);
#else
    return impl2float(val) <= impl2float(rhs.val);
#endif
  }

  KOKKOS_FUNCTION
  bool operator>=(floating_point_wrapper rhs) const {
#ifdef __CUDA_ARCH__
    return static_cast<bool>(val >= rhs.val);
#else
    return impl2float(val) >= impl2float(rhs.val);
#endif
  }

  KOKKOS_FUNCTION
  friend bool operator==(const volatile floating_point_wrapper& lhs,
                         const volatile floating_point_wrapper& rhs) {
    floating_point_wrapper tmp_lhs = lhs, tmp_rhs = rhs;
    return tmp_lhs == tmp_rhs;
  }

  KOKKOS_FUNCTION
  friend bool operator!=(const volatile floating_point_wrapper& lhs,
                         const volatile floating_point_wrapper& rhs) {
    floating_point_wrapper tmp_lhs = lhs, tmp_rhs = rhs;
    return tmp_lhs != tmp_rhs;
  }

  KOKKOS_FUNCTION
  friend bool operator<(const volatile floating_point_wrapper& lhs,
                        const volatile floating_point_wrapper& rhs) {
    floating_point_wrapper tmp_lhs = lhs, tmp_rhs = rhs;
    return tmp_lhs < tmp_rhs;
  }

  KOKKOS_FUNCTION
  friend bool operator>(const volatile floating_point_wrapper& lhs,
                        const volatile floating_point_wrapper& rhs) {
    floating_point_wrapper tmp_lhs = lhs, tmp_rhs = rhs;
    return tmp_lhs > tmp_rhs;
  }

  KOKKOS_FUNCTION
  friend bool operator<=(const volatile floating_point_wrapper& lhs,
                         const volatile floating_point_wrapper& rhs) {
    floating_point_wrapper tmp_lhs = lhs, tmp_rhs = rhs;
    return tmp_lhs <= tmp_rhs;
  }

  KOKKOS_FUNCTION
  friend bool operator>=(const volatile floating_point_wrapper& lhs,
                         const volatile floating_point_wrapper& rhs) {
    floating_point_wrapper tmp_lhs = lhs, tmp_rhs = rhs;
    return tmp_lhs >= tmp_rhs;
  }

  // Insertion and extraction operators
  friend std::ostream& operator<<(std::ostream& os,
                                  const floating_point_wrapper& x) {
    const std::string out = std::to_string(static_cast<double>(x));
    os << out;
    return os;
  }

  friend std::istream& operator>>(std::istream& is, floating_point_wrapper& x) {
    std::string in;
    is >> in;
    x = std::stod(in);
    return is;
  }
};
}  // namespace Impl

/************************** half conversions **********************************/
KOKKOS_INLINE_FUNCTION
half_t cast_to_half(half_t val) { return val; }

// CUDA before 11.1 only has the half <-> float conversions marked host device
// So we will largely convert to float on the host for conversion
// But still call the correct functions on the device
#if (CUDA_VERSION < 11100)

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

/************************** bhalf conversions *********************************/
#if CUDA_VERSION >= 11000 && CUDA_VERISON < 11200
KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(bhalf_t val) { return val; }

KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(float val) { return bhalf_t(__float2bfloat16(val)); }

KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(bool val) {
  return cast_to_bhalf(static_cast<float>(val));
}

KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(double val) {
  // double2bfloat16 was only introduced in CUDA 11 too
  return bhalf_t(__float2bfloat16(static_cast<float>(val)));
}

KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(short val) {
  return bhalf_t(__float2bfloat16(static_cast<float>(val)));
}

KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(unsigned short val) {
  return bhalf_t(__float2bfloat16(static_cast<float>(val)));
}

KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(int val) {
  return bhalf_t(__float2bfloat16(static_cast<float>(val)));
}

KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(unsigned int val) {
  return bhalf_t(__float2bfloat16(static_cast<float>(val)));
}

KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(long long val) {
  return bhalf_t(__float2bfloat16(static_cast<float>(val)));
}

KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(unsigned long long val) {
  return bhalf_t(__float2bfloat16(static_cast<float>(val)));
}

KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(long val) {
  return cast_to_bhalf(static_cast<long long>(val));
}

KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(unsigned long val) {
  return cast_to_bhalf(static_cast<unsigned long long>(val));
}

template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, float>::value, T>
cast_from_bhalf(bhalf_t val) {
  return __bfloat162float(bhalf_t::impl_type(val));
}

template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, bool>::value, T>
cast_from_bhalf(bhalf_t val) {
  return static_cast<T>(cast_from_bhalf<float>(val));
}

template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, double>::value, T>
cast_from_bhalf(bhalf_t val) {
  return static_cast<T>(__bfloat162float(bhalf_t::impl_type(val)));
}

template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, short>::value, T>
cast_from_bhalf(bhalf_t val) {
  return static_cast<T>(__bfloat162float(bhalf_t::impl_type(val)));
}

template <class T>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<std::is_same<T, unsigned short>::value, T>
    cast_from_bhalf(bhalf_t val) {
  return static_cast<T>(__bfloat162float(bhalf_t::impl_type(val)));
}
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, int>::value, T>
cast_from_bhalf(bhalf_t val) {
  return static_cast<T>(__bfloat162float(bhalf_t::impl_type(val)));
}

template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, unsigned>::value, T>
cast_from_bhalf(bhalf_t val) {
  return static_cast<T>(__bfloat162float(bhalf_t::impl_type(val)));
}

template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, long long>::value, T>
cast_from_bhalf(bhalf_t val) {
  return static_cast<T>(__bfloat162float(bhalf_t::impl_type(val)));
}

template <class T>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<std::is_same<T, unsigned long long>::value, T>
    cast_from_bhalf(bhalf_t val) {
  return static_cast<T>(__bfloat162float(bhalf_t::impl_type(val)));
}

template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, long>::value, T>
cast_from_bhalf(bhalf_t val) {
  return static_cast<T>(cast_from_bhalf<long long>(val));
}

template <class T>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<std::is_same<T, unsigned long>::value, T>
    cast_from_bhalf(bhalf_t val) {
  return static_cast<T>(cast_from_bhalf<unsigned long long>(val));
}
#endif  // CUDA_VERSION >= 11000 && CUDA_VERISON < 11200

#if CUDA_VERISON >= 11200
KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(bhalf_t val) { return val; }
KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(float val) { return __float2bfloat16(val); }
KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(double val) { return __double2bfloat16(val); }
KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(short val) { return __short2bfloat16_rn(val); }
KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(unsigned short val) { return __ushort2bfloat16_rn(val); }
KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(int val) { return __int2bfloat16_rn(val); }
KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(unsigned int val) { return __uint2bfloat16_rn(val); }
KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(long long val) { return __ll2bfloat16_rn(val); }
KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(unsigned long long val) { return __ull2bfloat16_rn(val); }
KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(long val) {
  return cast_to_bhalf(static_cast<long long>(val));
}
KOKKOS_INLINE_FUNCTION
bhalf_t cast_to_bhalf(unsigned long val) {
  return cast_to_bhalf(static_cast<unsigned long long>(val));
}

template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, float>::value, T>
cast_from_bhalf(bhalf_t val) {
  return __bfloat162float(val);
}
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, double>::value, T>
cast_from_bhalf(bhalf_t val) {
  return __bfloat162double(val);
}
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, short>::value, T>
cast_from_bhalf(bhalf_t val) {
  return __bfloat162short_rz(val);
}
template <class T>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<std::is_same<T, unsigned short>::value, T>
    cast_from_bhalf(bhalf_t val) {
  return __bfloat162ushort_rz(val);
}
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, int>::value, T>
cast_from_bhalf(bhalf_t val) {
  return __bfloat162int_rz(val);
}
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, unsigned int>::value, T>
cast_from_bhalf(bhalf_t val) {
  return __bfloat162uint_rz(val);
}
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, long long>::value, T>
cast_from_bhalf(bhalf_t val) {
  return __bfloat162ll_rz(val);
}
template <class T>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<std::is_same<T, unsigned long long>::value, T>
    cast_from_bhalf(bhalf_t val) {
  return __bfloat162ull_rz(val);
}
template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, long>::value, T>
cast_from_bhalf(bhalf_t val) {
  return static_cast<T>(cast_from_bhalf<long long>(val));
}
template <class T>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<std::is_same<T, unsigned long>::value, T>
    cast_from_bhalf(bhalf_t val) {
  return static_cast<T>(cast_from_bhalf<unsigned long long>(val));
}
#endif  // CUDA_VERSION >= 11200
}  // namespace Experimental

#if (CUDA_VERSION >= 11000)
template <>
struct reduction_identity<Kokkos::Experimental::bhalf_t> {
  KOKKOS_FORCEINLINE_FUNCTION constexpr static float sum() noexcept {
    return 0.0F;
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static float prod() noexcept {
    return 1.0F;
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static float max() noexcept {
    return -0x7f7f;
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static float min() noexcept {
    return 0x7f7f;
  }
};
#endif  // CUDA_VERSION >= 11000

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
