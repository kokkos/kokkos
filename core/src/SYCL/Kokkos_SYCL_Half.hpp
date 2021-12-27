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

#ifndef KOKKOS_SYCL_HALF_HPP_
#define KOKKOS_SYCL_HALF_HPP_

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_SYCL
#include <iosfwd>  // istream & ostream for extraction and insertion ops
#include <Kokkos_NumericTraits.hpp>  // reduction_identity

#ifndef KOKKOS_IMPL_HALF_TYPE_DEFINED
// Make sure no one else tries to define half_t
#define KOKKOS_IMPL_HALF_TYPE_DEFINED

namespace Kokkos {
namespace Impl {
struct half_impl_t {
  using type = sycl::half;
};
}  // namespace Impl
namespace Experimental {

class alignas(2) half_t {
 public:
  using impl_type = Kokkos::Impl::half_impl_t::type;

 private:
  impl_type val;

 public:
  KOKKOS_FUNCTION
  constexpr half_t() : val(0.0F) {}

  KOKKOS_INLINE_FUNCTION
  half_t(const volatile half_t& rhs) {
    val = const_cast<const impl_type&>(rhs.val);
  }

  // Don't support implicit conversion back to impl_type.
  // impl_type is a storage only type on host.
  KOKKOS_FUNCTION
  explicit operator impl_type() const { return val; }
  KOKKOS_FUNCTION
  explicit operator float() const { return val; }
  KOKKOS_FUNCTION
  explicit operator bool() const { return val; }
  KOKKOS_FUNCTION
  explicit operator double() const { return val; }
  KOKKOS_FUNCTION
  explicit operator short() const { return val; }
  KOKKOS_FUNCTION
  explicit operator int() const { return val; }
  KOKKOS_FUNCTION
  explicit operator long() const { return val; }
  KOKKOS_FUNCTION
  explicit operator long long() const { return val; }
  KOKKOS_FUNCTION
  explicit operator unsigned short() const { return val; }
  KOKKOS_FUNCTION
  explicit operator unsigned int() const { return val; }
  KOKKOS_FUNCTION
  explicit operator unsigned long() const { return val; }
  KOKKOS_FUNCTION
  explicit operator unsigned long long() const { return val; }

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
  constexpr half_t(impl_type rhs) : val(rhs) {}
  KOKKOS_FUNCTION
  constexpr half_t(float rhs) : val(rhs) {}
  KOKKOS_FUNCTION
  constexpr half_t(double rhs) : val(rhs) {}
  KOKKOS_FUNCTION
  explicit constexpr half_t(bool rhs) : val(rhs) {}
  KOKKOS_FUNCTION
  constexpr half_t(short rhs) : val(rhs) {}
  KOKKOS_FUNCTION
  constexpr half_t(int rhs) : val(rhs) {}
  KOKKOS_FUNCTION
  constexpr half_t(long rhs) : val(rhs) {}
  KOKKOS_FUNCTION
  constexpr half_t(long long rhs) : val(rhs) {}
  KOKKOS_FUNCTION
  constexpr half_t(unsigned short rhs) : val(rhs) {}
  KOKKOS_FUNCTION
  constexpr half_t(unsigned int rhs) : val(rhs) {}
  KOKKOS_FUNCTION
  constexpr half_t(unsigned long rhs) : val(rhs) {}
  KOKKOS_FUNCTION
  constexpr half_t(unsigned long long rhs) : val(rhs) {}

  // Unary operators
  KOKKOS_FUNCTION
  half_t operator+() const {
    half_t tmp = *this;
    tmp.val    = +tmp.val;
    return tmp;
  }

  KOKKOS_FUNCTION
  half_t operator-() const {
    half_t tmp = *this;
    tmp.val    = -tmp.val;
    return tmp;
  }

  // Prefix operators
  KOKKOS_FUNCTION
  half_t& operator++() {
    ++val;
    return *this;
  }

  KOKKOS_FUNCTION
  half_t& operator--() {
    --val;
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

  KOKKOS_FUNCTION half_t& operator=(const half_t& rhs) = default;

  template <class T>
  KOKKOS_FUNCTION half_t& operator=(T rhs) {
    val = rhs;
    return *this;
  }

  KOKKOS_FUNCTION
  void operator=(half_t rhs) volatile {
    impl_type new_val = rhs.val;
    volatile uint16_t* val_ptr =
        reinterpret_cast<volatile uint16_t*>(const_cast<impl_type*>(&val));
    *val_ptr = reinterpret_cast<uint16_t&>(new_val);
  }

  template <class T>
  KOKKOS_FUNCTION void operator=(T rhs) volatile {
    impl_type new_val = rhs;
    volatile uint16_t* val_ptr =
        reinterpret_cast<volatile uint16_t*>(const_cast<impl_type*>(&val));
    *val_ptr = reinterpret_cast<uint16_t&>(new_val);
  }

  // Compound operators
  KOKKOS_FUNCTION
  half_t& operator+=(half_t rhs) {
    val += rhs.val;
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
    val -= rhs.val;
    return *this;
  }

  KOKKOS_FUNCTION
  void operator-=(const volatile half_t& rhs) volatile {
    half_t tmp_rhs = rhs;
    half_t tmp_lhs = *this;

    tmp_lhs -= tmp_rhs;
    *this = tmp_lhs;
  }

  // Compound operators: upcast overloads for -=
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
    val *= rhs.val;
    return *this;
  }

  KOKKOS_FUNCTION
  void operator*=(const volatile half_t& rhs) volatile {
    half_t tmp_rhs = rhs;
    half_t tmp_lhs = *this;

    tmp_lhs *= tmp_rhs;
    *this = tmp_lhs;
  }

  // Compound operators: upcast overloads for *=
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
    val /= rhs.val;
    return *this;
  }

  KOKKOS_FUNCTION
  void operator/=(const volatile half_t& rhs) volatile {
    half_t tmp_rhs = rhs;
    half_t tmp_lhs = *this;

    tmp_lhs /= tmp_rhs;
    *this = tmp_lhs;
  }

  // Compound operators: upcast overloads for /=
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
    lhs.val += rhs.val;
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
    lhs.val -= rhs.val;
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
    lhs.val *= rhs.val;
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
    lhs.val /= rhs.val;
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
  bool operator!() const { return static_cast<bool>(!val); }

  // NOTE: Loses short-circuit evaluation
  KOKKOS_FUNCTION
  bool operator&&(half_t rhs) const {
    return static_cast<bool>(val && rhs.val);
  }

  // NOTE: Loses short-circuit evaluation
  KOKKOS_FUNCTION
  bool operator||(half_t rhs) const {
    return static_cast<bool>(val || rhs.val);
  }

  // Comparison operators
  KOKKOS_FUNCTION
  bool operator==(half_t rhs) const {
    return static_cast<bool>(val == rhs.val);
  }

  KOKKOS_FUNCTION
  bool operator!=(half_t rhs) const {
    return static_cast<bool>(val != rhs.val);
  }

  KOKKOS_FUNCTION
  bool operator<(half_t rhs) const { return static_cast<bool>(val < rhs.val); }

  KOKKOS_FUNCTION
  bool operator>(half_t rhs) const { return static_cast<bool>(val > rhs.val); }

  KOKKOS_FUNCTION
  bool operator<=(half_t rhs) const {
    return static_cast<bool>(val <= rhs.val);
  }

  KOKKOS_FUNCTION
  bool operator>=(half_t rhs) const {
    return static_cast<bool>(val >= rhs.val);
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

KOKKOS_INLINE_FUNCTION
half_t cast_to_half(half_t val) { return val; }

KOKKOS_INLINE_FUNCTION
half_t cast_to_half(float val) { return half_t(val); }

KOKKOS_INLINE_FUNCTION
half_t cast_to_half(bool val) { return half_t::impl_type(val); }

KOKKOS_INLINE_FUNCTION
half_t cast_to_half(double val) { return half_t::impl_type(val); }

KOKKOS_INLINE_FUNCTION
half_t cast_to_half(short val) { return half_t::impl_type(val); }

KOKKOS_INLINE_FUNCTION
half_t cast_to_half(unsigned short val) { return half_t::impl_type(val); }

KOKKOS_INLINE_FUNCTION
half_t cast_to_half(int val) { return half_t::impl_type(val); }

KOKKOS_INLINE_FUNCTION
half_t cast_to_half(unsigned int val) { return half_t::impl_type(val); }

KOKKOS_INLINE_FUNCTION
half_t cast_to_half(long long val) { return half_t::impl_type(val); }

KOKKOS_INLINE_FUNCTION
half_t cast_to_half(unsigned long long val) { return half_t::impl_type(val); }

KOKKOS_INLINE_FUNCTION
half_t cast_to_half(long val) { return half_t::impl_type(val); }

KOKKOS_INLINE_FUNCTION
half_t cast_to_half(unsigned long val) { return half_t::impl_type(val); }

template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, float>::value, T>
cast_from_half(half_t val) {
  return static_cast<float>(static_cast<half_t::impl_type>(val));
}

template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, bool>::value, T>
cast_from_half(half_t val) {
  return static_cast<half_t::impl_type>(val);
}

template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, double>::value, T>
cast_from_half(half_t val) {
  return static_cast<half_t::impl_type>(val);
}

template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, short>::value, T>
cast_from_half(half_t val) {
  return static_cast<half_t::impl_type>(val);
}

template <class T>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<std::is_same<T, unsigned short>::value, T>
    cast_from_half(half_t val) {
  return static_cast<half_t::impl_type>(val);
}

template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, int>::value, T>
cast_from_half(half_t val) {
  return static_cast<half_t::impl_type>(val);
}

template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, unsigned>::value, T>
cast_from_half(half_t val) {
  return static_cast<half_t::impl_type>(val);
}

template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, long long>::value, T>
cast_from_half(half_t val) {
  return static_cast<half_t::impl_type>(val);
}

template <class T>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<std::is_same<T, unsigned long long>::value, T>
    cast_from_half(half_t val) {
  return static_cast<half_t::impl_type>(val);
}

template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_same<T, long>::value, T>
cast_from_half(half_t val) {
  return static_cast<half_t::impl_type>(val);
}

template <class T>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<std::is_same<T, unsigned long>::value, T>
    cast_from_half(half_t val) {
  return static_cast<half_t::impl_type>(val);
}

}  // namespace Experimental

template <>
struct reduction_identity<Kokkos::Experimental::half_t> {
  KOKKOS_FORCEINLINE_FUNCTION constexpr static Kokkos::Experimental::half_t
  sum() noexcept {
    return 0.0F;
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static Kokkos::Experimental::half_t
  prod() noexcept {
    return 1.0F;
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static Kokkos::Experimental::half_t
  max() noexcept {
    return std::numeric_limits<
        Kokkos::Experimental::half_t::impl_type>::lowest();
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static Kokkos::Experimental::half_t
  min() noexcept {
    return std::numeric_limits<Kokkos::Experimental::half_t::impl_type>::max();
  }
};

}  // namespace Kokkos
#endif  // KOKKOS_IMPL_HALF_TYPE_DEFINED
#endif  // KOKKOS_ENABLE_SYCL
#endif
