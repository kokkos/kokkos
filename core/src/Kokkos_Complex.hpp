/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
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
#ifndef KOKKOS_COMPLEX_HPP
#define KOKKOS_COMPLEX_HPP

#include <Kokkos_Atomic.hpp>
#include <Kokkos_NumericTraits.hpp>
#include <complex>
#include <iostream>

namespace Kokkos {

/// \class complex
/// \brief Partial reimplementation of std::complex that works as the
///   result of a Kokkos::parallel_reduce.
/// \tparam RealType The type of the real and imaginary parts of the
///   complex number.  As with std::complex, this is only defined for
///   \c float, \c double, and <tt>long double</tt>.  The latter is
///   currently forbidden in CUDA device kernels.
template <class RealType>
class alignas(2 * sizeof(RealType)) complex {
 private:
  RealType re_{};
  RealType im_{};

 public:
  //! The type of the real or imaginary parts of this complex number.
  using value_type = RealType;

  //! Default constructor (initializes both real and imaginary parts to zero).
  KOKKOS_INLINE_FUNCTION
  complex() noexcept = default;

  //! Copy constructor.
  KOKKOS_INLINE_FUNCTION
  complex(const complex&) noexcept = default;

  KOKKOS_INLINE_FUNCTION
  complex& operator=(const complex&) noexcept = default;

  /// \brief Conversion constructor from std::complex.
  ///
  /// This constructor cannot be called in a CUDA device function,
  /// because std::complex's methods and nonmember functions are not
  /// marked as CUDA device functions.
  template <class RType>
  KOKKOS_INLINE_FUNCTION
  // We can use this aspect of the standard to avoid calling non-device-marked
  // functions `std::real` and `std::imag`: "For any object z of type
  // complex<T>, reinterpret_cast<T(&)[2]>(z)[0] is the real part of z and
  // reinterpret_cast<T(&)[2]>(z)[1] is the imaginary part of z."
  // Now we don't have to provide a whole bunch of the overloads of things
  // taking either Kokkos::complex or std::complex
  complex(const std::complex<RType>& src) noexcept
      : re_(reinterpret_cast<const RType (&)[2]>(src)[0]),
        im_(reinterpret_cast<const RType (&)[2]>(src)[1]) {}

  /// \brief Conversion operator to std::complex.
  ///
  /// This operator cannot be called in a CUDA device function,
  /// because std::complex's methods and nonmember functions are not
  /// marked as CUDA device functions.
  // TODO: make explicit.  DJS 2019-08-28
  operator std::complex<RealType>() const noexcept {
    return std::complex<RealType>(re_, im_);
  }

  /// \brief Constructor that takes just the real part, and sets the
  ///   imaginary part to zero.
  template <class RType>
  KOKKOS_INLINE_FUNCTION complex(const RType& val) noexcept
      : re_(val), im_(static_cast<RType>(0)) {}

  // BUG HCC WORKAROUND
  KOKKOS_INLINE_FUNCTION
  complex(const RealType& re, const RealType& im) noexcept : re_(re), im_(im) {}

  //! Constructor that takes the real and imaginary parts.
  template <class RealType1, class RealType2>
  KOKKOS_INLINE_FUNCTION complex(const RealType1& re,
                                 const RealType2& im) noexcept
      : re_(re), im_(im) {}

  //! Assignment operator.
  template <class RType>
  KOKKOS_INLINE_FUNCTION complex& operator=(
      const complex<RType>& src) noexcept {
    re_ = src.re_;
    im_ = src.im_;
    return *this;
  }

  //! Assignment operator (from a real number).
  template <class RType>
  KOKKOS_INLINE_FUNCTION complex& operator=(const RType& val) noexcept {
    re_ = val;
    im_ = RealType(0);
    return *this;
  }

  /// \brief Assignment operator from std::complex.
  ///
  /// This constructor cannot be called in a CUDA device function,
  /// because std::complex's methods and nonmember functions are not
  /// marked as CUDA device functions.
  template <class RType>
  complex& operator=(const std::complex<RType>& src) noexcept {
    *this = complex(src);
    return *this;
  }

  //! The imaginary part of this complex number.
  KOKKOS_INLINE_FUNCTION
  KOKKOS_CONSTEXPR_14 RealType& imag() noexcept { return im_; }

  //! The real part of this complex number.
  KOKKOS_INLINE_FUNCTION
  KOKKOS_CONSTEXPR_14 RealType& real() noexcept { return re_; }

  //! The imaginary part of this complex number.
  KOKKOS_INLINE_FUNCTION
  constexpr RealType imag() const noexcept { return im_; }

  //! The real part of this complex number.
  KOKKOS_INLINE_FUNCTION
  constexpr RealType real() const noexcept { return re_; }

  //! Set the imaginary part of this complex number.
  KOKKOS_INLINE_FUNCTION
  KOKKOS_CONSTEXPR_14
  void imag(RealType v) noexcept { im_ = v; }

  //! Set the real part of this complex number.
  KOKKOS_INLINE_FUNCTION
  KOKKOS_CONSTEXPR_14
  void real(RealType v) noexcept { re_ = v; }

  template <typename RType>
  KOKKOS_CONSTEXPR_14 KOKKOS_INLINE_FUNCTION complex& operator+=(
      const complex<RType>& src) noexcept {
    static_assert(std::is_convertible<RType, RealType>::value,
                  "RType must be convertible to RealType");
    re_ += src.re_;
    im_ += src.im_;
    return *this;
  }

  template <typename RType>
  KOKKOS_CONSTEXPR_14 KOKKOS_INLINE_FUNCTION complex& operator+=(
      const RType& src) noexcept {
    static_assert(std::is_convertible<RType, RealType>::value,
                  "RType must be convertible to RealType");
    re_ += src;
    return *this;
  }

  template <typename RType>
  KOKKOS_CONSTEXPR_14 KOKKOS_INLINE_FUNCTION complex& operator-=(
      const complex<RType>& src) noexcept {
    static_assert(std::is_convertible<RType, RealType>::value,
                  "RType must be convertible to RealType");
    re_ -= src.re_;
    im_ -= src.im_;
    return *this;
  }

  template <typename RType>
  KOKKOS_CONSTEXPR_14 KOKKOS_INLINE_FUNCTION complex& operator-=(
      const RType& src) noexcept {
    static_assert(std::is_convertible<RType, RealType>::value,
                  "RType must be convertible to RealType");
    re_ -= src;
    return *this;
  }

  template <typename RType>
  KOKKOS_CONSTEXPR_14 KOKKOS_INLINE_FUNCTION complex& operator*=(
      const complex<RType>& src) noexcept {
    static_assert(std::is_convertible<RType, RealType>::value,
                  "RType must be convertible to RealType");
    const RealType realPart = re_ * src.re_ - im_ * src.im_;
    const RealType imagPart = re_ * src.im_ + im_ * src.re_;
    re_                     = realPart;
    im_                     = imagPart;
    return *this;
  }

  template <typename RType>
  KOKKOS_CONSTEXPR_14 KOKKOS_INLINE_FUNCTION complex& operator*=(
      const RType& src) noexcept {
    static_assert(std::is_convertible<RType, RealType>::value,
                  "RType must be convertible to RealType");
    re_ *= src;
    im_ *= src;
    return *this;
  }

  template <typename RType>
  // Conditional noexcept, just in case RType throws on divide-by-zero
  KOKKOS_CONSTEXPR_14 KOKKOS_INLINE_FUNCTION complex& operator/=(
      const complex<RType>& y) noexcept(noexcept(RealType{} / RealType{})) {
    static_assert(std::is_convertible<RType, RealType>::value,
                  "RType must be convertible to RealType");

    // Scale (by the "1-norm" of y) to avoid unwarranted overflow.
    // If the real part is +/-Inf and the imaginary part is -/+Inf,
    // this won't change the result.
    const RealType s = std::fabs(y.real()) + std::fabs(y.imag());

    // If s is 0, then y is zero, so x/y == real(x)/0 + i*imag(x)/0.
    // In that case, the relation x/y == (x/s) / (y/s) doesn't hold,
    // because y/s is NaN.
    // TODO mark this branch unlikely
    if (s == RealType(0)) {
      this->re_ /= s;
      this->im_ /= s;
    } else {
      const complex x_scaled(this->re_ / s, this->im_ / s);
      const complex y_conj_scaled(y.re_ / s, -(y.im_) / s);
      const RealType y_scaled_abs =
          y_conj_scaled.re_ * y_conj_scaled.re_ +
          y_conj_scaled.im_ * y_conj_scaled.im_;  // abs(y) == abs(conj(y))
      *this = x_scaled * y_conj_scaled;
      *this /= y_scaled_abs;
    }
    return *this;
  }

  KOKKOS_CONSTEXPR_14
  KOKKOS_INLINE_FUNCTION complex& operator/=(
      const std::complex<RealType>& y) noexcept(noexcept(RealType{} /
                                                         RealType{})) {
    // Scale (by the "1-norm" of y) to avoid unwarranted overflow.
    // If the real part is +/-Inf and the imaginary part is -/+Inf,
    // this won't change the result.
    const RealType s = std::fabs(y.real()) + std::fabs(y.imag());

    // If s is 0, then y is zero, so x/y == real(x)/0 + i*imag(x)/0.
    // In that case, the relation x/y == (x/s) / (y/s) doesn't hold,
    // because y/s is NaN.
    if (s == RealType(0)) {
      this->re_ /= s;
      this->im_ /= s;
    } else {
      const complex x_scaled(this->re_ / s, this->im_ / s);
      const complex y_conj_scaled(y.re_ / s, -(y.im_) / s);
      const RealType y_scaled_abs =
          y_conj_scaled.re_ * y_conj_scaled.re_ +
          y_conj_scaled.im_ * y_conj_scaled.im_;  // abs(y) == abs(conj(y))
      *this = x_scaled * y_conj_scaled;
      *this /= y_scaled_abs;
    }
    return *this;
  }

  template <typename RType>
  KOKKOS_CONSTEXPR_14 KOKKOS_INLINE_FUNCTION complex& operator/=(
      const RType& src) noexcept(noexcept(RealType{} / RType{})) {
    static_assert(std::is_convertible<RType, RealType>::value,
                  "RType must be convertible to RealType");

    re_ /= src;
    im_ /= src;
    return *this;
  }

  //---------------------------------------------------------------------------
  // Hidden friend comparison operators
  //---------------------------------------------------------------------------

  friend KOKKOS_INLINE_FUNCTION bool operator==(const complex& a,
                                                const complex& b) noexcept {
    return a.real() == b.real() && a.imag() == b.imag();
  }

  template <typename RType>
  friend KOKKOS_INLINE_FUNCTION constexpr bool operator==(
      const complex& a, const complex<RType>& b) noexcept {
    //----------------------------------------//
    using common_type = typename std::common_type<RealType, RType>::type;
    return common_type(a.real()) == common_type(b.real()) &&
           common_type(a.imag()) == common_type(b.imag());
  }

  template <typename RType>
  friend constexpr bool operator==(const complex& a,
                                   const std::complex<RType>& b) noexcept {
    using common_type = typename std::common_type<RealType, RType>::type;
    return common_type(a.real()) == common_type(b.real()) &&
           common_type(a.imag()) == common_type(b.imag());
  }

  template <typename RType>
  friend constexpr bool operator==(const std::complex<RType>& a,
                                   const complex& b) noexcept {
    return b == a;
  }

  template <typename RType>
  friend KOKKOS_INLINE_FUNCTION bool operator==(const complex& a,
                                                const RType b) noexcept {
    using common_type = typename std::common_type<RealType, RType>::type;
    return (common_type(a.real()) == common_type(b)) &&
           (common_type(a.imag()) == common_type(0));
  }

  template <typename RType>
  friend KOKKOS_INLINE_FUNCTION bool operator==(const RType& a,
                                                const complex b) noexcept {
    return b == a;
  }

  friend KOKKOS_INLINE_FUNCTION bool operator!=(const complex& a,
                                                const complex& b) noexcept {
    return a.real() != b.real() || a.imag() != b.imag();
  }

  template <typename RType>
  friend KOKKOS_INLINE_FUNCTION bool operator!=(
      const complex& a, const complex<RType>& b) noexcept {
    //----------------------------------------//
    using common_type = typename std::common_type<RealType, RType>::type;
    return common_type(a.real()) != common_type(b.real()) ||
           common_type(a.imag()) != common_type(b.imag());
  }

  template <typename RType>
  friend inline constexpr bool operator!=(
      const complex& a, const std::complex<RealType>& b) noexcept {
    //----------------------------------------//
    using common_type = typename std::common_type<RealType, RType>::type;
    return common_type(a.real()) != common_type(b.real()) ||
           common_type(a.imag()) != common_type(b.imag());
  }

  template <typename RType>
  friend inline bool operator!=(const std::complex<RealType>& a,
                                const complex& b) {
    return b != a;
  }

  template <typename RType>
  friend KOKKOS_INLINE_FUNCTION bool operator!=(const complex& a,
                                                const RType& b) noexcept {
    using common_type = typename std::common_type<RealType, RType>::type;
    return (common_type(a.real()) != common_type(b)) ||
           (common_type(a.imag()) != common_type(0));
  }

  template <typename RType>
  friend KOKKOS_FORCEINLINE_FUNCTION bool operator!=(
      const RType& a, const complex& b) noexcept {
    //----------------------------------------//
    return a != b;
  }

  //---------------------------------------------------------------------------
  // TODO: refactor Kokkos reductions to remove dependency on
  // volatile member overloads since they are being deprecated in c++20
  //---------------------------------------------------------------------------

  //! Copy constructor from volatile.
  template <class RType>
  KOKKOS_INLINE_FUNCTION complex(const volatile complex<RType>& src) noexcept
      : re_(src.re_), im_(src.im_) {}

  /// \brief Assignment operator, for volatile <tt>*this</tt> and
  ///   nonvolatile input.
  ///
  /// \param src [in] Input; right-hand side of the assignment.
  ///
  /// This operator returns \c void instead of <tt>volatile
  /// complex& </tt>.  See Kokkos Issue #177 for the
  /// explanation.  In practice, this means that you should not chain
  /// assignments with volatile lvalues.
  template <class RType>
  KOKKOS_INLINE_FUNCTION void operator=(
      const complex<RType>& src) volatile noexcept {
    re_ = src.re_;
    im_ = src.im_;
    // We deliberately do not return anything here.  See explanation
    // in public documentation above.
  }

  //! Assignment operator.
  template <class RType>
  KOKKOS_INLINE_FUNCTION volatile complex& operator=(
      const volatile complex<RType>& src) volatile noexcept {
    re_ = src.re_;
    im_ = src.im_;
    return *this;
  }

  //! Assignment operator.
  template <class RType>
  KOKKOS_INLINE_FUNCTION complex& operator=(
      const volatile complex<RType>& src) noexcept {
    re_ = src.re_;
    im_ = src.im_;
    return *this;
  }

  //! Assignment operator (from a real number).
  template <class RType>
  KOKKOS_INLINE_FUNCTION void operator=(const RType& val) volatile noexcept {
    re_ = val;
    im_ = RealType(0);
  }

  //! The imaginary part of this complex number (volatile overload).
  KOKKOS_INLINE_FUNCTION
  volatile RealType& imag() volatile noexcept { return im_; }

  //! The real part of this complex number (volatile overload).
  KOKKOS_INLINE_FUNCTION
  volatile RealType& real() volatile noexcept { return re_; }

  //! The imaginary part of this complex number (volatile overload).
  KOKKOS_INLINE_FUNCTION
  RealType imag() const volatile noexcept { return im_; }

  //! The real part of this complex number (volatile overload).
  KOKKOS_INLINE_FUNCTION
  RealType real() const volatile noexcept { return re_; }

  template <typename RType>
  KOKKOS_INLINE_FUNCTION void operator+=(
      const volatile complex<RType>& src) volatile noexcept {
    static_assert(std::is_convertible<RType, RealType>::value,
                  "RType must be convertible to RealType");
    re_ += src.re_;
    im_ += src.im_;
  }

  template <typename RType>
  KOKKOS_INLINE_FUNCTION void operator+=(
      const volatile RType& src) volatile noexcept {
    static_assert(std::is_convertible<RType, RealType>::value,
                  "RType must be convertible to RealType");
    re_ += src;
  }

  template <typename RType>
  KOKKOS_INLINE_FUNCTION void operator*=(
      const volatile complex<RType>& src) volatile noexcept {
    static_assert(std::is_convertible<RType, RealType>::value,
                  "RType must be convertible to RealType");
    const RealType realPart = re_ * src.re_ - im_ * src.im_;
    const RealType imagPart = re_ * src.im_ + im_ * src.re_;

    re_ = realPart;
    im_ = imagPart;
  }

  template <typename RType>
  KOKKOS_INLINE_FUNCTION void operator*=(
      const volatile RType& src) volatile noexcept {
    static_assert(std::is_convertible<RType, RealType>::value,
                  "RType must be convertible to RealType");
    re_ *= src;
    im_ *= src;
  }
};

//! Binary + operator for complex complex.
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION
    complex<typename std::common_type<RealType1, RealType2>::type>
    operator+(const complex<RealType1>& x,
              const complex<RealType2>& y) noexcept {
  return complex<typename std::common_type<RealType1, RealType2>::type>(
      x.real() + y.real(), x.imag() + y.imag());
}

//! Binary + operator for complex scalar.
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION
    complex<typename std::common_type<RealType1, RealType2>::type>
    operator+(const complex<RealType1>& x, const RealType2& y) noexcept {
  return complex<typename std::common_type<RealType1, RealType2>::type>(
      x.real() + y, x.imag());
}

//! Binary + operator for scalar complex.
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION
    complex<typename std::common_type<RealType1, RealType2>::type>
    operator+(const RealType1& x, const complex<RealType2>& y) noexcept {
  return complex<typename std::common_type<RealType1, RealType2>::type>(
      x + y.real(), y.imag());
}

//! Unary + operator for complex.
template <class RealType>
KOKKOS_INLINE_FUNCTION complex<RealType> operator+(
    const complex<RealType>& x) noexcept {
  return complex<RealType>{+x.real(), +x.imag()};
}

//! Binary - operator for complex.
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION
    complex<typename std::common_type<RealType1, RealType2>::type>
    operator-(const complex<RealType1>& x,
              const complex<RealType2>& y) noexcept {
  return complex<typename std::common_type<RealType1, RealType2>::type>(
      x.real() - y.real(), x.imag() - y.imag());
}

//! Binary - operator for complex scalar.
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION
    complex<typename std::common_type<RealType1, RealType2>::type>
    operator-(const complex<RealType1>& x, const RealType2& y) noexcept {
  return complex<typename std::common_type<RealType1, RealType2>::type>(
      x.real() - y, x.imag());
}

//! Binary - operator for scalar complex.
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION
    complex<typename std::common_type<RealType1, RealType2>::type>
    operator-(const RealType1& x, const complex<RealType2>& y) noexcept {
  return complex<typename std::common_type<RealType1, RealType2>::type>(
      x - y.real(), -y.imag());
}

//! Unary - operator for complex.
template <class RealType>
KOKKOS_INLINE_FUNCTION complex<RealType> operator-(
    const complex<RealType>& x) noexcept {
  return complex<RealType>(-x.real(), -x.imag());
}

//! Binary * operator for complex.
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION
    complex<typename std::common_type<RealType1, RealType2>::type>
    operator*(const complex<RealType1>& x,
              const complex<RealType2>& y) noexcept {
  return complex<typename std::common_type<RealType1, RealType2>::type>(
      x.real() * y.real() - x.imag() * y.imag(),
      x.real() * y.imag() + x.imag() * y.real());
}

/// \brief Binary * operator for std::complex and complex.
///
/// This needs to exist because template parameters can't be deduced when
/// conversions occur.  We could probably fix this using hidden friends patterns
///
/// This function cannot be called in a CUDA device function, because
/// std::complex's methods and nonmember functions are not marked as
/// CUDA device functions.
template <class RealType1, class RealType2>
inline complex<typename std::common_type<RealType1, RealType2>::type> operator*(
    const std::complex<RealType1>& x, const complex<RealType2>& y) {
  return complex<typename std::common_type<RealType1, RealType2>::type>(
      x.real() * y.real() - x.imag() * y.imag(),
      x.real() * y.imag() + x.imag() * y.real());
}

/// \brief Binary * operator for RealType times complex.
///
/// This function exists because the compiler doesn't know that
/// RealType and complex<RealType> commute with respect to operator*.
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION
    complex<typename std::common_type<RealType1, RealType2>::type>
    operator*(const RealType1& x, const complex<RealType2>& y) noexcept {
  return complex<typename std::common_type<RealType1, RealType2>::type>(
      x * y.real(), x * y.imag());
}

/// \brief Binary * operator for RealType times complex.
///
/// This function exists because the compiler doesn't know that
/// RealType and complex<RealType> commute with respect to operator*.
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION
    complex<typename std::common_type<RealType1, RealType2>::type>
    operator*(const complex<RealType1>& y, const RealType2& x) noexcept {
  return complex<typename std::common_type<RealType1, RealType2>::type>(
      x * y.real(), x * y.imag());
}

//! Imaginary part of a complex number.
template <class RealType>
KOKKOS_INLINE_FUNCTION RealType imag(const complex<RealType>& x) noexcept {
  return x.imag();
}

//! Real part of a complex number.
template <class RealType>
KOKKOS_INLINE_FUNCTION RealType real(const complex<RealType>& x) noexcept {
  return x.real();
}

//! Absolute value (magnitude) of a complex number.
template <class RealType>
KOKKOS_INLINE_FUNCTION RealType abs(const complex<RealType>& x) {
#ifndef __CUDA_ARCH__
  using std::hypot;
#endif
  return hypot(x.real(), x.imag());
}

//! Power of a complex number
template <class RealType>
KOKKOS_INLINE_FUNCTION Kokkos::complex<RealType> pow(const complex<RealType>& x,
                                                     const RealType& e) {
  RealType r   = abs(x);
  RealType phi = std::atan(x.imag() / x.real());
  return std::pow(r, e) *
         Kokkos::complex<RealType>(std::cos(phi * e), std::sin(phi * e));
}

//! Square root of a complex number.
template <class RealType>
KOKKOS_INLINE_FUNCTION Kokkos::complex<RealType> sqrt(
    const complex<RealType>& x) {
  RealType r   = abs(x);
  RealType phi = std::atan(x.imag() / x.real());
  return std::sqrt(r) *
         Kokkos::complex<RealType>(std::cos(phi * 0.5), std::sin(phi * 0.5));
}

//! Conjugate of a complex number.
template <class RealType>
KOKKOS_INLINE_FUNCTION complex<RealType> conj(
    const complex<RealType>& x) noexcept {
  return complex<RealType>(real(x), -imag(x));
}

//! Exponential of a complex number.
template <class RealType>
KOKKOS_INLINE_FUNCTION complex<RealType> exp(const complex<RealType>& x) {
  return std::exp(x.real()) *
         complex<RealType>(std::cos(x.imag()), std::sin(x.imag()));
}

/// This function cannot be called in a CUDA device function,
/// because std::complex's methods and nonmember functions are not
/// marked as CUDA device functions.
template <class RealType>
inline complex<RealType> exp(const std::complex<RealType>& c) {
  return complex<RealType>(std::exp(c.real()) * std::cos(c.imag()),
                           std::exp(c.real()) * std::sin(c.imag()));
}

//! Binary operator / for complex and real numbers
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION
    complex<typename std::common_type<RealType1, RealType2>::type>
    operator/(const complex<RealType1>& x,
              const RealType2& y) noexcept(noexcept(RealType1{} /
                                                    RealType2{})) {
  return complex<typename std::common_type<RealType1, RealType2>::type>(
      real(x) / y, imag(x) / y);
}

//! Binary operator / for complex.
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION
    complex<typename std::common_type<RealType1, RealType2>::type>
    operator/(const complex<RealType1>& x,
              const complex<RealType2>& y) noexcept(noexcept(RealType1{} /
                                                             RealType2{})) {
  // Scale (by the "1-norm" of y) to avoid unwarranted overflow.
  // If the real part is +/-Inf and the imaginary part is -/+Inf,
  // this won't change the result.
  typedef
      typename std::common_type<RealType1, RealType2>::type common_real_type;
  const common_real_type s = std::fabs(real(y)) + std::fabs(imag(y));

  // If s is 0, then y is zero, so x/y == real(x)/0 + i*imag(x)/0.
  // In that case, the relation x/y == (x/s) / (y/s) doesn't hold,
  // because y/s is NaN.
  if (s == 0.0) {
    return complex<common_real_type>(real(x) / s, imag(x) / s);
  } else {
    const complex<common_real_type> x_scaled(real(x) / s, imag(x) / s);
    const complex<common_real_type> y_conj_scaled(real(y) / s, -imag(y) / s);
    const RealType1 y_scaled_abs =
        real(y_conj_scaled) * real(y_conj_scaled) +
        imag(y_conj_scaled) * imag(y_conj_scaled);  // abs(y) == abs(conj(y))
    complex<common_real_type> result = x_scaled * y_conj_scaled;
    result /= y_scaled_abs;
    return result;
  }
}

//! Binary operator / for complex and real numbers
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION
    complex<typename std::common_type<RealType1, RealType2>::type>
    operator/(const RealType1& x,
              const complex<RealType2>& y) noexcept(noexcept(RealType1{} /
                                                             RealType2{})) {
  return complex<typename std::common_type<RealType1, RealType2>::type>(x) / y;
}

template <class RealType>
std::ostream& operator<<(std::ostream& os, const complex<RealType>& x) {
  const std::complex<RealType> x_std(Kokkos::real(x), Kokkos::imag(x));
  os << x_std;
  return os;
}

template <class RealType>
std::istream& operator>>(std::istream& is, complex<RealType>& x) {
  std::complex<RealType> x_std;
  is >> x_std;
  x = x_std;  // only assigns on success of above
  return is;
}

template <class T>
struct reduction_identity<Kokkos::complex<T> > {
  typedef reduction_identity<T> t_red_ident;
  KOKKOS_FORCEINLINE_FUNCTION constexpr static Kokkos::complex<T>
  sum() noexcept {
    return Kokkos::complex<T>(t_red_ident::sum(), t_red_ident::sum());
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static Kokkos::complex<T>
  prod() noexcept {
    return Kokkos::complex<T>(t_red_ident::prod(), t_red_ident::sum());
  }
};

}  // namespace Kokkos

#endif  // KOKKOS_COMPLEX_HPP
