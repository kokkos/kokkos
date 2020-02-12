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

// Experimental unified task-data parallel manycore LDRD

#ifndef KOKKOS_IMPL_OPTIONALREF_HPP
#define KOKKOS_IMPL_OPTIONALREF_HPP

#include <Kokkos_Macros.hpp>

#include <Kokkos_Core_fwd.hpp>

#include <Kokkos_PointerOwnership.hpp>
#include <impl/Kokkos_Error.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
namespace Kokkos {
namespace Impl {

struct InPlaceTag {};

template <class T>
struct OptionalRef {
 private:
  ObservingRawPtr<T> m_value = nullptr;

 public:
  using value_type = T;

  KOKKOS_INLINE_FUNCTION
  OptionalRef() = default;

  KOKKOS_INLINE_FUNCTION
  OptionalRef(OptionalRef const&) = default;

  KOKKOS_INLINE_FUNCTION
  OptionalRef(OptionalRef&&) = default;

  KOKKOS_INLINE_FUNCTION
  OptionalRef& operator=(OptionalRef const&) = default;

  KOKKOS_INLINE_FUNCTION
  // Can't return a reference to volatile OptionalRef, since GCC issues a
  // warning about reference to volatile not accessing the underlying value
  void operator=(OptionalRef const volatile& other) volatile noexcept {
    m_value = other.m_value;
  }

  KOKKOS_INLINE_FUNCTION
  OptionalRef& operator=(OptionalRef&&) = default;

  KOKKOS_INLINE_FUNCTION
  ~OptionalRef() = default;

  KOKKOS_INLINE_FUNCTION
  explicit OptionalRef(T& arg_value) : m_value(&arg_value) {}

  KOKKOS_INLINE_FUNCTION
  explicit OptionalRef(std::nullptr_t) : m_value(nullptr) {}

  KOKKOS_INLINE_FUNCTION
  OptionalRef& operator=(T& arg_value) {
    m_value = &arg_value;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  OptionalRef& operator=(std::nullptr_t) {
    m_value = nullptr;
    return *this;
  }

  //----------------------------------------

  KOKKOS_INLINE_FUNCTION
  OptionalRef<typename std::add_volatile<T>::type>
  as_volatile() volatile noexcept {
    return OptionalRef<typename std::add_volatile<T>::type>(*(*this));
  }

  KOKKOS_INLINE_FUNCTION
  OptionalRef<
      typename std::add_volatile<typename std::add_const<T>::type>::type>
  as_volatile() const volatile noexcept {
    return OptionalRef<
        typename std::add_volatile<typename std::add_const<T>::type>::type>(
        *(*this));
  }

  //----------------------------------------

  KOKKOS_INLINE_FUNCTION
  T& operator*() & {
    KOKKOS_EXPECTS(this->has_value());
    return *m_value;
  }

  KOKKOS_INLINE_FUNCTION
  T const& operator*() const& {
    KOKKOS_EXPECTS(this->has_value());
    return *m_value;
  }

  KOKKOS_INLINE_FUNCTION
  T volatile& operator*() volatile& {
    KOKKOS_EXPECTS(this->has_value());
    return *m_value;
  }

  KOKKOS_INLINE_FUNCTION
  T const volatile& operator*() const volatile& {
    KOKKOS_EXPECTS(this->has_value());
    return *m_value;
  }

  KOKKOS_INLINE_FUNCTION
  T&& operator*() && {
    KOKKOS_EXPECTS(this->has_value());
    return std::move(*m_value);
  }

  KOKKOS_INLINE_FUNCTION
  T* operator->() {
    KOKKOS_EXPECTS(this->has_value());
    return m_value;
  }

  KOKKOS_INLINE_FUNCTION
  T const* operator->() const {
    KOKKOS_EXPECTS(this->has_value());
    return m_value;
  }

  KOKKOS_INLINE_FUNCTION
  T volatile* operator->() volatile {
    KOKKOS_EXPECTS(this->has_value());
    return m_value;
  }

  KOKKOS_INLINE_FUNCTION
  T const volatile* operator->() const volatile {
    KOKKOS_EXPECTS(this->has_value());
    return m_value;
  }

  KOKKOS_INLINE_FUNCTION
  T* get() { return m_value; }

  KOKKOS_INLINE_FUNCTION
  T const* get() const { return m_value; }

  KOKKOS_INLINE_FUNCTION
  T volatile* get() volatile { return m_value; }

  KOKKOS_INLINE_FUNCTION
  T const volatile* get() const volatile { return m_value; }

  //----------------------------------------

  KOKKOS_INLINE_FUNCTION
  operator bool() { return m_value != nullptr; }

  KOKKOS_INLINE_FUNCTION
  operator bool() const { return m_value != nullptr; }

  KOKKOS_INLINE_FUNCTION
  operator bool() volatile { return m_value != nullptr; }

  KOKKOS_INLINE_FUNCTION
  operator bool() const volatile { return m_value != nullptr; }

  KOKKOS_INLINE_FUNCTION
  bool has_value() { return m_value != nullptr; }

  KOKKOS_INLINE_FUNCTION
  bool has_value() const { return m_value != nullptr; }

  KOKKOS_INLINE_FUNCTION
  bool has_value() volatile { return m_value != nullptr; }

  KOKKOS_INLINE_FUNCTION
  bool has_value() const volatile { return m_value != nullptr; }
};

}  // end namespace Impl
}  // end namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #ifndef KOKKOS_IMPL_OPTIONALREF_HPP */
