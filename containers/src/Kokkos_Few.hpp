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
// Questions? Contact  H. Carter Edwards (hcedwar@sandia.gov)
// 
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_CORE_EXP_FEW_HPP
#define KOKKOS_CORE_EXP_FEW_HPP

#include <Kokkos_Macros.hpp>
#include <type_traits>
#include <initializer_list>
#include <new>

namespace Kokkos { namespace Experimental {

/** \class Few
 *  \brief A std::array-like class for small tuples.
 *
 *  This class is useful for implementing types which can
 *  be copied to and from the device, including reduction types,
 *  for example 3D vectors and matrices.
 *  It also properly calls the constructors and destructors of
 *  its members, so it can hold non-POD members such as Views.
 *  Finally, it defines volatile forms of basic operators to
 *  allow it to be used as a reduction type.
 *  The reason this class is needed instead of using std:array
 *  is the volatile support and KOKKOS_INLINE_FUNCTION annotations.
 */

template <typename T, size_t n>
class Few {
  using UninitT = typename std::aligned_storage<sizeof(T), alignof(T)>::type;
  UninitT array_[n];

 public:
  enum { size = n };
  KOKKOS_INLINE_FUNCTION T* data() { return reinterpret_cast<T*>(array_); }
  KOKKOS_INLINE_FUNCTION T const* data() const {
    return reinterpret_cast<T const*>(array_);
  }
  KOKKOS_INLINE_FUNCTION T volatile* data() volatile {
    return reinterpret_cast<T volatile*>(array_);
  }
  KOKKOS_INLINE_FUNCTION T const volatile* data() const volatile {
    return reinterpret_cast<T const volatile*>(array_);
  }
  template <typename I0>
  KOKKOS_INLINE_FUNCTION T& operator[](I0 i) { return data()[i]; }
  template <typename I0>
  KOKKOS_INLINE_FUNCTION T const& operator[](I0 i) const { return data()[i]; }
  template <typename I0>
  KOKKOS_INLINE_FUNCTION T volatile& operator[](I0 i) volatile { return data()[i]; }
  template <typename I0>
  KOKKOS_INLINE_FUNCTION T const volatile& operator[](I0 i) const volatile {
    return data()[i];
  }
  Few(std::initializer_list<T> l) {
    size_t i = 0;
    for (auto it = l.begin(); it != l.end(); ++it) {
      new (data() + (i++)) T(*it);
    }
  }
  KOKKOS_INLINE_FUNCTION Few() {
    for (size_t i = 0; i < n; ++i) new (data() + i) T();
  }
  KOKKOS_INLINE_FUNCTION ~Few() {
    for (size_t i = 0; i < n; ++i) (data()[i]).~T();
  }
  KOKKOS_INLINE_FUNCTION void operator=(Few<T, n> const& rhs) volatile {
    for (size_t i = 0; i < n; ++i) data()[i] = rhs[i];
  }
  KOKKOS_INLINE_FUNCTION void operator=(Few<T, n> const& rhs) {
    for (size_t i = 0; i < n; ++i) data()[i] = rhs[i];
  }
  KOKKOS_INLINE_FUNCTION void operator=(Few<T, n> const volatile& rhs) {
    for (size_t i = 0; i < n; ++i) data()[i] = rhs[i];
  }
  KOKKOS_INLINE_FUNCTION Few(Few<T, n> const& rhs) {
    for (size_t i = 0; i < n; ++i) new (data() + i) T(rhs[i]);
  }
  KOKKOS_INLINE_FUNCTION Few(Few<T, n> const volatile& rhs) {
    for (size_t i = 0; i < n; ++i) new (data() + i) T(rhs[i]);
  }
};

}} // namespace Kokkos::Experimental

#endif //KOKKOS_CORE_EXP_FEW_HPP

