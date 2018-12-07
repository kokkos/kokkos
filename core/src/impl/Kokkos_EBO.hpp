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

#ifndef KOKKOS_EBO_HPP
#define KOKKOS_EBO_HPP

//----------------------------------------------------------------------------

#include <Kokkos_Macros.hpp>

#include <Kokkos_Core_fwd.hpp>
//----------------------------------------------------------------------------


#include <utility>
#include <type_traits>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template <class T, bool>
struct EBOBaseImpl;

template <class T>
struct EBOBaseImpl<T, true> {

  template <class... Args,
    int=typename std::enable_if<
      std::is_constructible<T, Args...>::value,
      int
    >::type(0)
  >
  KOKKOS_FORCEINLINE_FUNCTION
  EBOBaseImpl(
    Args&&... args
  ) noexcept(noexcept(ebo_base_t(std::forward<Args>(args)...)))
  {
    // still call the constructor
    auto intentionally_unused = T(std::forward<Args>(args)...);
  }

  // TODO noexcept in the right places?
  KOKKOS_FORCEINLINE_FUNCTION
  EBOBaseImpl() = default;

  KOKKOS_FORCEINLINE_FUNCTION
  EBOBaseImpl(EBOBaseImpl const&) = default;

  KOKKOS_FORCEINLINE_FUNCTION
  EBOBaseImpl(EBOBaseImpl&&) = default;

  KOKKOS_FORCEINLINE_FUNCTION
  EBOBaseImpl& operator=(EBOBaseImpl const&) = default;

  KOKKOS_FORCEINLINE_FUNCTION
  EBOBaseImpl& operator=(EBOBaseImpl&&) = default;

  KOKKOS_FORCEINLINE_FUNCTION
  ~EBOBaseImpl() = default;

  KOKKOS_INLINE_FUNCTION
  T& _ebo_data_member() & {
    return *reinterpret_cast<T*>(this);
  }

  KOKKOS_INLINE_FUNCTION
  T const& _ebo_data_member() const & {
    return *reinterpret_cast<T const*>(this);
  }

  KOKKOS_INLINE_FUNCTION
  T volatile& _ebo_data_member() volatile & {
    return *reinterpret_cast<T volatile*>(this);
  }

  KOKKOS_INLINE_FUNCTION
  T const volatile& _ebo_data_member() const volatile & {
    return *reinterpret_cast<T const volatile*>(this);
  }

  KOKKOS_INLINE_FUNCTION
  T&& _ebo_data_member() && {
    return std::move(*reinterpret_cast<T*>(this));
  }

};

template <class T>
struct EBOBaseImpl<T, false> {

  T m_ebo_object;

  template <class... Args,
    int=typename std::enable_if<
      std::is_constructible<T, Args...>::value,
      int
    >::type(0)
  >
  KOKKOS_FORCEINLINE_FUNCTION
  EBOBaseImpl(
    Args&&... args
  ) noexcept(noexcept(T(std::forward<Args>(args)...)))
    : m_ebo_object(std::forward<Args>(args)...)
  { }

  // TODO noexcept in the right places?

  // We need to make this only get generated if it's used so that it doesn't
  // generate a host call from device code
  template <
    class _forceGenerateOnFirstUse = void,
    class=typename std::enable_if<std::is_void<_forceGenerateOnFirstUse>::value>::type
  >
  KOKKOS_FORCEINLINE_FUNCTION
  EBOBaseImpl() : m_ebo_object() { }

  KOKKOS_FORCEINLINE_FUNCTION
  EBOBaseImpl(EBOBaseImpl const&) = default;

  KOKKOS_FORCEINLINE_FUNCTION
  EBOBaseImpl(EBOBaseImpl&&) = default;

  KOKKOS_FORCEINLINE_FUNCTION
  EBOBaseImpl& operator=(EBOBaseImpl const&) = default;

  KOKKOS_FORCEINLINE_FUNCTION
  EBOBaseImpl& operator=(EBOBaseImpl&&) = default;

  KOKKOS_FORCEINLINE_FUNCTION
  ~EBOBaseImpl() = default;

  KOKKOS_INLINE_FUNCTION
  T& _ebo_data_member() & {
    return m_ebo_object;
  }

  KOKKOS_INLINE_FUNCTION
  T const& _ebo_data_member() const & {
    return m_ebo_object;
  }

  KOKKOS_INLINE_FUNCTION
  T volatile& _ebo_data_member() volatile & {
    return m_ebo_object;
  }

  KOKKOS_INLINE_FUNCTION
  T const volatile& _ebo_data_member() const volatile & {
    return m_ebo_object;
  }

  KOKKOS_INLINE_FUNCTION
  T&& _ebo_data_member() && {
    return m_ebo_object;
  }

};

/**
 *
 * @tparam T
 */
template <class T>
struct StandardLayoutNoUniqueAddressMemberEmulation
  : EBOBaseImpl<T, std::is_empty<T>::value>
{
private:

  using ebo_base_t = EBOBaseImpl<T, std::is_empty<T>::value>;

public:

  using ebo_base_t::ebo_base_t;

  KOKKOS_FORCEINLINE_FUNCTION
  T& no_unique_address_data_member() & {
    return this->ebo_base_t::_ebo_data_member();
  }

  KOKKOS_FORCEINLINE_FUNCTION
  T const& no_unique_address_data_member() const & {
    return this->ebo_base_t::_ebo_data_member();
  }

  KOKKOS_FORCEINLINE_FUNCTION
  T volatile& no_unique_address_data_member() volatile & {
    return this->ebo_base_t::_ebo_data_member();
  }

  KOKKOS_FORCEINLINE_FUNCTION
  T const volatile& no_unique_address_data_member() const volatile & {
    return this->ebo_base_t::_ebo_data_member();
  }

  KOKKOS_FORCEINLINE_FUNCTION
  T&& no_unique_address_data_member() && {
    return this->ebo_base_t::_ebo_data_member();
  }
};

/**
 *
 * @tparam T
 */
template <class T>
class NoUniqueAddressMemberEmulation
  : private StandardLayoutNoUniqueAddressMemberEmulation<T>
{
private:

  using base_t = StandardLayoutNoUniqueAddressMemberEmulation<T>;

public:

  using base_t::base_t;
  using base_t::no_unique_address_data_member;

};


} // end namespace Impl
} // end namespace Kokkos


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------


#endif /* #ifndef KOKKOS_EBO_HPP */

