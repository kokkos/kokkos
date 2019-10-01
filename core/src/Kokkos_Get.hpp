/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
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

#ifndef KOKKOS_CORE_GET_HPP
#define KOKKOS_CORE_GET_HPP

#include <Properties/Kokkos_Detection.hpp>

#include <utility>  // std::get

namespace Kokkos {

//==============================================================================
// <editor-fold desc="get Niebloid implementation details"> {{{1

namespace Impl {
namespace _get_impl_ignore_adl {

// Poison-pill overload
template <class T, size_t I>
void get(T&&) = delete;

KOKKOS_DECLARE_DETECTION_ARCHETYPE(_has_adl_free_function_get_archetype,
                                   (class T, class IType), (T, IType),
                                   decltype(get<IType::value>(declval<T>())));
template <class T, size_t I>
using has_adl_free_function_get =
    Kokkos::Impl::is_detected<_has_adl_free_function_get_archetype, T,
                              std::integral_constant<size_t, I>>;

KOKKOS_DECLARE_DETECTION_ARCHETYPE(
    _has_std_get_archetype, (class T, class IType), (T, IType),
    decltype(std::get<IType::value>(declval<T>())));
template <class T, size_t I>
using has_std_get =
    Kokkos::Impl::is_detected<_has_std_get_archetype, T,
                              std::integral_constant<size_t, I>>;

KOKKOS_DECLARE_DETECTION_ARCHETYPE(
    _has_intrusive_get_archetype, (class T, class IType), (T, IType),
    decltype(declval<T>().template get<IType::value>()));

template <class T, size_t I>
using has_intrusive_get =
    Kokkos::Impl::is_detected<_has_intrusive_get_archetype, T,
                              std::integral_constant<size_t, I>>;

template <size_t I>
struct _get_impl {
  template <class T, class = typename std::enable_if<
                         has_intrusive_get<T, I>::value>::type>
  KOKKOS_INLINE_FUNCTION constexpr auto operator()(T&& val) const
      noexcept(noexcept(((T &&) val).template get<I>()))
          -> decltype(((T &&) val).template get<I>()) {
    return ((T &&) val).template get<I>();
  }
  template <class T, class = typename std::enable_if<
                         !has_intrusive_get<T, I>::value &&
                         has_adl_free_function_get<T, I>::value>::type>
  KOKKOS_INLINE_FUNCTION constexpr auto operator()(T&& val) const
      noexcept(noexcept(get<I>((T &&) val))) -> decltype(get<I>((T &&) val)) {
    return get<I>((T &&) val);
  }

  // Last resort, try std::get, and drop the KOKKOS_INLINE_FUNCTION
  template <class T, class = typename std::enable_if<
                         !has_intrusive_get<T, I>::value &&
                         !has_adl_free_function_get<T, I>::value &&
                         has_std_get<T, I>::value>::type>
  constexpr auto operator()(T&& val) const
      noexcept(noexcept(std::get<I>((T &&) val)))
          -> decltype(std::get<I>((T &&) val)) {
    return std::get<I>((T &&) val);
  }
};

}  // namespace _get_impl_ignore_adl
}  // end namespace Impl

// </editor-fold> end get Niebloid implementation details }}}1
//==============================================================================
/**
 *  std::get drop-in replacement, except that it's device marked and doesn't
 *  participate in ADL.
 */
#if defined(KOKKOS_ENABLE_CXX11)
// We can't use a Niebloid here because it requires variable templates
// This will have some warnings in the std::get case
template <size_t I, class T>
KOKKOS_INLINE_FUNCTION constexpr auto get(T&& val) const
    noexcept(noexcept(Impl::_get_impl_ignore_adl::_get_impl<I>{}((T &&) val)))
        ->decltype(Impl::_get_impl_ignore_adl::_get_impl<I>{}((T &&) val)) {
  return Impl::_get_impl_ignore_adl::_get_impl<I>{}((T &&) val);
}
#elif defined(KOKKOS_ENABLE_CXX14) || defined(KOKKOS_ENABLE_CXX17) || \
    defined(KOKKOS_ENABLE_CXX20)
template <size_t I>
#if defined(KOKKOS_ENABLE_CXX17) || defined(KOKKOS_ENABLE_CXX20)
inline
#endif
    constexpr Impl::_get_impl_ignore_adl::_get_impl<I>
        get = {};
#endif

}  // end namespace Kokkos

#endif  // KOKKOS_CORE_GET_HPP
