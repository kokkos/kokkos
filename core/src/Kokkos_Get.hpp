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

// Needs to be outside of namespace Kokkos to avoid detecting the get Niebloid
// in namespace Kokkos
namespace Impl {
namespace _get_impl_disable_adl {

//------------------------------------------------------------------------------

// Poison-pill overload
template <class T, size_t I>
void get(T&&) = delete;

//------------------------------------------------------------------------------

KOKKOS_IMPL_DECLARE_DETECTION_ARCHETYPE(
    _has_adl_free_function_get_archetype, (class T, class IType), (T, IType),
    decltype(get<IType::value>(Kokkos::Impl::declval<T>())));
template <class T, size_t I>
using has_adl_free_function_get =
    Kokkos::Impl::is_detected<_has_adl_free_function_get_archetype, T,
                              std::integral_constant<size_t, I>>;
template <class T, size_t I>
using adl_free_function_get_result_t =
    Kokkos::Impl::detected_t<_has_adl_free_function_get_archetype, T,
                             std::integral_constant<size_t, I>>;

//------------------------------------------------------------------------------

KOKKOS_IMPL_DECLARE_DETECTION_ARCHETYPE(
    _has_std_get_archetype, (class T, class IType), (T, IType),
    decltype(std::get<IType::value>(Kokkos::Impl::declval<T>())));
template <class T, size_t I>
using has_std_get =
    Kokkos::Impl::is_detected<_has_std_get_archetype, T,
                              std::integral_constant<size_t, I>>;

template <class T, size_t I>
using std_get_result_t =
    Kokkos::Impl::detected_t<_has_std_get_archetype, T,
                             std::integral_constant<size_t, I>>;

//------------------------------------------------------------------------------

KOKKOS_IMPL_DECLARE_DETECTION_ARCHETYPE(
    _has_intrusive_get_archetype, (class T, class IType), (T, IType),
    decltype(Kokkos::Impl::declval<T>().template get<IType::value>()));

template <class T, size_t I>
using has_intrusive_get =
    Kokkos::Impl::is_detected<_has_intrusive_get_archetype, T,
                              std::integral_constant<size_t, I>>;

template <class T, size_t I>
using intrusive_get_result_t =
    Kokkos::Impl::detected_t<_has_intrusive_get_archetype, T,
                             std::integral_constant<size_t, I>>;

//------------------------------------------------------------------------------

template <size_t I>
struct _get_niebloid {
  template <class T>
  KOKKOS_INLINE_FUNCTION constexpr
      typename std::enable_if<has_intrusive_get<T, I>::value,
                              intrusive_get_result_t<T, I>>::type
      operator()(T&& val) const
      noexcept(noexcept(((T &&) val).template get<I>())) {
    return ((T &&) val).template get<I>();
  }

  // If there's no member function, try to find the free function with ADL.
  // For now, let's try assuming this is device-marked, and require that
  // std::get *does not* work (since it would have to be device-marked)
  template <class T>
  KOKKOS_INLINE_FUNCTION constexpr
      typename std::enable_if<!has_intrusive_get<T, I>::value &&
                                  has_adl_free_function_get<T, I>::value &&
                                  !has_std_get<T, I>::value,
                              adl_free_function_get_result_t<T, I>>::type
      operator()(T&& val) const noexcept(noexcept(get<I>((T &&) val))) {
    return get<I>((T &&) val);
  }

  // Last resort, try std::get, and drop the KOKKOS_INLINE_FUNCTION, since
  // it's not device-marked.  The ADL free function version will *also* work for
  // things like std::pair in namespace std, but since those won't be
  // device-marked, we should use the one here without device markings instead.
  template <class T>
  constexpr inline typename std::enable_if<!has_intrusive_get<T, I>::value &&
                                               has_std_get<T, I>::value,
                                           std_get_result_t<T, I>>::type
  operator()(T&& val) const noexcept(noexcept(std::get<I>((T &&) val))) {
    return std::get<I>((T &&) val);
  }
};

//------------------------------------------------------------------------------

}  // namespace _get_impl_disable_adl
}  // namespace Impl

// </editor-fold> end get Niebloid implementation details }}}1
//==============================================================================

/**
 *  std::get drop-in replacement, except that it's device marked and doesn't
 *  participate in ADL.
 */
#if defined(KOKKOS_ENABLE_CXX11)
// We can't use a Niebloid here because it requires variable templates
template <size_t I, class T,
          class = typename std::enable_if<
              Impl::_get_impl_disable_adl::has_intrusive_get<T, I>::value ||
              !Impl::_get_impl_disable_adl::has_std_get<T, I>::value>::type>
KOKKOS_INLINE_FUNCTION constexpr auto get(T&& val) noexcept(
    noexcept(Impl::_get_impl_disable_adl::_get_niebloid<I>{}((T &&) val)))
    -> decltype(Impl::_get_impl_disable_adl::_get_niebloid<I>{}((T &&) val)) {
  return Impl::_get_impl_disable_adl::_get_niebloid<I>{}((T &&) val);
}
// Non-device-marked "overload" that goes through std::get
template <size_t I, class T>
constexpr typename std::enable_if<
    !Impl::_get_impl_disable_adl::has_intrusive_get<T, I>::value &&
        Impl::_get_impl_disable_adl::has_std_get<T, I>::value,
    Impl::_get_impl_disable_adl::std_get_result_t<T, I>>::type
get(T&& val) noexcept(
    noexcept(Impl::_get_impl_disable_adl::_get_niebloid<I>{}((T &&) val))) {
  return Impl::_get_impl_disable_adl::_get_niebloid<I>{}((T &&) val);
}
#elif defined(KOKKOS_ENABLE_CXX14) || defined(KOKKOS_ENABLE_CXX17) || \
    defined(KOKKOS_ENABLE_CXX20)
template <size_t I>
#if defined(KOKKOS_ENABLE_CXX17) || defined(KOKKOS_ENABLE_CXX20)
inline
#endif
    constexpr Impl::_get_impl_disable_adl::_get_niebloid<I>
        get = {};
#endif

//==============================================================================
// <editor-fold desc="A trait for the availability of Kokkos::get"> {{{1

namespace Impl {

KOKKOS_IMPL_DECLARE_DETECTION_ARCHETYPE_2PARAMS(
    _has_kokkos_get_archetype, T, IType,
    decltype(Kokkos::get<IType::value>(declval<T>())));

template <class T, size_t I>
using has_kokkos_get = is_detected<_has_kokkos_get_archetype, T,
                                   std::integral_constant<size_t, I>>;

template <class T, size_t I>
using kokkos_get_result_t =
    detected_t<_has_kokkos_get_archetype, T, std::integral_constant<size_t, I>>;

// Consider things that use std::get only to be not device supported.  Assume
// that anything else is user error (missing KOKKOS_INLINE_FUNCTION somewhere)
// and should result in the usual warnings.  This isn't a perfect assumption,
// but it's a reasonable working one.
template <class T, size_t I>
using has_device_supported_kokkos_get = std::integral_constant<
    bool, has_kokkos_get<T, I>::value &&
              (Impl::_get_impl_disable_adl::has_intrusive_get<T, I>::value ||
               !Impl::_get_impl_disable_adl::has_std_get<T, I>::value)>;

}  // end namespace Impl

// </editor-fold> end A trait for the availability of Kokkos::get }}}1
//==============================================================================
}  // end namespace Kokkos

#endif  // KOKKOS_CORE_GET_HPP
