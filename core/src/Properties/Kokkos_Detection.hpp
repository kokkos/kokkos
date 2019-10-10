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

/**
 *  Kokkos detection idiom implementation.
 *
 *  If you're unfamiliar with the detection idiom, it may be helpful to read
 *  about the idiom in general rather than reading the code here directly.
 *  A few recommended articles:
 *
 *    - https://blog.tartanllama.xyz/detection-idiom/
 *    - https://people.eecs.berkeley.edu/~brock/blog/detection_idiom.php
 *    - https://en.cppreference.com/w/cpp/experimental/is_detected
 *    - https://www.youtube.com/watch?v=a0FliKwcwXE
 *    - https://www.youtube.com/watch?v=U3jGdnRL3KI
 *    - http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/n4436.pdf
 *
 *  The macro-based versions of the idiom are to incorporate bug workarounds
 *  for various compilers.
 *
 *  @todo change the macros in this file over to real detection idiom
 *        once we drop support for compilers that don't work with it
 */

#ifndef KOKKOS_PROPERTIES_DETECTION_HPP
#define KOKKOS_PROPERTIES_DETECTION_HPP

#include <Kokkos_Macros.hpp>
#include <Kokkos_Core_fwd.hpp>


namespace Kokkos {
namespace Impl {

// A declval implementation that won't make nvcc barf
template <class T>
KOKKOS_FUNCTION T&& _declval(int) noexcept;
template <class T>
KOKKOS_FUNCTION T _declval(long) noexcept;

#if defined(KOKKOS_ENABLE_CXX17) || defined(KOKKOS_ENABLE_CXX20)
using std::void_t;
#else
template <class T>
using void_t = void;
#endif

template <class T>
KOKKOS_FUNCTION decltype(Kokkos::Impl::_declval<T>(0)) declval() noexcept;

// A type to return from is_detected when detection fails.  Delete everything
// in order to maximize the probability that use of a is_detected result type
// will lead to a compilation error.
struct nonesuch {
  nonesuch()                = delete;
  ~nonesuch()               = delete;
  nonesuch(nonesuch const&) = delete;
  void operator=(nonesuch const&) = delete;
};

#ifdef KOKKOS_COMPILER_SUPPORTS_DETECTION_IDIOM

// primary template handles all types not supporting the archetypal Op
template <class Default, class _always_void, template <class...> class Op,
          class... Args>
struct _detector {
  constexpr static auto value = false;
  using type                  = Default;
};

// specialization recognizes and handles only types supporting Op
template <class Default, template <class...> class Op, class... Args>
struct _detector<Default, void_t<Op<Args...>>, Op, Args...> {
  constexpr static auto value = true;
  using type                  = Op<Args...>;
};

#define KOKKOS_IMPL_DECLARE_DETECTION_ARCHETYPE(name, params, params_no_intro, \
                                                ...)                           \
  template <KOKKOS_IMPL_PP_REMOVE_PARENS(params)>                              \
  using name = __VA_ARGS__;

#else

namespace _detection_impl {

template <class Default, template <class...> class WorkaroundImpl,
          class... Args>
struct detector_workaround_impl {
  // This could be done with inheritance, but let's just do it this way
  // to preempt some mistakes that could be made involving that and that
  // wouldn't work with the "real" detection idiom
  using _impl                 = WorkaroundImpl<Default, void, Args...>;
  using type                  = typename _impl::type;
  static constexpr auto value = _impl::value;
};

}  // end namespace _detection_impl

#define KOKKOS_IMPL_DECLARE_DETECTION_ARCHETYPE(name, params, params_no_intro, \
                                                ...)                           \
  template <class Default, class _k__always_void,                              \
            KOKKOS_IMPL_PP_REMOVE_PARENS(params)>                              \
  struct name##__impl {                                                        \
    static constexpr bool value = false;                                       \
    using type                  = Default;                                     \
  };                                                                           \
  template <class Default, KOKKOS_IMPL_PP_REMOVE_PARENS(params)>               \
  struct name##__impl<Default, ::Kokkos::Impl::void_t<__VA_ARGS__>,            \
                      KOKKOS_IMPL_PP_REMOVE_PARENS(params_no_intro)> {         \
    static constexpr bool value = true;                                        \
    using type                  = __VA_ARGS__;                                 \
  };                                                                           \
  template <class Default, class _always_void,                                 \
            KOKKOS_IMPL_PP_REMOVE_PARENS(params)>                              \
  using name = name##__impl<Default, _always_void,                             \
                            KOKKOS_IMPL_PP_REMOVE_PARENS(params_no_intro)>

template <class Default, class _always_void, template <class...> class Op,
          class... Args>
using _detector =
    _detection_impl::detector_workaround_impl<Default, Op, Args...>;

#endif

#define KOKKOS_IMPL_DECLARE_DETECTION_ARCHETYPE_1PARAM(name, param, ...) \
  KOKKOS_IMPL_DECLARE_DETECTION_ARCHETYPE(name, (class param), (param),  \
                                          __VA_ARGS__)

#define KOKKOS_IMPL_DECLARE_DETECTION_ARCHETYPE_2PARAMS(name, param1, param2, \
                                                        ...)                  \
  KOKKOS_IMPL_DECLARE_DETECTION_ARCHETYPE(name, (class param1, class param2), \
                                          (param1, param2), __VA_ARGS__)

template <template <class...> class Op, class... Args>
using is_detected = _detector<nonesuch, void, Op, Args...>;

template <template <class...> class Op, class... Args>
using detected_t = typename is_detected<Op, Args...>::type;

template <class Default, template <class...> class Op, class... Args>
using detected_or = _detector<Default, void, Op, Args...>;

template <class Default, template <class...> class Op, class... Args>
using detected_or_t = typename detected_or<Default, Op, Args...>::type;

template <class Expected, template <class...> class Op, class... Args>
using is_detected_exact = std::is_same<Expected, detected_t<Op, Args...>>;

template <class To, template <class...> class Op, class... Args>
using is_detected_convertible =
    std::is_convertible<detected_t<Op, Args...>, To>;
}  // end namespace Impl
}  // end namespace Kokkos

#endif  // KOKKOS_PROPERTIES_DETECTION_HPP
