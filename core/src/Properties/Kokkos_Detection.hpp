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

#ifndef KOKKOS_PROPERTIES_DETECTION_HPP
#define KOKKOS_PROPERTIES_DETECTION_HPP

#include <Kokkos_Core_fwd.hpp>

namespace Kokkos {
namespace Impl {

// A declval implementation that won't make nvcc barf
template <class T>
KOKKOS_FUNCTION
T&& _declval(int) noexcept;
template <class T>
KOKKOS_FUNCTION
T _declval(long) noexcept;

template <class T>
KOKKOS_FUNCTION
decltype(Kokkos::Impl::_declval<T>(0))
declval() noexcept;

// A void_t implementation that works with gcc-4.9 (workaround for bug 64395, also in EDG)
// From: http://stackoverflow.com/questions/35753920/why-does-the-void-t-detection-idiom-not-work-with-gcc-4-9
namespace _void_t_impl {

template <class U>
struct make_void {
  template <class V>
  struct _make_void_impl {
    using type = void;
  };
  using type = typename _make_void_impl<U>::type;
};

} // end namepace _void_t_impl

template <class T>
using void_t = typename _void_t_impl::make_void<T>::type;

/*
template <class>
using void_t = void;
 */

// Large pieces taken or adapted from http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/n4436.pdf

// primary template handles all types not supporting the archetypal Op
template <
  class Default,
  class _always_void,
  template <class...> class Op,
  class... Args
>
struct _detector {
  constexpr static auto value = false;
  using type = Default;
};

// specialization recognizes and handles only types supporting Op
template <
  class Default,
  template <class...> class Op,
  class... Args
>
struct _detector<Default, void_t<Op<Args...>>, Op, Args...> {
  constexpr static auto value = true;
  using type = Op<Args...>;
};

struct nonesuch {
  nonesuch() = delete;
  ~nonesuch() = delete;
  nonesuch(nonesuch const&) = delete;
  void operator=(nonesuch const&) = delete;
};

template <template <class...> class Op, class... Args>
using is_detected = _detector<nonesuch, void, Op, Args...>;

template <template <class...> class Op, class... Args>
using detected_t = typename is_detected<Op, Args...>::type;

template <class Default, template <class...> class Op, class... Args>
using detected_or = _detector<Default, void, Op, Args...>;

template <class Default, template <class...> class Op, class... Args>
using detected_or_t = typename detected_or<Default, Op, Args...>::type;

template <class Expected, template<class...> class Op, class... Args>
using is_detected_exact = std::is_same<Expected, detected_t<Op, Args...>>;

template <class To, template <class...> class Op, class... Args>
using is_detected_convertible = std::is_convertible<detected_t<Op, Args...>, To>;

} // end namespace Impl
} // end namespace Kokkos


#endif //KOKKOS_PROPERTIES_DETECTION_HPP
