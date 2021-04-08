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

#ifndef KOKKOS_CORE_IMPL_UTILITIES_HPP
#define KOKKOS_CORE_IMPL_UTILITIES_HPP

#include <Kokkos_Macros.hpp>
#include <cstdint>
#include <type_traits>
#include <initializer_list>  // in-order comma operator fold emulation
#include <utility>           // integer_sequence and friends

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template <typename T>
struct identity {
  using type = T;
};

template <typename T>
using identity_t = typename identity<T>::type;

// Make a type dependent on something in order to avoid ODR-using it (e.g.,
// in order to avoid requiring it to be complete).
template <class T, class /*Ignored*/>
struct dependent_identity {
  using type = T;
};

struct not_a_type {
  not_a_type()                  = delete;
  ~not_a_type()                 = delete;
  not_a_type(not_a_type const&) = delete;
  void operator=(not_a_type const&) = delete;
};

#if defined(__cpp_lib_void_t)
// since C++17
using std::void_t;
#else
template <class...>
using void_t = void;
#endif

//==============================================================================
// <editor-fold desc="remove_cvref_t"> {{{1

#if defined(__cpp_lib_remove_cvref)
// since C++20
using std::remove_cvref;
using std::remove_cvref_t;
#else
template <class T>
struct remove_cvref {
  using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

template <class T>
using remove_cvref_t = typename remove_cvref<T>::type;
#endif

// </editor-fold> end remove_cvref_t }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="is_specialization_of"> {{{1

template <class Type, template <class...> class Template, class Enable = void>
struct is_specialization_of : std::false_type {};

template <template <class...> class Template, class... Args>
struct is_specialization_of<Template<Args...>, Template> : std::true_type {};

// </editor-fold> end is_specialization_of }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="Folding emulation"> {{{1

// acts like void for comma fold emulation
struct _fold_comma_emulation_return {};

template <class... Ts>
constexpr KOKKOS_INLINE_FUNCTION _fold_comma_emulation_return
emulate_fold_comma_operator(Ts&&...) noexcept {
  return _fold_comma_emulation_return{};
}

#define KOKKOS_IMPL_FOLD_COMMA_OPERATOR(expr)                                \
  ::Kokkos::Impl::emulate_fold_comma_operator(                               \
      ::std::initializer_list<::Kokkos::Impl::_fold_comma_emulation_return>{ \
          ((expr), ::Kokkos::Impl::_fold_comma_emulation_return{})...})

// </editor-fold> end Folding emulation }}}1
//==============================================================================

//==============================================================================
// destruct_delete is a unique_ptr deleter for objects
// created by placement new into already allocated memory
// by only calling the destructor on the object.
//
// Because unique_ptr never calls its deleter with a nullptr value,
// no need to check if p == nullptr.
//
// Note:  This differs in interface from std::default_delete in that the
// function call operator is templated instead of the class, to make
// it easier to use and disallow specialization.
struct destruct_delete {
  template <typename T>
  KOKKOS_INLINE_FUNCTION constexpr void operator()(T* p) const noexcept {
    p->~T();
  }
};
//==============================================================================

//==============================================================================
// <editor-fold desc="type_list"> {{{1

// An intentionally uninstantiateable type_list for metaprogramming purposes
template <class...>
struct type_list;

// </editor-fold> end type_list }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="MSVC linearize base workaround"> {{{1

// MSVC workaround: more than two base classes (and more than one if the
// hierarchy gets too deep) causes problems with EBO, so we need to linearize
// the inheritance hierarchy to avoid losing EBO and ending up with an object
// representation that is larger than it needs to be.
// Note: by convention, the nested template in a higher-order metafunction like
// GetBase is named apply, so we use that name here (this convention grew out
// of Boost MPL)
template <template <class> class GetBase, class...>
struct linearize_bases;
template <template <class> class GetBase, class T, class... Ts>
struct linearize_bases<GetBase, T, Ts...> : GetBase<T>::template apply<Ts...> {
};
template <template <class> class GetBase>
struct linearize_bases<GetBase> {};

// </editor-fold> end MSVC linearize base workaround }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="repeated_type"> {{{1

template <class T, std::size_t I>
using repeated_type = T;

template <class T, T val, std::size_t I>
struct repeated_value { static constexpr T value = val; };

// </editor-fold> end repeated_type }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="Tools for iterating through flags in an enum"> {{{1

template <class T, T Flag>
struct next_flag {
  // Since we can't do casts like this in constexpr, we still have to use an
  // enum here
  enum : std::underlying_type_t<T> {
    underlying_value = std::underlying_type_t<T>(Flag << 1)
  };
  static constexpr T value = static_cast<T>(underlying_value);
};

template <class T, T Flag>
/* KOKKOS_INLINE_VARIABLE */
constexpr T next_flag_v = next_flag<T, Flag>::value;

// </editor-fold> end Tools for iterating through flags in an enum }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="add_restrict"> {{{1

template <class T>
struct add_restrict;

template <class T>
struct add_restrict<T*> {
  using type = T* KOKKOS_RESTRICT;
};

template <class T>
struct add_restrict<T&> {
  using type = T& KOKKOS_RESTRICT;
};

template <class T>
using add_restrict_t = typename add_restrict<T>::type;

// </editor-fold> end add_restrict }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="align_ptr"> {{{1

template <class T>
struct align_ptr;

template <class T>
struct align_ptr<T*> {
  using type = T* KOKKOS_IMPL_ALIGN_PTR(KOKKOS_MEMORY_ALIGNMENT);
};

template <class T>
using align_ptr_t = typename align_ptr<T>::type;

// </editor-fold> end add_restrict }}}1
//==============================================================================

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_CORE_IMPL_UTILITIES_HPP
