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

#include "CompileTestCommon.hpp"

#include <Properties/Kokkos_Detection.hpp>

#include <type_traits>

#define KOKKOS_STATIC_TEST_DETECT_EXPRESSION_T(type, ...) \
  template <class T> using KOKKOS_PP_CAT(_generated_archetype_, __LINE__) = decltype(__VA_ARGS__); \
  KOKKOS_STATIC_TEST(is_detected<KOKKOS_PP_CAT(_generated_archetype_, __LINE__), type>::value)

#define KOKKOS_STATIC_TEST_DETECT_EXPRESSION_T_U(typeT, typeU, ...) \
  template <class T, class U> using KOKKOS_PP_CAT(_generated_archetype_, __LINE__) = decltype(__VA_ARGS__); \
  KOKKOS_STATIC_TEST(is_detected<KOKKOS_PP_CAT(_generated_archetype_, __LINE__), typeT, typeU>::value)

struct A { };
struct B : A { };
struct C { };

struct MyTestCase1 {

  using my_member_type = C;

  static constexpr auto my_static_data_member = int(42);

  KOKKOS_FUNCTION
  void my_method();

  KOKKOS_FUNCTION
  int my_int_method(double, float);

  KOKKOS_FUNCTION
  double my_method_with_overloads();

  KOKKOS_FUNCTION
  B& my_method_with_overloads(my_member_type);

  KOKKOS_FUNCTION
  void my_method_with_overloads(int**);

  KOKKOS_FUNCTION
  my_member_type my_method_with_overloads(my_member_type, double);

  KOKKOS_FUNCTION
  MyTestCase1& my_method_with_overloads(A const&);

};


// TODO test these in other scopes that could trip up compilers, like function scope or dependent class template scope

// hopefully this isn't too lazy; it should make the tests a bit more readable
using namespace Kokkos::Impl;

//==============================================================================
// <editor-fold desc="Test member type detection"> {{{1

//template <class T>
//using _my_member_type_archetype = typename T::my_member_type;
KOKKOS_DECLARE_DETECTION_ARCHETYPE_1PARAM(
  _my_member_type_archetype, T,
  typename T::my_member_type
);

KOKKOS_STATIC_TEST(
  is_detected<_my_member_type_archetype, MyTestCase1>::value
);

KOKKOS_STATIC_TEST(
  is_detected_exact<C, _my_member_type_archetype, MyTestCase1>::value
);

KOKKOS_STATIC_TEST(
  is_detected_convertible<C, _my_member_type_archetype, MyTestCase1>::value
);

// </editor-fold> end Test member type detection }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="Test method detection"> {{{1

//template <class T>
//using _my_method_archetype_1 = decltype(T{}.my_method());
KOKKOS_DECLARE_DETECTION_ARCHETYPE_1PARAM(
  _my_method_archetype_1, T,
  decltype(T{}.my_method())
);

//template <class T>
//using _my_method_archetype_2 = decltype(declval<T>().my_method());
KOKKOS_DECLARE_DETECTION_ARCHETYPE_1PARAM(
  _my_method_archetype_2, T,
  decltype(declval<T>().my_method())
);

KOKKOS_STATIC_TEST(
  is_detected<_my_method_archetype_1, MyTestCase1>::value
);

KOKKOS_STATIC_TEST(
  is_detected<_my_method_archetype_2, MyTestCase1>::value
);

KOKKOS_STATIC_TEST(
  is_detected_exact<void, _my_method_archetype_1, MyTestCase1>::value
);

KOKKOS_STATIC_TEST(
  is_detected_exact<void, _my_method_archetype_2, MyTestCase1>::value
);

template <class T, class U, class V>
nonesuch _my_int_method_archetype_fn(long) { return { }; }


//KOKKOS_STATIC_TEST_DETECT_EXPRESSION_T(
//  MyTestCase1,
//  declval<T>().my_method_with_overloads(declval<C>())
//);
//
//KOKKOS_STATIC_TEST_DETECT_EXPRESSION_T_U(
//  MyTestCase1, C,
//  declval<T>().my_method_with_overloads(declval<U>())
//);


template <class U>
struct OuterClass {

  // Things like this don't work with intel or cuda (probably a bug in the EDG frontend)

  KOKKOS_DECLARE_DETECTION_ARCHETYPE_1PARAM(
    _inner_method_archetype, T,
    decltype(declval<T>().my_method_with_overloads(declval<U>()))
  );
  //template <class T>
  //using _inner_method_archetype =
  //decltype(declval<T>().my_method_with_overloads(declval<U>()));

  template <class T, class UProtected>
  using _inner_method_reversed_archetype_protected =
  decltype(declval<UProtected>().my_method_with_overloads(declval<T>()));

  template <class T>
  using _inner_method_reversed_archetype = detected_t<
    _inner_method_reversed_archetype_protected, T, U
  >;

  // test the compiler's ability to handle indirection with this pattern
  // Should be the last overload when T = MyClass and U = C
  //   (since the detected argument type should resolve to the second overload,
  //   which returns B&), and not detected otherwise
  template <class T>
  using _overload_nested_dependent_type_archetype = decltype(
  declval<T>().my_method_with_overloads(
    declval<detected_t<_inner_method_archetype, T>>()
  )
  );

  // This is how to do archetypes to workaround the bug in intel and cuda:
  template <class T, class USafe>
  using _inner_method_archetype_safe =
  decltype(declval<T>().my_method_with_overloads(declval<USafe>()));

  template <class T>
  using inner_method_detected = is_detected<_inner_method_archetype_safe, T, U>;

};

#ifdef TESTS_THAT_DONT_WORK_ON_INTEL_AND_CUDA
// These don't work on compilers with EDG frontends, even though they should.
// I've left them here as an example of what not to do and included a workaround
// below

// Should be the third overload
KOKKOS_STATIC_TEST(
  is_detected<OuterClass<int**>::template _inner_method_archetype, MyTestCase1>::value
);
KOKKOS_STATIC_TEST(
  std::is_same<
    void,
    detected_t<OuterClass<int**>::template _inner_method_archetype, MyTestCase1>
  >::value
);

// The hardest test: should be the last overload
KOKKOS_STATIC_TEST(
  is_detected_convertible<
    MyTestCase1,
    OuterClass<C>::template _overload_nested_dependent_type_archetype,
    MyTestCase1
  >::value
);
#endif

KOKKOS_STATIC_TEST(
  OuterClass<int**>::template inner_method_detected<MyTestCase1>::value
);

// </editor-fold> end Test method detection }}}1
//==============================================================================


//==============================================================================
// <editor-fold desc="Test free function detection"> {{{1

int my_free_function(A, B, C);

A my_overloaded_free_function(C);
B my_overloaded_free_function(A, B);
MyTestCase1 my_overloaded_free_function(MyTestCase1);

template <class T>
typename std::enable_if<
  std::is_convertible<T, A>::value,
  C
>::type
my_overloaded_free_function(T);

//------------------------------------------------------------------------------

template <class... Ts>
using _free_function_archetype = decltype(my_free_function(declval<Ts>()...));

template <class... Ts>
using _free_function_overload_archetype = decltype(my_overloaded_free_function(declval<Ts>()...));

template <class T, class U>
using _free_function_overload_archetype_workaround = decltype(my_overloaded_free_function(declval<T>(), declval<U>()));

KOKKOS_STATIC_TEST(
  is_detected_exact<int, _free_function_archetype, A, B, C>::value
);

KOKKOS_STATIC_TEST(
  is_detected_exact<A, _free_function_overload_archetype, C>::value
);

KOKKOS_STATIC_TEST(
  is_detected_exact<B, _free_function_overload_archetype, B, B>::value
);

KOKKOS_STATIC_TEST(
  is_detected_exact<C, _free_function_overload_archetype, B>::value
);

//KOKKOS_STATIC_TEST(
//  not is_detected_exact<C, _free_function_overload_archetype, int>::value
//);

//KOKKOS_STATIC_TEST(
//  not is_detected<_free_function_overload_archetype, B, A>::value
//);

//KOKKOS_STATIC_TEST(
//  not is_detected<_free_function_overload_archetype_workaround, B, A>::value
//);

// </editor-fold> end Test free function detection }}}1
//==============================================================================
#endif