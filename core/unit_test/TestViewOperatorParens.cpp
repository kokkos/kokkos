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

#include <Kokkos_Core.hpp>

#define STATIC_ASSERT(cond) static_assert(cond, "")

// clang-format off
using V0 = Kokkos::View<int>;
using V1 = Kokkos::View<int*>;
using V2 = Kokkos::View<int**>;
using V3 = Kokkos::View<int***>;
using V4 = Kokkos::View<int****>;
using V5 = Kokkos::View<int*****>;
using V6 = Kokkos::View<int******>;
using V7 = Kokkos::View<int*******>;
using V8 = Kokkos::View<int********>;

template <class V> using access_view_operator_parens_0_args_t = decltype(std::declval<V const&>().operator()());
template <class V> using access_view_operator_parens_1_args_t = decltype(std::declval<V const&>().operator()(1));
template <class V> using access_view_operator_parens_2_args_t = decltype(std::declval<V const&>().operator()(1, 2));
template <class V> using access_view_operator_parens_3_args_t = decltype(std::declval<V const&>().operator()(1, 2, 3));
template <class V> using access_view_operator_parens_4_args_t = decltype(std::declval<V const&>().operator()(1, 2, 3, 4));
template <class V> using access_view_operator_parens_5_args_t = decltype(std::declval<V const&>().operator()(1, 2, 3, 4, 5));
template <class V> using access_view_operator_parens_6_args_t = decltype(std::declval<V const&>().operator()(1, 2, 3, 4, 5, 6));
template <class V> using access_view_operator_parens_7_args_t = decltype(std::declval<V const&>().operator()(1, 2, 3, 4, 5, 6, 7));
template <class V> using access_view_operator_parens_8_args_t = decltype(std::declval<V const&>().operator()(1, 2, 3, 4, 5, 6, 7, 8));

STATIC_ASSERT(( Kokkos::is_detected<access_view_operator_parens_0_args_t, V0>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_0_args_t, V1>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_0_args_t, V2>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_0_args_t, V3>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_0_args_t, V4>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_0_args_t, V5>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_0_args_t, V6>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_0_args_t, V7>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_0_args_t, V8>::value));

STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_1_args_t, V0>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_operator_parens_1_args_t, V1>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_1_args_t, V2>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_1_args_t, V3>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_1_args_t, V4>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_1_args_t, V5>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_1_args_t, V6>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_1_args_t, V7>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_1_args_t, V8>::value));

STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_2_args_t, V0>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_2_args_t, V1>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_operator_parens_2_args_t, V2>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_2_args_t, V3>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_2_args_t, V4>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_2_args_t, V5>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_2_args_t, V6>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_2_args_t, V7>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_2_args_t, V8>::value));

STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_3_args_t, V0>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_3_args_t, V1>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_3_args_t, V2>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_operator_parens_3_args_t, V3>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_3_args_t, V4>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_3_args_t, V5>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_3_args_t, V6>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_3_args_t, V7>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_3_args_t, V8>::value));

STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_4_args_t, V0>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_4_args_t, V1>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_4_args_t, V2>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_4_args_t, V3>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_operator_parens_4_args_t, V4>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_4_args_t, V5>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_4_args_t, V6>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_4_args_t, V7>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_4_args_t, V8>::value));

STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_5_args_t, V0>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_5_args_t, V1>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_5_args_t, V2>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_5_args_t, V3>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_5_args_t, V4>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_operator_parens_5_args_t, V5>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_5_args_t, V6>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_5_args_t, V7>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_5_args_t, V8>::value));

STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_6_args_t, V0>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_6_args_t, V1>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_6_args_t, V2>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_6_args_t, V3>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_6_args_t, V4>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_6_args_t, V5>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_operator_parens_6_args_t, V6>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_6_args_t, V7>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_6_args_t, V8>::value));

STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_7_args_t, V0>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_7_args_t, V1>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_7_args_t, V2>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_7_args_t, V3>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_7_args_t, V4>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_7_args_t, V5>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_7_args_t, V6>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_operator_parens_7_args_t, V7>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_7_args_t, V8>::value));

STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_8_args_t, V0>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_8_args_t, V1>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_8_args_t, V2>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_8_args_t, V3>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_8_args_t, V4>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_8_args_t, V5>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_8_args_t, V6>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_operator_parens_8_args_t, V7>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_operator_parens_8_args_t, V8>::value));

template <class V> using access_view_member_function_0_args_t = decltype(std::declval<V const&>().access());
template <class V> using access_view_member_function_1_args_t = decltype(std::declval<V const&>().access(1));
template <class V> using access_view_member_function_2_args_t = decltype(std::declval<V const&>().access(1, 2));
template <class V> using access_view_member_function_3_args_t = decltype(std::declval<V const&>().access(1, 2, 3));
template <class V> using access_view_member_function_4_args_t = decltype(std::declval<V const&>().access(1, 2, 3, 4));
template <class V> using access_view_member_function_5_args_t = decltype(std::declval<V const&>().access(1, 2, 3, 4, 5));
template <class V> using access_view_member_function_6_args_t = decltype(std::declval<V const&>().access(1, 2, 3, 4, 5, 6));
template <class V> using access_view_member_function_7_args_t = decltype(std::declval<V const&>().access(1, 2, 3, 4, 5, 6, 7));
template <class V> using access_view_member_function_8_args_t = decltype(std::declval<V const&>().access(1, 2, 3, 4, 5, 6, 7, 8));
template <class V> using access_view_member_function_9_args_t = decltype(std::declval<V const&>().access(1, 2, 3, 4, 5, 6, 7, 8, 9));

STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_0_args_t, V0>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_0_args_t, V1>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_0_args_t, V2>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_0_args_t, V3>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_0_args_t, V4>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_0_args_t, V5>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_0_args_t, V6>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_0_args_t, V7>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_0_args_t, V8>::value));

STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_1_args_t, V0>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_1_args_t, V1>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_1_args_t, V2>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_1_args_t, V3>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_1_args_t, V4>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_1_args_t, V5>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_1_args_t, V6>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_1_args_t, V7>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_1_args_t, V8>::value));

STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_2_args_t, V0>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_2_args_t, V1>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_2_args_t, V2>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_2_args_t, V3>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_2_args_t, V4>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_2_args_t, V5>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_2_args_t, V6>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_2_args_t, V7>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_2_args_t, V8>::value));

STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_3_args_t, V0>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_3_args_t, V1>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_3_args_t, V2>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_3_args_t, V3>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_3_args_t, V4>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_3_args_t, V5>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_3_args_t, V6>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_3_args_t, V7>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_3_args_t, V8>::value));

STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_4_args_t, V0>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_4_args_t, V1>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_4_args_t, V2>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_4_args_t, V3>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_4_args_t, V4>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_4_args_t, V5>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_4_args_t, V6>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_4_args_t, V7>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_4_args_t, V8>::value));

STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_5_args_t, V0>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_5_args_t, V1>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_5_args_t, V2>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_5_args_t, V3>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_5_args_t, V4>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_5_args_t, V5>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_5_args_t, V6>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_5_args_t, V7>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_5_args_t, V8>::value));

STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_6_args_t, V0>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_6_args_t, V1>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_6_args_t, V2>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_6_args_t, V3>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_6_args_t, V4>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_6_args_t, V5>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_6_args_t, V6>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_6_args_t, V7>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_6_args_t, V8>::value));

STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_7_args_t, V0>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_7_args_t, V1>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_7_args_t, V2>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_7_args_t, V3>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_7_args_t, V4>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_7_args_t, V5>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_7_args_t, V6>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_7_args_t, V7>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_7_args_t, V8>::value));

STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_8_args_t, V0>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_8_args_t, V1>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_8_args_t, V2>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_8_args_t, V3>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_8_args_t, V4>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_8_args_t, V5>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_8_args_t, V6>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_8_args_t, V7>::value));
STATIC_ASSERT(( Kokkos::is_detected<access_view_member_function_8_args_t, V8>::value));

STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_9_args_t, V0>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_9_args_t, V1>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_9_args_t, V2>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_9_args_t, V3>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_9_args_t, V4>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_9_args_t, V5>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_9_args_t, V6>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_9_args_t, V7>::value));
STATIC_ASSERT((!Kokkos::is_detected<access_view_member_function_9_args_t, V8>::value));
// clang-format on
