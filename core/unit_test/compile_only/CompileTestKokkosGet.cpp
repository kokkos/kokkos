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

#include <Kokkos_Get.hpp>
#include <Kokkos_Pair.hpp>

#include <type_traits>

KOKKOS_STATIC_TEST(
    Kokkos::Impl::has_kokkos_get<std::pair<int, char>, 0>::value
);
KOKKOS_STATIC_TEST(
    Kokkos::Impl::has_kokkos_get<std::pair<int, char>, 1>::value
);
KOKKOS_STATIC_TEST(
    Kokkos::Impl::has_kokkos_get<std::pair<int, int>, 0>::value
);
KOKKOS_STATIC_TEST(
    Kokkos::Impl::has_kokkos_get<std::pair<int, int>, 1>::value
);
KOKKOS_STATIC_TEST(
    Kokkos::Impl::has_kokkos_get<Kokkos::pair<int, int>, 0>::value
);
KOKKOS_STATIC_TEST(
    Kokkos::Impl::has_kokkos_get<Kokkos::pair<int, int>, 1>::value
);

KOKKOS_STATIC_TEST(
    std::is_same<
      Kokkos::Impl::kokkos_get_result_t<std::pair<int, char> const&, 0>,
          int const&
          >::value
);
KOKKOS_STATIC_TEST(
    std::is_same<
        Kokkos::Impl::kokkos_get_result_t<std::pair<int, char> const&, 1>,
        char const&
    >::value
);

KOKKOS_STATIC_TEST(
    Kokkos::Impl::has_kokkos_get<std::array<int, 4>, 0>::value
);
KOKKOS_STATIC_TEST(
    Kokkos::Impl::has_kokkos_get<std::array<int, 4>, 3>::value
);
KOKKOS_STATIC_TEST(
    !Kokkos::Impl::has_kokkos_get<double, 42>::value
);
