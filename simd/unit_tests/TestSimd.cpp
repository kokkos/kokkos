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

#include <gtest/gtest.h>

#include <Kokkos_Simd.hpp>

#include <impl/Kokkos_Error.hpp>

class check_using_gtest {
 public:
  void operator()(bool condition, char const* message) const
  {
    GTEST_TEST_BOOLEAN_(condition, message, false, true, GTEST_NONFATAL_FAILURE_);
  }
};

class check_using_kokkos {
 public:
  KOKKOS_INLINE_FUNCTION void operator()(bool condition, char const* message) const
  {
    if (!condition) ::Kokkos::abort(message);
  }
};

#define KOKKOS_SIMD_CHECK(checker, expression) \
  checker(expression, #expression)

template <class Checker>
class check_equality {
  Checker m_checker;
 public:
  template <class T>
  KOKKOS_INLINE_FUNCTION void operator()(
      T const& expected_result,
      T const& computed_result) const
  {
    // use the simd_mask reducer functions first
    KOKKOS_SIMD_CHECK(m_checker, all_of(expected_result == computed_result));
    KOKKOS_SIMD_CHECK(m_checker, none_of(expected_result != computed_result));
    // double-check that with manual comparison of each entry
    for (std::size_t i = 0; i < expected_result.size(); ++i) {
      KOKKOS_SIMD_CHECK(m_checker, expected_result[i] == computed_result[i]);
    }
  }
};

class plus {
 public:
  template <class T>
  KOKKOS_INLINE_FUNCTION auto operator()(T const& a, T const& b) const
  {
    return a + b;
  }
};

TEST(simd, plus)
{
  using simd_type = Kokkos::Experimental::simd<double, Kokkos::Experimental::simd_abi::host_native>;
  check_equality<check_using_gtest> equality_checker;
  equality_checker(simd_type(2.0), simd_type(1.0) + simd_type(1.0));
}
