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

#if defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_CUDA_LAMBDA)
#if defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)
#error "Macro bug: KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA shouldn't be defined"
#endif
#else
#if !defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)
#error "Macro bug: KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA should be defined"
#endif
#endif

#define KOKKOS_PRAGMA_UNROLL(a)

namespace TestCompilerMacros {

template <class DEVICE_TYPE>
struct AddFunctor {
  using execution_space = DEVICE_TYPE;
  using type            = typename Kokkos::View<int**, execution_space>;
  type a, b;
  int length;

  AddFunctor(type a_, type b_) : a(a_), b(b_), length(a.extent(1)) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(int i) const {
#ifdef KOKKOS_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
#ifdef KOKKOS_ENABLE_PRAGMA_IVDEP
#pragma ivdep
#endif
#ifdef KOKKOS_ENABLE_PRAGMA_VECTOR
#pragma vector always
#endif
#ifdef KOKKOS_ENABLE_PRAGMA_LOOPCOUNT
#pragma loop count(128)
#endif
#ifndef KOKKOS_ENABLE_DEBUG
#ifdef KOKKOS_ENABLE_PRAGMA_SIMD
#pragma simd
#endif
#endif
    for (int j = 0; j < length; j++) {
      a(i, j) += b(i, j);
    }
  }
};

template <class DeviceType>
bool Test() {
  using type = typename Kokkos::View<int**, DeviceType>;
  type a("A", 1024, 128);
  type b("B", 1024, 128);

  AddFunctor<DeviceType> f(a, b);
  Kokkos::parallel_for(1024, f);
  DeviceType().fence();

  return true;
}

}  // namespace TestCompilerMacros

namespace Test {
TEST(TEST_CATEGORY, compiler_macros) {
  ASSERT_TRUE((TestCompilerMacros::Test<TEST_EXECSPACE>()));
}
}  // namespace Test
