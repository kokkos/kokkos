
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

#ifdef _NVHPC_CUDA
#include<nv/target>

//#define SUPPRESS_WARNING //if you paste this in warnings for both foo and bar disappear
#ifdef SUPPRESS_WARNING
#pragma push
#pragma diag_suppress implicit_return_from_non_void_function
#endif
__host__ __device__ int foo() {
  if target (nv::target::is_host) {
    return 0;
  }
  if target (nv::target::is_device) {
    return 1;
  }
}
#ifdef SUPPRESS_WARNING
#pragma pop
#endif
__host__ __device__ int bar() {
  if target (nv::target::is_host) {
    return 0;
  }
  if target (nv::target::is_device) {
    return 1;
  }
}
__global__ void baz() {
  (void)foo();
  (void)bar();
}
#endif


#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <TestDefaultDeviceType_Category.hpp>

namespace Test {

void ldg_failure() {
  Kokkos::View<double*> a("A",10);
  Kokkos::deep_copy(a,1.0);
  Kokkos::parallel_for(1, KOKKOS_LAMBDA(int) {
    KOKKOS_IF_ON_DEVICE(
      double* ptr = &a(0);
      printf("%lf %lf\n",__ldg(ptr),*ptr);
      )
  });
  Kokkos::fence();
}


void subview_failure() {
  Kokkos::View<int*> a("A",10);
  Kokkos::parallel_for(10, KOKKOS_LAMBDA(int i) { a[i] = i+1; });
  Kokkos::pair<int,int> p(1,5);
  auto h_b = Kokkos::subview(a,p);
  auto h_c = Kokkos::subview(a,Kokkos::pair<int,int>(2,6));
  Kokkos::parallel_for(1, KOKKOS_LAMBDA(int) {
    Kokkos::pair<int,int> p(1,5);
    auto b = Kokkos::subview(a,p);
    auto c = Kokkos::subview(a,Kokkos::pair<int,int>(2,6));
    printf("Device: %p %p %p %i %i %i\n",a.data(),b.data(),c.data(),a(0),b(0),c(0));
    printf("Host: %p %p %p %i %i %i\n",a.data(),h_b.data(),h_c.data(),a(0),h_b(0),h_c(0));
  });
  Kokkos::fence();
}

//#define NVHPC_LABS
// unresolved ptxas external labs
#ifdef NVHPC_LABS
void abs_fail() {
  Kokkos::parallel_for(1, KOKKOS_LAMBDA(int) {
    long long a = 1;
    using std::labs;
    long long b = labs(a);
    printf("%li %li\n",a,b);
  });
}
#endif

//#define NVHPC_EXP2
//Unhandled builtin: 90 (exp2)
//NVC++-F-0000-Internal compiler error. Unhandled builtin function.       0 
#ifdef NVHPC_EXP2
void exp2_fail() {
  Kokkos::parallel_for(1, KOKKOS_LAMBDA(int) {
    double a = 1;
    using std::exp2;
    double b = exp2(a);
    printf("%li %li\n",a,b);
  });
}
#endif


//#define NVHPC_MEMORY_POOL_FAIL
// NVC++-F-0000-Internal compiler error. Basic LLVM base data type required      23
#ifdef NVHPC_MEMORY_POOL_FAIL
#include<TestMemoryPool.hpp>
#endif

TEST(defaultdevicetype, development_test) {
  ldg_failure();
  subview_failure();
#ifdef NVHPC_LABS
  abs_fail();
#endif
#ifdef NVHPC_EXP2
  exp2_fail();
#endif
}
}  // namespace Test
