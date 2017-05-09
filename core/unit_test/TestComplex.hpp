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
// Questions? Contact  H. Carter Edwards (hcedwar@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include<Kokkos_Core.hpp>
#include<cstdio>

namespace Test {

template<class ExecSpace>
struct TestComplexConstruction {
  Kokkos::View<Kokkos::complex<double>*,ExecSpace> d_results;
  typename Kokkos::View<Kokkos::complex<double>*,ExecSpace>::HostMirror h_results;
  
  void testit () {
    d_results = Kokkos::View<Kokkos::complex<double>*,ExecSpace>("TestComplexConstruction",10);
    h_results = Kokkos::create_mirror_view(a);
   
    Kokkos::parallel_for(1, *this);
    Kokkos::fence();
    Kokkos::deep_copy(h_results,d_results);

    ASSERT_EQ(h_results(0).real(),1.5);  ASSERT_EQ(h_results(0).real(),2.5);
    ASSERT_EQ(h_results(1).real(),1.5);  ASSERT_EQ(h_results(1).real(),2.5);
    ASSERT_EQ(h_results(2).real(),0.0);  ASSERT_EQ(h_results(2).real(),0.0);
    ASSERT_EQ(h_results(3).real(),3.5);  ASSERT_EQ(h_results(3).real(),0.0);
    ASSERT_EQ(h_results(4).real(),4.5);  ASSERT_EQ(h_results(4).real(),5.5);
    ASSERT_EQ(h_results(5).real(),1.5);  ASSERT_EQ(h_results(5).real(),2.5);
    ASSERT_EQ(h_results(6).real(),4.5);  ASSERT_EQ(h_results(6).real(),5.5);
  }

  KOKKOS_INLINE_FUNCTION 
  void operator() (const int &i ) const {
    Kokkos::complex<double> a(1.5,2.5);
    d_results(0) = a;
    Kokkos::complex<double> b(a);
    d_results(1) = b;
    Kokkos::complex<double> c = Kokkos::complex<double>();
    d_results(2) = c;
    Kokkos::complex<double> d(3.5);
    d_results(3) = d; 
    volatile Kokkos::complex<double> a_v(4.5,5.5);
    d_results(4) = a_v;
    volatile Kokkos::complex<double> b_v(a);
    d_results(5) = b_v;
    Kokkos::complex<double> e(a_v);
    d_results(6) = e;
  } 
};

TEST_F(TEST_CATEGORY, complex_construction) {
  TestComplexConstruction<TEST_EXECSPACE> test;
  test.testit();
} 

} // namespace Test


