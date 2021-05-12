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

#ifndef KOKKOS_MATHEMATICAL_SPECIAL_FUNCTIONS_HPP
#define KOKKOS_MATHEMATICAL_SPECIAL_FUNCTIONS_HPP

#include <Kokkos_Macros.hpp>
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_NumericTraits.hpp>

namespace Kokkos {
namespace Experimental {

//! Compute exponential integral E1(x) (x > 0).
template<class RealType>
KOKKOS_INLINE_FUNCTION RealType expint(const RealType& x) {
//This function is a conversion of the corresponding Fortran program in
//S. Zhang & J. Jin "Computation of Special Functions" (Wiley, 1996).
  using Kokkos::Experimental::infinity;
  using Kokkos::Experimental::epsilon;
  using Kokkos::Experimental::fabs;
  using Kokkos::Experimental::pow;
  using Kokkos::Experimental::log;
  using Kokkos::Experimental::exp;

  RealType e1;

  if (x < 0) {
    e1 = -infinity<RealType>::value;
  }
  else if (x == 0.0) {
    e1 = infinity<RealType>::value;
  }
  else if (x <= 1.0) {
    e1 = 1.0;
    RealType r = 1.0;
    for (int k=1; k<=25; k++) {
      RealType k_real = static_cast<RealType> (k);
      r  = -r*k_real*x/pow(k_real+1.0,2.0);
      e1 = e1+r;
      if (fabs(r) <= fabs(e1)*epsilon<RealType>::value) 
        break;
    }
    e1 = -0.5772156649015328-log(x)+x*e1;
  }
  else {
    int m = 20 + static_cast<int>(80.0/x);
    RealType t0 = 0.0;
    for (int k=m; k>=1; k--) {
      RealType k_real = static_cast<RealType> (k);
      t0 = k_real/(1.0+k_real/(x+t0));
    }
    e1 = exp(-x)*(1.0/(x+t0));
  }
  return e1;
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
