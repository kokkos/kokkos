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

#include <Kokkos_MathematicalConstants.hpp>

// NOTE These out-of class definitions are only required with C++14.  Since
// C++17, a static data member declared constexpr is implicitly inline.

#if !defined(KOKKOS_ENABLE_CXX17)
namespace Kokkos {
namespace Experimental {
#define OUT_OF_CLASS_DEFINITION_MATH_CONSTANT(TRAIT) \
  constexpr float TRAIT<float>::value;               \
  constexpr double TRAIT<double>::value;             \
  constexpr long double TRAIT<long double>::value

OUT_OF_CLASS_DEFINITION_MATH_CONSTANT(e);
OUT_OF_CLASS_DEFINITION_MATH_CONSTANT(log2e);
OUT_OF_CLASS_DEFINITION_MATH_CONSTANT(log10e);
OUT_OF_CLASS_DEFINITION_MATH_CONSTANT(pi);
OUT_OF_CLASS_DEFINITION_MATH_CONSTANT(inv_pi);
OUT_OF_CLASS_DEFINITION_MATH_CONSTANT(inv_sqrtpi);
OUT_OF_CLASS_DEFINITION_MATH_CONSTANT(ln2);
OUT_OF_CLASS_DEFINITION_MATH_CONSTANT(ln10);
OUT_OF_CLASS_DEFINITION_MATH_CONSTANT(sqrt2);
OUT_OF_CLASS_DEFINITION_MATH_CONSTANT(sqrt3);
OUT_OF_CLASS_DEFINITION_MATH_CONSTANT(inv_sqrt3);
OUT_OF_CLASS_DEFINITION_MATH_CONSTANT(egamma);
OUT_OF_CLASS_DEFINITION_MATH_CONSTANT(phi);

}  // namespace Experimental
}  // namespace Kokkos
#endif
