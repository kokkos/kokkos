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

#include <Kokkos_View.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <algorithm>
#include <cstdio>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  {
    const auto fun = KOKKOS_LAMBDA(double& d) { d++; };
    Kokkos::View<double*> v("label", 10);

    using exespace = Kokkos::DefaultExecutionSpace;
#if defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)
    Kokkos::Experimental::for_each(exespace(), v, fun);
    Kokkos::Experimental::for_each("some-label", exespace(), v, fun);

    Kokkos::Experimental::for_each(exespace(), Kokkos::Experimental::begin(v),
                                   Kokkos::Experimental::end(v), fun);
    Kokkos::Experimental::for_each("some-label", exespace(),
                                   Kokkos::Experimental::begin(v),
                                   Kokkos::Experimental::end(v), fun);
    Kokkos::Experimental::for_each_n(exespace(), v, 3, fun);
    Kokkos::Experimental::for_each_n("some-label", exespace(), v, 3, fun);

    // note: it is possible to use std::for_each as well
    std::for_each(Kokkos::Experimental::begin(v), Kokkos::Experimental::end(v),
                  fun);
#endif
  }

  Kokkos::finalize();
}
