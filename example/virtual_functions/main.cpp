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

#include <classes.hpp>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  {
    Foo* f_1 = (Foo*)Kokkos::kokkos_malloc(sizeof(Foo_1));
    Foo* f_2 = (Foo*)Kokkos::kokkos_malloc(sizeof(Foo_2));

    Kokkos::parallel_for(
        "CreateObjects", 1, KOKKOS_LAMBDA(const int&) {
          new ((Foo_1*)f_1) Foo_1();
          new ((Foo_2*)f_2) Foo_2();
        });

    int value_1, value_2;
    Kokkos::parallel_reduce(
        "CheckValues", 1,
        KOKKOS_LAMBDA(const int&, int& lsum) { lsum = f_1->value(); }, value_1);

    Kokkos::parallel_reduce(
        "CheckValues", 1,
        KOKKOS_LAMBDA(const int&, int& lsum) { lsum = f_2->value(); }, value_2);

    printf("Values: %i %i\n", value_1, value_2);

    Kokkos::parallel_for(
        "DestroyObjects", 1, KOKKOS_LAMBDA(const int&) {
          f_1->~Foo();
          f_2->~Foo();
        });

    Kokkos::kokkos_free(f_1);
    Kokkos::kokkos_free(f_2);
  }

  Kokkos::finalize();
}
