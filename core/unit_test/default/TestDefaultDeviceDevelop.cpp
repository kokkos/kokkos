
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

#include <Kokkos_Core.hpp>

#include <TestDefaultDeviceType_Category.hpp>

#include <Kokkos_Core.hpp>
void foo(Kokkos::View<float**> A, int R, int occ) {
  Kokkos::RangePolicy<> p(0, A.extent(0));
  auto const p_occ = Kokkos::Experimental::prefer(
      p, Kokkos::Experimental::DesiredOccupancy{Kokkos::AUTO});
  const int M = A.extent_int(1);
  Kokkos::parallel_for(
      "Bench", p_occ, KOKKOS_LAMBDA(int i) {
        for (int r = 0; r < R; r++) {
          float f = 0.;
          for (int m = 0; m < M; m++) {
            f += A(i, m);
            A(i, m) += f;
          }
        }
      });
}
int romain() {
  {
    int argc     = 1;
    char* argv[] = {"", "", "", ""};
    int N        = (argc > 1) ? atoi(argv[1]) : 1000000;
    int M        = (argc > 2) ? atoi(argv[2]) : 25;
    int R        = (argc > 3) ? atoi(argv[3]) : 20;
    int K        = (argc > 4) ? atoi(argv[4]) : 10000;
    Kokkos::View<float**> A("A", N, M);
    foo(A, R, 100);
    Kokkos::fence();
    Kokkos::Timer timer;
    for (int k = 0; k < K; k++) foo(A, R, 100);
    Kokkos::fence();
    double time = timer.seconds();
    printf("%lf\n", time);
    foo(A, R, 33);
    Kokkos::fence();
    timer.reset();
    for (int k = 0; k < K; k++) foo(A, R, 33);
    Kokkos::fence();
    time = timer.seconds();
    printf("%lf\n", time);
  }
  return 0;
}

struct TF {
  KOKKOS_FUNCTION void operator()(const int) const {}
};
namespace Test {
void foo() {
  romain();
  Kokkos::finalize();
  exit(0);
  TF f;
  using namespace Kokkos::Tools::Experimental;
  VariableInfo info;
  info.category      = StatisticalCategory::kokkos_value_categorical;
  info.type          = ValueType::kokkos_value_int64;
  info.valueQuantity = CandidateValueType::kokkos_value_unbounded;
  size_t id          = declare_input_type("dogggo", info);
  auto ctx           = get_new_context_id();
  auto v             = make_variable_value(id, int64_t(1));
  begin_context(ctx);
  set_input_values(ctx, 1, &v);
  constexpr const int data_size = 100000000;
  Kokkos::View<float*> a("a", data_size);
  Kokkos::View<float*> b("b", data_size);
  Kokkos::View<float*> c("c", data_size);
  int scalar = 1.0;
  for (int x = 0; x < 10000; ++x) {
    Kokkos::parallel_for(
        "puppies",
        Kokkos::Experimental::prefer(
            Kokkos::RangePolicy<>(0, data_size),
            Kokkos::Experimental::DesiredOccupancy{Kokkos::AUTO}),
        KOKKOS_LAMBDA(int index) { a[index] = b[index] + scalar * c[index]; });
  }
  end_context(ctx);
}
TEST(defaultdevicetype, development_test) { foo(); }

}  // namespace Test
