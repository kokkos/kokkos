
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

#ifndef TESTHALFOPERATOR_HPP_
#define TESTHALFOPERATOR_HPP_
namespace Test {
#define FP16_EPSILON 0.0009765625F
using namespace Kokkos::Experimental;
using ExecutionSpace = TEST_EXECSPACE;
using ScalarType     = double;
using ViewType       = Kokkos::View<ScalarType*, ExecutionSpace>;
using ViewTypeHost   = Kokkos::View<ScalarType*, Kokkos::HostSpace>;
KOKKOS_FUNCTION
const half_t& accept_ref(const half_t& a) { return a; }

enum OP_TESTS {
  ASSIGN,
  ASSIGN_CHAINED,
  UNA,
  UNS,
  PREFIX_INC,
  PREFIX_DEC,
  POSTFIX_INC,
  POSTFIX_DEC,
  CADD,
  CSUB,
  CMUL,
  CDIV,
  ADD,
  SUB,
  MUL,
  DIV,
  NEG,
  AND,
  OR,
  EQ,
  NEQ,
  LT,
  GT,
  LE,
  GE,
  TW,
  PASS_BY_REF,
  AO___HALF,
  AO_HALF_T,
  N_OP_TESTS
};

template <class view_type>
struct Functor_TestHalfOperators {
  half_t lhs, rhs;
  double d_lhs, d_rhs;
  view_type actual_lhs, expected_lhs;

  Functor_TestHalfOperators(half_t lhs = 0, half_t rhs = 0)
      : lhs(lhs), rhs(rhs) {
    actual_lhs   = view_type("actual_lhs", N_OP_TESTS);
    expected_lhs = view_type("expected_lhs", N_OP_TESTS);
    d_lhs        = cast_from_half<double>(lhs);
    d_rhs        = cast_from_half<double>(rhs);

    if (std::is_same<view_type, ViewTypeHost>::value) {
      auto run_on_host = *this;
      run_on_host(0);
    } else {
      Kokkos::parallel_for("Test::Functor_TestHalfOperators",
                           Kokkos::RangePolicy<ExecutionSpace>(0, 1), *this);
    }
  }

  KOKKOS_FUNCTION
  void operator()(int) const {
    half_t tmp_lhs, tmp2_lhs, *tmp_ptr;
    double tmp_d_lhs;
#if defined(HALF_IMPL_TYPE)
    half_t::impl_type half_tmp;
#else
    half_t half_tmp;
#endif  // HALF_IMPL_TYPE

    tmp_lhs              = lhs;
    actual_lhs(ASSIGN)   = cast_from_half<double>(tmp_lhs);
    expected_lhs(ASSIGN) = d_lhs;

    tmp_lhs  = 0;
    tmp2_lhs = tmp_lhs           = lhs;
    actual_lhs(ASSIGN_CHAINED)   = cast_from_half<double>(tmp2_lhs);
    expected_lhs(ASSIGN_CHAINED) = d_lhs;

    actual_lhs(UNA)   = cast_from_half<double>(+lhs);
    expected_lhs(UNA) = +d_lhs;

    actual_lhs(UNS)   = cast_from_half<double>(-lhs);
    expected_lhs(UNS) = -d_lhs;

    tmp_lhs                  = lhs;
    tmp_d_lhs                = d_lhs;
    actual_lhs(PREFIX_INC)   = cast_from_half<double>(++tmp_lhs);
    expected_lhs(PREFIX_INC) = ++tmp_d_lhs;

    actual_lhs(PREFIX_DEC)   = cast_from_half<double>(--tmp_lhs);
    expected_lhs(PREFIX_DEC) = --tmp_d_lhs;

    // if (lhs != tmp_lhs) {
    //  printf("tmp_lhs = %f, lhs = %f\n", __half2float(tmp_lhs),
    //  __half2float(lhs)); Kokkos::abort("Error in half_t prefix operators");
    //}

    actual_lhs(POSTFIX_INC)   = cast_from_half<double>(tmp_lhs++);
    expected_lhs(POSTFIX_INC) = tmp_d_lhs++;

    actual_lhs(POSTFIX_DEC)   = cast_from_half<double>(tmp_lhs--);
    expected_lhs(POSTFIX_DEC) = tmp_d_lhs--;

    // if (lhs != tmp_lhs) {
    //  printf("tmp_lhs = %f, lhs = %f\n", __half2float(tmp_lhs),
    //  __half2float(lhs)); Kokkos::abort("Error in half_t postfix operators");
    //}

    tmp_lhs = lhs;
    tmp_lhs += rhs;
    actual_lhs(CADD)   = cast_from_half<double>(tmp_lhs);
    expected_lhs(CADD) = d_lhs;
    expected_lhs(CADD) += d_rhs;

    tmp_lhs = lhs;
    tmp_lhs -= rhs;
    actual_lhs(CSUB)   = cast_from_half<double>(tmp_lhs);
    expected_lhs(CSUB) = d_lhs;
    expected_lhs(CSUB) -= d_rhs;

    tmp_lhs = lhs;
    tmp_lhs *= rhs;
    actual_lhs(CMUL)   = cast_from_half<double>(tmp_lhs);
    expected_lhs(CMUL) = d_lhs;
    expected_lhs(CMUL) *= d_rhs;

    tmp_lhs = lhs;
    tmp_lhs /= rhs;
    actual_lhs(CDIV)   = cast_from_half<double>(tmp_lhs);
    expected_lhs(CDIV) = d_lhs;
    expected_lhs(CDIV) /= d_rhs;

    actual_lhs(ADD)   = cast_from_half<double>(lhs + rhs);
    expected_lhs(ADD) = d_lhs + d_rhs;

    actual_lhs(SUB)   = cast_from_half<double>(lhs - rhs);
    expected_lhs(SUB) = d_lhs - d_rhs;

    actual_lhs(MUL)   = cast_from_half<double>(lhs * rhs);
    expected_lhs(MUL) = d_lhs * d_rhs;

    actual_lhs(DIV)   = cast_from_half<double>(lhs / rhs);
    expected_lhs(DIV) = d_lhs / d_rhs;

    actual_lhs(NEG)   = cast_from_half<double>(!lhs);
    expected_lhs(NEG) = !d_lhs;

    actual_lhs(AND)   = cast_from_half<double>(half_t(0) && lhs);
    expected_lhs(AND) = double(0) && d_lhs;

    actual_lhs(OR)   = cast_from_half<double>(lhs || half_t(1));
    expected_lhs(OR) = d_lhs || double(1);

    actual_lhs(EQ)   = lhs == rhs;
    expected_lhs(EQ) = d_lhs == d_rhs;

    actual_lhs(NEQ)   = lhs != rhs;
    expected_lhs(NEQ) = d_lhs != d_rhs;

    actual_lhs(LT)   = lhs < rhs;
    expected_lhs(LT) = d_lhs < d_rhs;

    actual_lhs(GT)   = lhs > rhs;
    expected_lhs(GT) = d_lhs > d_rhs;

    actual_lhs(LE)   = lhs <= rhs;
    expected_lhs(LE) = d_lhs <= d_rhs;

    actual_lhs(GE)   = lhs >= rhs;
    expected_lhs(GE) = d_lhs >= d_rhs;

    // actual_lhs(TW)   = lhs <=> rhs;  // Need C++20?
    // expected_lhs(TW) = d_lhs <=> d_rhs;  // Need C++20?

    actual_lhs(PASS_BY_REF)   = cast_from_half<double>(accept_ref(lhs));
    expected_lhs(PASS_BY_REF) = d_lhs;

    half_tmp = lhs;
    // half_tmp = cast_from_half<float>(lhs);
    tmp_ptr = &(tmp_lhs = half_tmp);
    if (tmp_ptr != &tmp_lhs)
      Kokkos::abort("Error in half_t address-of operator");
    actual_lhs(AO___HALF)   = cast_from_half<double>(*tmp_ptr);
    expected_lhs(AO___HALF) = d_lhs;

    tmp2_lhs = lhs;
    tmp_ptr  = &(tmp_lhs = tmp2_lhs);
    if (tmp_ptr != &tmp_lhs)
      Kokkos::abort("Error in half_t address-of operator");
    actual_lhs(AO_HALF_T)   = cast_from_half<double>(tmp_ptr[0]);
    expected_lhs(AO_HALF_T) = d_lhs;
  }
};

void __test_half_operators(half_t lhs, half_t rhs) {
  double epsilon = half_is_float ? FLT_EPSILON : FP16_EPSILON;
  Functor_TestHalfOperators<ViewType> f_device(lhs, rhs);    // Run on device
  Functor_TestHalfOperators<ViewTypeHost> f_host(lhs, rhs);  // Run on host
  typename ViewType::HostMirror f_device_actual_lhs =
      Kokkos::create_mirror_view(f_device.actual_lhs);
  typename ViewType::HostMirror f_device_expected_lhs =
      Kokkos::create_mirror_view(f_device.expected_lhs);

  ExecutionSpace().fence();
  Kokkos::deep_copy(f_device_actual_lhs, f_device.actual_lhs);
  Kokkos::deep_copy(f_device_expected_lhs, f_device.expected_lhs);
  for (int op_test = 0; op_test < N_OP_TESTS; op_test++) {
    // printf("%lf\n", actual_lhs(op));
    ASSERT_NEAR(f_device_actual_lhs(op_test), f_device_expected_lhs(op_test),
                epsilon);
    ASSERT_NEAR(f_host.actual_lhs(op_test), f_host.expected_lhs(op_test),
                epsilon);
  }
}

void test_half_operators() {
  half_t lhs = 0.23458, rhs = 0.67898;
  for (int i = -3; i < 2; i++) {
    __test_half_operators(lhs + cast_to_half(i + 1), rhs + cast_to_half(i));
  }
}

TEST(TEST_CATEGORY, half_operators) { test_half_operators(); }
}  // namespace Test
#endif
