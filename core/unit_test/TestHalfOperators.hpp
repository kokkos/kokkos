
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
// TODO: Remove ifndef once https://github.com/kokkos/kokkos/pull/3480 merges
#ifndef KOKKOS_ENABLE_SYCL
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
  CADD_H_H,
  CADD_H_S,
  CSUB_H_H,
  CSUB_H_S,
  CMUL_H_H,
  CMUL_H_S,
  CDIV_H_H,
  CDIV_H_S,
  ADD_H_H,
  ADD_H_S,
  ADD_S_H,
  ADD_H_D,
  ADD_D_H,
  ADD_H_I,  // TODO: SI, I, LI, LLI, USI, UI, ULI, ULLI
  ADD_I_H,
  ADD_H_S_SZ,
  ADD_S_H_SZ,
  ADD_H_D_SZ,
  ADD_D_H_SZ,
  ADD_H_I_SZ,
  ADD_I_H_SZ,
  SUB_H_H,
  SUB_H_S,
  SUB_S_H,
  SUB_H_D,
  SUB_D_H,  // TODO: SI, I, LI, LLI, USI, UI, ULI, ULLI
  SUB_H_S_SZ,
  SUB_S_H_SZ,
  SUB_H_D_SZ,
  SUB_D_H_SZ,
  MUL_H_H,
  MUL_H_S,
  MUL_S_H,
  MUL_H_D,
  MUL_D_H,  // TODO: SI, I, LI, LLI, USI, UI, ULI, ULLI
  MUL_H_S_SZ,
  MUL_S_H_SZ,
  MUL_H_D_SZ,
  MUL_D_H_SZ,
  DIV_H_H,
  DIV_H_S,
  DIV_S_H,
  DIV_H_D,
  DIV_D_H,  // TODO: SI, I, LI, LLI, USI, UI, ULI, ULLI
  DIV_H_S_SZ,
  DIV_S_H_SZ,
  DIV_H_D_SZ,
  DIV_D_H_SZ,
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
  AO_IMPL_HALF,
  AO_HALF_T,
  N_OP_TESTS
};

template <class view_type>
struct Functor_TestHalfOperators {
  half_t h_lhs, h_rhs;
  double d_lhs, d_rhs;
  view_type actual_lhs, expected_lhs;

  Functor_TestHalfOperators(half_t lhs = half_t(0), half_t rhs = half_t(0))
      : h_lhs(lhs), h_rhs(rhs) {
    actual_lhs   = view_type("actual_lhs", N_OP_TESTS);
    expected_lhs = view_type("expected_lhs", N_OP_TESTS);
    d_lhs        = cast_from_half<double>(h_lhs);
    d_rhs        = cast_from_half<double>(h_rhs);

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
    using half_impl_type = Kokkos::Impl::half_impl_t::type;
    half_impl_type half_tmp;

    tmp_lhs              = h_lhs;
    actual_lhs(ASSIGN)   = cast_from_half<double>(tmp_lhs);
    expected_lhs(ASSIGN) = d_lhs;

    tmp_lhs  = 0;
    tmp2_lhs = tmp_lhs           = h_lhs;
    actual_lhs(ASSIGN_CHAINED)   = cast_from_half<double>(tmp2_lhs);
    expected_lhs(ASSIGN_CHAINED) = d_lhs;

    actual_lhs(UNA)   = cast_from_half<double>(+h_lhs);
    expected_lhs(UNA) = +d_lhs;

    actual_lhs(UNS)   = cast_from_half<double>(-h_lhs);
    expected_lhs(UNS) = -d_lhs;

    tmp_lhs                  = h_lhs;
    tmp_d_lhs                = d_lhs;
    actual_lhs(PREFIX_INC)   = cast_from_half<double>(++tmp_lhs);
    expected_lhs(PREFIX_INC) = ++tmp_d_lhs;

    actual_lhs(PREFIX_DEC)   = cast_from_half<double>(--tmp_lhs);
    expected_lhs(PREFIX_DEC) = --tmp_d_lhs;

    // if (h_lhs != tmp_lhs) {
    //  printf("tmp_lhs = %f, h_lhs = %f\n", __half2float(tmp_lhs),
    //  __half2float(h_lhs)); Kokkos::abort("Error in half_t prefix operators");
    //}

    actual_lhs(POSTFIX_INC)   = cast_from_half<double>(tmp_lhs++);
    expected_lhs(POSTFIX_INC) = tmp_d_lhs++;

    actual_lhs(POSTFIX_DEC)   = cast_from_half<double>(tmp_lhs--);
    expected_lhs(POSTFIX_DEC) = tmp_d_lhs--;

    // if (h_lhs != tmp_lhs) {
    //  printf("tmp_lhs = %f, h_lhs = %f\n", __half2float(tmp_lhs),
    //  __half2float(h_lhs)); Kokkos::abort("Error in half_t postfix
    //  operators");
    //}

    tmp_lhs = h_lhs;
    tmp_lhs += h_rhs;
    actual_lhs(CADD_H_H)   = cast_from_half<double>(tmp_lhs);
    expected_lhs(CADD_H_H) = d_lhs;
    expected_lhs(CADD_H_H) += d_rhs;

    tmp_lhs = h_lhs;
    tmp_lhs += static_cast<float>(d_rhs);
    actual_lhs(CADD_H_S)   = cast_from_half<double>(tmp_lhs);
    expected_lhs(CADD_H_S) = d_lhs;
    expected_lhs(CADD_H_S) += d_rhs;

    tmp_lhs = h_lhs;
    tmp_lhs -= h_rhs;
    actual_lhs(CSUB_H_H)   = cast_from_half<double>(tmp_lhs);
    expected_lhs(CSUB_H_H) = d_lhs;
    expected_lhs(CSUB_H_H) -= d_rhs;

    tmp_lhs = h_lhs;
    tmp_lhs -= static_cast<float>(d_rhs);
    actual_lhs(CSUB_H_S)   = cast_from_half<double>(tmp_lhs);
    expected_lhs(CSUB_H_S) = d_lhs;
    expected_lhs(CSUB_H_S) -= d_rhs;

    tmp_lhs = h_lhs;
    tmp_lhs *= h_rhs;
    actual_lhs(CMUL_H_H)   = cast_from_half<double>(tmp_lhs);
    expected_lhs(CMUL_H_H) = d_lhs;
    expected_lhs(CMUL_H_H) *= d_rhs;

    tmp_lhs = h_lhs;
    tmp_lhs *= static_cast<float>(d_rhs);
    actual_lhs(CMUL_H_S)   = cast_from_half<double>(tmp_lhs);
    expected_lhs(CMUL_H_S) = d_lhs;
    expected_lhs(CMUL_H_S) *= d_rhs;

    tmp_lhs = h_lhs;
    tmp_lhs /= h_rhs;
    actual_lhs(CDIV_H_H)   = cast_from_half<double>(tmp_lhs);
    expected_lhs(CDIV_H_H) = d_lhs;
    expected_lhs(CDIV_H_H) /= d_rhs;

    tmp_lhs = h_lhs;
    tmp_lhs /= static_cast<float>(d_rhs);
    actual_lhs(CDIV_H_S)   = cast_from_half<double>(tmp_lhs);
    expected_lhs(CDIV_H_S) = d_lhs;
    expected_lhs(CDIV_H_S) /= d_rhs;

    actual_lhs(ADD_H_H)   = cast_from_half<double>(h_lhs + h_rhs);
    expected_lhs(ADD_H_H) = d_lhs + d_rhs;
    auto add_h_s          = h_lhs + static_cast<float>(d_rhs);
    actual_lhs(ADD_H_S)   = add_h_s;
    expected_lhs(ADD_H_S) = d_lhs + d_rhs;
    auto add_s_h          = static_cast<float>(d_lhs) + h_rhs;
    actual_lhs(ADD_S_H)   = add_s_h;
    expected_lhs(ADD_S_H) = d_lhs + d_rhs;
    auto add_h_d          = h_lhs + d_rhs;
    actual_lhs(ADD_H_D)   = add_h_d;
    expected_lhs(ADD_H_D) = d_lhs + d_rhs;
    auto add_d_h          = d_lhs + h_rhs;
    actual_lhs(ADD_D_H)   = add_d_h;
    expected_lhs(ADD_D_H) = d_lhs + d_rhs;
    auto add_h_i          = h_lhs + int(d_rhs);
    actual_lhs(ADD_H_I)   = cast_from_half<double>(add_h_i);
    expected_lhs(ADD_H_I) = d_lhs + int(d_rhs);
    auto add_i_h          = int(d_lhs) + h_rhs;
    actual_lhs(ADD_I_H)   = cast_from_half<double>(add_i_h);
    expected_lhs(ADD_I_H) = int(d_lhs) + d_rhs;
    // TODO: SI, I, LI, LLI, USI, UI, ULI, ULLI

    actual_lhs(ADD_H_S_SZ)   = sizeof(add_h_s);
    expected_lhs(ADD_H_S_SZ) = sizeof(float);
    actual_lhs(ADD_S_H_SZ)   = sizeof(add_s_h);
    expected_lhs(ADD_S_H_SZ) = sizeof(float);
    actual_lhs(ADD_H_D_SZ)   = sizeof(add_h_d);
    expected_lhs(ADD_H_D_SZ) = sizeof(double);
    actual_lhs(ADD_D_H_SZ)   = sizeof(add_d_h);
    expected_lhs(ADD_D_H_SZ) = sizeof(double);
    actual_lhs(ADD_H_I_SZ)   = sizeof(add_h_i);
    expected_lhs(ADD_H_I_SZ) = sizeof(half_t);
    actual_lhs(ADD_I_H_SZ)   = sizeof(add_i_h);
    expected_lhs(ADD_I_H_SZ) = sizeof(half_t);

    actual_lhs(SUB_H_H)   = cast_from_half<double>(h_lhs - h_rhs);
    expected_lhs(SUB_H_H) = d_lhs - d_rhs;
    auto sub_h_s          = h_lhs - static_cast<float>(d_rhs);
    actual_lhs(SUB_H_S)   = sub_h_s;
    expected_lhs(SUB_H_S) = d_lhs - d_rhs;
    auto sub_s_h          = static_cast<float>(d_lhs) - h_rhs;
    actual_lhs(SUB_S_H)   = sub_s_h;
    expected_lhs(SUB_S_H) = d_lhs - d_rhs;
    auto sub_h_d          = h_lhs - d_rhs;
    actual_lhs(SUB_H_D)   = sub_h_d;
    expected_lhs(SUB_H_D) = d_lhs - d_rhs;
    auto sub_d_h          = d_lhs - h_rhs;
    actual_lhs(SUB_D_H)   = sub_d_h;
    expected_lhs(SUB_D_H) = d_lhs - d_rhs;
    // TODO: SI, I, LI, LLI, USI, UI, ULI, ULLI

    actual_lhs(SUB_H_S_SZ)   = sizeof(sub_h_s);
    expected_lhs(SUB_H_S_SZ) = sizeof(float);
    actual_lhs(SUB_S_H_SZ)   = sizeof(sub_s_h);
    expected_lhs(SUB_S_H_SZ) = sizeof(float);
    actual_lhs(SUB_H_D_SZ)   = sizeof(sub_h_d);
    expected_lhs(SUB_H_D_SZ) = sizeof(double);
    actual_lhs(SUB_D_H_SZ)   = sizeof(sub_d_h);
    expected_lhs(SUB_D_H_SZ) = sizeof(double);

    actual_lhs(MUL_H_H)   = cast_from_half<double>(h_lhs * h_rhs);
    expected_lhs(MUL_H_H) = d_lhs * d_rhs;
    auto mul_h_s          = h_lhs * static_cast<float>(d_rhs);
    actual_lhs(MUL_H_S)   = mul_h_s;
    expected_lhs(MUL_H_S) = d_lhs * d_rhs;
    auto mul_s_h          = static_cast<float>(d_lhs) * h_rhs;
    actual_lhs(MUL_S_H)   = mul_s_h;
    expected_lhs(MUL_S_H) = d_lhs * d_rhs;
    auto mul_h_d          = h_lhs * d_rhs;
    actual_lhs(MUL_H_D)   = mul_h_d;
    expected_lhs(MUL_H_D) = d_lhs * d_rhs;
    auto mul_d_h          = d_lhs * h_rhs;
    actual_lhs(MUL_D_H)   = mul_d_h;
    expected_lhs(MUL_D_H) = d_lhs * d_rhs;
    // TODO: SI, I, LI, LLI, USI, UI, ULI, ULLI

    actual_lhs(MUL_H_S_SZ)   = sizeof(mul_h_s);
    expected_lhs(MUL_H_S_SZ) = sizeof(float);
    actual_lhs(MUL_S_H_SZ)   = sizeof(mul_s_h);
    expected_lhs(MUL_S_H_SZ) = sizeof(float);
    actual_lhs(MUL_H_D_SZ)   = sizeof(mul_h_d);
    expected_lhs(MUL_H_D_SZ) = sizeof(double);
    actual_lhs(MUL_D_H_SZ)   = sizeof(mul_d_h);
    expected_lhs(MUL_D_H_SZ) = sizeof(double);

    actual_lhs(DIV_H_H)   = cast_from_half<double>(h_lhs / h_rhs);
    expected_lhs(DIV_H_H) = d_lhs / d_rhs;
    auto div_h_s          = h_lhs / static_cast<float>(d_rhs);
    actual_lhs(DIV_H_S)   = div_h_s;
    expected_lhs(DIV_H_S) = d_lhs / d_rhs;
    auto div_s_h          = static_cast<float>(d_lhs) / h_rhs;
    actual_lhs(DIV_S_H)   = div_s_h;
    expected_lhs(DIV_S_H) = d_lhs / d_rhs;
    auto div_h_d          = h_lhs / d_rhs;
    actual_lhs(DIV_H_D)   = div_h_d;
    expected_lhs(DIV_H_D) = d_lhs / d_rhs;
    auto div_d_h          = d_lhs / h_rhs;
    actual_lhs(DIV_D_H)   = div_d_h;
    expected_lhs(DIV_D_H) = d_lhs / d_rhs;
    // TODO: SI, I, LI, LLI, USI, UI, ULI, ULLI

    actual_lhs(DIV_H_S_SZ)   = sizeof(div_h_s);
    expected_lhs(DIV_H_S_SZ) = sizeof(float);
    actual_lhs(DIV_S_H_SZ)   = sizeof(div_s_h);
    expected_lhs(DIV_S_H_SZ) = sizeof(float);
    actual_lhs(DIV_H_D_SZ)   = sizeof(div_h_d);
    expected_lhs(DIV_H_D_SZ) = sizeof(double);
    actual_lhs(DIV_D_H_SZ)   = sizeof(div_d_h);
    expected_lhs(DIV_D_H_SZ) = sizeof(double);

    // TODO: figure out why operator{!,&&,||} are returning __nv_bool
    actual_lhs(NEG)   = static_cast<double>(!h_lhs);
    expected_lhs(NEG) = !d_lhs;

    actual_lhs(AND)   = static_cast<double>(half_t(0) && h_lhs);
    expected_lhs(AND) = double(0) && d_lhs;

    actual_lhs(OR)   = static_cast<double>(h_lhs || half_t(1));
    expected_lhs(OR) = d_lhs || double(1);

    actual_lhs(EQ)   = h_lhs == h_rhs;
    expected_lhs(EQ) = d_lhs == d_rhs;

    actual_lhs(NEQ)   = h_lhs != h_rhs;
    expected_lhs(NEQ) = d_lhs != d_rhs;

    actual_lhs(LT)   = h_lhs < h_rhs;
    expected_lhs(LT) = d_lhs < d_rhs;

    actual_lhs(GT)   = h_lhs > h_rhs;
    expected_lhs(GT) = d_lhs > d_rhs;

    actual_lhs(LE)   = h_lhs <= h_rhs;
    expected_lhs(LE) = d_lhs <= d_rhs;

    actual_lhs(GE)   = h_lhs >= h_rhs;
    expected_lhs(GE) = d_lhs >= d_rhs;

    // actual_lhs(TW)   = h_lhs <=> h_rhs;  // Need C++20?
    // expected_lhs(TW) = d_lhs <=> d_rhs;  // Need C++20?

    actual_lhs(PASS_BY_REF)   = cast_from_half<double>(accept_ref(h_lhs));
    expected_lhs(PASS_BY_REF) = d_lhs;

    half_tmp = cast_from_half<float>(h_lhs);
    tmp_ptr  = &(tmp_lhs = half_tmp);
    if (tmp_ptr != &tmp_lhs)
      Kokkos::abort("Error in half_t address-of operator");
    actual_lhs(AO_IMPL_HALF)   = cast_from_half<double>(*tmp_ptr);
    expected_lhs(AO_IMPL_HALF) = d_lhs;

    tmp2_lhs = h_lhs;
    tmp_ptr  = &(tmp_lhs = tmp2_lhs);
    if (tmp_ptr != &tmp_lhs)
      Kokkos::abort("Error in half_t address-of operator");
    actual_lhs(AO_HALF_T)   = cast_from_half<double>(tmp_ptr[0]);
    expected_lhs(AO_HALF_T) = d_lhs;

    // TODO: Check upcasting and downcasting in large expressions involving
    // integral and floating point types
  }
};

void __test_half_operators(half_t h_lhs, half_t h_rhs) {
  double epsilon = KOKKOS_HALF_T_IS_FLOAT ? FLT_EPSILON : FP16_EPSILON;
  Functor_TestHalfOperators<ViewType> f_device(h_lhs, h_rhs);  // Run on device
  Functor_TestHalfOperators<ViewTypeHost> f_host(h_lhs, h_rhs);  // Run on host
  typename ViewType::HostMirror f_device_actual_lhs =
      Kokkos::create_mirror_view(f_device.actual_lhs);
  typename ViewType::HostMirror f_device_expected_lhs =
      Kokkos::create_mirror_view(f_device.expected_lhs);

  ExecutionSpace().fence();
  Kokkos::deep_copy(f_device_actual_lhs, f_device.actual_lhs);
  Kokkos::deep_copy(f_device_expected_lhs, f_device.expected_lhs);
  for (int op_test = 0; op_test < N_OP_TESTS; op_test++) {
    // printf("op_test = %d\n", op_test);
    ASSERT_NEAR(f_device_actual_lhs(op_test), f_device_expected_lhs(op_test),
                epsilon);
    ASSERT_NEAR(f_host.actual_lhs(op_test), f_host.expected_lhs(op_test),
                epsilon);
  }

  // Check whether half_t is trivially copyable
  ASSERT_TRUE(std::is_trivially_copyable<half_t>::value);
  constexpr size_t n       = 2;
  constexpr size_t n_bytes = sizeof(half_t) * n;
  const half_t h_arr0 = half_t(0x89ab), h_arr1 = half_t(0xcdef);
  half_t h_arr[n];
  char c_arr[n_bytes], *h_arr_ptr = nullptr;
  size_t i;

  h_arr[0]  = h_arr0;
  h_arr[1]  = h_arr1;
  h_arr_ptr = reinterpret_cast<char*>(h_arr);

  std::memcpy(c_arr, h_arr, n_bytes);
  for (i = 0; i < n_bytes; i++) ASSERT_TRUE(c_arr[i] == h_arr_ptr[i]);

  std::memcpy(h_arr, c_arr, n_bytes);
  ASSERT_TRUE(h_arr[0] == h_arr0);
  ASSERT_TRUE(h_arr[1] == h_arr1);
}

void test_half_operators() {
  half_t h_lhs = half_t(0.23458), h_rhs = half_t(0.67898);
  for (int i = -3; i < 2; i++) {
    __test_half_operators(h_lhs + cast_to_half(i + 1), h_rhs + cast_to_half(i));
  }
}

TEST(TEST_CATEGORY, half_operators) { test_half_operators(); }
}  // namespace Test
#endif  // KOKKOS_ENABLE_SYCL
#endif  // TESTHALFOPERATOR_HPP_
