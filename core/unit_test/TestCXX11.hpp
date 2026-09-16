// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

namespace TestCXX11 {

struct FunctorAddTest {
  using execution_space = TEST_EXECSPACE;
  using view_type       = Kokkos::View<double**, execution_space>;
  using team_member = typename Kokkos::TeamPolicy<execution_space>::member_type;

  view_type a_, b_;

  FunctorAddTest(view_type& a, view_type& b) : a_(a), b_(b) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& i) const {
    b_(i, 0) = a_(i, 1) + a_(i, 2);
    b_(i, 1) = a_(i, 0) - a_(i, 3);
    b_(i, 2) = a_(i, 4) + a_(i, 0);
    b_(i, 3) = a_(i, 2) - a_(i, 1);
    b_(i, 4) = a_(i, 3) + a_(i, 4);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const team_member& dev) const {
    const int begin = dev.league_rank() * 4;
    const int end   = begin + 4;
    for (int i = begin + dev.team_rank(); i < end; i += dev.team_size()) {
      b_(i, 0) = a_(i, 1) + a_(i, 2);
      b_(i, 1) = a_(i, 0) - a_(i, 3);
      b_(i, 2) = a_(i, 4) + a_(i, 0);
      b_(i, 3) = a_(i, 2) - a_(i, 1);
      b_(i, 4) = a_(i, 3) + a_(i, 4);
    }
  }
};

template <bool PWRTest>
double AddTestFunctor() {
  using execution_space = TEST_EXECSPACE;
  Kokkos::View<double**, execution_space> a("A", 100, 5);
  Kokkos::View<double**, execution_space> b("B", 100, 5);
  typename Kokkos::View<double**, execution_space>::host_mirror_type h_a =
      Kokkos::create_mirror_view(a);
  typename Kokkos::View<double**, execution_space>::host_mirror_type h_b =
      Kokkos::create_mirror_view(b);

  for (int i = 0; i < 100; i++) {
    for (int j = 0; j < 5; j++) {
      h_a(i, j) = 0.1 * i / (1.1 * j + 1.0) + 0.5 * j;
    }
  }
  Kokkos::deep_copy(a, h_a);

  if (PWRTest == false) {
    Kokkos::parallel_for(100, FunctorAddTest(a, b));
  } else {
    using policy_type = Kokkos::TeamPolicy<execution_space>;
    Kokkos::parallel_for(policy_type(25, Kokkos::AUTO), FunctorAddTest(a, b));
  }
  Kokkos::deep_copy(h_b, b);

  double result = 0;
  for (int i = 0; i < 100; i++) {
    for (int j = 0; j < 5; j++) {
      result += h_b(i, j);
    }
  }

  return result;
}

template <bool PWRTest>
double AddTestLambda() {
  using execution_space = TEST_EXECSPACE;
  Kokkos::View<double**, execution_space> a("A", 100, 5);
  Kokkos::View<double**, execution_space> b("B", 100, 5);
  typename Kokkos::View<double**, execution_space>::host_mirror_type h_a =
      Kokkos::create_mirror_view(a);
  typename Kokkos::View<double**, execution_space>::host_mirror_type h_b =
      Kokkos::create_mirror_view(b);

  for (int i = 0; i < 100; i++) {
    for (int j = 0; j < 5; j++) {
      h_a(i, j) = 0.1 * i / (1.1 * j + 1.0) + 0.5 * j;
    }
  }
  Kokkos::deep_copy(a, h_a);

  if (PWRTest == false) {
    using policy_type = Kokkos::RangePolicy<execution_space>;
    Kokkos::parallel_for(
        policy_type(0, 100), KOKKOS_LAMBDA(const int& i) {
          b(i, 0) = a(i, 1) + a(i, 2);
          b(i, 1) = a(i, 0) - a(i, 3);
          b(i, 2) = a(i, 4) + a(i, 0);
          b(i, 3) = a(i, 2) - a(i, 1);
          b(i, 4) = a(i, 3) + a(i, 4);
        });
  } else {
    using policy_type = Kokkos::TeamPolicy<execution_space>;
    using team_member = typename policy_type::member_type;

    policy_type policy(25, Kokkos::AUTO);

    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(const team_member& dev) {
          const unsigned int begin = dev.league_rank() * 4;
          const unsigned int end   = begin + 4;
          for (unsigned int i = begin + dev.team_rank(); i < end;
               i += dev.team_size()) {
            b(i, 0) = a(i, 1) + a(i, 2);
            b(i, 1) = a(i, 0) - a(i, 3);
            b(i, 2) = a(i, 4) + a(i, 0);
            b(i, 3) = a(i, 2) - a(i, 1);
            b(i, 4) = a(i, 3) + a(i, 4);
          }
        });
  }
  Kokkos::deep_copy(h_b, b);

  double result = 0;
  for (int i = 0; i < 100; i++) {
    for (int j = 0; j < 5; j++) {
      result += h_b(i, j);
    }
  }

  return result;
}

struct FunctorReduceTest {
  using execution_space = TEST_EXECSPACE;
  using view_type       = Kokkos::View<double**, TEST_EXECSPACE>;
  using value_type      = double;
  using team_member = typename Kokkos::TeamPolicy<execution_space>::member_type;

  view_type a_;

  FunctorReduceTest(view_type& a) : a_(a) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& i, value_type& sum) const {
    sum += a_(i, 1) + a_(i, 2);
    sum += a_(i, 0) - a_(i, 3);
    sum += a_(i, 4) + a_(i, 0);
    sum += a_(i, 2) - a_(i, 1);
    sum += a_(i, 3) + a_(i, 4);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const team_member& dev, value_type& sum) const {
    const int begin = dev.league_rank() * 4;
    const int end   = begin + 4;
    for (int i = begin + dev.team_rank(); i < end; i += dev.team_size()) {
      sum += a_(i, 1) + a_(i, 2);
      sum += a_(i, 0) - a_(i, 3);
      sum += a_(i, 4) + a_(i, 0);
      sum += a_(i, 2) - a_(i, 1);
      sum += a_(i, 3) + a_(i, 4);
    }
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& update) const { update = 0.0; }

  KOKKOS_INLINE_FUNCTION
  void join(value_type& update, value_type const& input) const {
    update += input;
  }
};

template <bool PWRTest>
double ReduceTestFunctor() {
  using execution_space = TEST_EXECSPACE;
  using view_type       = Kokkos::View<double**, execution_space>;
  using unmanaged_result =
      Kokkos::View<double, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;

  view_type a("A", 100, 5);
  typename view_type::host_mirror_type h_a = Kokkos::create_mirror_view(a);

  for (int i = 0; i < 100; i++) {
    for (int j = 0; j < 5; j++) {
      h_a(i, j) = 0.1 * i / (1.1 * j + 1.0) + 0.5 * j;
    }
  }
  Kokkos::deep_copy(a, h_a);

  double result = 0.0;
  if (PWRTest == false) {
    Kokkos::parallel_reduce(100, FunctorReduceTest(a),
                            unmanaged_result(&result));
  } else {
    using policy_type = Kokkos::TeamPolicy<execution_space>;
    Kokkos::parallel_reduce(policy_type(25, Kokkos::AUTO), FunctorReduceTest(a),
                            unmanaged_result(&result));
  }
  Kokkos::fence();

  return result;
}

template <bool PWRTest>
double ReduceTestLambda() {
  using execution_space = TEST_EXECSPACE;
  using view_type       = Kokkos::View<double**, execution_space>;
  using unmanaged_result =
      Kokkos::View<double, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;

  view_type a("A", 100, 5);
  typename view_type::host_mirror_type h_a = Kokkos::create_mirror_view(a);

  for (int i = 0; i < 100; i++) {
    for (int j = 0; j < 5; j++) {
      h_a(i, j) = 0.1 * i / (1.1 * j + 1.0) + 0.5 * j;
    }
  }
  Kokkos::deep_copy(a, h_a);

  double result = 0.0;

  if (PWRTest == false) {
    using policy_type = Kokkos::RangePolicy<execution_space>;
    Kokkos::parallel_reduce(
        policy_type(0, 100),
        KOKKOS_LAMBDA(const int& i, double& sum) {
          sum += a(i, 1) + a(i, 2);
          sum += a(i, 0) - a(i, 3);
          sum += a(i, 4) + a(i, 0);
          sum += a(i, 2) - a(i, 1);
          sum += a(i, 3) + a(i, 4);
        },
        unmanaged_result(&result));
  } else {
    using policy_type = Kokkos::TeamPolicy<execution_space>;
    using team_member = typename policy_type::member_type;
    Kokkos::parallel_reduce(
        policy_type(25, Kokkos::AUTO),
        KOKKOS_LAMBDA(const team_member& dev, double& sum) {
          const unsigned int begin = dev.league_rank() * 4;
          const unsigned int end   = begin + 4;
          for (unsigned int i = begin + dev.team_rank(); i < end;
               i += dev.team_size()) {
            sum += a(i, 1) + a(i, 2);
            sum += a(i, 0) - a(i, 3);
            sum += a(i, 4) + a(i, 0);
            sum += a(i, 2) - a(i, 1);
            sum += a(i, 3) + a(i, 4);
          }
        },
        unmanaged_result(&result));
  }
  Kokkos::fence();

  return result;
}

double TestVariantLambda(int test) {
  switch (test) {
    case 1: return AddTestLambda<false>();
    case 2: return AddTestLambda<true>();
    case 3: return ReduceTestLambda<false>();
    case 4: return ReduceTestLambda<true>();
    default: Kokkos::abort("unreachable");
  }

  return 0;
}

double TestVariantFunctor(int test) {
  switch (test) {
    case 1: return AddTestFunctor<false>();
    case 2: return AddTestFunctor<true>();
    case 3: return ReduceTestFunctor<false>();
    case 4: return ReduceTestFunctor<true>();
    default: Kokkos::abort("unreachable");
  }

  return 0;
}

bool Test(int test) {
  double res_functor = TestVariantFunctor(test);
  double res_lambda  = TestVariantLambda(test);

  char testnames[5][256] = {" ", "AddTest", "AddTest TeamPolicy", "ReduceTest",
                            "ReduceTest TeamPolicy"};
  bool passed            = true;

  auto a = res_functor;
  auto b = res_lambda;
  // use a tolerant comparison because functors and lambdas vectorize
  // differently https://github.com/trilinos/Trilinos/issues/3233
  auto rel_err = (std::abs(b - a) / std::max(std::abs(a), std::abs(b)));
  auto tol     = 1e-14;
  if (rel_err > tol) {
    passed = false;

    std::cout << "CXX11 ( test = '" << testnames[test]
              << "' FAILED : relative error " << rel_err << " > tolerance "
              << tol << std::endl;
  }

  return passed;
}

}  // namespace TestCXX11

namespace Test {
TEST(TEST_CATEGORY, cxx11) {
  ASSERT_TRUE((TestCXX11::Test(1)));
  ASSERT_TRUE((TestCXX11::Test(2)));
  ASSERT_TRUE((TestCXX11::Test(3)));
  ASSERT_TRUE((TestCXX11::Test(4)));
}

}  // namespace Test
