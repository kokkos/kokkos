
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

#include <default/TestDefaultDeviceType_Category.hpp>
//#include <TestReduceCombinatorical.hpp>

namespace Test {

namespace ReduceCombinatorical {

struct FunctorScalarFoo {
  Kokkos::View<double*> result;

  FunctorScalarFoo(Kokkos::View<double*> r) : result(r) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& i, double& update) const { update += i; }

  KOKKOS_INLINE_FUNCTION
  void init(double& update) const { update = 0.0; }
};

 template <class PolicyType, class FunctorType, class ReturnType>
 struct MyParallelReduceAdaptor {
   typedef Kokkos::Impl::ParallelReduceReturnValue<void, ReturnType, FunctorType>
       return_value_adapter;
   typedef Kokkos::Impl::ParallelReduceFunctorType<
       FunctorType, PolicyType, typename return_value_adapter::value_type,
       typename PolicyType::execution_space>
       functor_adaptor;
   /*static inline*/ void execute(const std::string& label,
                                  const PolicyType& policy,
                                  const FunctorType& functor,
                                  ReturnType& return_value) {

     using DriverTypeOrg = Kokkos::Impl::ParallelReduce<
         typename functor_adaptor::functor_type, PolicyType,
         typename return_value_adapter::reducer_type>;

     /*using DriverTypeNew = Kokkos::Impl::ParallelReduce<
         Kokkos::Impl::CudaFunctorAdapter<
             Test::ReduceCombinatorical::FunctorScalarFoo,
             Kokkos::RangePolicy<Kokkos::Cuda>, double, void>,
         Kokkos::RangePolicy<Kokkos::Cuda>, Kokkos::InvalidType, Kokkos::Cuda>;*/
     using ft1 = Kokkos::Impl::CudaFunctorAdapter <
                FunctorType, PolicyType, double, void>; // works
     using ft2 = typename Kokkos::Impl::ParallelReduceFunctorType<
         FunctorType, PolicyType, typename return_value_adapter::value_type,
         typename PolicyType::execution_space>::functor_type; // fails
     using ft = typename functor_adaptor::functor_type; //fails
     using DriverTypeNew = Kokkos::Impl::ParallelReduce<ft2,
         PolicyType, typename return_value_adapter::reducer_type>;
     printf("Org: [ %s ]\nNew: [ %s ]\n IsSame: %s\n", typeid(DriverTypeOrg).name(),
            typeid(DriverTypeNew).name(),std::is_same<DriverTypeOrg,DriverTypeNew>::value?"true":"false");
     #if false
     DriverTypeOrg 
         #else
     DriverTypeNew
         #endif
         closure(functor, policy,
                        Kokkos::View<double, Kokkos::HostSpace>(&return_value));
     printf("Lets Start\n");

     closure.execute();
     printf("Lets Done\n");
   }
 };

 template<class FunctorType, class ReturnType>
 void myparallel_reduce(const size_t& policy, const FunctorType& functor,
     ReturnType& return_value) {
   typedef typename Kokkos::Impl::ParallelReducePolicyType<
       void, size_t, FunctorType>::policy_type policy_type;
   MyParallelReduceAdaptor<policy_type, FunctorType, ReturnType>().execute(
       "", policy_type(0, policy), functor, return_value);
   Kokkos::Impl::ParallelReduceFence<
       typename policy_type::execution_space,
       ReturnType>::fence(typename policy_type::execution_space(),
                          return_value);
 }
 void foo() { 
   Kokkos::View<double*> a("A", 5);

   double result;
   FunctorScalarFoo f(a);
   myparallel_reduce(1000, f, result);  
   printf("Result: %lf\n", result);
 }

}


TEST(defaultdevicetype, reduce_instantiation_a1) {
  //TestReduceCombinatoricalInstantiation<>::execute_a1();
  ReduceCombinatorical::foo();
}

}  // namespace Test
/*
template <unsigned RD>
struct MViewDimension;

template <>
struct MViewDimension<1> {
  enum { ArgN1 = 11 };
  enum { N1 = 11 };
};
template <>
struct MViewDimension<2> {
  enum { ArgN2 = 12 };
  enum { N2 = 12 };
};

template <>
struct MViewDimension<0> {
  enum { ArgN0 = 0 };
  unsigned N0;
  MViewDimension()                      = default;
  MViewDimension(const MViewDimension&) = default;
  MViewDimension& operator=(const MViewDimension&) = default;
  KOKKOS_INLINE_FUNCTION explicit MViewDimension(size_t V) : N0(V) {}
};

struct Foo : public MViewDimension<0>,
             public MViewDimension<1>,
             public MViewDimension<2> {
  using MViewDimension<0>::N0;
  using MViewDimension<1>::N1;
  using MViewDimension<2>::N2;

  using MViewDimension<0>::ArgN0;
  using MViewDimension<1>::ArgN1;
  using MViewDimension<2>::ArgN2;
  alignas(8) unsigned dummy;
};

using thingy_t = Kokkos::Impl::ViewDimension<0u>;

//#include <TestViewAPI_a.hpp>
void pfence(const char* str) {
  Kokkos::fence();
  printf("%s\n", str);
}

__global__ void fill(int* ptr, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) ptr[i] = 7;
}

template <class Functor>
__global__ void fillfct(const Functor f, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) f(i);
}

template <class Functor>
__global__ void run_driver(const Functor f, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 1)
    printf("HeyA: %i (%i %i %i)\n", i, int(sizeof(f)), int(sizeof(thingy_t)),
           int(sizeof(Kokkos::Impl::ViewDimension1<
                      size_t(Kokkos::Impl::variadic_size_t<1u, 0u>::value),
                      unsigned(Kokkos::Impl::rank_dynamic<0u>::value)>)));
  if (i < N) f();
}

struct FunctorT {
  Kokkos::View<int*> a;
  FunctorT(Kokkos::View<int*> a_) : a(a_) {
    printf("Constructed: %p\n", a.data());
  }
  KOKKOS_FUNCTION
  void operator()(const int i) const { a(i) = 9; }
};

template <class T>
struct DebugPrintState {
  static void print(const char* str, const T&) {
    printf("Kokkos::DebugPrintState %s [\"%s\"]\n", typeid(T).name(), str);
  }
};
template <>
struct DebugPrintState<FunctorT> {
  static void print(const char* str, const FunctorT& f) {
    printf("Kokkos::DebugPrintState %s [\"%s\"] %p\n", typeid(FunctorT).name(),
           str, f.a.data());
  }
};

template <>
struct DebugPrintState<Kokkos::Impl::ParallelFor<
    FunctorT, Kokkos::RangePolicy<Kokkos::Cuda>, Kokkos::Cuda>> {
  using T =
      Kokkos::Impl::ParallelFor<FunctorT, Kokkos::RangePolicy<Kokkos::Cuda>,
                                Kokkos::Cuda>;
  static void print(const char* str, const T& f) {
    printf(
        "Kokkos::DebugPrintState %s [\"%s\"]\n", typeid(T).name(),
        str);  //,
               //f.m_functor.a.data(),int(f.m_policy.begin()),int(f.m_policy.end()));
  }
};

void foo() {
  int N = 100;
  Kokkos::View<int*> a("A", N);
  pfence("ViewCreate");
  auto h_a1 = Kokkos::create_mirror_view(a);
  auto h_a2 = Kokkos::create_mirror_view(a);
  pfence("MirrorCreate");
  printf("%p %p %p\n", a.data(), h_a1.data(), h_a2.data());
  Kokkos::deep_copy(h_a2, 5);
  Kokkos::deep_copy(h_a1, 8);
  pfence("DeepCopy0");
  // Kokkos::deep_copy(a, h_a2);
  //  pfence("DeepCopy1");
  fillfct<<<(N + 31) / 32, 32>>>(
      KOKKOS_LAMBDA(const int i) { a(i) = 3; }, N);
  pfence("Fill0");
  Kokkos::deep_copy(h_a1, a);
  pfence("DeepCopy3");
  printf("%i %i\n", h_a1(0), h_a1(N - 1));
  // Kokkos::deep_copy(a, 7);
  auto lambda = KOKKOS_LAMBDA(const int i) { a(i) = 7; };
  FunctorT lambdaf(a);
  using policy_t = Kokkos::RangePolicy<Kokkos::Cuda>;
  policy_t policy(0, N);

  using FunctorType = decltype(lambda);
  Kokkos::Impl::ParallelFor<FunctorT, policy_t> closure(lambdaf, policy);
  // closure.execute();
  printf("PoilicyA: (%i %i %i)\n", int(sizeof(closure)), int(sizeof(thingy_t)),
         int(sizeof(Kokkos::Impl::ViewDimension1<
                    size_t(Kokkos::Impl::variadic_size_t<1u, 0u>::value),
                    unsigned(Kokkos::Impl::rank_dynamic<0u>::value)>)));
  run_driver<<<(N + 31) / 32, 32>>>(closure, N);
  // Kokkos::parallel_for(N, lambda);
  pfence("Fill1");
  Kokkos::deep_copy(h_a1, a);
  pfence("DeepCopy3");
  printf("%i %i\n", h_a1(0), h_a1(N - 1));

  fill<<<(N + 31) / 32, 32>>>(a.data(), N);
  pfence("Fill2");
  Kokkos::deep_copy(h_a1, a);
  pfence("DeepCopy3");
  printf("%i %i\n", h_a1(0), h_a1(N - 1));

  fillfct<<<(N + 31) / 32, 32>>>(
      KOKKOS_LAMBDA(const int i) { a(i) = 11; }, N);
  pfence("Fill3");
  Kokkos::deep_copy(h_a1, a);
  pfence("DeepCopy3");
  printf("%i %i\n", h_a1(0), h_a1(N - 1));
}

TEST(TEST_CATEGORY, mytest) { foo(); }*/