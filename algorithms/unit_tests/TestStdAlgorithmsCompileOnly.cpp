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

#include <std_algorithms/Kokkos_BeginEnd.hpp>
#include <Kokkos_StdAlgorithms.hpp>

namespace Test {
namespace stdalgos {
namespace compileonly {

template <class ValueType>
struct TrivialUnaryFunctor {
  KOKKOS_INLINE_FUNCTION
  ValueType operator()(const ValueType a) const { return a; }
};

template <class ValueType>
struct TrivialBinaryFunctor {
  KOKKOS_INLINE_FUNCTION
  ValueType operator()(const ValueType &a, const ValueType &b) const {
    return (a + b);
  }

  KOKKOS_INLINE_FUNCTION
  ValueType operator()(const volatile ValueType &a,
                       const volatile ValueType &b) const {
    return (a + b);
  }
};

template <class ValueType>
struct TrivialUnaryPredicate {
  KOKKOS_INLINE_FUNCTION
  bool operator()(const ValueType val) const {
    (void)val;
    return true;
  }
};

template <class ValueType>
struct TimesTwoFunctor {
  KOKKOS_INLINE_FUNCTION
  void operator()(ValueType &val) const { val *= (ValueType)2; }
};

template <class ValueType>
struct TrivialComparator {
  KOKKOS_INLINE_FUNCTION
  bool operator()(const ValueType &a, const ValueType &b) const {
    return a > b;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator()(const volatile ValueType &a,
                  const volatile ValueType &b) const {
    return a > b;
  }
};

template <class ValueType>
struct TrivialGenerator {
  KOKKOS_INLINE_FUNCTION
  ValueType operator()() const { return ValueType{}; }
};

template <class ValueType>
struct TrivialReduceJoinFunctor {
  KOKKOS_FUNCTION
  ValueType operator()(const ValueType &a, const ValueType &b) const {
    return a + b;
  }

  KOKKOS_FUNCTION
  ValueType operator()(const volatile ValueType &a,
                       const volatile ValueType &b) const {
    return a + b;
  }
};

template <class ValueType>
struct TrivialTransformReduceUnnaryTransformer {
  KOKKOS_FUNCTION
  ValueType operator()(const ValueType &a) const { return a; }
};

template <class ValueType>
struct TrivialTransformReduceBinaryTransformer {
  KOKKOS_FUNCTION
  ValueType operator()(const ValueType &a, const ValueType &b) const {
    return (a * b);
  }
};

// put all code here and don't call from main
// so that even if one runs the executable,
// nothing is run anyway
void all() {
  namespace KE = Kokkos::Experimental;
  Kokkos::DefaultExecutionSpace execution_space;

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    auto r = KE::find("shouldwork", execution_space, in1, 5.);
    (void)r;
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    auto r = KE::find_if("shouldwork", execution_space, in1,
                         TrivialUnaryPredicate<T>());
    (void)r;
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    auto r = KE::find_if_not("shouldwork", execution_space, in1,
                             TrivialUnaryPredicate<T>());
    (void)r;
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    auto r =
        KE::for_each("shouldwork", execution_space, in1, TimesTwoFunctor<T>());
    (void)r;
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    auto r = KE::for_each_n("shouldwork", execution_space, in1, 5,
                            TimesTwoFunctor<T>());
    (void)r;
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    auto r = KE::count_if("shouldwork", execution_space, in1,
                          TrivialUnaryPredicate<T>());
    (void)r;
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    auto r = KE::count("shouldwork", execution_space, in1, 3.);
    (void)r;
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    Kokkos::View<T *> in2("in2", 10);
    auto r = KE::mismatch("shouldwork", execution_space, in1, in2);
    (void)r;
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    auto r = KE::all_of("shouldwork", execution_space, in1,
                        TrivialUnaryPredicate<T>());
    (void)r;
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    auto r = KE::any_of("shouldwork", execution_space, in1,
                        TrivialUnaryPredicate<T>());
    (void)r;
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    auto r = KE::none_of("shouldwork", execution_space, in1,
                         TrivialUnaryPredicate<T>());
    (void)r;
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    Kokkos::View<T *> in2("in2", 10);
    auto r = KE::equal("shouldwork", execution_space, in1, in2);
    (void)r;
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    Kokkos::View<T *> in2("in2", 10);
    auto r =
        KE::lexicographical_compare("shouldwork", execution_space, in1, in2);
    (void)r;
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    Kokkos::View<T *> in2("in2", 10);
    auto r = KE::lexicographical_compare("shouldwork", execution_space, in1,
                                         in2, TrivialComparator<T>());
    (void)r;
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    auto r = KE::adjacent_find("shouldwork", execution_space, in1,
                               TrivialBinaryFunctor<T>());
    (void)r;
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    Kokkos::View<T *> in2("in2", 10);
    auto r = KE::search("shouldwork", execution_space, in1, in2);
    (void)r;
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    Kokkos::View<T *> in2("in2", 10);
    auto r = KE::search("shouldwork", execution_space, in1, in2,
                        TrivialBinaryFunctor<T>());
    (void)r;
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    Kokkos::View<T *> in2("in2", 10);
    auto r = KE::find_first_of("shouldwork", execution_space, in1, in2);
    (void)r;
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    Kokkos::View<T *> in2("in2", 10);
    auto r = KE::find_first_of("shouldwork", execution_space, in1, in2,
                               TrivialBinaryFunctor<T>());
    (void)r;
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    auto r = KE::search_n("shouldwork", execution_space, in1, 5, 3.0);
    (void)r;
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    auto r = KE::search_n("shouldwork", execution_space, in1, 5, 3.0,
                          TrivialBinaryFunctor<T>());
    (void)r;
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    Kokkos::View<T *> in2("in2", 10);
    auto r = KE::find_end("shouldwork", execution_space, in1, in2);
    (void)r;
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    Kokkos::View<T *> in2("in2", 10);
    auto r = KE::find_end("shouldwork", execution_space, in1, in2,
                          TrivialBinaryFunctor<T>());
    (void)r;
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    Kokkos::View<T *> in2("in2", 10);
    auto r =
        KE::replace_copy("shouldwork", execution_space, in1, in2, 3.0, 11.0);
    (void)r;
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    Kokkos::View<T *> in2("in2", 10);
    auto r = KE::replace_copy_if("shouldwork", execution_space, in1, in2,
                                 TrivialUnaryPredicate<T>(), 11.0);
    (void)r;
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    KE::replace("shouldwork", execution_space, in1, 3.0, 11.0);
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    KE::replace_if("shouldwork", execution_space, in1,
                   TrivialUnaryPredicate<T>(), 11.0);
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    Kokkos::View<T *> in2("in2", 10);
    auto r = KE::copy("shouldwork", execution_space, in1, in2);
    (void)r;
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    Kokkos::View<T *> in2("in2", 10);
    auto r = KE::copy_n("shouldwork", execution_space, in1, 3, in2);
    (void)r;
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    Kokkos::View<T *> in2("in2", 10);
    auto r = KE::copy_backward("shouldwork", execution_space, in1, in2);
    (void)r;
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    Kokkos::View<T *> in2("in2", 10);
    auto r = KE::copy_if("shouldwork", execution_space, in1, in2,
                         TrivialUnaryPredicate<T>());
    (void)r;
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    KE::fill("shouldwork", execution_space, in1, 22.0);
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    auto r = KE::fill_n("shouldwork", execution_space, in1, 5, 22.0);
    (void)r;
  }

  {
    // this code is from https://github.com/kokkos/kokkos/issues/4711
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    Kokkos::View<T *> in2("in2", 10);
    Kokkos::View<T *> out("out", 10);
    TrivialBinaryFunctor<T> func;
    KE::transform("shouldwork", execution_space, in1, in2, out, func);
    // KOKKOS_LAMBDA(T x1, T x2) {
    //   return x1 + x2;
    // });
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    KE::generate("shouldwork", execution_space, in1, TrivialGenerator<T>());
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    KE::generate_n("shouldwork", execution_space, in1, 5,
                   TrivialGenerator<T>());
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    auto r = KE::is_sorted_until("shouldwork", execution_space, in1);
    (void)r;
  }

#ifndef KOKKOS_ENABLE_OPENMPTARGET
  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    auto r = KE::is_sorted_until("shouldwork", execution_space, in1,
                                 TrivialComparator<T>());
    (void)r;
  }
#endif

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    auto r = KE::is_sorted("shouldwork", execution_space, in1);
    (void)r;
  }

#ifndef KOKKOS_ENABLE_OPENMPTARGET
  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    auto r = KE::is_sorted("shouldwork", execution_space, in1,
                           TrivialComparator<T>());
    (void)r;
  }
#endif

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    auto r = KE::min_element("shouldwork", execution_space, in1);
    (void)r;
  }

#ifndef KOKKOS_ENABLE_OPENMPTARGET
  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    auto r = KE::min_element("shouldwork", execution_space, in1,
                             TrivialComparator<T>());
    (void)r;
  }
#endif

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    auto r = KE::max_element("shouldwork", execution_space, in1);
    (void)r;
  }

#ifndef KOKKOS_ENABLE_OPENMPTARGET
  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    auto r = KE::max_element("shouldwork", execution_space, in1,
                             TrivialComparator<T>());
    (void)r;
  }
#endif

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    auto r = KE::minmax_element("shouldwork", execution_space, in1);
    (void)r;
  }

#ifndef KOKKOS_ENABLE_OPENMPTARGET
  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    auto r = KE::minmax_element("shouldwork", execution_space, in1,
                                TrivialComparator<T>());
    (void)r;
  }
#endif

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    auto r = KE::is_partitioned("shouldwork", execution_space, in1,
                                TrivialUnaryPredicate<T>());
    (void)r;
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    Kokkos::View<T *> in2("in2", 10);
    Kokkos::View<T *> in3("in3", 10);
    auto r = KE::partition_copy("shouldwork", execution_space, in1, in2, in3,
                                TrivialUnaryPredicate<T>());
    (void)r;
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    auto r = KE::partition_point("shouldwork", execution_space, in1,
                                 TrivialUnaryPredicate<T>());
    (void)r;
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    Kokkos::View<T *> in2("in2", 10);
    auto r = KE::adjacent_difference("shouldwork", execution_space, in1, in2);
    (void)r;
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    Kokkos::View<T *> in2("in2", 10);
    auto r = KE::adjacent_difference("shouldwork", execution_space, in1, in2,
                                     TrivialBinaryFunctor<T>());
    (void)r;
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    Kokkos::View<T *> in2("in2", 10);
    auto r = KE::exclusive_scan("shouldwork", execution_space, in1, in2, 3.3);
    (void)r;
  }

#ifndef KOKKOS_ENABLE_OPENMPTARGET
  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    Kokkos::View<T *> in2("in2", 10);
    auto r = KE::transform_exclusive_scan("shouldwork", execution_space, in1,
                                          in2, 3.3, TrivialBinaryFunctor<T>(),
                                          TrivialUnaryFunctor<T>());
    (void)r;
  }
#endif

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    Kokkos::View<T *> in2("in2", 10);
    auto r = KE::inclusive_scan("shouldwork", execution_space, in1, in2);
    (void)r;
  }

#ifndef KOKKOS_ENABLE_OPENMPTARGET
  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    Kokkos::View<T *> in2("in2", 10);
    auto r = KE::transform_inclusive_scan("shouldwork", execution_space, in1,
                                          in2, TrivialBinaryFunctor<T>(),
                                          TrivialUnaryFunctor<T>());
    (void)r;
  }
#endif

#ifndef KOKKOS_ENABLE_OPENMPTARGET
  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    auto r = KE::reduce("shouldwork", execution_space, in1);
    (void)r;
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    auto r = KE::reduce("shouldwork", execution_space, in1, 33.0,
                        TrivialReduceJoinFunctor<T>());
    (void)r;
  }
#endif

#ifndef KOKKOS_ENABLE_OPENMPTARGET
  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    Kokkos::View<T *> in2("in2", 10);
    auto r =
        KE::transform_reduce("shouldwork", execution_space, in1, in2, 33.0);
    (void)r;
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    Kokkos::View<T *> in2("in2", 10);
    auto r = KE::transform_reduce("shouldwork", execution_space, in1, in2, 33.0,
                                  TrivialReduceJoinFunctor<T>(),
                                  TrivialTransformReduceBinaryTransformer<T>());
    (void)r;
  }

  {
    using T = double;
    Kokkos::View<T *> in1("in1", 10);
    auto r = KE::transform_reduce("shouldwork", execution_space, in1, 33.0,
                                  TrivialReduceJoinFunctor<T>(),
                                  TrivialTransformReduceUnnaryTransformer<T>());
    (void)r;
  }
#endif
}

}  // namespace compileonly
}  // namespace stdalgos
}  // namespace Test

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  Kokkos::finalize();
  return 0;
}
