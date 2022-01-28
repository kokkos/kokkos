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
struct TrivialBinaryPredicate {
  KOKKOS_INLINE_FUNCTION
  bool operator()(const ValueType val, const ValueType val2) const {
    (void)val;
    (void)val2;
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

namespace KE = Kokkos::Experimental;
using KE::begin;
using KE::end;
using T = double;
Kokkos::View<T *> in1("in1", 10);
Kokkos::View<T *> in2("in2", 10);
Kokkos::DefaultExecutionSpace execution_space;
std::string const label = "trivial";

#define KOKKOS_TEST_ALGORITHM_MACRO_ITERATOR_BE(ALGO)                   \
  (void)KE::ALGO(execution_space, /*--*/ KE::begin(in1), KE::end(in1)); \
  (void)KE::ALGO(label, execution_space, KE::begin(in1), KE::end(in1));

#define KOKKOS_TEST_ALGORITHM_MACRO_ITERATOR_BEB(ALGO)                 \
  (void)KE::ALGO(execution_space, /*--*/ KE::begin(in1), KE::end(in1), \
                 KE::begin(in2));                                      \
  (void)KE::ALGO(label, execution_space, KE::begin(in1), KE::end(in1), \
                 KE::begin(in2));

#define KOKKOS_TEST_ALGORITHM_MACRO_ITERATOR_BEBE(ALGO)                \
  (void)KE::ALGO(execution_space, /*--*/ KE::begin(in1), KE::end(in1), \
                 KE::begin(in2), KE::end(in2));                        \
  (void)KE::ALGO(label, execution_space, KE::begin(in1), KE::end(in1), \
                 KE::begin(in2), KE::end(in2));

#define KOKKOS_TEST_ALGORITHM_MACRO_ITERATOR_B_ANY(ALGO, ARG)  \
  (void)KE::ALGO(execution_space, /*--*/ KE::begin(in1), ARG); \
  (void)KE::ALGO(label, execution_space, KE::begin(in1), ARG);

#define KOKKOS_TEST_ALGORITHM_MACRO_ITERATOR_B_ANY_ANY(ALGO, ARG1, ARG2) \
  (void)KE::ALGO(execution_space, /*--*/ KE::begin(in1), ARG1, ARG2);    \
  (void)KE::ALGO(label, execution_space, KE::begin(in1), ARG1, ARG2);

#define KOKKOS_TEST_ALGORITHM_MACRO_ITERATOR_BE_ANY(ALGO, ARG1)               \
  (void)KE::ALGO(execution_space, /*--*/ KE::begin(in1), KE::end(in1), ARG1); \
  (void)KE::ALGO(label, execution_space, KE::begin(in1), KE::end(in1), ARG1);

#define KOKKOS_TEST_ALGORITHM_MACRO_ITERATOR_BE_ANY_ANY(ALGO, ARG1, ARG2)    \
  (void)KE::ALGO(execution_space, /*--*/ KE::begin(in1), KE::end(in1), ARG1, \
                 ARG2);                                                      \
  (void)KE::ALGO(label, execution_space, KE::begin(in1), KE::end(in1), ARG1, \
                 ARG2);

#define KOKKOS_TEST_ALGORITHM_MACRO_ITERATOR_BE_ANY_ANY_ANY(ALGO, ARG1, ARG2, \
                                                            ARG3)             \
  (void)KE::ALGO(execution_space, /*--*/ KE::begin(in1), KE::end(in1), ARG1,  \
                 ARG2, ARG3);                                                 \
  (void)KE::ALGO(label, execution_space, KE::begin(in1), KE::end(in1), ARG1,  \
                 ARG2, ARG3);

#define KOKKOS_TEST_ALGORITHM_MACRO_ITERATOR_BEB_ANY(ALGO, ARG)        \
  (void)KE::ALGO(execution_space, /*--*/ KE::begin(in1), KE::end(in1), \
                 KE::begin(in2), ARG);                                 \
  (void)KE::ALGO(label, execution_space, KE::begin(in1), KE::end(in1), \
                 KE::begin(in2), ARG);

#define KOKKOS_TEST_ALGORITHM_MACRO_ITERATOR_BEBE_ANY(ALGO, ARG)       \
  (void)KE::ALGO(execution_space, /*--*/ KE::begin(in1), KE::end(in1), \
                 KE::begin(in2), KE::end(in2), ARG);                   \
  (void)KE::ALGO(label, execution_space, KE::begin(in1), KE::end(in1), \
                 KE::begin(in2), KE::end(in2), ARG);

#define KOKKOS_TEST_ALGORITHM_MACRO_VIEW(ALGO) \
  (void)KE::ALGO(execution_space, /*--*/ in1); \
  (void)KE::ALGO(label, execution_space, in1);

#define KOKKOS_TEST_ALGORITHM_MACRO_VIEW_VIEW(ALGO) \
  (void)KE::ALGO(execution_space, /*--*/ in1, in2); \
  (void)KE::ALGO(label, execution_space, in1, in2);

#define KOKKOS_TEST_ALGORITHM_MACRO_VIEW_ANY(ALGO, ARG) \
  (void)KE::ALGO(execution_space, /*--*/ in1, ARG);     \
  (void)KE::ALGO(label, execution_space, in1, ARG);

#define KOKKOS_TEST_ALGORITHM_MACRO_VIEW_VIEW_ANY(ALGO, ARG) \
  (void)KE::ALGO(execution_space, /*--*/ in1, in2, ARG);     \
  (void)KE::ALGO(label, execution_space, in1, in2, ARG);

#define KOKKOS_TEST_ALGORITHM_MACRO_VIEW_ANY_ANY(ALGO, ARG1, ARG2) \
  (void)KE::ALGO(execution_space, /*--*/ in1, ARG1, ARG2);         \
  (void)KE::ALGO(label, execution_space, in1, ARG1, ARG2);

#define KOKKOS_TEST_ALGORITHM_MACRO_VIEW_ANY_ANY_ANY(ALGO, ARG1, ARG2, ARG3) \
  (void)KE::ALGO(execution_space, /*--*/ in1, ARG1, ARG2, ARG3);             \
  (void)KE::ALGO(label, execution_space, in1, ARG1, ARG2, ARG3);

void non_modifying_seq_ops() {
  KOKKOS_TEST_ALGORITHM_MACRO_ITERATOR_BE_ANY(find, T{5});
  KOKKOS_TEST_ALGORITHM_MACRO_VIEW_ANY(find, T{5});

  KOKKOS_TEST_ALGORITHM_MACRO_ITERATOR_BE_ANY(find_if,
                                              TrivialUnaryPredicate<T>());
  KOKKOS_TEST_ALGORITHM_MACRO_VIEW_ANY(find_if, TrivialUnaryPredicate<T>());

  KOKKOS_TEST_ALGORITHM_MACRO_ITERATOR_BE_ANY(find_if_not,
                                              TrivialUnaryPredicate<T>());
  KOKKOS_TEST_ALGORITHM_MACRO_VIEW_ANY(find_if_not, TrivialUnaryPredicate<T>());

  KOKKOS_TEST_ALGORITHM_MACRO_ITERATOR_BE_ANY(for_each, TimesTwoFunctor<T>());
  KOKKOS_TEST_ALGORITHM_MACRO_VIEW_ANY(for_each, TimesTwoFunctor<T>());

  KOKKOS_TEST_ALGORITHM_MACRO_ITERATOR_B_ANY_ANY(for_each_n, 3,
                                                 TimesTwoFunctor<T>());
  KOKKOS_TEST_ALGORITHM_MACRO_VIEW_ANY_ANY(for_each_n, 3, TimesTwoFunctor<T>());

  KOKKOS_TEST_ALGORITHM_MACRO_ITERATOR_BE_ANY(count_if,
                                              TrivialUnaryPredicate<T>());
  KOKKOS_TEST_ALGORITHM_MACRO_VIEW_ANY(count_if, TrivialUnaryPredicate<T>());

  KOKKOS_TEST_ALGORITHM_MACRO_ITERATOR_BE_ANY(count, T{22});
  KOKKOS_TEST_ALGORITHM_MACRO_VIEW_ANY(count, T{22});

  KOKKOS_TEST_ALGORITHM_MACRO_ITERATOR_BEBE(mismatch);
  KOKKOS_TEST_ALGORITHM_MACRO_ITERATOR_BEBE_ANY(mismatch,
                                                TrivialBinaryPredicate<T>());
  KOKKOS_TEST_ALGORITHM_MACRO_VIEW_VIEW(mismatch);
  KOKKOS_TEST_ALGORITHM_MACRO_VIEW_VIEW_ANY(mismatch,
                                            TrivialBinaryPredicate<T>());

  KOKKOS_TEST_ALGORITHM_MACRO_ITERATOR_BE_ANY(all_of,
                                              TrivialUnaryPredicate<T>());
  KOKKOS_TEST_ALGORITHM_MACRO_VIEW_ANY(all_of, TrivialUnaryPredicate<T>());

  KOKKOS_TEST_ALGORITHM_MACRO_ITERATOR_BE_ANY(any_of,
                                              TrivialUnaryPredicate<T>());
  KOKKOS_TEST_ALGORITHM_MACRO_VIEW_ANY(any_of, TrivialUnaryPredicate<T>());

  KOKKOS_TEST_ALGORITHM_MACRO_ITERATOR_BE_ANY(none_of,
                                              TrivialUnaryPredicate<T>());
  KOKKOS_TEST_ALGORITHM_MACRO_VIEW_ANY(none_of, TrivialUnaryPredicate<T>());

  KOKKOS_TEST_ALGORITHM_MACRO_ITERATOR_BEB(equal);
  KOKKOS_TEST_ALGORITHM_MACRO_ITERATOR_BEB_ANY(equal,
                                               TrivialBinaryPredicate<T>());
  KOKKOS_TEST_ALGORITHM_MACRO_VIEW_VIEW(equal);
  KOKKOS_TEST_ALGORITHM_MACRO_VIEW_VIEW_ANY(equal, TrivialBinaryPredicate<T>());
  KOKKOS_TEST_ALGORITHM_MACRO_ITERATOR_BEBE(equal);
  KOKKOS_TEST_ALGORITHM_MACRO_ITERATOR_BEBE_ANY(equal,
                                                TrivialBinaryPredicate<T>());

  KOKKOS_TEST_ALGORITHM_MACRO_ITERATOR_BEBE(lexicographical_compare);
  KOKKOS_TEST_ALGORITHM_MACRO_ITERATOR_BEBE_ANY(lexicographical_compare,
                                                TrivialComparator<T>());
  KOKKOS_TEST_ALGORITHM_MACRO_VIEW_VIEW(lexicographical_compare);
  KOKKOS_TEST_ALGORITHM_MACRO_VIEW_VIEW_ANY(lexicographical_compare,
                                            TrivialComparator<T>());

  KOKKOS_TEST_ALGORITHM_MACRO_ITERATOR_BE(adjacent_find);
  KOKKOS_TEST_ALGORITHM_MACRO_VIEW(adjacent_find);
  KOKKOS_TEST_ALGORITHM_MACRO_ITERATOR_BE_ANY(adjacent_find,
                                              TrivialBinaryFunctor<T>());
  KOKKOS_TEST_ALGORITHM_MACRO_VIEW_ANY(adjacent_find,
                                       TrivialBinaryFunctor<T>());

  KOKKOS_TEST_ALGORITHM_MACRO_ITERATOR_BEBE(search);
  KOKKOS_TEST_ALGORITHM_MACRO_VIEW_VIEW(search);
  KOKKOS_TEST_ALGORITHM_MACRO_ITERATOR_BEBE_ANY(search,
                                                TrivialBinaryFunctor<T>());
  KOKKOS_TEST_ALGORITHM_MACRO_VIEW_VIEW_ANY(search, TrivialBinaryFunctor<T>());

  KOKKOS_TEST_ALGORITHM_MACRO_ITERATOR_BEBE(find_first_of);
  KOKKOS_TEST_ALGORITHM_MACRO_VIEW_VIEW(find_first_of);
  KOKKOS_TEST_ALGORITHM_MACRO_ITERATOR_BEBE_ANY(find_first_of,
                                                TrivialBinaryFunctor<T>());
  KOKKOS_TEST_ALGORITHM_MACRO_VIEW_VIEW_ANY(find_first_of,
                                            TrivialBinaryFunctor<T>());

  KOKKOS_TEST_ALGORITHM_MACRO_ITERATOR_BE_ANY_ANY(search_n, 2, T{22});
  KOKKOS_TEST_ALGORITHM_MACRO_VIEW_ANY_ANY(search_n, 2, T{22});
  KOKKOS_TEST_ALGORITHM_MACRO_ITERATOR_BE_ANY_ANY_ANY(
      search_n, 2, T{22}, TrivialBinaryPredicate<T>());
  KOKKOS_TEST_ALGORITHM_MACRO_VIEW_ANY_ANY_ANY(search_n, 2, T{22},
                                               TrivialBinaryPredicate<T>());

  KOKKOS_TEST_ALGORITHM_MACRO_ITERATOR_BEBE(find_end);
  KOKKOS_TEST_ALGORITHM_MACRO_VIEW_VIEW(find_end);
  KOKKOS_TEST_ALGORITHM_MACRO_ITERATOR_BEBE_ANY(find_end,
                                                TrivialBinaryFunctor<T>());
  KOKKOS_TEST_ALGORITHM_MACRO_VIEW_VIEW_ANY(find_end,
                                            TrivialBinaryFunctor<T>());
}

void modifying_seq_ops() {
  //     auto r = KE::replace_copy("shouldwork", execution_space, in1,
  //     in2, 3.0, 11.0); auto r = KE::replace_copy_if("shouldwork",
  //     execution_space, in1, in2,
  //                                  TrivialUnaryPredicate<T>(), 11.0);
  //     KE::replace("shouldwork", execution_space, in1, 3.0, 11.0);
  //     KE::replace_if("shouldwork", execution_space, in1,
  //                    TrivialUnaryPredicate<T>(), 11.0);
  //     auto r = KE::copy("shouldwork", execution_space, in1, in2);
  //     auto r = KE::copy_n("shouldwork", execution_space, in1, 3, in2);
  //     auto r = KE::copy_backward("shouldwork", execution_space, in1, in2);
  //     auto r = KE::copy_if("shouldwork", execution_space, in1, in2,
  //                          TrivialUnaryPredicate<T>());
  //     KE::fill("shouldwork", execution_space, in1, 22.0);
  //     auto r = KE::fill_n("shouldwork", execution_space, in1, 5, 22.0);
  //     KE::transform("shouldwork", execution_space, in1, in2, out, func);
  //     KE::generate("shouldwork", execution_space, in1,
  //     TrivialGenerator<T>()); KE::generate_n("shouldwork", execution_space,
  //     in1, 5,
  //                    TrivialGenerator<T>());
}

//     auto r = KE::is_sorted_until("shouldwork", execution_space, in1);

// #ifndef KOKKOS_ENABLE_OPENMPTARGET
//     auto r = KE::is_sorted_until("shouldwork", execution_space, in1,
//                                  TrivialComparator<T>());
// #endif

//   {
//     auto r = KE::is_sorted("shouldwork", execution_space, in1);
//   }

// #ifndef KOKKOS_ENABLE_OPENMPTARGET
//   {
//     auto r = KE::is_sorted("shouldwork", execution_space, in1,
//                            TrivialComparator<T>());
//   }
// #endif

//   {
//     auto r = KE::min_element("shouldwork", execution_space, in1);
//   }

// #ifndef KOKKOS_ENABLE_OPENMPTARGET
//   {
//     auto r = KE::min_element("shouldwork", execution_space, in1,
//                              TrivialComparator<T>());
//   }
// #endif

//   {
//     auto r = KE::max_element("shouldwork", execution_space, in1);
//   }

// #ifndef KOKKOS_ENABLE_OPENMPTARGET
//     auto r = KE::max_element("shouldwork", execution_space, in1,
//                              TrivialComparator<T>());
// #endif

//   {
//     auto r = KE::minmax_element("shouldwork", execution_space, in1);
//   }

// #ifndef KOKKOS_ENABLE_OPENMPTARGET
//   {
//     auto r = KE::minmax_element("shouldwork", execution_space, in1,
//                                 TrivialComparator<T>());

// #endif

//   {
//     auto r = KE::is_partitioned("shouldwork", execution_space, in1,
//                                 TrivialUnaryPredicate<T>());
//   }

//   {
//     auto r = KE::partition_copy("shouldwork", execution_space, in1, in2, in3,
//                                 TrivialUnaryPredicate<T>());
//   }

//   {
//     auto r = KE::partition_point("shouldwork", execution_space, in1,
//                                  TrivialUnaryPredicate<T>());
//}

//   {
//     auto r = KE::adjacent_difference("shouldwork", execution_space, in1,
//     in2);
//   }

//   {
//     auto r = KE::adjacent_difference("shouldwork", execution_space, in1, in2,
//                                      TrivialBinaryFunctor<T>());
//   }

//   {
//     auto r = KE::exclusive_scan("shouldwork", execution_space, in1,
//     in2, 3.3);
//   }

// #ifndef KOKKOS_ENABLE_OPENMPTARGET
//   {
//     auto r = KE::transform_exclusive_scan("shouldwork", execution_space, in1,
//                                           in2, 3.3,
//                                           TrivialBinaryFunctor<T>(),
//                                           TrivialUnaryFunctor<T>());
//   }
// #endif

//   {
//     auto r = KE::inclusive_scan("shouldwork", execution_space, in1, in2);
//   }

// #ifndef KOKKOS_ENABLE_OPENMPTARGET
//   {
//     auto r = KE::transform_inclusive_scan("shouldwork", execution_space, in1,
//                                           in2, TrivialBinaryFunctor<T>(),
//                                           TrivialUnaryFunctor<T>());
//   }
// #endif

// #ifndef KOKKOS_ENABLE_OPENMPTARGET
//   {
//     auto r = KE::reduce("shouldwork", execution_space, in1);
//   }

//   {
//     auto r = KE::reduce("shouldwork", execution_space, in1, 33.0,
//                         TrivialReduceJoinFunctor<T>());
//   }
// #endif

// #ifndef KOKKOS_ENABLE_OPENMPTARGET
//   {
//     auto r = KE::transform_reduce("shouldwork", execution_space, in1,
//     in2, 33.0);
//   }

//   {
//     auto r = KE::transform_reduce("shouldwork", execution_space, in1,
//     in2, 33.0,
//                                   TrivialReduceJoinFunctor<T>(),
//                                   TrivialTransformReduceBinaryTransformer<T>());
//   }

//   {
//     auto r = KE::transform_reduce("shouldwork", execution_space, in1, 33.0,
//                                   TrivialReduceJoinFunctor<T>(),
//                                   TrivialTransformReduceUnnaryTransformer<T>());
//   }
// #endif

}  // namespace compileonly
}  // namespace stdalgos
}  // namespace Test

int main(int argc, char *argv[]) { return 0; }
