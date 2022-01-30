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
struct TrivialTransformReduceUnaryTransformer {
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
using count_type = std::size_t;
using T          = double;
Kokkos::View<T *> in1("in1", 10);
Kokkos::View<T *> in2("in2", 10);
Kokkos::View<T *> in3("in3", 10);
Kokkos::DefaultExecutionSpace exe_space;
std::string const label = "trivial";

//
// just iterators
//
#define TEST_ALGO_MACRO_B1E1(ALGO)                                \
  (void)KE::ALGO(exe_space, /*--*/ KE::begin(in1), KE::end(in1)); \
  (void)KE::ALGO(label, exe_space, KE::begin(in1), KE::end(in1));

#define TEST_ALGO_MACRO_B1E1B2(ALGO)                             \
  (void)KE::ALGO(exe_space, /*--*/ KE::begin(in1), KE::end(in1), \
                 KE::begin(in2));                                \
  (void)KE::ALGO(label, exe_space, KE::begin(in1), KE::end(in1), \
                 KE::begin(in2));

#define TEST_ALGO_MACRO_B1E1B2E2(ALGO)                           \
  (void)KE::ALGO(exe_space, /*--*/ KE::begin(in1), KE::end(in1), \
                 KE::begin(in2), KE::end(in2));                  \
  (void)KE::ALGO(label, exe_space, KE::begin(in1), KE::end(in1), \
                 KE::begin(in2), KE::end(in2));

#define TEST_ALGO_MACRO_B1E1E2(ALGO)                             \
  (void)KE::ALGO(exe_space, /*--*/ KE::begin(in1), KE::end(in1), \
                 KE::end(in2));                                  \
  (void)KE::ALGO(label, exe_space, KE::begin(in1), KE::end(in1), KE::end(in2));

#define TEST_ALGO_MACRO_B1E1E2B3(ALGO)                                         \
  (void)KE::ALGO(exe_space, /*--*/ KE::begin(in1), KE::end(in1), KE::end(in2), \
                 KE::begin(in3));                                              \
  (void)KE::ALGO(label, exe_space, KE::begin(in1), KE::end(in1), KE::end(in2), \
                 KE::begin(in3));

#define TEST_ALGO_MACRO_B1E1E1B2(ALGO)                                         \
  (void)KE::ALGO(exe_space, /*--*/ KE::begin(in1), KE::end(in1), KE::end(in1), \
                 KE::begin(in2));                                              \
  (void)KE::ALGO(label, exe_space, KE::begin(in1), KE::end(in1), KE::end(in1), \
                 KE::begin(in2));

//
// iterators and params
//
#define TEST_ALGO_MACRO_B1_ANY(ALGO, ...)                        \
  (void)KE::ALGO(exe_space, /*--*/ KE::begin(in1), __VA_ARGS__); \
  (void)KE::ALGO(label, exe_space, KE::begin(in1), __VA_ARGS__);

#define TEST_ALGO_MACRO_B1E1_ANY(ALGO, ...)                                    \
  (void)KE::ALGO(exe_space, /*--*/ KE::begin(in1), KE::end(in1), __VA_ARGS__); \
  (void)KE::ALGO(label, exe_space, KE::begin(in1), KE::end(in1), __VA_ARGS__);

#define TEST_ALGO_MACRO_B1E1B2_ANY(ALGO, ...)                    \
  (void)KE::ALGO(exe_space, /*--*/ KE::begin(in1), KE::end(in1), \
                 KE::begin(in2), __VA_ARGS__);                   \
  (void)KE::ALGO(label, exe_space, KE::begin(in1), KE::end(in1), \
                 KE::begin(in2), __VA_ARGS__);

#define TEST_ALGO_MACRO_B1_ARG_B2(ALGO, ARG)                             \
  (void)KE::ALGO(exe_space, /*--*/ KE::begin(in1), ARG, KE::begin(in2)); \
  (void)KE::ALGO(label, exe_space, KE::begin(in1), ARG, KE::begin(in2));

#define TEST_ALGO_MACRO_B1E1B2B3_ANY(ALGO, ...)                  \
  (void)KE::ALGO(exe_space, /*--*/ KE::begin(in1), KE::end(in1), \
                 KE::begin(in2), KE::begin(in3), __VA_ARGS__);   \
  (void)KE::ALGO(label, exe_space, KE::begin(in1), KE::end(in1), \
                 KE::begin(in2), KE::begin(in3), __VA_ARGS__);

#define TEST_ALGO_MACRO_B1E1B2E2_ANY(ALGO, ARG)                  \
  (void)KE::ALGO(exe_space, /*--*/ KE::begin(in1), KE::end(in1), \
                 KE::begin(in2), KE::end(in2), ARG);             \
  (void)KE::ALGO(label, exe_space, KE::begin(in1), KE::end(in1), \
                 KE::begin(in2), KE::end(in2), ARG);

//
// views only
//
#define TEST_ALGO_MACRO_V1(ALGO)         \
  (void)KE::ALGO(exe_space, /*--*/ in1); \
  (void)KE::ALGO(label, exe_space, in1);

#define TEST_ALGO_MACRO_V1V2(ALGO)            \
  (void)KE::ALGO(exe_space, /*--*/ in1, in2); \
  (void)KE::ALGO(label, exe_space, in1, in2);

#define TEST_ALGO_MACRO_V1V2V3(ALGO)               \
  (void)KE::ALGO(exe_space, /*--*/ in1, in2, in3); \
  (void)KE::ALGO(label, exe_space, in1, in2, in3);

//
// views and params
//
#define TEST_ALGO_MACRO_V1_ANY(ALGO, ...)             \
  (void)KE::ALGO(exe_space, /*--*/ in1, __VA_ARGS__); \
  (void)KE::ALGO(label, exe_space, in1, __VA_ARGS__);

#define TEST_ALGO_MACRO_V1V2_ANY(ALGO, ...)                \
  (void)KE::ALGO(exe_space, /*--*/ in1, in2, __VA_ARGS__); \
  (void)KE::ALGO(label, exe_space, in1, in2, __VA_ARGS__);

#define TEST_ALGO_MACRO_V1V2V3_ANY(ALGO, ...)                   \
  (void)KE::ALGO(exe_space, /*--*/ in1, in2, in3, __VA_ARGS__); \
  (void)KE::ALGO(label, exe_space, in1, in2, in3, __VA_ARGS__);

#define TEST_ALGO_MACRO_V1_ARG_V2(ALGO, ARG)       \
  (void)KE::ALGO(exe_space, /*--*/ in1, ARG, in2); \
  (void)KE::ALGO(label, exe_space, in1, ARG, in2);

void non_modifying_seq_ops() {
  TEST_ALGO_MACRO_B1E1_ANY(find, T{});
  TEST_ALGO_MACRO_V1_ANY(find, T{});

  TEST_ALGO_MACRO_B1E1_ANY(find_if, TrivialUnaryPredicate<T>());
  TEST_ALGO_MACRO_V1_ANY(find_if, TrivialUnaryPredicate<T>());

  TEST_ALGO_MACRO_B1E1_ANY(find_if_not, TrivialUnaryPredicate<T>());
  TEST_ALGO_MACRO_V1_ANY(find_if_not, TrivialUnaryPredicate<T>());

  TEST_ALGO_MACRO_B1E1_ANY(for_each, TimesTwoFunctor<T>());
  TEST_ALGO_MACRO_V1_ANY(for_each, TimesTwoFunctor<T>());

  TEST_ALGO_MACRO_B1_ANY(for_each_n, count_type{}, TimesTwoFunctor<T>());
  TEST_ALGO_MACRO_V1_ANY(for_each_n, count_type{}, TimesTwoFunctor<T>());

  TEST_ALGO_MACRO_B1E1_ANY(count_if, TrivialUnaryPredicate<T>());
  TEST_ALGO_MACRO_V1_ANY(count_if, TrivialUnaryPredicate<T>());

  TEST_ALGO_MACRO_B1E1_ANY(count, T{});
  TEST_ALGO_MACRO_V1_ANY(count, T{});

  TEST_ALGO_MACRO_B1E1B2E2(mismatch);
  TEST_ALGO_MACRO_B1E1B2E2_ANY(mismatch, TrivialBinaryPredicate<T>());
  TEST_ALGO_MACRO_V1V2(mismatch);
  TEST_ALGO_MACRO_V1V2_ANY(mismatch, TrivialBinaryPredicate<T>());

  TEST_ALGO_MACRO_B1E1_ANY(all_of, TrivialUnaryPredicate<T>());
  TEST_ALGO_MACRO_V1_ANY(all_of, TrivialUnaryPredicate<T>());

  TEST_ALGO_MACRO_B1E1_ANY(any_of, TrivialUnaryPredicate<T>());
  TEST_ALGO_MACRO_V1_ANY(any_of, TrivialUnaryPredicate<T>());

  TEST_ALGO_MACRO_B1E1_ANY(none_of, TrivialUnaryPredicate<T>());
  TEST_ALGO_MACRO_V1_ANY(none_of, TrivialUnaryPredicate<T>());

  TEST_ALGO_MACRO_B1E1B2(equal);
  TEST_ALGO_MACRO_B1E1B2_ANY(equal, TrivialBinaryPredicate<T>());
  TEST_ALGO_MACRO_V1V2(equal);
  TEST_ALGO_MACRO_V1V2_ANY(equal, TrivialBinaryPredicate<T>());
  TEST_ALGO_MACRO_B1E1B2E2(equal);
  TEST_ALGO_MACRO_B1E1B2E2_ANY(equal, TrivialBinaryPredicate<T>());

  TEST_ALGO_MACRO_B1E1B2E2(lexicographical_compare);
  TEST_ALGO_MACRO_B1E1B2E2_ANY(lexicographical_compare, TrivialComparator<T>());
  TEST_ALGO_MACRO_V1V2(lexicographical_compare);
  TEST_ALGO_MACRO_V1V2_ANY(lexicographical_compare, TrivialComparator<T>());

  TEST_ALGO_MACRO_B1E1(adjacent_find);
  TEST_ALGO_MACRO_V1(adjacent_find);
  TEST_ALGO_MACRO_B1E1_ANY(adjacent_find, TrivialBinaryFunctor<T>());
  TEST_ALGO_MACRO_V1_ANY(adjacent_find, TrivialBinaryFunctor<T>());

  TEST_ALGO_MACRO_B1E1B2E2(search);
  TEST_ALGO_MACRO_V1V2(search);
  TEST_ALGO_MACRO_B1E1B2E2_ANY(search, TrivialBinaryFunctor<T>());
  TEST_ALGO_MACRO_V1V2_ANY(search, TrivialBinaryFunctor<T>());

  TEST_ALGO_MACRO_B1E1B2E2(find_first_of);
  TEST_ALGO_MACRO_V1V2(find_first_of);
  TEST_ALGO_MACRO_B1E1B2E2_ANY(find_first_of, TrivialBinaryFunctor<T>());
  TEST_ALGO_MACRO_V1V2_ANY(find_first_of, TrivialBinaryFunctor<T>());

  TEST_ALGO_MACRO_B1E1_ANY(search_n, count_type{}, T{});
  TEST_ALGO_MACRO_V1_ANY(search_n, count_type{}, T{});
  TEST_ALGO_MACRO_B1E1_ANY(search_n, count_type{}, T{},
                           TrivialBinaryPredicate<T>());
  TEST_ALGO_MACRO_V1_ANY(search_n, count_type{}, T{},
                         TrivialBinaryPredicate<T>());

  TEST_ALGO_MACRO_B1E1B2E2(find_end);
  TEST_ALGO_MACRO_V1V2(find_end);
  TEST_ALGO_MACRO_B1E1B2E2_ANY(find_end, TrivialBinaryFunctor<T>());
  TEST_ALGO_MACRO_V1V2_ANY(find_end, TrivialBinaryFunctor<T>());
}

void modifying_seq_ops() {
  TEST_ALGO_MACRO_B1E1B2_ANY(replace_copy, T{}, T{});
  TEST_ALGO_MACRO_V1V2_ANY(replace_copy, T{}, T{});

  TEST_ALGO_MACRO_B1E1B2_ANY(replace_copy_if, TrivialUnaryPredicate<T>(), T{});
  TEST_ALGO_MACRO_V1V2_ANY(replace_copy_if, TrivialUnaryPredicate<T>(), T{});

  TEST_ALGO_MACRO_B1E1_ANY(replace, T{}, T{});
  TEST_ALGO_MACRO_V1_ANY(replace, T{}, T{});

  TEST_ALGO_MACRO_B1E1_ANY(replace_if, TrivialUnaryPredicate<T>(), T{});
  TEST_ALGO_MACRO_V1_ANY(replace_if, TrivialUnaryPredicate<T>(), T{});

  TEST_ALGO_MACRO_B1E1B2(copy);
  TEST_ALGO_MACRO_V1V2(copy);

  TEST_ALGO_MACRO_B1_ARG_B2(copy_n, count_type{});
  TEST_ALGO_MACRO_V1_ARG_V2(copy_n, count_type{});

  TEST_ALGO_MACRO_B1E1B2(copy_backward);
  TEST_ALGO_MACRO_V1V2(copy_backward);

  TEST_ALGO_MACRO_B1E1B2_ANY(copy_if, TrivialUnaryPredicate<T>());
  TEST_ALGO_MACRO_V1V2_ANY(copy_if, TrivialUnaryPredicate<T>());

  TEST_ALGO_MACRO_B1E1_ANY(fill, T{});
  TEST_ALGO_MACRO_V1_ANY(fill, T{});

  TEST_ALGO_MACRO_B1_ANY(fill_n, count_type{}, T{});
  TEST_ALGO_MACRO_V1_ANY(fill_n, count_type{}, T{});

  TEST_ALGO_MACRO_B1E1B2_ANY(transform, TrivialUnaryFunctor<T>{});
  TEST_ALGO_MACRO_V1V2_ANY(transform, TrivialUnaryFunctor<T>{});

  TEST_ALGO_MACRO_B1E1B2_ANY(transform, TrivialUnaryFunctor<T>{});
  TEST_ALGO_MACRO_B1E1B2B3_ANY(transform, TrivialBinaryFunctor<T>{});
  TEST_ALGO_MACRO_V1V2_ANY(transform, TrivialUnaryFunctor<T>{});
  TEST_ALGO_MACRO_V1V2V3_ANY(transform, TrivialBinaryFunctor<T>{});

  TEST_ALGO_MACRO_B1E1_ANY(generate, TrivialGenerator<T>{});
  TEST_ALGO_MACRO_V1_ANY(generate, TrivialGenerator<T>{});

  TEST_ALGO_MACRO_B1_ANY(generate_n, count_type{}, TrivialGenerator<T>{});
  TEST_ALGO_MACRO_V1_ANY(generate_n, count_type{}, TrivialGenerator<T>{});

  TEST_ALGO_MACRO_B1E1B2(reverse_copy);
  TEST_ALGO_MACRO_V1V2(reverse_copy);

  TEST_ALGO_MACRO_B1E1(reverse);
  TEST_ALGO_MACRO_V1(reverse);

  TEST_ALGO_MACRO_B1E1B2(move);
  TEST_ALGO_MACRO_V1V2(move);

  TEST_ALGO_MACRO_B1E1E2(move_backward);
  TEST_ALGO_MACRO_V1V2(move_backward);

  TEST_ALGO_MACRO_B1E1B2(swap_ranges);
  TEST_ALGO_MACRO_V1V2(swap_ranges);

  TEST_ALGO_MACRO_B1E1(unique);
  TEST_ALGO_MACRO_V1(unique);
  TEST_ALGO_MACRO_B1E1_ANY(unique, TrivialBinaryPredicate<T>{});
  TEST_ALGO_MACRO_V1_ANY(unique, TrivialBinaryPredicate<T>{});

  TEST_ALGO_MACRO_B1E1B2(unique_copy);
  TEST_ALGO_MACRO_V1V2(unique_copy);
  TEST_ALGO_MACRO_B1E1B2_ANY(unique_copy, TrivialBinaryPredicate<T>{});
  TEST_ALGO_MACRO_V1V2_ANY(unique_copy, TrivialBinaryPredicate<T>{});

  TEST_ALGO_MACRO_B1E1E2(rotate);
  TEST_ALGO_MACRO_V1_ANY(rotate, count_type{});

  TEST_ALGO_MACRO_B1E1E1B2(rotate_copy);
  TEST_ALGO_MACRO_V1_ARG_V2(rotate_copy, count_type{});

  TEST_ALGO_MACRO_B1E1_ANY(remove_if, TrivialUnaryPredicate<T>{});
  TEST_ALGO_MACRO_V1_ANY(remove_if, TrivialUnaryPredicate<T>{});

  TEST_ALGO_MACRO_B1E1_ANY(remove, T{});
  TEST_ALGO_MACRO_V1_ANY(remove, T{});

  TEST_ALGO_MACRO_B1E1B2_ANY(remove_copy, T{});
  TEST_ALGO_MACRO_V1V2_ANY(remove_copy, T{});

  TEST_ALGO_MACRO_B1E1B2_ANY(remove_copy_if, TrivialUnaryPredicate<T>());
  TEST_ALGO_MACRO_V1V2_ANY(remove_copy_if, TrivialUnaryPredicate<T>());

  TEST_ALGO_MACRO_B1E1_ANY(shift_left, count_type{});
  TEST_ALGO_MACRO_V1_ANY(shift_left, count_type{});

  TEST_ALGO_MACRO_B1E1_ANY(shift_right, count_type{});
  TEST_ALGO_MACRO_V1_ANY(shift_right, count_type{});
}

void sorting_ops() {
  TEST_ALGO_MACRO_B1E1(is_sorted_until);
  TEST_ALGO_MACRO_V1(is_sorted_until);

#ifndef KOKKOS_ENABLE_OPENMPTARGET
  TEST_ALGO_MACRO_B1E1_ANY(is_sorted_until, TrivialComparator<T>());
  TEST_ALGO_MACRO_V1_ANY(is_sorted_until, TrivialComparator<T>());
#endif

  TEST_ALGO_MACRO_B1E1(is_sorted);
  TEST_ALGO_MACRO_V1(is_sorted);

#ifndef KOKKOS_ENABLE_OPENMPTARGET
  TEST_ALGO_MACRO_B1E1_ANY(is_sorted, TrivialComparator<T>());
  TEST_ALGO_MACRO_V1_ANY(is_sorted, TrivialComparator<T>());
#endif
}

void minmax_ops() {
  TEST_ALGO_MACRO_B1E1(min_element);
  TEST_ALGO_MACRO_V1(min_element);
  TEST_ALGO_MACRO_B1E1(max_element);
  TEST_ALGO_MACRO_V1(max_element);
  TEST_ALGO_MACRO_B1E1(minmax_element);
  TEST_ALGO_MACRO_V1(minmax_element);

#ifndef KOKKOS_ENABLE_OPENMPTARGET
  TEST_ALGO_MACRO_B1E1_ANY(min_element, TrivialComparator<T>());
  TEST_ALGO_MACRO_V1_ANY(min_element, TrivialComparator<T>());
  TEST_ALGO_MACRO_B1E1_ANY(max_element, TrivialComparator<T>());
  TEST_ALGO_MACRO_V1_ANY(max_element, TrivialComparator<T>());
  TEST_ALGO_MACRO_B1E1_ANY(minmax_element, TrivialComparator<T>());
  TEST_ALGO_MACRO_V1_ANY(minmax_element, TrivialComparator<T>());
#endif
}

void partitionig_ops() {
  TEST_ALGO_MACRO_B1E1_ANY(is_partitioned, TrivialUnaryPredicate<T>());
  TEST_ALGO_MACRO_V1_ANY(is_partitioned, TrivialUnaryPredicate<T>());

  TEST_ALGO_MACRO_B1E1B2B3_ANY(partition_copy, TrivialUnaryPredicate<T>());
  TEST_ALGO_MACRO_V1V2V3_ANY(partition_copy, TrivialUnaryPredicate<T>());

  TEST_ALGO_MACRO_B1E1_ANY(partition_point, TrivialUnaryPredicate<T>());
  TEST_ALGO_MACRO_V1_ANY(partition_point, TrivialUnaryPredicate<T>());
}

void numeric() {
  TEST_ALGO_MACRO_B1E1B2(adjacent_difference);
  TEST_ALGO_MACRO_B1E1B2_ANY(adjacent_difference, TrivialBinaryFunctor<T>());
  TEST_ALGO_MACRO_V1V2(adjacent_difference);
  TEST_ALGO_MACRO_V1V2_ANY(adjacent_difference, TrivialBinaryFunctor<T>());

  TEST_ALGO_MACRO_B1E1B2_ANY(exclusive_scan, T{});
  TEST_ALGO_MACRO_V1V2_ANY(exclusive_scan, T{});
#ifndef KOKKOS_ENABLE_OPENMPTARGET
  TEST_ALGO_MACRO_B1E1B2_ANY(exclusive_scan, T{}, TrivialBinaryFunctor<T>());
  TEST_ALGO_MACRO_V1V2_ANY(exclusive_scan, T{}, TrivialBinaryFunctor<T>());

  TEST_ALGO_MACRO_B1E1B2_ANY(transform_exclusive_scan, T{},
                             TrivialBinaryFunctor<T>(),
                             TrivialUnaryFunctor<T>());
  TEST_ALGO_MACRO_V1V2_ANY(transform_exclusive_scan, T{},
                           TrivialBinaryFunctor<T>(), TrivialUnaryFunctor<T>());
#endif

  TEST_ALGO_MACRO_B1E1B2(inclusive_scan);
  TEST_ALGO_MACRO_V1V2(inclusive_scan);
#ifndef KOKKOS_ENABLE_OPENMPTARGET
  TEST_ALGO_MACRO_B1E1B2_ANY(inclusive_scan, TrivialBinaryFunctor<T>());
  TEST_ALGO_MACRO_V1V2_ANY(inclusive_scan, TrivialBinaryFunctor<T>());
  TEST_ALGO_MACRO_B1E1B2_ANY(inclusive_scan, TrivialBinaryFunctor<T>(), T{});
  TEST_ALGO_MACRO_V1V2_ANY(inclusive_scan, TrivialBinaryFunctor<T>(), T{});

  TEST_ALGO_MACRO_B1E1B2_ANY(transform_inclusive_scan,
                             TrivialBinaryFunctor<T>(),
                             TrivialUnaryFunctor<T>());
  TEST_ALGO_MACRO_V1V2_ANY(transform_inclusive_scan, TrivialBinaryFunctor<T>(),
                           TrivialUnaryFunctor<T>());
  TEST_ALGO_MACRO_B1E1B2_ANY(transform_inclusive_scan,
                             TrivialBinaryFunctor<T>(),
                             TrivialUnaryFunctor<T>(), T{});
  TEST_ALGO_MACRO_V1V2_ANY(transform_inclusive_scan, TrivialBinaryFunctor<T>(),
                           TrivialUnaryFunctor<T>(), T{});
#endif

#ifndef KOKKOS_ENABLE_OPENMPTARGET
  TEST_ALGO_MACRO_B1E1(reduce);
  TEST_ALGO_MACRO_V1(reduce);
  TEST_ALGO_MACRO_B1E1_ANY(reduce, T{});
  TEST_ALGO_MACRO_V1_ANY(reduce, T{});
  TEST_ALGO_MACRO_B1E1_ANY(reduce, T{}, TrivialReduceJoinFunctor<T>());
  TEST_ALGO_MACRO_V1_ANY(reduce, T{}, TrivialReduceJoinFunctor<T>());

  TEST_ALGO_MACRO_B1E1B2_ANY(transform_reduce, T{});
  TEST_ALGO_MACRO_V1V2_ANY(transform_reduce, T{});
  TEST_ALGO_MACRO_B1E1B2_ANY(transform_reduce, T{},
                             TrivialReduceJoinFunctor<T>(),
                             TrivialTransformReduceBinaryTransformer<T>());
  TEST_ALGO_MACRO_V1V2_ANY(transform_reduce, T{}, TrivialReduceJoinFunctor<T>(),
                           TrivialTransformReduceBinaryTransformer<T>());

  TEST_ALGO_MACRO_B1E1_ANY(transform_reduce, T{}, TrivialReduceJoinFunctor<T>(),
                           TrivialTransformReduceUnaryTransformer<T>());
  TEST_ALGO_MACRO_V1_ANY(transform_reduce, T{}, TrivialReduceJoinFunctor<T>(),
                         TrivialTransformReduceUnaryTransformer<T>());
#endif
}

}  // namespace compileonly
}  // namespace stdalgos
}  // namespace Test

int main() { return 0; }
