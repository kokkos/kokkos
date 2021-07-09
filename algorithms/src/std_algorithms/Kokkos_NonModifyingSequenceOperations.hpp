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

#ifndef KOKKOS_NON_MODIFYING_SEQUENCE_OPERATIONS_HPP
#define KOKKOS_NON_MODIFYING_SEQUENCE_OPERATIONS_HPP

#include <Kokkos_Core.hpp>
#include "Kokkos_BeginEnd.hpp"
#include "Kokkos_StdAlgorithmsConstraints.hpp"

/// \file Kokkos_NonModifyingSequenceOperations.hpp
/// \brief Kokkos non-modifying sequence operations

namespace Kokkos {
namespace Experimental {

// see https://github.com/kokkos/kokkos/issues/4075
// all_of (note: this might depend on find_if)
// any_of (note: this might depend on find_if)
// none_of (note: this might depend on find_if)
// for_each
// for_each_n
// count
// count_if
// mismatch

// -------------------
// for_each
// -------------------
template <class IteratorType, class UnaryFunctorType>
struct ForEach {
  IteratorType m_first;
  UnaryFunctorType m_functor;

  KOKKOS_INLINE_FUNCTION
  void operator()(int i) const {
    auto element = m_first + i;
    m_functor(*element);
  }

  ForEach(IteratorType _first, UnaryFunctorType _functor)
      : m_first(_first), m_functor(_functor) {}
};

template <class IteratorType, class UnaryFunctorType>
UnaryFunctorType for_each(const std::string& label, IteratorType first,
                          IteratorType end, UnaryFunctorType functor) {
  const auto numOfElements = end - first;
  Kokkos::parallel_for(label, numOfElements,
                       ForEach<IteratorType, UnaryFunctorType>(first, functor));
  return functor;
}

template <class IteratorType, class UnaryFunctorType>
UnaryFunctorType for_each(IteratorType first, IteratorType end,
                          UnaryFunctorType functor) {
  return for_each("_for_each_default_label_1", first, end, std::move(functor));
}

template <class DataType, class... Properties, class UnaryFunctorType>
UnaryFunctorType for_each(const std::string& label,
                          const Kokkos::View<DataType, Properties...>& v,
                          UnaryFunctorType functor) {
  using ViewInType = Kokkos::View<DataType, Properties...>;
  static_assert(
      is_admissible_to_kokkos_std_non_modifying_sequence_op<ViewInType>::value,
      "Currently, Kokkos::Experimental::for_each only accepts 1D Views.");

  return for_each(label, ::Kokkos::Experimental::begin(v),
                  ::Kokkos::Experimental::end(v), std::move(functor));
}

template <class DataType, class... Properties, class UnaryFunctorType>
UnaryFunctorType for_each(const Kokkos::View<DataType, Properties...>& v,
                          UnaryFunctorType functor) {
  return for_each("_for_each_default_label_2", v, std::move(functor));
}

// -------------------
// for_each_n
// -------------------
template <class IteratorType, class SizeType, class UnaryFunctorType>
IteratorType for_each_n(const std::string& label, IteratorType first,
                        SizeType n, UnaryFunctorType functor) {
  if (n <= 0) return first;

  auto last = first + n;
  for_each(label, first, last, std::move(functor));
  return last;
}

template <class IteratorType, class SizeType, class UnaryFunctorType>
IteratorType for_each_n(IteratorType first, SizeType n,
                        UnaryFunctorType functor) {
  return for_each_n("_for_each_n_default_label_1", first, n, functor);
}

template <class DataType, class... Properties, class SizeType,
          class UnaryFunctorType>
auto for_each_n(const std::string& label,
                const Kokkos::View<DataType, Properties...>& v, SizeType n,
                UnaryFunctorType functor) {
  using ViewInType = Kokkos::View<DataType, Properties...>;
  static_assert(
      is_admissible_to_kokkos_std_non_modifying_sequence_op<ViewInType>::value,
      "Currently, Kokkos::Experimental::for_each_n only accepts 1D Views.");

  return for_each_n(label, begin(v), n, std::move(functor));
}

template <class DataType, class... Properties, class SizeType,
          class UnaryFunctorType>
auto for_each_n(const Kokkos::View<DataType, Properties...>& v, SizeType n,
                UnaryFunctorType functor) {
  return for_each_n("_for_each_n_default_label_2", v, n, std::move(functor));
}

// -------------------
// count_if
// -------------------
template <class IteratorType, class Predicate>
struct CountIf {
  IteratorType m_first;
  Predicate m_predicate;

  KOKKOS_INLINE_FUNCTION
  void operator()(int i, std::size_t& lsum) const {
    auto element = m_first + i;
    if (m_predicate(*element)) {
      lsum++;
    }
  }

  CountIf(IteratorType _first, Predicate _predicate)
      : m_first(_first), m_predicate(_predicate) {}
};

template <class IteratorType, class Predicate>
std::size_t count_if(IteratorType first, IteratorType last,
                     Predicate predicate) {
  const auto numOfElements = last - first;
  std::size_t count        = 0;

  Kokkos::parallel_reduce(
      numOfElements, CountIf<IteratorType, Predicate>(first, predicate), count);

  return count;
}

template <class DataType, class... Properties, class Predicate>
std::size_t count_if(const Kokkos::View<DataType, Properties...>& v,
                     Predicate predicate) {
  using ViewInType = Kokkos::View<DataType, Properties...>;
  static_assert(
      is_admissible_to_kokkos_std_non_modifying_sequence_op<ViewInType>::value,
      "Currently, Kokkos::Experimental::count_if only accepts 1D Views.");

  return Kokkos::Experimental::count_if(begin(v), end(v), std::move(predicate));
}

// -------------------
// count
// -------------------
template <class T>
struct EqualsVal {
  const T& m_value;

  KOKKOS_INLINE_FUNCTION
  bool operator()(const T& val) const { return val == m_value; }

  EqualsVal(const T& _value) : m_value(_value) {}
};

// FIXME: ensure T is the value type of InputIterator
// see:
// __glibcxx_function_requires(_EqualOpConcept<
// typename iterator_traits<_InputIterator>::value_type, _Tp>)
template <class IteratorType, class T>
std::size_t count(IteratorType first, IteratorType last, const T& value) {
  return Kokkos::Experimental::count_if(
      // FIXME: is std::equal_to usable here? also see:
      // __gnu_cxx::__ops::__iter_equals_val(__value));
      first, last, EqualsVal<T>(value));
}

template <class DataType, class... Properties, class T>
std::size_t count(const Kokkos::View<DataType, Properties...>& v,
                  const T& value) {
  using ViewInType = Kokkos::View<DataType, Properties...>;
  static_assert(
      is_admissible_to_kokkos_std_non_modifying_sequence_op<ViewInType>::value,
      "Currently, Kokkos::Experimental::count only accepts 1D Views.");

  return Kokkos::Experimental::count_if(begin(v), end(v), EqualsVal<T>(value));
}

// -------------------
// mismatch
// -------------------
template <class IteratorType1, class IteratorType2>
Kokkos::pair<IteratorType1, IteratorType2> mismatch(IteratorType1 first_1,
                                                    IteratorType1 last_1,
                                                    IteratorType2 first_2,
                                                    IteratorType2 last_2) {
  while (first_1 != last_1 && first_2 != last_2 && *first_1 == *first_2) {
    ++first_1;
    ++first_2;
  }
  return pair<IteratorType1, IteratorType2>(first_1, first_2);
}

// -------------------
// find_if
// -------------------
template <class IteratorType, class Predicate>
IteratorType find_if(IteratorType first, IteratorType last,
                     Predicate predicate) {
  while (first != last && !predicate(*first)) ++first;
  return first;
}

template <class DataType, class... Properties, class Predicate>
auto find_if(const Kokkos::View<DataType, Properties...>& v,
             Predicate predicate) {
  using ViewInType = Kokkos::View<DataType, Properties...>;
  static_assert(
      is_admissible_to_kokkos_std_non_modifying_sequence_op<ViewInType>::value,
      "Currently, Kokkos::Experimental::find_if only accepts 1D Views.");

  return Kokkos::Experimental::find_if(begin(v), end(v), std::move(predicate));
}

// -------------------
// any_of
// -------------------
template <class InputIterator, class Predicate>
bool any_of(InputIterator first, InputIterator last, Predicate predicate) {
  return Kokkos::Experimental::find_if(first, last, predicate) != last;
}

// -------------------
// none_of
// -------------------
template <class IteratorType, class Predicate>
bool none_of(IteratorType first, IteratorType last, Predicate predicate) {
  return Kokkos::Experimental::find_if(first, last, predicate) == last;
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
