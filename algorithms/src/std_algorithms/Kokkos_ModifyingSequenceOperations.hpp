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

#ifndef KOKKOS_MODIFYING_SEQUENCE_OPERATIONS_HPP
#define KOKKOS_MODIFYING_SEQUENCE_OPERATIONS_HPP

#include <Kokkos_Core.hpp>
#include "Kokkos_BeginEnd.hpp"
#include "Kokkos_StdAlgorithmsConstraints.hpp"

/// \file Kokkos_ModifyingSequenceOperations.hpp
/// \brief Kokkos modifying sequence operations

namespace Kokkos {
namespace Experimental {

// see https://github.com/kokkos/kokkos/issues/4075
//
// copy
// copy_if
// copy_n
// copy_backward
// move
// move_backward
// fill
// fill_n
// transform
// generate
// generate_n
// swap
// swap_ranges
// iter_swap
// reverse_copy

// -------------------
// copy
// -------------------
template <class InputIterator, class OutputIterator>
struct Copy {
  InputIterator m_first;
  OutputIterator m_dest_first;

  KOKKOS_INLINE_FUNCTION
  void operator()(int i) const { *(m_dest_first + i) = *(m_first + i); }

  Copy(InputIterator _first, OutputIterator _dest_first)
      : m_first(_first), m_dest_first(_dest_first) {}
};

template <class InputIterator, class OutputIterator>
OutputIterator copy(InputIterator first, InputIterator last,
                    OutputIterator d_first) {
  const auto numOfElements = last - first;
  Kokkos::parallel_for(numOfElements,
                       Copy<InputIterator, OutputIterator>(first, d_first));
  return d_first + numOfElements;
}

template <class DataType1, class... Properties1, class DataType2,
          class... Properties2>
auto copy(const Kokkos::View<DataType1, Properties1...>& source,
          Kokkos::View<DataType2, Properties2...>& dest) {
  using ViewInType1 = Kokkos::View<DataType1, Properties1...>;
  using ViewInType2 = Kokkos::View<DataType2, Properties2...>;
  static_assert(is_admissible_to_kokkos_std_non_modifying_sequence_op<
                    ViewInType1>::value and
                    is_admissible_to_kokkos_std_non_modifying_sequence_op<
                        ViewInType2>::value,
                "Currently, Kokkos::Experimental::copy only accepts 1D Views.");
  return Kokkos::Experimental::copy(cbegin(source), cend(source), begin(dest));
}

// -------------------
// copy_n
// -------------------
template <class InputIterator, class Size, class OutputIterator>
OutputIterator copy_n(InputIterator first, Size count, OutputIterator result) {
  if (count < 0) {
    return result;
  }

  return Kokkos::Experimental::copy(first, first + count, result);
}

// -------------------
// copy_backward
// -------------------
template <class IteratorType1, class IteratorType2>
struct CopyBackward {
  IteratorType1 m_last;
  IteratorType2 m_dest_last;

  KOKKOS_INLINE_FUNCTION
  void operator()(int i) const { *(m_dest_last - i) = *(m_last - i); }

  CopyBackward(IteratorType1 _last, IteratorType2 _dest_last)
      : m_last(_last), m_dest_last(_dest_last) {}
};

template <class IteratorType1, class IteratorType2>
IteratorType2 copy_backward(IteratorType1 first, IteratorType1 last,
                            IteratorType2 d_last) {
  const auto num_elements = last - first;
  Kokkos::parallel_for(
      num_elements, CopyBackward<IteratorType1, IteratorType2>(last, d_last));
  return d_last - num_elements;
}

// -------------------
// copy_if
// -------------------
template <class InputIterator, class OutputIterator, class Predicate>
OutputIterator copy_if(InputIterator first, InputIterator last,
                       OutputIterator d_first, Predicate pred) {
  for (; first != last; ++first)
    if (pred(*first)) {
      *d_first = *first;
      ++d_first;
    }
  return d_first;
}

// -------------------
// reverse_copy
// -------------------
template <class InputIterator, class OutputIterator>
struct ReverseCopy {
  InputIterator m_last;
  OutputIterator m_dest_first;

  KOKKOS_INLINE_FUNCTION
  void operator()(int i) const { *(m_dest_first + i) = *(m_last - 1 - i); }

  ReverseCopy(InputIterator _last, OutputIterator _dest_first)
      : m_last(_last), m_dest_first(_dest_first) {}
};

template <class InputIterator, class OutputIterator>
OutputIterator reverse_copy(InputIterator first, InputIterator last,
                            OutputIterator d_first) {
  const auto numOfElements = last - first;
  Kokkos::parallel_for(
      numOfElements, ReverseCopy<InputIterator, OutputIterator>(last, d_first));
  return d_first + numOfElements;
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
