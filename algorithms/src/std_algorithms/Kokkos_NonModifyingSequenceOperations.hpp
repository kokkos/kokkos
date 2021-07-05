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
template <class PointerType, class FunctorType>
struct ForEach {
  PointerType m_first;
  FunctorType m_functor;

  KOKKOS_INLINE_FUNCTION
  void operator()(int i) const {
    auto element = m_first + i;
    m_functor(*element);
  }

  ForEach(PointerType _first, FunctorType _functor)
      : m_first(_first), m_functor(_functor) {}
};

template <class PointerType, class FunctorType>
FunctorType for_each(const std::string& label, PointerType first,
                     PointerType end, FunctorType functor) {
  const auto numOfElements = end - first;
  Kokkos::parallel_for(label, numOfElements,
                       ForEach<PointerType, FunctorType>(first, functor));
  return functor;
}

template <class PointerType, class FunctorType>
FunctorType for_each(PointerType first, PointerType end, FunctorType functor) {
  return for_each("", first, end, std::move(functor));
}

template <class DataType, class... Properties, class FunctorType>
FunctorType for_each(const std::string& label,
                     Kokkos::View<DataType, Properties...> v,
                     FunctorType functor) {
  using ViewInType = Kokkos::View<DataType, Properties...>;
  static_assert(is_admissible_view_to_kokkos_std_non_modifying_sequence_op<
                    ViewInType>::value,
                "Currently, Kokkos::Experimental::for_each only accepts 1D "
                "contiguous Views.");

  KOKKOS_EXPECTS(v.span_is_contiguous());
  return for_each(label, ::Kokkos::Experimental::begin(v),
                  ::Kokkos::Experimental::end(v), std::move(functor));
}

template <class DataType, class... Properties, class FunctorType>
FunctorType for_each(Kokkos::View<DataType, Properties...> v,
                     FunctorType functor) {
  return for_each("", v, std::move(functor));
}

// -------------------
// for_each_n
// -------------------
template <class PointerType, class SizeType, class FunctorType>
PointerType for_each_n(const std::string& label, PointerType first, SizeType n,
                       FunctorType functor) {
  if (n <= 0) return first;

  auto last = first + n;
  for_each(label, first, last, std::move(functor));
  return last;
}

template <class PointerType, class SizeType, class FunctorType>
PointerType for_each_n(PointerType first, SizeType n, FunctorType functor) {
  return for_each_n("", first, n, functor);
}

template <class DataType, class... Properties, class SizeType,
          class FunctorType>
void for_each_n(const std::string& label,
                Kokkos::View<DataType, Properties...> v, SizeType n,
                FunctorType functor) {
  using ViewInType = Kokkos::View<DataType, Properties...>;
  static_assert(is_admissible_view_to_kokkos_std_non_modifying_sequence_op<
                    ViewInType>::value,
                "Currently, Kokkos::Experimental::for_each_n only accepts 1D "
                "contiguous Views.");

  KOKKOS_EXPECTS(v.span_is_contiguous());
  for_each_n(label, begin(v), n, std::move(functor));
}

template <class DataType, class... Properties, class SizeType,
          class FunctorType>
void for_each_n(Kokkos::View<DataType, Properties...> v, SizeType n,
                FunctorType functor) {
  for_each_n("", v, n, std::move(functor));
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
