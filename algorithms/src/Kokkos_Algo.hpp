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

#ifndef KOKKOS_ALGO_HPP
#define KOKKOS_ALGO_HPP

#include <Kokkos_Core.hpp>

/// \file Kokkos_Algo.hpp
/// \brief Kokkos counterparts for Standard C++ Library algorithms

namespace Kokkos {
namespace Experimental {

template <typename T, typename enable = void>
struct view_admissible_to_kokkos_std_algorithms : std::false_type {};

template <typename T>
struct view_admissible_to_kokkos_std_algorithms<
    T, std::enable_if_t< ::Kokkos::is_view<T>::value and T::rank == 1> >
    : std::true_type {};

template <class DataType, class... Properties>
auto begin(const Kokkos::View<DataType, Properties...>& v)
    -> decltype(v.data()) {
  using ViewInType = Kokkos::View<DataType, Properties...>;
  static_assert(view_admissible_to_kokkos_std_algorithms<ViewInType>::value,
                "Currently, Kokkos::Experimental::begin only accepts 1D "
                "contiguous Views.");

  KOKKOS_EXPECTS(v.span_is_contiguous());
  return v.data();
}

template <class DataType, class... Properties>
auto end(const Kokkos::View<DataType, Properties...>& v) -> decltype(v.data()) {
  using ViewInType = Kokkos::View<DataType, Properties...>;
  static_assert(
      view_admissible_to_kokkos_std_algorithms<ViewInType>::value,
      "Currently, Kokkos::Experimental::end only accepts 1D contiguous Views.");

  KOKKOS_EXPECTS(v.span_is_contiguous());
  return v.data() + v.size();
}

template <class PointerType, class FunctorType>
FunctorType for_each(PointerType data, PointerType end, FunctorType functor) {
  const auto numOfElements = end - data;
  Kokkos::parallel_for(
      numOfElements, KOKKOS_LAMBDA(const int i) {
        auto element = data + i;
        functor(*element);
      });
  return functor;
}

template <class DataType, class... Properties, class FunctorType>
FunctorType for_each(Kokkos::View<DataType, Properties...> v,
                     FunctorType functor) {
  using ViewInType = Kokkos::View<DataType, Properties...>;
  static_assert(view_admissible_to_kokkos_std_algorithms<ViewInType>::value,
                "Currently, Kokkos::Experimental::for_each only accepts 1D "
                "contiguous Views.");

  KOKKOS_EXPECTS(v.span_is_contiguous());
  return for_each(::Kokkos::Experimental::begin(v),
                  ::Kokkos::Experimental::end(v), std::move(functor));
}

template <class PointerType, class SizeType, class FunctorType>
PointerType for_each_n(PointerType data, SizeType n, FunctorType functor) {
  if (n <= 0) return data;

  auto last = data + n;
  for_each(data, last, std::move(functor));
  return last;
}

template <class DataType, class... Properties, class SizeType,
          class FunctorType>
void for_each_n(Kokkos::View<DataType, Properties...> v, SizeType n,
                FunctorType functor) {
  for_each_n(begin(v), n, std::move(functor));
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
