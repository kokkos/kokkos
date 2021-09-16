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

#ifndef KOKKOS_STD_ALGORITHMS_CONSTRAINTS_HPP_
#define KOKKOS_STD_ALGORITHMS_CONSTRAINTS_HPP_

#include <Kokkos_DetectionIdiom.hpp>
#include <Kokkos_View.hpp>

namespace Kokkos {
namespace Experimental {

template <typename T, typename enable = void>
struct is_admissible_to_kokkos_std_algorithms : std::false_type {};

template <typename T>
struct is_admissible_to_kokkos_std_algorithms<
    T, std::enable_if_t< ::Kokkos::is_view<T>::value && T::rank == 1 &&
                         (std::is_same<typename T::traits::array_layout,
                                       Kokkos::LayoutLeft>::value ||
                          std::is_same<typename T::traits::array_layout,
                                       Kokkos::LayoutRight>::value ||
                          std::is_same<typename T::traits::array_layout,
                                       Kokkos::LayoutStride>::value)> >
    : std::true_type {};

template <class ViewType>
KOKKOS_INLINE_FUNCTION constexpr void
static_assert_is_admissible_to_kokkos_std_algorithms(const ViewType&) {
  static_assert(is_admissible_to_kokkos_std_algorithms<ViewType>::value,
                "Currently, Kokkos standard algorithms only accept 1D Views.");
}

//
// is_iterator
//
template <class T>
using iterator_category_t = typename T::iterator_category;

template <class T>
using is_iterator = Kokkos::is_detected<iterator_category_t, T>;

//
// are_iterators
//
template <class... Args>
struct are_iterators;

template <class T>
struct are_iterators<T> {
  static constexpr bool value = is_iterator<T>::value;
};

template <class Head, class... Tail>
struct are_iterators<Head, Tail...> {
  static constexpr bool value =
      are_iterators<Head>::value && are_iterators<Tail...>::value;
};

//
// are_random_access_iterators
//
template <class... Args>
struct are_random_access_iterators;

template <class T>
struct are_random_access_iterators<T> {
  static constexpr bool value =
      is_iterator<T>::value &&
      std::is_same<typename T::iterator_category,
                   std::random_access_iterator_tag>::value;
};

template <class Head, class... Tail>
struct are_random_access_iterators<Head, Tail...> {
  static constexpr bool value = are_random_access_iterators<Head>::value &&
                                are_random_access_iterators<Tail...>::value;
};

//
// are_accessible_iterators
//
template <class... Args>
struct are_accessible_iterators;

template <class ExeSpace, class IteratorType>
struct are_accessible_iterators<ExeSpace, IteratorType> {
  using view_type = typename IteratorType::view_type;
  static constexpr bool value =
      SpaceAccessibility<ExeSpace,
                         typename view_type::memory_space>::accessible;
};

template <class ExeSpace, class Head, class... Tail>
struct are_accessible_iterators<ExeSpace, Head, Tail...> {
  static constexpr bool value =
      are_accessible_iterators<ExeSpace, Head>::value &&
      are_accessible_iterators<ExeSpace, Tail...>::value;
};

template <class ExecutionSpace, class... IteratorTypes>
KOKKOS_INLINE_FUNCTION constexpr void
static_assert_random_access_and_accessible() {
  static_assert(
      are_random_access_iterators<IteratorTypes...>::value,
      "Currently, Kokkos standard algorithms require random access iterators.");

  static_assert(
      are_accessible_iterators<ExecutionSpace, IteratorTypes...>::value,
      "Incompatible view/iterator and execution space");
}

//
// have matching difference_type
//
template <class... Args>
struct iterators_have_matching_difference_type;

template <class T>
struct iterators_have_matching_difference_type<T> {
  static constexpr bool value = true;
};

template <class T1, class T2>
struct iterators_have_matching_difference_type<T1, T2> {
  static constexpr bool value =
      std::is_same<typename T1::difference_type,
                   typename T2::difference_type>::value;
};

template <class T1, class T2, class... Tail>
struct iterators_have_matching_difference_type<T1, T2, Tail...> {
  static constexpr bool value =
      iterators_have_matching_difference_type<T1, T2>::value &&
      iterators_have_matching_difference_type<T2, Tail...>::value;
};

template <class... IteratorTypes>
KOKKOS_INLINE_FUNCTION constexpr void
static_assert_iterators_have_matching_difference_type() {
  static_assert(
      iterators_have_matching_difference_type<IteratorTypes...>::value,
      "Iterators do not have matching difference_type");
}

//
// not_openmptarget
//
template <class ExeSpace>
struct not_openmptarget {
#if not defined KOKKOS_ENABLE_OPENMPTARGET
  static constexpr bool value = true;
#else
  static constexpr bool value =
      !std::is_same<std::remove_cv_t<std::remove_reference_t<ExeSpace> >,
                    ::Kokkos::Experimental::OpenMPTarget>::value;
#endif
};

template <class ExecutionSpace>
KOKKOS_INLINE_FUNCTION constexpr void static_assert_is_not_opemnptarget(
    const ExecutionSpace&) {
  static_assert(not_openmptarget<ExecutionSpace>::value,
                "Currently, Kokkos standard algorithms do not support custom "
                "comparators in OpenMPTarget");
}

//
// valid range
//
template <class IteratorType>
void expect_valid_range(IteratorType first, IteratorType last) {
  // this is a no-op for release
  KOKKOS_EXPECTS(last >= first);
  // avoid compiler complaining when KOKKOS_EXPECTS is no-op
  (void)first;
  (void)last;
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
