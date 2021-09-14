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

#ifndef KOKKOS_STD_SORTING_OPERATIONS_HPP
#define KOKKOS_STD_SORTING_OPERATIONS_HPP

#include <Kokkos_Core.hpp>
#include "Kokkos_BeginEnd.hpp"
#include "Kokkos_Constraints.hpp"
#include "Kokkos_NonModifyingSequenceOperations.hpp"
#include "Kokkos_HelperPredicates.hpp"
#include <string>

namespace Kokkos {
namespace Experimental {

// ------------------------------------------
// begin Impl namespace
namespace Impl {

template <class IndexType, class IteratorType, class IndicatorViewType,
          class ComparatorType>
struct StdIsSortedUntilFunctor {
  IteratorType m_first;
  IndicatorViewType m_indicator;
  ComparatorType m_comparator;

  KOKKOS_INLINE_FUNCTION
  void operator()(const IndexType i, int& update, const bool final) const {
    const auto val_i   = *(m_first + i);
    const auto val_ip1 = *(m_first + i + 1);

    if (m_comparator(val_ip1, val_i)) {
      update += 1;
    } else {
      update += 0;
    }

    if (final) {
      m_indicator(i) = update;
    }
  }

  KOKKOS_INLINE_FUNCTION
  StdIsSortedUntilFunctor(IteratorType _first1, IndicatorViewType indicator,
                          ComparatorType comparator)
      : m_first(_first1),
        m_indicator(indicator),
        m_comparator(::Kokkos::Experimental::move(comparator)) {}
};

template <class IndexType, class IteratorType, class ComparatorType>
struct StdIsSortedFunctor {
  IteratorType m_first;
  ComparatorType m_comparator;

  KOKKOS_INLINE_FUNCTION
  void operator()(const IndexType i, int& update, const bool final) const {
    const auto val_i   = *(m_first + i);
    const auto val_ip1 = *(m_first + i + 1);

    if (m_comparator(val_ip1, val_i)) {
      update += 1;
    } else {
      update += 0;
    }

    if (final) {
      // no op in this case
    }
  }

  KOKKOS_INLINE_FUNCTION
  StdIsSortedFunctor(IteratorType _first1, ComparatorType comparator)
      : m_first(_first1),
        m_comparator(::Kokkos::Experimental::move(comparator)) {}
};

// impl functions

template <class ExecutionSpace, class IteratorType, class ComparatorType>
IteratorType is_sorted_until_impl(const std::string& label,
                                  const ExecutionSpace& ex, IteratorType first,
                                  IteratorType last, ComparatorType comp) {
  static_assert_random_access_and_accessible<ExecutionSpace, IteratorType>();

  const auto num_elements = last - first;

  if (num_elements <= 1) {
    return last;
  }

  using indicator_type = ::Kokkos::View<int*, ExecutionSpace>;
  using index_type     = int;
  using functor_type   = StdIsSortedUntilFunctor<index_type, IteratorType,
                                               indicator_type, ComparatorType>;

  const auto num_elements_minus_one = num_elements - 1;
  indicator_type indicator("is_sorted_until_indicator_helper",
                           num_elements_minus_one);

  ::Kokkos::parallel_scan(
      label, RangePolicy<ExecutionSpace>(ex, 0, num_elements_minus_one),
      functor_type(first, indicator, ::Kokkos::Experimental::move(comp)));

  constexpr int sentinel_value = 1;
  auto r                       = ::Kokkos::Experimental::find(
      ex, ::Kokkos::Experimental::cbegin(indicator),
      ::Kokkos::Experimental::cend(indicator), sentinel_value);
  const auto shift = r - ::Kokkos::Experimental::cbegin(indicator);
  return first + (shift + 1);
}

template <class ExecutionSpace, class IteratorType>
IteratorType is_sorted_until_impl(const std::string& label,
                                  const ExecutionSpace& ex, IteratorType first,
                                  IteratorType last) {
  using value_type = typename IteratorType::value_type;
  using pred_t = Impl::StdAlgoLessThanBinaryPredicate<value_type, value_type>;
  return is_sorted_until_impl(label, ex, first, last, pred_t());
}

template <class ExecutionSpace, class IteratorType, class ComparatorType>
bool is_sorted_impl(const std::string& label, const ExecutionSpace& ex,
                    IteratorType first, IteratorType last,
                    ComparatorType comp) {
  static_assert_random_access_and_accessible<ExecutionSpace, IteratorType>();

  const auto num_elements = last - first;
  if (num_elements <= 1) {
    return true;
  }

  const auto num_elements_minus_one = num_elements - 1;
  using index_type                  = int;
  using functor_type =
      StdIsSortedFunctor<index_type, IteratorType, ComparatorType>;

  int result = 0;
  ::Kokkos::parallel_scan(
      label, RangePolicy<ExecutionSpace>(ex, 0, num_elements_minus_one),
      functor_type(first, ::Kokkos::Experimental::move(comp)), result);
  if (result == 0) {
    return true;
  } else {
    return false;
  }
}

template <class ExecutionSpace, class IteratorType>
bool is_sorted_impl(const std::string& label, const ExecutionSpace& ex,
                    IteratorType first, IteratorType last) {
  using value_type = typename IteratorType::value_type;
  using pred_t = Impl::StdAlgoLessThanBinaryPredicate<value_type, value_type>;
  return is_sorted_impl(label, ex, first, last, pred_t());
}

}  // namespace Impl
// ------------------------------------------

// ----------------------------------
// is_sorted_until public API
// ----------------------------------
template <class ExecutionSpace, class IteratorType>
IteratorType is_sorted_until(const ExecutionSpace& ex, IteratorType first,
                             IteratorType last) {
  return Impl::is_sorted_until_impl(
      "kokkos_is_sorted_until_iterator_api_default", ex, first, last);
}

template <class ExecutionSpace, class IteratorType>
IteratorType is_sorted_until(const std::string& label, const ExecutionSpace& ex,
                             IteratorType first, IteratorType last) {
  return Impl::is_sorted_until_impl(label, ex, first, last);
}

template <class ExecutionSpace, class DataType, class... Properties>
auto is_sorted_until(const ExecutionSpace& ex,
                     const ::Kokkos::View<DataType, Properties...>& view) {
  static_assert_is_admissible_to_kokkos_std_algorithms(view);

  namespace KE = ::Kokkos::Experimental;
  return Impl::is_sorted_until_impl("kokkos_is_sorted_until_view_api_default",
                                    ex, KE::cbegin(view), KE::cend(view));
}

template <class ExecutionSpace, class DataType, class... Properties>
auto is_sorted_until(const std::string& label, const ExecutionSpace& ex,
                     const ::Kokkos::View<DataType, Properties...>& view) {
  static_assert_is_admissible_to_kokkos_std_algorithms(view);

  namespace KE = ::Kokkos::Experimental;
  return Impl::is_sorted_until_impl(label, ex, KE::cbegin(view),
                                    KE::cend(view));
}

template <class ExecutionSpace, class IteratorType, class ComparatorType>
IteratorType is_sorted_until(const ExecutionSpace& ex, IteratorType first,
                             IteratorType last, ComparatorType comp) {
  static_assert_is_not_opemnptarget(ex);
  return Impl::is_sorted_until_impl(
      "kokkos_is_sorted_until_iterator_api_default", ex, first, last, comp);
}

template <class ExecutionSpace, class IteratorType, class ComparatorType>
IteratorType is_sorted_until(const std::string& label, const ExecutionSpace& ex,
                             IteratorType first, IteratorType last,
                             ComparatorType comp) {
  static_assert_is_not_opemnptarget(ex);

  return Impl::is_sorted_until_impl(label, ex, first, last, comp);
}

template <class ExecutionSpace, class DataType, class... Properties,
          class ComparatorType>
auto is_sorted_until(const ExecutionSpace& ex,
                     const ::Kokkos::View<DataType, Properties...>& view,
                     ComparatorType comp) {
  static_assert_is_admissible_to_kokkos_std_algorithms(view);
  static_assert_is_not_opemnptarget(ex);

  namespace KE = ::Kokkos::Experimental;
  return Impl::is_sorted_until_impl("kokkos_is_sorted_until_view_api_default",
                                    ex, KE::cbegin(view), KE::cend(view), comp);
}

template <class ExecutionSpace, class DataType, class... Properties,
          class ComparatorType>
auto is_sorted_until(const std::string& label, const ExecutionSpace& ex,
                     const ::Kokkos::View<DataType, Properties...>& view,
                     ComparatorType comp) {
  static_assert_is_admissible_to_kokkos_std_algorithms(view);
  static_assert_is_not_opemnptarget(ex);

  namespace KE = ::Kokkos::Experimental;
  return Impl::is_sorted_until_impl(label, ex, KE::cbegin(view), KE::cend(view),
                                    comp);
}

// ----------------------------------
// is_sorted public API
// ----------------------------------
template <class ExecutionSpace, class IteratorType>
bool is_sorted(const ExecutionSpace& ex, IteratorType first,
               IteratorType last) {
  return Impl::is_sorted_impl("kokkos_is_sorted_iterator_api_default", ex,
                              first, last);
}

template <class ExecutionSpace, class IteratorType>
bool is_sorted(const std::string& label, const ExecutionSpace& ex,
               IteratorType first, IteratorType last) {
  return Impl::is_sorted_impl(label, ex, first, last);
}

template <class ExecutionSpace, class DataType, class... Properties>
bool is_sorted(const ExecutionSpace& ex,
               const ::Kokkos::View<DataType, Properties...>& view) {
  static_assert_is_admissible_to_kokkos_std_algorithms(view);

  namespace KE = ::Kokkos::Experimental;
  return Impl::is_sorted_impl("kokkos_is_sorted_view_api_default", ex,
                              KE::cbegin(view), KE::cend(view));
}

template <class ExecutionSpace, class DataType, class... Properties>
bool is_sorted(const std::string& label, const ExecutionSpace& ex,
               const ::Kokkos::View<DataType, Properties...>& view) {
  static_assert_is_admissible_to_kokkos_std_algorithms(view);

  namespace KE = ::Kokkos::Experimental;
  return Impl::is_sorted_impl(label, ex, KE::cbegin(view), KE::cend(view));
}

template <class ExecutionSpace, class IteratorType, class ComparatorType>
bool is_sorted(const ExecutionSpace& ex, IteratorType first, IteratorType last,
               ComparatorType comp) {
  static_assert_is_not_opemnptarget(ex);
  return Impl::is_sorted_impl("kokkos_is_sorted_iterator_api_default", ex,
                              first, last, comp);
}

template <class ExecutionSpace, class IteratorType, class ComparatorType>
bool is_sorted(const std::string& label, const ExecutionSpace& ex,
               IteratorType first, IteratorType last, ComparatorType comp) {
  static_assert_is_not_opemnptarget(ex);
  return Impl::is_sorted_impl(label, ex, first, last, comp);
}

template <class ExecutionSpace, class DataType, class... Properties,
          class ComparatorType>
bool is_sorted(const ExecutionSpace& ex,
               const ::Kokkos::View<DataType, Properties...>& view,
               ComparatorType comp) {
  static_assert_is_admissible_to_kokkos_std_algorithms(view);
  static_assert_is_not_opemnptarget(ex);

  namespace KE = ::Kokkos::Experimental;
  return Impl::is_sorted_impl("kokkos_is_sorted_view_api_default", ex,
                              KE::cbegin(view), KE::cend(view), comp);
}

template <class ExecutionSpace, class DataType, class... Properties,
          class ComparatorType>
bool is_sorted(const std::string& label, const ExecutionSpace& ex,
               const ::Kokkos::View<DataType, Properties...>& view,
               ComparatorType comp) {
  static_assert_is_admissible_to_kokkos_std_algorithms(view);
  static_assert_is_not_opemnptarget(ex);

  namespace KE = ::Kokkos::Experimental;
  return Impl::is_sorted_impl(label, ex, KE::cbegin(view), KE::cend(view),
                              comp);
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
