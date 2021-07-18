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

#ifndef KOKKOS_RANDOM_ACCESS_ITERATOR_HPP
#define KOKKOS_RANDOM_ACCESS_ITERATOR_HPP

#include <iterator>
#include <Kokkos_View.hpp>

namespace Kokkos {
namespace Experimental {

template <class enable, class ViewT, class... Args>
class RandomAccessIterator;

template <class DataType, class... Args>
class RandomAccessIterator<
    std::enable_if_t<
        Kokkos::View<DataType, Args...>::rank == 1 and
        (std::is_same<typename Kokkos::View<DataType, Args...>::array_layout,
                      Kokkos::LayoutStride>::value or
         std::is_same<typename Kokkos::View<DataType, Args...>::array_layout,
                      Kokkos::LayoutLeft>::value or
         std::is_same<typename Kokkos::View<DataType, Args...>::array_layout,
                      Kokkos::LayoutRight>::value)>,
    Kokkos::View<DataType, Args...> >
    : public std::iterator<
          std::random_access_iterator_tag,
          typename Kokkos::View<DataType, Args...>::value_type, ptrdiff_t,
          typename Kokkos::View<DataType, Args...>::pointer_type,
          typename Kokkos::View<DataType, Args...>::reference_type> {
 public:
  using view_t          = Kokkos::View<DataType, Args...>;
  using iterator_t      = RandomAccessIterator<void, view_t>;
  using difference_type = ptrdiff_t;
  using value_type      = typename view_t::value_type;
  using reference       = typename view_t::reference_type;

  // delete default cnstr because m_view is a const member
  KOKKOS_INLINE_FUNCTION RandomAccessIterator() = default;

  explicit KOKKOS_INLINE_FUNCTION RandomAccessIterator(const view_t viewIn)
      : m_view(viewIn) {}
  explicit KOKKOS_INLINE_FUNCTION RandomAccessIterator(const view_t viewIn,
                                                       ptrdiff_t current_index)
      : m_view(viewIn), m_currentIndex(current_index) {}

  KOKKOS_INLINE_FUNCTION
  iterator_t& operator++() {
    ++m_currentIndex;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  iterator_t operator++(int) {
    auto tmp = *this;
    ++*this;
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  iterator_t& operator--() {
    --m_currentIndex;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  iterator_t operator--(int) {
    auto tmp = *this;
    --*this;
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  iterator_t operator+(difference_type n) const {
    return iterator_t(m_view, m_currentIndex + n);
  }

  KOKKOS_INLINE_FUNCTION
  iterator_t operator-(difference_type n) const {
    return iterator_t(m_view, m_currentIndex - n);
  }

  KOKKOS_INLINE_FUNCTION
  difference_type operator-(iterator_t it) const {
    return m_currentIndex - it.m_currentIndex;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator==(iterator_t other) const {
    return m_currentIndex == other.m_currentIndex &&
           m_view.data() == other.m_view.data();
  }

  KOKKOS_INLINE_FUNCTION
  bool operator!=(iterator_t other) const {
    return m_currentIndex != other.m_currentIndex ||
           m_view.data() != other.m_view.data();
  }

  KOKKOS_INLINE_FUNCTION
  bool operator<(iterator_t other) const {
    return m_currentIndex < other.m_currentIndex;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator<=(iterator_t other) const {
    return m_currentIndex <= other.m_currentIndex;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator>(iterator_t other) const {
    return m_currentIndex > other.m_currentIndex;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator>=(iterator_t other) const {
    return m_currentIndex >= other.m_currentIndex;
  }

  KOKKOS_INLINE_FUNCTION
  reference operator*() const { return m_view(m_currentIndex); }

 private:
  const view_t m_view;
  ptrdiff_t m_currentIndex = 0;
};

}  // namespace Experimental
}  // namespace Kokkos

#endif
