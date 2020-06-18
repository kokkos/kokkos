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

#include <gtest/gtest.h>

#include <stdexcept>
#include <sstream>
#include <iostream>
#include <time.h>

#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

#include <Kokkos_Core.hpp>
#define OFFSET_LIST_MAX_SIZE 100

namespace Kokkos {

struct LayoutSelective {
  //! Tag this class as a kokkos array layout
  using array_layout = LayoutSelective;

  size_t offset_list[OFFSET_LIST_MAX_SIZE];
  size_t list_size;

  enum { is_extent_constructible = false };

  LayoutSelective() {
    for (int i = 0; i < OFFSET_LIST_MAX_SIZE; i++) {
      offset_list[i] = i;
    }
  }
  LayoutSelective(LayoutSelective const& rhs) {
    list_size = rhs.list_size;
    for (int i = 0; i < list_size; i++) {
      offset_list[i] = rhs.offset_list[i];
    }
  }
  LayoutSelective(LayoutSelective&&) = default;
  LayoutSelective& operator=(LayoutSelective const&) = default;
  LayoutSelective& operator=(LayoutSelective&&) = default;

  KOKKOS_INLINE_FUNCTION
  explicit LayoutSelective(const size_t ol_[], const size_t size_) {
    list_size = size_;
    for (int i = 0; i < list_size; i++) {
      offset_list[i] = ol_[i];
    }
  }

  KOKKOS_INLINE_FUNCTION
  size_t offset(size_t ndx) const {
    KOKKOS_ASSERT(ndx >= 0 && ndx < list_size);
    return offset_list[ndx];
  }
};

namespace Impl {
template <class Dimension>
struct ViewOffset<Dimension, Kokkos::LayoutSelective, void> {
 public:
  using is_mapping_plugin = std::true_type;
  using is_regular        = std::false_type;

  typedef size_t size_type;
  typedef Dimension dimension_type;
  typedef Kokkos::LayoutSelective array_layout;

  //----------------------------------------
  dimension_type m_dim;
  array_layout m_selective;

  // rank 1
  template <typename I0>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(I0 const& i0) const {
    return m_selective.offset(i0);
  }

  // This ViewOffset and the underlying layout only supports rank 1 Views

  //----------------------------------------

  KOKKOS_INLINE_FUNCTION
  array_layout layout() const { return array_layout(); }

  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_0() const {
    return m_dim.N0;
  }

  /* Cardinality of the domain index space */
  KOKKOS_INLINE_FUNCTION
  constexpr size_type size() const { return m_dim.N0; }

 public:
  /* Span of the range space, largest stride * dimension */
  KOKKOS_INLINE_FUNCTION
  constexpr size_type span() const { return m_dim.N0; }

  KOKKOS_INLINE_FUNCTION constexpr bool span_is_contiguous() const {
    return false;
  }

  /* Strides of dimensions */
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_0() const { return 1; }

  // Stride with [ rank ] value is the total length
  template <typename iType>
  KOKKOS_INLINE_FUNCTION void stride(iType* const s) const {
    if (0 < dimension_type::rank) {
      s[0] = 1;
    }
    for (int i = 1; i < 8; i++) s[i] = 0;
    s[dimension_type::rank] = span();
  }

  //----------------------------------------
  ViewOffset()                  = default;
  ViewOffset(const ViewOffset&) = default;
  ViewOffset& operator=(const ViewOffset&) = default;

  ViewOffset(std::integral_constant<unsigned, 0> const&,
             Kokkos::LayoutSelective const& rhs)
      : m_dim(rhs.list_size, 0, 0, 0, 0, 0, 0, 0), m_selective(rhs) {}
};

}  // namespace Impl
}  // namespace Kokkos

namespace Test {

class InnerClass {
 public:
  long data[100];

  KOKKOS_INLINE_FUNCTION
  InnerClass() {
    for (int i = 0; i < 100; i++) {
      data[i] = (long)i;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void update(long d) {
    for (int i = 0; i < 100; i++) {
      data[i] += d;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void set(long d) {
    for (int i = 0; i < 100; i++) {
      data[i] = d;
    }
  }
};

template <class ExecutionSpace>
struct TestLayout {
  using Layout    = Kokkos::LayoutRight;
  using SubLayout = Kokkos::LayoutSelective;

  // Allocate y, x vectors and Matrix A on device.
  using ViewVectorType = Kokkos::View<InnerClass*, Layout>;
  using SubViewVectorType =
      Kokkos::View<InnerClass*, SubLayout, Kokkos::MemoryUnmanaged>;
  const int N = 100;

  void run_test() {
    ViewVectorType a("a", N);
    Kokkos::parallel_for(
        N, KOKKOS_LAMBDA(const int i) { a(i).update((double)i); });
    size_t offsets[] = {20, 40};
    SubLayout sl(offsets, 2);
    SubViewVectorType b(a.data(), sl);
    Kokkos::parallel_for(
        2, KOKKOS_LAMBDA(const int i) { b(i).set(20 * (i + 1)); });

    auto a_h = Kokkos::create_mirror_view(a);
    Kokkos::deep_copy(a_h, a);
    ASSERT_EQ(a_h(20).data[0], 20);
    ASSERT_EQ(a_h(40).data[0], 40);
  }
};

TEST(TEST_CATEGORY, view_irregular_layout) {
  TestLayout<TEST_EXECSPACE> tl;
  tl.run_test();
}

}  // namespace Test
