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

#ifndef KOKKOS_ALGORITHMS_UNITTESTS_TESTSORT_HPP
#define KOKKOS_ALGORITHMS_UNITTESTS_TESTSORT_HPP

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_DynamicView.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_Sort.hpp>
#include <unordered_set>

namespace Test {

namespace Impl {

template <class ExecutionSpace, class Scalar>
struct is_sorted_struct {
  using value_type      = unsigned int;
  using execution_space = ExecutionSpace;

  Kokkos::View<Scalar*, ExecutionSpace> keys;

  is_sorted_struct(Kokkos::View<Scalar*, ExecutionSpace> keys_) : keys(keys_) {}
  KOKKOS_INLINE_FUNCTION
  void operator()(int i, unsigned int& count) const {
    if (keys(i) > keys(i + 1)) count++;
  }
};

template <class ExecutionSpace, class Scalar>
struct sum {
  using value_type      = double;
  using execution_space = ExecutionSpace;

  Kokkos::View<Scalar*, ExecutionSpace> keys;

  sum(Kokkos::View<Scalar*, ExecutionSpace> keys_) : keys(keys_) {}
  KOKKOS_INLINE_FUNCTION
  void operator()(int i, double& count) const { count += keys(i); }
};

template <class ExecutionSpace, class Scalar>
struct bin3d_is_sorted_struct {
  using value_type      = unsigned int;
  using execution_space = ExecutionSpace;

  Kokkos::View<Scalar * [3], ExecutionSpace> keys;

  int max_bins;
  Scalar min;
  Scalar max;

  bin3d_is_sorted_struct(Kokkos::View<Scalar * [3], ExecutionSpace> keys_,
                         int max_bins_, Scalar min_, Scalar max_)
      : keys(keys_), max_bins(max_bins_), min(min_), max(max_) {}
  KOKKOS_INLINE_FUNCTION
  void operator()(int i, unsigned int& count) const {
    int ix1 = int((keys(i, 0) - min) / max * max_bins);
    int iy1 = int((keys(i, 1) - min) / max * max_bins);
    int iz1 = int((keys(i, 2) - min) / max * max_bins);
    int ix2 = int((keys(i + 1, 0) - min) / max * max_bins);
    int iy2 = int((keys(i + 1, 1) - min) / max * max_bins);
    int iz2 = int((keys(i + 1, 2) - min) / max * max_bins);

    if (ix1 > ix2)
      count++;
    else if (ix1 == ix2) {
      if (iy1 > iy2)
        count++;
      else if ((iy1 == iy2) && (iz1 > iz2))
        count++;
    }
  }
};

template <class ExecutionSpace, class Scalar>
struct sum3D {
  using value_type      = double;
  using execution_space = ExecutionSpace;

  Kokkos::View<Scalar * [3], ExecutionSpace> keys;

  sum3D(Kokkos::View<Scalar * [3], ExecutionSpace> keys_) : keys(keys_) {}
  KOKKOS_INLINE_FUNCTION
  void operator()(int i, double& count) const {
    count += keys(i, 0);
    count += keys(i, 1);
    count += keys(i, 2);
  }
};

template <class ExecutionSpace, typename KeyType>
void test_1D_sort_impl(unsigned int n, bool force_kokkos) {
  using KeyViewType = Kokkos::View<KeyType*, ExecutionSpace>;
  KeyViewType keys("Keys", n);

  // Test sorting array with all numbers equal
  ExecutionSpace exec;
  Kokkos::deep_copy(exec, keys, KeyType(1));
  Kokkos::sort(exec, keys, force_kokkos);

  Kokkos::Random_XorShift64_Pool<ExecutionSpace> g(1931);
  Kokkos::fill_random(keys, g,
                      Kokkos::Random_XorShift64_Pool<
                          ExecutionSpace>::generator_type::MAX_URAND);

  double sum_before       = 0.0;
  double sum_after        = 0.0;
  unsigned int sort_fails = 0;

  Kokkos::parallel_reduce(Kokkos::RangePolicy<ExecutionSpace>(exec, 0, n),
                          sum<ExecutionSpace, KeyType>(keys), sum_before);

  Kokkos::sort(exec, keys, force_kokkos);

  Kokkos::parallel_reduce(Kokkos::RangePolicy<ExecutionSpace>(exec, 0, n),
                          sum<ExecutionSpace, KeyType>(keys), sum_after);
  Kokkos::parallel_reduce(Kokkos::RangePolicy<ExecutionSpace>(exec, 0, n - 1),
                          is_sorted_struct<ExecutionSpace, KeyType>(keys),
                          sort_fails);

  double ratio   = sum_before / sum_after;
  double epsilon = 1e-10;
  unsigned int equal_sum =
      (ratio > (1.0 - epsilon)) && (ratio < (1.0 + epsilon)) ? 1 : 0;

  ASSERT_EQ(sort_fails, 0u);
  ASSERT_EQ(equal_sum, 1u);
}

template <class ExecutionSpace, typename KeyType>
void test_3D_sort_impl(unsigned int n) {
  using KeyViewType = Kokkos::View<KeyType * [3], ExecutionSpace>;

  KeyViewType keys("Keys", n * n * n);

  Kokkos::Random_XorShift64_Pool<ExecutionSpace> g(1931);
  Kokkos::fill_random(keys, g, 100.0);

  double sum_before       = 0.0;
  double sum_after        = 0.0;
  unsigned int sort_fails = 0;

  ExecutionSpace exec;
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<ExecutionSpace>(exec, 0, keys.extent(0)),
      sum3D<ExecutionSpace, KeyType>(keys), sum_before);

  int bin_1d = 1;
  while (bin_1d * bin_1d * bin_1d * 4 < (int)keys.extent(0)) bin_1d *= 2;
  int bin_max[3]                          = {bin_1d, bin_1d, bin_1d};
  typename KeyViewType::value_type min[3] = {0, 0, 0};
  typename KeyViewType::value_type max[3] = {100, 100, 100};

  using BinOp = Kokkos::BinOp3D<KeyViewType>;
  BinOp bin_op(bin_max, min, max);
  Kokkos::BinSort<KeyViewType, BinOp> Sorter(keys, bin_op, false);
  Sorter.create_permute_vector(exec);
  Sorter.sort(exec, keys);

  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<ExecutionSpace>(exec, 0, keys.extent(0)),
      sum3D<ExecutionSpace, KeyType>(keys), sum_after);
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<ExecutionSpace>(exec, 0, keys.extent(0) - 1),
      bin3d_is_sorted_struct<ExecutionSpace, KeyType>(keys, bin_1d, min[0],
                                                      max[0]),
      sort_fails);

  double ratio   = sum_before / sum_after;
  double epsilon = 1e-10;
  unsigned int equal_sum =
      (ratio > (1.0 - epsilon)) && (ratio < (1.0 + epsilon)) ? 1 : 0;

  if (sort_fails)
    printf("3D Sort Sum: %f %f Fails: %u\n", sum_before, sum_after, sort_fails);

  ASSERT_EQ(sort_fails, 0u);
  ASSERT_EQ(equal_sum, 1u);
}

//----------------------------------------------------------------------------

template <class ExecutionSpace, typename KeyType>
void test_dynamic_view_sort_impl(unsigned int n) {
  using KeyDynamicViewType =
      Kokkos::Experimental::DynamicView<KeyType*, ExecutionSpace>;
  using KeyViewType = Kokkos::View<KeyType*, ExecutionSpace>;

  const size_t upper_bound    = 2 * n;
  const size_t min_chunk_size = 1024;

  KeyDynamicViewType keys("Keys", min_chunk_size, upper_bound);

  keys.resize_serial(n);

  KeyViewType keys_view("KeysTmp", n);

  // Test sorting array with all numbers equal
  ExecutionSpace exec;
  Kokkos::deep_copy(exec, keys_view, KeyType(1));
  Kokkos::deep_copy(keys, keys_view);
  Kokkos::sort(exec, keys, 0 /* begin */, n /* end */);

  Kokkos::Random_XorShift64_Pool<ExecutionSpace> g(1931);
  Kokkos::fill_random(keys_view, g,
                      Kokkos::Random_XorShift64_Pool<
                          ExecutionSpace>::generator_type::MAX_URAND);

  exec.fence();
  Kokkos::deep_copy(keys, keys_view);

  double sum_before       = 0.0;
  double sum_after        = 0.0;
  unsigned int sort_fails = 0;

  Kokkos::parallel_reduce(Kokkos::RangePolicy<ExecutionSpace>(exec, 0, n),
                          sum<ExecutionSpace, KeyType>(keys_view), sum_before);

  Kokkos::sort(exec, keys, 0 /* begin */, n /* end */);

  exec.fence();  // Need this fence to prevent BusError with Cuda
  Kokkos::deep_copy(keys_view, keys);

  Kokkos::parallel_reduce(Kokkos::RangePolicy<ExecutionSpace>(exec, 0, n),
                          sum<ExecutionSpace, KeyType>(keys_view), sum_after);
  Kokkos::parallel_reduce(Kokkos::RangePolicy<ExecutionSpace>(exec, 0, n - 1),
                          is_sorted_struct<ExecutionSpace, KeyType>(keys_view),
                          sort_fails);

  double ratio   = sum_before / sum_after;
  double epsilon = 1e-10;
  unsigned int equal_sum =
      (ratio > (1.0 - epsilon)) && (ratio < (1.0 + epsilon)) ? 1 : 0;

  if (sort_fails != 0 || equal_sum != 1) {
    std::cout << " N = " << n << " ; sum_before = " << sum_before
              << " ; sum_after = " << sum_after << " ; ratio = " << ratio
              << std::endl;
  }

  ASSERT_EQ(sort_fails, 0u);
  ASSERT_EQ(equal_sum, 1u);
}

//----------------------------------------------------------------------------

template <class ExecutionSpace>
void test_issue_1160_impl() {
  Kokkos::View<int*, ExecutionSpace> element_("element", 10);
  Kokkos::View<double*, ExecutionSpace> x_("x", 10);
  Kokkos::View<double*, ExecutionSpace> v_("y", 10);

  auto h_element = Kokkos::create_mirror_view(element_);
  auto h_x       = Kokkos::create_mirror_view(x_);
  auto h_v       = Kokkos::create_mirror_view(v_);

  h_element(0) = 9;
  h_element(1) = 8;
  h_element(2) = 7;
  h_element(3) = 6;
  h_element(4) = 5;
  h_element(5) = 4;
  h_element(6) = 3;
  h_element(7) = 2;
  h_element(8) = 1;
  h_element(9) = 0;

  for (int i = 0; i < 10; ++i) {
    h_v.access(i, 0) = h_x.access(i, 0) = double(h_element(i));
  }
  ExecutionSpace exec;
  Kokkos::deep_copy(exec, element_, h_element);
  Kokkos::deep_copy(exec, x_, h_x);
  Kokkos::deep_copy(exec, v_, h_v);

  using KeyViewType = decltype(element_);
  using BinOp       = Kokkos::BinOp1D<KeyViewType>;

  int begin = 3;
  int end   = 8;
  auto max  = h_element(begin);
  auto min  = h_element(end - 1);
  BinOp binner(end - begin, min, max);

  Kokkos::BinSort<KeyViewType, BinOp> Sorter(element_, begin, end, binner,
                                             false);
  Sorter.create_permute_vector(exec);
  Sorter.sort(exec, element_, begin, end);

  Sorter.sort(exec, x_, begin, end);
  Sorter.sort(exec, v_, begin, end);

  Kokkos::deep_copy(exec, h_element, element_);
  Kokkos::deep_copy(exec, h_x, x_);
  Kokkos::deep_copy(exec, h_v, v_);
  exec.fence();

  ASSERT_EQ(h_element(0), 9);
  ASSERT_EQ(h_element(1), 8);
  ASSERT_EQ(h_element(2), 7);
  ASSERT_EQ(h_element(3), 2);
  ASSERT_EQ(h_element(4), 3);
  ASSERT_EQ(h_element(5), 4);
  ASSERT_EQ(h_element(6), 5);
  ASSERT_EQ(h_element(7), 6);
  ASSERT_EQ(h_element(8), 1);
  ASSERT_EQ(h_element(9), 0);

  for (int i = 0; i < 10; ++i) {
    ASSERT_EQ(h_element(i), int(h_x.access(i, 0)));
    ASSERT_EQ(h_element(i), int(h_v.access(i, 0)));
  }
}

template <class ExecutionSpace>
void test_issue_4978_impl() {
  Kokkos::View<long long*, ExecutionSpace> element_("element", 9);

  auto h_element = Kokkos::create_mirror_view(element_);

  h_element(0) = LLONG_MIN;
  h_element(1) = 0;
  h_element(2) = 3;
  h_element(3) = 2;
  h_element(4) = 1;
  h_element(5) = 3;
  h_element(6) = 6;
  h_element(7) = 4;
  h_element(8) = 3;

  ExecutionSpace exec;
  Kokkos::deep_copy(exec, element_, h_element);

  Kokkos::sort(exec, element_);

  Kokkos::deep_copy(exec, h_element, element_);
  exec.fence();

  ASSERT_EQ(h_element(0), LLONG_MIN);
  ASSERT_EQ(h_element(1), 0);
  ASSERT_EQ(h_element(2), 1);
  ASSERT_EQ(h_element(3), 2);
  ASSERT_EQ(h_element(4), 3);
  ASSERT_EQ(h_element(5), 3);
  ASSERT_EQ(h_element(6), 3);
  ASSERT_EQ(h_element(7), 4);
  ASSERT_EQ(h_element(8), 6);
}

template <class ExecutionSpace, class T>
void test_sort_integer_overflow() {
  // array with two extrema in reverse order to expose integer overflow bug in
  // bin calculation
  T a[2]  = {Kokkos::Experimental::finite_max<T>::value,
            Kokkos::Experimental::finite_min<T>::value};
  auto vd = Kokkos::create_mirror_view_and_copy(
      ExecutionSpace(), Kokkos::View<T[2], Kokkos::HostSpace>(a));
  Kokkos::sort(vd, /*force using Kokkos bin sort*/ true);
  auto vh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vd);
  EXPECT_TRUE(std::is_sorted(vh.data(), vh.data() + 2))
      << "view (" << vh[0] << ", " << vh[1] << ") is not sorted";
}

// Comparator for sorting in descending order
template <typename Key>
struct GreaterThan {
  KOKKOS_FUNCTION constexpr bool operator()(const Key& lhs,
                                            const Key& rhs) const {
    return lhs > rhs;
  }
};

// Functor to test sort_team: each team responsible for sorting one array
template <typename ExecSpace, typename KeyViewType, typename OffsetViewType>
struct TeamSortFunctor {
  using TeamMem  = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  using SizeType = typename KeyViewType::size_type;
  using KeyType  = typename KeyViewType::non_const_value_type;
  TeamSortFunctor(const KeyViewType& keys_, const OffsetViewType& offsets_,
                  bool sortDescending_)
      : keys(keys_), offsets(offsets_), sortDescending(sortDescending_) {}
  KOKKOS_INLINE_FUNCTION void operator()(const TeamMem& t) const {
    int i          = t.league_rank();
    SizeType begin = offsets(i);
    SizeType end   = offsets(i + 1);
    if (sortDescending)
      Kokkos::Experimental::sort_team(
          t, Kokkos::subview(keys, Kokkos::make_pair(begin, end)),
          GreaterThan<KeyType>());
    else
      Kokkos::Experimental::sort_team(
          t, Kokkos::subview(keys, Kokkos::make_pair(begin, end)));
  }
  KeyViewType keys;
  OffsetViewType offsets;
  bool sortDescending;
};

// Functor to test sort_by_key_team: each team responsible for sorting one array
template <typename ExecSpace, typename KeyViewType, typename ValueViewType,
          typename OffsetViewType>
struct TeamSortByKeyFunctor {
  using TeamMem  = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  using SizeType = typename KeyViewType::size_type;
  using KeyType  = typename KeyViewType::non_const_value_type;
  TeamSortByKeyFunctor(const KeyViewType& keys_, const ValueViewType& values_,
                       const OffsetViewType& offsets_, bool sortDescending_)
      : keys(keys_),
        values(values_),
        offsets(offsets_),
        sortDescending(sortDescending_) {}
  KOKKOS_INLINE_FUNCTION void operator()(const TeamMem& t) const {
    int i          = t.league_rank();
    SizeType begin = offsets(i);
    SizeType end   = offsets(i + 1);
    if (sortDescending) {
      Kokkos::Experimental::sort_by_key_team(
          t, Kokkos::subview(keys, Kokkos::make_pair(begin, end)),
          Kokkos::subview(values, Kokkos::make_pair(begin, end)),
          GreaterThan<KeyType>());
    } else {
      Kokkos::Experimental::sort_by_key_team(
          t, Kokkos::subview(keys, Kokkos::make_pair(begin, end)),
          Kokkos::subview(values, Kokkos::make_pair(begin, end)));
    }
  }
  KeyViewType keys;
  ValueViewType values;
  OffsetViewType offsets;
  bool sortDescending;
};

// Functor to test sort_thread: each thread (multiple vector lanes) responsible
// for sorting one array
template <typename ExecSpace, typename KeyViewType, typename OffsetViewType>
struct ThreadSortFunctor {
  using TeamMem  = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  using SizeType = typename KeyViewType::size_type;
  using KeyType  = typename KeyViewType::non_const_value_type;
  ThreadSortFunctor(const KeyViewType& keys_, const OffsetViewType& offsets_,
                    bool sortDescending_)
      : keys(keys_), offsets(offsets_), sortDescending(sortDescending_) {}
  KOKKOS_INLINE_FUNCTION void operator()(const TeamMem& t) const {
    int i = t.league_rank() * t.team_size() + t.team_rank();
    // Number of arrays to sort doesn't have to be divisible by team size, so
    // some threads may be idle.
    if (i < offsets.extent_int(0) - 1) {
      SizeType begin = offsets(i);
      SizeType end   = offsets(i + 1);
      if (sortDescending)
        Kokkos::Experimental::sort_thread(
            t, Kokkos::subview(keys, Kokkos::make_pair(begin, end)),
            GreaterThan<KeyType>());
      else
        Kokkos::Experimental::sort_thread(
            t, Kokkos::subview(keys, Kokkos::make_pair(begin, end)));
    }
  }
  KeyViewType keys;
  OffsetViewType offsets;
  bool sortDescending;
};

// Functor to test sort_by_key_thread
template <typename ExecSpace, typename KeyViewType, typename ValueViewType,
          typename OffsetViewType>
struct ThreadSortByKeyFunctor {
  using TeamMem  = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  using SizeType = typename KeyViewType::size_type;
  using KeyType  = typename KeyViewType::non_const_value_type;
  ThreadSortByKeyFunctor(const KeyViewType& keys_, const ValueViewType& values_,
                         const OffsetViewType& offsets_, bool sortDescending_)
      : keys(keys_),
        values(values_),
        offsets(offsets_),
        sortDescending(sortDescending_) {}
  KOKKOS_INLINE_FUNCTION void operator()(const TeamMem& t) const {
    int i = t.league_rank() * t.team_size() + t.team_rank();
    // Number of arrays to sort doesn't have to be divisible by team size, so
    // some threads may be idle.
    if (i < offsets.extent_int(0) - 1) {
      SizeType begin = offsets(i);
      SizeType end   = offsets(i + 1);
      if (sortDescending) {
        Kokkos::Experimental::sort_by_key_thread(
            t, Kokkos::subview(keys, Kokkos::make_pair(begin, end)),
            Kokkos::subview(values, Kokkos::make_pair(begin, end)),
            GreaterThan<KeyType>());
      } else {
        Kokkos::Experimental::sort_by_key_thread(
            t, Kokkos::subview(keys, Kokkos::make_pair(begin, end)),
            Kokkos::subview(values, Kokkos::make_pair(begin, end)));
      }
    }
  }
  KeyViewType keys;
  ValueViewType values;
  OffsetViewType offsets;
  bool sortDescending;
};

// Randomly generate n arrays of up to k elements each, packed into a single 1D
// view. Like a row in a StaticCrsGraph, array i begins at offsets(i)
// (inclusive) and ends at offsets(i+1) (exclusive)
template <typename KeyType, typename KeyViewType, typename OffsetViewType>
void randomPackedArrays(unsigned n, unsigned k, KeyViewType& keys,
                        OffsetViewType& offsets, KeyType minKey,
                        KeyType maxKey) {
  offsets          = OffsetViewType("Offsets", n + 1);
  auto offsetsHost = Kokkos::create_mirror_view(Kokkos::HostSpace(), offsets);
  Kokkos::Random_XorShift64_Pool<Kokkos::DefaultHostExecutionSpace> gHost(1931);
  Kokkos::fill_random(offsetsHost, gHost, k);
  Kokkos::DefaultHostExecutionSpace().fence();
  offsetsHost(0) = 0;
  // Prefix-sum
  for (unsigned i = 1; i <= n; i++) {
    offsetsHost(i) += offsetsHost(i - 1);
  }
  Kokkos::deep_copy(offsets, offsetsHost);
  unsigned totalLength = offsetsHost(n);
  keys                 = KeyViewType("Keys", totalLength);
  Kokkos::Random_XorShift64_Pool<typename KeyViewType::execution_space> g(1931);
  Kokkos::fill_random(keys, g, minKey, maxKey);
}

// Same as above, but also generate values array with same length as keys
template <typename KeyType, typename KeyViewType, typename ValueType,
          typename ValueViewType, typename OffsetViewType>
void randomPackedArrays(unsigned n, unsigned k, KeyViewType& keys,
                        ValueViewType& values, OffsetViewType& offsets,
                        KeyType minKey, KeyType maxKey, ValueType minVal,
                        ValueType maxVal) {
  offsets          = OffsetViewType("Offsets", n + 1);
  auto offsetsHost = Kokkos::create_mirror_view(Kokkos::HostSpace(), offsets);
  Kokkos::Random_XorShift64_Pool<Kokkos::DefaultHostExecutionSpace> gHost(1931);
  Kokkos::fill_random(offsetsHost, gHost, k);
  Kokkos::DefaultHostExecutionSpace().fence();
  offsetsHost(0) = 0;
  // Prefix-sum
  for (unsigned i = 1; i <= n; i++) {
    offsetsHost(i) += offsetsHost(i - 1);
  }
  Kokkos::deep_copy(offsets, offsetsHost);
  unsigned totalLength = offsetsHost(n);
  keys                 = KeyViewType("Keys", totalLength);
  values               = ValueViewType("Values", totalLength);
  Kokkos::Random_XorShift64_Pool<typename KeyViewType::execution_space> g(1931);
  Kokkos::fill_random(keys, g, minKey, maxKey);
  Kokkos::fill_random(values, g, minVal, maxVal);
}

template <class ExecutionSpace, typename KeyType>
void test_nested_sort_impl(unsigned narray, unsigned n, bool useTeams,
                           bool customCompare, KeyType minKey, KeyType maxKey) {
  using KeyViewType    = Kokkos::View<KeyType*, ExecutionSpace>;
  using OffsetViewType = Kokkos::View<unsigned*, ExecutionSpace>;
  using TeamPol        = Kokkos::TeamPolicy<ExecutionSpace>;
  KeyViewType keys;
  OffsetViewType offsets;
  randomPackedArrays<KeyType, KeyViewType, OffsetViewType>(
      narray, n, keys, offsets, minKey, maxKey);
  // note: doing create_mirror because we always want this to be a separate
  // copy, even if keys is already host-accessible. keysHost becomes the correct
  // result to compare against.
  auto keysHost = Kokkos::create_mirror(Kokkos::HostSpace(), keys);
  Kokkos::deep_copy(keysHost, keys);
  auto offsetsHost =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), offsets);
  // Sort the same arrays on host to compare against
  for (unsigned i = 0; i < narray; i++) {
    KeyType* begin = keysHost.data() + offsetsHost(i);
    KeyType* end   = keysHost.data() + offsetsHost(i + 1);
    if (customCompare)
      std::sort(begin, end,
                [](const KeyType& a, const KeyType& b) { return a > b; });
    else
      std::sort(begin, end);
  }
  if (useTeams) {
    int vectorLen = std::min<int>(4, TeamPol::vector_length_max());
    TeamPol policy(narray, Kokkos::AUTO(), vectorLen);
    Kokkos::parallel_for(
        policy, TeamSortFunctor<ExecutionSpace, KeyViewType, OffsetViewType>(
                    keys, offsets, customCompare));
  } else {
    ThreadSortFunctor<ExecutionSpace, KeyViewType, OffsetViewType> functor(
        keys, offsets, customCompare);
    int vectorLen = std::min<int>(4, TeamPol::vector_length_max());
    TeamPol dummy(1, Kokkos::AUTO(), vectorLen);
    int teamSize =
        dummy.team_size_recommended(functor, Kokkos::ParallelForTag());
    int numTeams = (narray + teamSize - 1) / teamSize;
    Kokkos::parallel_for(TeamPol(numTeams, teamSize, vectorLen), functor);
  }
  auto keysOut = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), keys);
  for (unsigned i = 0; i < keys.extent(0); i++) {
    EXPECT_EQ(keysOut(i), keysHost(i));
  }
}

template <class ExecutionSpace, typename KeyType, typename ValueType>
void test_nested_sort_by_key_impl(unsigned narray, unsigned n, bool useTeams,
                                  bool customCompare, KeyType minKey,
                                  KeyType maxKey, ValueType minVal,
                                  ValueType maxVal) {
  using KeyViewType    = Kokkos::View<KeyType*, ExecutionSpace>;
  using ValueViewType  = Kokkos::View<ValueType*, ExecutionSpace>;
  using OffsetViewType = Kokkos::View<unsigned*, ExecutionSpace>;
  using TeamPol        = Kokkos::TeamPolicy<ExecutionSpace>;
  KeyViewType keys;
  ValueViewType values;
  OffsetViewType offsets;
  randomPackedArrays<KeyType, KeyViewType, ValueType, ValueViewType,
                     OffsetViewType>(narray, n, keys, values, offsets, minKey,
                                     maxKey, minVal, maxVal);
  // note: doing create_mirror because we always want this to be a separate
  // copy, even if keys/vals are already host-accessible. keysHost and valsHost
  // becomes the correct result to compare against.
  auto keysHost   = Kokkos::create_mirror(Kokkos::HostSpace(), keys);
  auto valuesHost = Kokkos::create_mirror(Kokkos::HostSpace(), values);
  Kokkos::deep_copy(keysHost, keys);
  Kokkos::deep_copy(valuesHost, values);
  auto offsetsHost =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), offsets);
  // Sort the same arrays on host to compare against
  for (unsigned i = 0; i < narray; i++) {
    // std:: doesn't have a sort_by_key, so sort a vector of key-value pairs
    // instead
    using KV = std::pair<KeyType, ValueType>;
    std::vector<KV> keysAndValues(offsetsHost(i + 1) - offsetsHost(i));
    for (unsigned j = 0; j < keysAndValues.size(); j++) {
      keysAndValues[j].first  = keysHost(offsetsHost(i) + j);
      keysAndValues[j].second = valuesHost(offsetsHost(i) + j);
    }
    if (customCompare) {
      std::sort(keysAndValues.begin(), keysAndValues.end(),
                [](const KV& a, const KV& b) { return a.first > b.first; });
    } else {
      std::sort(keysAndValues.begin(), keysAndValues.end(),
                [](const KV& a, const KV& b) { return a.first < b.first; });
    }
    // Copy back from pairs to views
    for (unsigned j = 0; j < keysAndValues.size(); j++) {
      keysHost(offsetsHost(i) + j)   = keysAndValues[j].first;
      valuesHost(offsetsHost(i) + j) = keysAndValues[j].second;
    }
  }
  if (useTeams) {
    int vectorLen = std::min<int>(4, TeamPol::vector_length_max());
    TeamPol policy(narray, Kokkos::AUTO(), vectorLen);
    Kokkos::parallel_for(
        policy, TeamSortByKeyFunctor<ExecutionSpace, KeyViewType, ValueViewType,
                                     OffsetViewType>(keys, values, offsets,
                                                     customCompare));
  } else {
    ThreadSortByKeyFunctor<ExecutionSpace, KeyViewType, ValueViewType,
                           OffsetViewType>
        functor(keys, values, offsets, customCompare);
    int vectorLen = std::min<int>(4, TeamPol::vector_length_max());
    TeamPol dummy(1, Kokkos::AUTO(), vectorLen);
    int teamSize =
        dummy.team_size_recommended(functor, Kokkos::ParallelForTag());
    int numTeams = (narray + teamSize - 1) / teamSize;
    Kokkos::parallel_for(TeamPol(numTeams, teamSize, vectorLen), functor);
  }
  auto keysOut = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), keys);
  auto valuesOut =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
  // First, compare keys since they will always match exactly
  for (unsigned i = 0; i < keys.extent(0); i++) {
    EXPECT_EQ(keysOut(i), keysHost(i));
  }
  // Kokkos::sort_by_key_X is not stable, so if a key happens to
  // appear more than once, the order of the values may not match exactly.
  // But the set of values for a given key should be identical.
  unsigned keyStart = 0;
  while (keyStart < keys.extent(0)) {
    KeyType key     = keysHost(keyStart);
    unsigned keyEnd = keyStart + 1;
    while (keyEnd < keys.extent(0) && keysHost(keyEnd) == key) keyEnd++;
    std::unordered_multiset<ValueType> correctVals;
    std::unordered_multiset<ValueType> outputVals;
    for (unsigned i = keyStart; i < keyEnd; i++) {
      correctVals.insert(valuesHost(i));
      outputVals.insert(valuesOut(i));
    }
    // Check one value at a time that they match
    for (auto it = correctVals.begin(); it != correctVals.end(); it++) {
      ValueType val = *it;
      EXPECT_TRUE(outputVals.find(val) != outputVals.end());
      EXPECT_EQ(correctVals.count(val), outputVals.count(val));
    }
    keyStart = keyEnd;
  }
}

//----------------------------------------------------------------------------

template <class ExecutionSpace, typename KeyType>
void test_1D_sort(unsigned int N) {
  test_1D_sort_impl<ExecutionSpace, KeyType>(N * N * N, true);
  test_1D_sort_impl<ExecutionSpace, KeyType>(N * N * N, false);
}

template <class ExecutionSpace, typename KeyType>
void test_3D_sort(unsigned int N) {
  test_3D_sort_impl<ExecutionSpace, KeyType>(N);
}

template <class ExecutionSpace, typename KeyType>
void test_dynamic_view_sort(unsigned int N) {
  test_dynamic_view_sort_impl<ExecutionSpace, KeyType>(N * N);
}

template <class ExecutionSpace>
void test_issue_1160_sort() {
  test_issue_1160_impl<ExecutionSpace>();
}

template <class ExecutionSpace>
void test_issue_4978_sort() {
  test_issue_4978_impl<ExecutionSpace>();
}

template <class ExecutionSpace, typename KeyType>
void test_sort(unsigned int N) {
  test_1D_sort<ExecutionSpace, KeyType>(N);
  test_3D_sort<ExecutionSpace, KeyType>(N);
// FIXME_OPENMPTARGET: OpenMPTarget doesn't support DynamicView yet.
#ifndef KOKKOS_ENABLE_OPENMPTARGET
  test_dynamic_view_sort<ExecutionSpace, KeyType>(N);
#endif
  test_issue_1160_sort<ExecutionSpace>();
  test_issue_4978_sort<ExecutionSpace>();
  test_sort_integer_overflow<ExecutionSpace, long long>();
  test_sort_integer_overflow<ExecutionSpace, unsigned long long>();
  test_sort_integer_overflow<ExecutionSpace, int>();
}

template <class ExecutionSpace, typename KeyType>
void test_nested_sort(unsigned int N, KeyType minKey, KeyType maxKey) {
  // 2nd arg: true = team-level, false = thread-level.
  // 3rd arg: true = custom comparator, false = default comparator.
  test_nested_sort_impl<ExecutionSpace, KeyType>(N, N, true, false, minKey,
                                                 maxKey);
  test_nested_sort_impl<ExecutionSpace, KeyType>(N, N, true, true, minKey,
                                                 maxKey);
  test_nested_sort_impl<ExecutionSpace, KeyType>(N, N, false, false, minKey,
                                                 maxKey);
  test_nested_sort_impl<ExecutionSpace, KeyType>(N, N, false, true, minKey,
                                                 maxKey);
}

template <class ExecutionSpace, typename KeyType, typename ValueType>
void test_nested_sort_by_key(unsigned int N, KeyType minKey, KeyType maxKey,
                             ValueType minVal, ValueType maxVal) {
  // 2nd arg: true = team-level, false = thread-level.
  // 3rd arg: true = custom comparator, false = default comparator.
  test_nested_sort_by_key_impl<ExecutionSpace, KeyType, ValueType>(
      N, N, true, false, minKey, maxKey, minVal, maxVal);
  test_nested_sort_by_key_impl<ExecutionSpace, KeyType, ValueType>(
      N, N, true, true, minKey, maxKey, minVal, maxVal);
  test_nested_sort_by_key_impl<ExecutionSpace, KeyType, ValueType>(
      N, N, false, false, minKey, maxKey, minVal, maxVal);
  test_nested_sort_by_key_impl<ExecutionSpace, KeyType, ValueType>(
      N, N, false, true, minKey, maxKey, minVal, maxVal);
}

}  // namespace Impl
}  // namespace Test
#endif /* KOKKOS_ALGORITHMS_UNITTESTS_TESTSORT_HPP */
