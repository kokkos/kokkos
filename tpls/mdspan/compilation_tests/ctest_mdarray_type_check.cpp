//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER
#include "ctest_common.hpp"

#include <experimental/mdarray>
#include <complex>

#if defined(MDSPAN_IMPL_USE_CONCEPTS) && MDSPAN_HAS_CXX_20
#include <concepts>
#endif

namespace KokkosEx = MDSPAN_IMPL_STANDARD_NAMESPACE::MDSPAN_IMPL_PROPOSED_NAMESPACE;

using E1 = Kokkos::extents<int32_t, Kokkos::dynamic_extent, 3>;
using M1 = KokkosEx::mdarray<float, E1, Kokkos::layout_left>;

#if defined(MDSPAN_IMPL_USE_CONCEPTS) && MDSPAN_HAS_CXX_20
template<class T>
constexpr bool is_copyable = std::copyable<T>;
#else
template<class T>
constexpr bool is_copyable =
  std::is_assignable<T,T>::value &&
  std::is_move_assignable<T>::value &&
  std::is_copy_constructible<T>::value &&
  std::is_move_constructible<T>::value;
#endif

MDSPAN_STATIC_TEST(
  std::is_same<typename M1::element_type, float>::value &&
  std::is_same<typename M1::index_type, int32_t>::value &&
  std::is_same<typename M1::size_type, uint32_t>::value &&
  std::is_same<typename M1::rank_type, size_t>::value &&
  std::is_same<typename M1::layout_type, Kokkos::layout_left>::value &&
  std::is_same<typename M1::container_type, std::vector<float>>::value &&
  std::is_same<decltype(M1::rank()), typename M1::rank_type>::value &&
  std::is_same<decltype(M1::rank_dynamic()), typename M1::rank_type>::value &&
  std::is_same<decltype(M1::static_extent(0)), size_t>::value &&
  std::is_same<decltype(M1::static_extent(1)), size_t>::value &&
  std::is_same<decltype(std::declval<M1>().extent(0)), typename E1::index_type>::value &&
  std::is_same<decltype(std::declval<M1>().extent(1)), typename E1::index_type>::value &&
  (M1::rank()==2) &&
  (M1::rank_dynamic()==1) &&
  (M1::static_extent(0) == Kokkos::dynamic_extent) &&
  (M1::static_extent(1) == 3) &&
  is_copyable<M1> &&
  std::is_default_constructible<M1>::value
);

using E2 = Kokkos::extents<int64_t, Kokkos::dynamic_extent, 3, Kokkos::dynamic_extent>;
using M2 = KokkosEx::mdarray<std::complex<double>, E2, Kokkos::layout_right, std::array<std::complex<double>,24>>;

MDSPAN_STATIC_TEST(
  std::is_same<typename M2::element_type, std::complex<double>>::value &&
  std::is_same<typename M2::index_type, int64_t>::value &&
  std::is_same<typename M2::size_type, uint64_t>::value &&
  std::is_same<typename M2::rank_type, size_t>::value &&
  std::is_same<typename M2::layout_type, Kokkos::layout_right>::value &&
  std::is_same<typename M2::container_type, std::array<std::complex<double>,24>>::value &&
  std::is_same<decltype(M2::rank()), typename E2::rank_type>::value &&
  std::is_same<decltype(M2::rank_dynamic()), typename E2::rank_type>::value &&
  std::is_same<decltype(M2::static_extent(0)), size_t>::value &&
  std::is_same<decltype(M2::static_extent(1)), size_t>::value &&
  std::is_same<decltype(M2::static_extent(2)), size_t>::value &&
  std::is_same<decltype(std::declval<M2>().extent(0)), typename E2::index_type>::value &&
  std::is_same<decltype(std::declval<M2>().extent(1)), typename E2::index_type>::value &&
  std::is_same<decltype(std::declval<M2>().extent(2)), typename E2::index_type>::value &&
  (M2::rank()==3) &&
  (M2::rank_dynamic()==2) &&
  (M2::static_extent(0) == Kokkos::dynamic_extent) &&
  (M2::static_extent(1) == 3) &&
  (M2::static_extent(2) == Kokkos::dynamic_extent) &&
  is_copyable<M2> &&
  std::is_default_constructible<M2>::value
);

using E3 = Kokkos::extents<uint32_t, Kokkos::dynamic_extent, 3, Kokkos::dynamic_extent>;
using M3 = KokkosEx::mdarray<uint32_t, E3>;

MDSPAN_STATIC_TEST(
  std::is_same<typename M3::element_type, uint32_t>::value &&
  std::is_same<typename M3::index_type, uint32_t>::value &&
  std::is_same<typename M3::size_type, uint32_t>::value &&
  std::is_same<typename M3::rank_type, size_t>::value &&
  std::is_same<typename M3::layout_type, Kokkos::layout_right>::value &&
  std::is_same<typename M3::container_type, std::vector<uint32_t>>::value &&
  std::is_same<decltype(M3::rank()), typename E3::rank_type>::value &&
  std::is_same<decltype(M3::rank_dynamic()), typename E3::rank_type>::value &&
  std::is_same<decltype(M3::static_extent(0)), size_t>::value &&
  std::is_same<decltype(M3::static_extent(1)), size_t>::value &&
  std::is_same<decltype(M3::static_extent(2)), size_t>::value &&
  std::is_same<decltype(std::declval<M3>().extent(0)), typename E3::index_type>::value &&
  std::is_same<decltype(std::declval<M3>().extent(1)), typename E3::index_type>::value &&
  std::is_same<decltype(std::declval<M3>().extent(2)), typename E3::index_type>::value &&
  (M3::rank()==3) &&
  (M3::rank_dynamic()==2) &&
  (M3::static_extent(0) == Kokkos::dynamic_extent) &&
  (M3::static_extent(1) == 3) &&
  (M3::static_extent(2) == Kokkos::dynamic_extent) &&
  is_copyable<M3> &&
  std::is_default_constructible<M3>::value
);
