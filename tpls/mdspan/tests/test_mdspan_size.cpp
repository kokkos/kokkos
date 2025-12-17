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
#include <mdspan/mdspan.hpp>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

namespace {


template<class Extents>
std::size_t product_of_extents(const Extents& e)
{
  size_t prod = 1;
  for(std::size_t r = 0; r < e.rank(); ++r) {
    prod *= static_cast<std::size_t>(e.extent(r));
  }
  return prod;
}

template<class Extents>
void test_mdspan_size(std::vector<char>& storage, Extents&& e)
{
  using extents_type = std::remove_cv_t<std::remove_reference_t<Extents>>;
  using mdspan_t = Kokkos::mdspan<char, extents_type>;
  using size_type = typename mdspan_t::size_type;
  const size_type min_storage_size = static_cast<size_type>(product_of_extents(e));
  if(storage.size() < min_storage_size) {
    storage.resize(min_storage_size);
  }
  mdspan_t m(storage.data(), std::forward<Extents>(e));

  static_assert(std::is_same<decltype(m.size()), size_type>::value,
		"The return type of mdspan::size() must be size_type.");

  // m.size() must not overflow, as long as the product of extents
  // is representable as a value of type size_type.
  ASSERT_EQ( min_storage_size, m.size());
}

TEST(TestMdspan, MdspanSizeReturnTypeAndPrecondition)
{
  (void) test_info_;
  std::vector<char> storage;

  static_assert(std::numeric_limits<std::int8_t>::max() == 127, "max int8_t != 127");
  test_mdspan_size(storage, Kokkos::extents<std::int8_t, 12, 11>{}); // 12 * 11 == 132
}

} // namespace (anonymous)
