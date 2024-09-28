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

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

/*
template <class ElementType, class MemorySpace>
class ReferenceCountedDataHandle {
 public:
  using value_type   = ElementType;
  using pointer      = value_type*;
  using reference    = value_type&;
  using memory_space = MemorySpace;
  ...
}
*/

namespace {
using element_t = float;
using mem_t     = typename TEST_EXECSPACE::memory_space;
using data_handle_t =
    Kokkos::Impl::ReferenceCountedDataHandle<element_t, mem_t>;
using const_data_handle_t =
    Kokkos::Impl::ReferenceCountedDataHandle<const element_t, mem_t>;
;
}  // namespace

TEST(TEST_CATEGORY, RefCountedDataHandle_Typedefs) {
  Kokkos::parallel_for(
      Kokkos::RangePolicy<TEST_EXECSPACE>(0, 1), KOKKOS_LAMBDA(int) {
        static_assert(
            std::is_same_v<typename data_handle_t::value_type, element_t>);
        static_assert(
            std::is_same_v<typename data_handle_t::pointer, element_t*>);
        static_assert(
            std::is_same_v<typename data_handle_t::reference, element_t&>);
        static_assert(
            std::is_same_v<typename data_handle_t::memory_space, mem_t>);
      });
}

TEST(TEST_CATEGORY, RefCountedDataHandle) {
  auto shared_alloc =
      Kokkos::Impl::make_shared_allocation_record<element_t, mem_t,
                                                  TEST_EXECSPACE>(
          100, "Test", mem_t(), nullptr,
          std::integral_constant<bool, false>(),   // padding
          std::integral_constant<bool, true>(),    // init
          std::integral_constant<bool, false>());  // sequential_host_init

  element_t* ptr         = static_cast<element_t*>(shared_alloc->data());
  const element_t* c_ptr = ptr;
  data_handle_t dh(shared_alloc);
  ASSERT_EQ(dh.use_count(), 1);
  ASSERT_EQ(dh.get_label(), std::string("Test"));
  ASSERT_EQ(dh.get(), ptr);
  ASSERT_EQ(dh.has_record(), true);
  {
    element_t* ptr_tmp(dh);
    ASSERT_EQ(ptr_tmp, ptr);
    static_assert(!std::is_convertible_v<data_handle_t, element_t*>);
  }
  {
    const_data_handle_t c_dh(dh);
    ASSERT_EQ(dh.use_count(), 2);
    ASSERT_EQ(c_dh.use_count(), 2);
  }
  ASSERT_EQ(dh.use_count(), 1);

  data_handle_t um_dh(ptr);
  ASSERT_EQ(um_dh.get(), ptr);
  ASSERT_EQ(um_dh.has_record(), false);

  data_handle_t dh_offset(dh, ptr + 5);
  ASSERT_EQ(dh_offset.use_count(), 2);
  ASSERT_EQ(dh_offset.get(), ptr + 5);
  ASSERT_EQ(dh_offset.get_label(), std::string("Test"));
  ASSERT_EQ(dh_offset.has_record(), true);
  {
    element_t* ptr_tmp(dh_offset);
    ASSERT_EQ(ptr_tmp, ptr + 5);
  }
  Kokkos::View<int, TEST_EXECSPACE> errors("Errors");
  Kokkos::parallel_for(Kokkos::RangePolicy<TEST_EXECSPACE>(0, 1), KOKKOS_LAMBDA(int) {
    // default ctor and non-const to const
    {
      data_handle_t dh2(dh);
      if(dh2.get() != ptr) errors() += 1;
      const_data_handle_t c_dh2(dh);
      const_data_handle_t c_dh3(c_dh2);
      static_assert(!std::is_constructible_v<data_handle_t, const_data_handle_t>);
}
// ctor from pointer
{
  data_handle_t dh2(ptr);
  if (dh2.get() != ptr) errors() += 2;
  const_data_handle_t c_dh1(ptr);
  if (c_dh1.get() != ptr) errors() += 4;
  const_data_handle_t c_dh2(c_ptr);
  if (c_dh2.get() != ptr) errors() += 8;
  static_assert(!std::is_constructible_v<data_handle_t, decltype(c_ptr)>);
}
// ctor for subviews
{
  data_handle_t dh2(dh, ptr + 5);
  if (dh2.get() != ptr + 5) errors() += 16;
}
});
int h_errors = 0;
Kokkos::deep_copy(h_errors, errors);
ASSERT_FALSE(h_errors & 1);
ASSERT_FALSE(h_errors & 2);
ASSERT_FALSE(h_errors & 4);
ASSERT_FALSE(h_errors & 8);
ASSERT_FALSE(h_errors & 16);
}
