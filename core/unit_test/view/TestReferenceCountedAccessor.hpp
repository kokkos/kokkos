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
template <class ElementType, class MemorySpace, class NestedAccessor>
class ReferenceCountedAccessor {
 public:
  using element_type     = ElementType;
  using data_handle_type = ReferenceCountedDataHandle<ElementType, MemorySpace>;
  using reference        = typename NestedAccessor::reference;
  using offset_policy    = ReferenceCountedAccessor;
}
*/

namespace {
using element_t      = float;
using mem_t          = typename TEST_EXECSPACE::memory_space;
using defacc_t       = Kokkos::default_accessor<element_t>;
using const_defacc_t = Kokkos::default_accessor<const element_t>;
using acc_t =
    Kokkos::Impl::ReferenceCountedAccessor<element_t, mem_t, defacc_t>;
using const_acc_t =
    Kokkos::Impl::ReferenceCountedAccessor<const element_t, mem_t,
                                           const_defacc_t>;
using data_handle_t       = typename acc_t::data_handle_type;
using const_data_handle_t = typename const_acc_t::data_handle_type;
}  // namespace

TEST(TEST_CATEGORY, RefCountedAcc_Typedefs) {
  Kokkos::parallel_for(
      Kokkos::RangePolicy<TEST_EXECSPACE>(0, 1), KOKKOS_LAMBDA(int) {
        static_assert(std::is_same_v<typename acc_t::element_type, element_t>);
        static_assert(
            std::is_same_v<
                typename acc_t::data_handle_type,
                Kokkos::Impl::ReferenceCountedDataHandle<element_t, mem_t>>);
        static_assert(std::is_same_v<typename acc_t::reference,
                                     typename defacc_t::reference>);
        static_assert(std::is_same_v<
                      typename acc_t::offset_policy,
                      Kokkos::Impl::ReferenceCountedAccessor<
                          element_t, mem_t, typename defacc_t::offset_policy>>);
      });
}

TEST(TEST_CATEGORY, RefCountedAcc_Ctors) {
  Kokkos::parallel_for(Kokkos::RangePolicy<TEST_EXECSPACE>(0, 1), KOKKOS_LAMBDA(int) {
      // default ctor and non-const to const
      {
        acc_t acc;
        const_acc_t c_acc(acc);
        static_assert(!std::is_constructible_v<acc_t, const_acc_t>);
}
// from default_accessor
{
  defacc_t defacc;
  const_defacc_t c_defacc;
  acc_t acc(defacc);
  const_acc_t c_acc1(defacc);
  const_acc_t c_acc2(c_defacc);
  static_assert(!std::is_constructible_v<acc_t, const_defacc_t>);
}
});
}

TEST(TEST_CATEGORY, RefCountedAcc_ConversionToDefaultAcc) {
  Kokkos::parallel_for(
      Kokkos::RangePolicy<TEST_EXECSPACE>(0, 1), KOKKOS_LAMBDA(int) {
        // default ctor and non-const to const
        acc_t acc;
        const_acc_t c_acc;
        defacc_t defacc(acc);
        const_defacc_t c_defacc1(acc);
        const_defacc_t c_defacc2(c_acc);
        static_assert(!std::is_constructible_v<defacc_t, const_acc_t>);
      });
}

TEST(TEST_CATEGORY, RefCountedAcc_Access) {
  element_t* ptr =
      (element_t*)Kokkos::kokkos_malloc<TEST_EXECSPACE::memory_space>(
          100 * sizeof(element_t));
  // Gonna use unmanaged data handles here (i.e. not actually referfence
  // counted)
  data_handle_t dh(ptr);
  const_data_handle_t cdh(ptr);

  Kokkos::View<int, TEST_EXECSPACE> errors("Errors");
  Kokkos::parallel_for(
      Kokkos::RangePolicy<TEST_EXECSPACE>(0, 1), KOKKOS_LAMBDA(int) {
        acc_t acc;
        const_acc_t c_acc;
        if (&acc.access(dh, 5) != ptr + 5) errors() += 1;
        if (&c_acc.access(cdh, 5) != ptr + 5) errors() += 2;
      });
  int h_errors = 0;
  Kokkos::deep_copy(h_errors, errors);
  ASSERT_FALSE(h_errors & 1);
  ASSERT_FALSE(h_errors & 2);
  Kokkos::kokkos_free(ptr);
}
