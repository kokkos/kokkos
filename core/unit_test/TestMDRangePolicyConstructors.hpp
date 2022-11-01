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

#include <Kokkos_Core.hpp>

namespace {

#ifndef KOKKOS_COMPILER_NVHPC       // FIXME_NVHPC
#ifndef KOKKOS_ENABLE_OPENMPTARGET  // FIXME_OPENMPTARGET
TEST(TEST_CATEGORY_DEATH, policy_bounds_unsafe_narrowing_conversions) {
  using Policy = Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>,
                                       Kokkos::IndexType<unsigned>>;

  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_DEATH(
      {
        (void)Policy({-1, 0}, {2, 3});
      },
      "unsafe narrowing conversion");
}
#endif
#endif

template <class... Args>
struct DummyPolicy : Kokkos::Impl::PolicyTraits<Args...> {
  using execution_policy = DummyPolicy;

  using base_t = Kokkos::Impl::PolicyTraits<Args...>;
  using base_t::base_t;
};
// For a more informative static assertion:
template <size_t>
struct static_assert_dummy_policy_must_be_size_one;
template <>
struct static_assert_dummy_policy_must_be_size_one<1> {};
template <size_t, size_t>
struct static_assert_dummy_policy_must_be_size_of_desired_occupancy;
template <>
struct static_assert_dummy_policy_must_be_size_of_desired_occupancy<
    sizeof(Kokkos::Experimental::DesiredOccupancy),
    sizeof(Kokkos::Experimental::DesiredOccupancy)> {};

// EBO failure with VS 16.11.3 and CUDA 11.4.2
#if !(defined(_WIN32) && defined(KOKKOS_ENABLE_CUDA))
TEST(TEST_CATEGORY, desired_occupancy_empty_base_optimization) {
  DummyPolicy<TEST_EXECSPACE> const policy{};
  static_assert(sizeof(decltype(policy)) == 1, "");
  static_assert_dummy_policy_must_be_size_one<sizeof(decltype(policy))>
      _assert1{};
  (void)&_assert1;  // avoid unused variable warning

  using Kokkos::Experimental::DesiredOccupancy;
  auto policy_with_occ =
      Kokkos::Experimental::prefer(policy, DesiredOccupancy{50});
  static_assert(sizeof(decltype(policy_with_occ)) == sizeof(DesiredOccupancy),
                "");
  static_assert_dummy_policy_must_be_size_of_desired_occupancy<
      sizeof(decltype(policy_with_occ)), sizeof(DesiredOccupancy)>
      _assert2{};
  (void)&_assert2;  // avoid unused variable warning
}
#endif

template <class T>
void more_md_range_policy_construction_test() {
  (void)Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>>{
      Kokkos::Array<T, 2>{}, Kokkos::Array<T, 2>{}};

  (void)Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>>{{{T(0), T(0)}},
                                                               {{T(2), T(2)}}};

  (void)Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>>{{T(0), T(0)},
                                                               {T(2), T(2)}};
}

TEST(TEST_CATEGORY, md_range_policy_construction_from_arrays) {
  {
    // Check that construction from Kokkos::Array of long compiles for backwards
    // compability.  This was broken in
    // https://github.com/kokkos/kokkos/pull/3527/commits/88ea8eec6567c84739d77bdd25fdbc647fae28bb#r512323639
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>> p1(
        Kokkos::Array<long, 2>{{0, 1}}, Kokkos::Array<long, 2>{{2, 3}});
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>> p2(
        Kokkos::Array<long, 2>{{0, 1}}, Kokkos::Array<long, 2>{{2, 3}});
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>> p3(
        Kokkos::Array<long, 2>{{0, 1}}, Kokkos::Array<long, 2>{{2, 3}},
        Kokkos::Array<long, 1>{{4}});
  }
  {
    // Check that construction from Kokkos::Array of the specified index type
    // works.
    using index_type = unsigned long long;
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>,
                          Kokkos::IndexType<index_type>>
        p1(Kokkos::Array<index_type, 2>{{0, 1}},
           Kokkos::Array<index_type, 2>{{2, 3}});
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>,
                          Kokkos::IndexType<index_type>>
        p2(Kokkos::Array<index_type, 2>{{0, 1}},
           Kokkos::Array<index_type, 2>{{2, 3}});
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>,
                          Kokkos::IndexType<index_type>>
        p3(Kokkos::Array<index_type, 2>{{0, 1}},
           Kokkos::Array<index_type, 2>{{2, 3}},
           Kokkos::Array<index_type, 1>{{4}});
  }
  {
    // Check that construction from double-braced initliazer list
    // works.
    using index_type = unsigned long long;
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>> p1({{0, 1}},
                                                              {{2, 3}});
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>,
                          Kokkos::IndexType<index_type>>
        p2({{0, 1}}, {{2, 3}});
  }

  more_md_range_policy_construction_test<char>();
  more_md_range_policy_construction_test<int>();
  more_md_range_policy_construction_test<unsigned long>();
  more_md_range_policy_construction_test<std::int64_t>();
}

}  // namespace
