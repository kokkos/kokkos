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

#include <Kokkos_Core.hpp>
#include <TestHPX_Category.hpp>

// These tests specifically check that work dispatched to independent instances
// is synchronized correctly on fences. A previous bug that this protects
// against is work being mistakenly dispatched to the default instance, but the
// fence fencing the independent instance. In that case these tests will fail.

namespace {
inline constexpr int n = 1 << 10;

TEST(hpx, independent_instances_synchronization_parallel_for_range_policy) {
  Kokkos::View<int *, Kokkos::Experimental::HPX> a("a", n);

  Kokkos::Experimental::HPX instance{
      Kokkos::Experimental::HPX::instance_mode::independent};
  Kokkos::RangePolicy<Kokkos::Experimental::HPX> policy(instance, 0, n);
  Kokkos::parallel_for(
      "parallel_for_range_policy", policy,
      KOKKOS_LAMBDA(const auto i) { a[i] = i; });

  instance.fence();

  for (int i = 0; i < n; ++i) {
    ASSERT_EQ(a[i], i);
  }
}

TEST(hpx, independent_instances_synchronization_parallel_for_mdrange_policy) {
  Kokkos::View<int *, Kokkos::Experimental::HPX> a("a", n);

  Kokkos::Experimental::HPX instance{
      Kokkos::Experimental::HPX::instance_mode::independent};
  Kokkos::MDRangePolicy<Kokkos::Experimental::HPX, Kokkos::Rank<2>> policy(
      instance, {{0, 0}}, {{n, 1}});
  Kokkos::parallel_for(
      "parallel_for_mdrange_policy", policy,
      KOKKOS_LAMBDA(const auto i, const auto) { a[i] = i; });

  instance.fence();

  for (int i = 0; i < n; ++i) {
    ASSERT_EQ(a[i], i);
  }
}

TEST(hpx, independent_instances_synchronization_parallel_for_team_policy) {
  Kokkos::View<int *, Kokkos::Experimental::HPX> a("a", n);

  Kokkos::Experimental::HPX instance{
      Kokkos::Experimental::HPX::instance_mode::independent};
  Kokkos::TeamPolicy<Kokkos::Experimental::HPX> policy(instance, n, 1);
  Kokkos::parallel_for(
      "parallel_for_team_policy", policy, KOKKOS_LAMBDA(const auto &handle) {
        a[handle.league_rank()] = handle.league_rank();
      });

  instance.fence();

  for (int i = 0; i < n; ++i) {
    ASSERT_EQ(a[i], i);
  }
}

TEST(hpx, independent_instances_synchronization_parallel_reduce_range_policy) {
  Kokkos::View<int *, Kokkos::Experimental::HPX> a("a", n);
  Kokkos::View<int, Kokkos::Experimental::HPX> b("b");

  Kokkos::Experimental::HPX instance{
      Kokkos::Experimental::HPX::instance_mode::independent};
  Kokkos::RangePolicy<Kokkos::Experimental::HPX> policy(instance, 0, n);
  Kokkos::parallel_reduce(
      "parallel_reduce_range_policy", policy,
      KOKKOS_LAMBDA(const int i, int &) { a[i] = i; }, b);

  instance.fence();

  for (int i = 0; i < n; ++i) {
    ASSERT_EQ(a[i], i);
  }
}

TEST(hpx,
     independent_instances_synchronization_parallel_reduce_mdrange_policy) {
  Kokkos::View<int *, Kokkos::Experimental::HPX> a("a", n);
  Kokkos::View<int, Kokkos::Experimental::HPX> b("b");

  Kokkos::Experimental::HPX instance{
      Kokkos::Experimental::HPX::instance_mode::independent};
  Kokkos::MDRangePolicy<Kokkos::Experimental::HPX, Kokkos::Rank<2>> policy(
      instance, {{0, 0}}, {{n, 1}});
  Kokkos::parallel_reduce(
      "parallel_reduce_mdrange_policy", policy,
      KOKKOS_LAMBDA(const int i, const int, int &) { a[i] = i; }, b);

  instance.fence();

  for (int i = 0; i < n; ++i) {
    ASSERT_EQ(a[i], i);
  }
}

TEST(hpx, independent_instances_synchronization_parallel_reduce_team_policy) {
  Kokkos::View<int *, Kokkos::Experimental::HPX> a("a", n);
  Kokkos::View<int, Kokkos::Experimental::HPX> b("b");

  Kokkos::Experimental::HPX instance{
      Kokkos::Experimental::HPX::instance_mode::independent};
  Kokkos::TeamPolicy<Kokkos::Experimental::HPX> policy(instance, n, 1);
  Kokkos::parallel_reduce(
      "parallel_reduce_team_policy", policy,
      KOKKOS_LAMBDA(const decltype(policy)::member_type &handle, int &) {
        a[handle.league_rank()] = handle.league_rank();
      },
      b);

  instance.fence();

  for (int i = 0; i < n; ++i) {
    ASSERT_EQ(a[i], i);
  }
}

TEST(hpx, independent_instances_synchronization_parallel_scan_range_policy) {
  Kokkos::View<int *, Kokkos::Experimental::HPX> a("a", n);
  Kokkos::View<int *, Kokkos::Experimental::HPX> b("b", n);

  Kokkos::Experimental::HPX instance{
      Kokkos::Experimental::HPX::instance_mode::independent};
  Kokkos::RangePolicy<Kokkos::Experimental::HPX> policy(instance, 0, n);
  Kokkos::parallel_scan(
      "parallel_scan_range_policy", policy,
      KOKKOS_LAMBDA(const int i, int &, bool final) {
        if (!final) {
          a[i] = i;
        }
      },
      b);

  instance.fence();

  for (int i = 0; i < n; ++i) {
    ASSERT_EQ(a[i], i);
  }
}
}  // namespace
