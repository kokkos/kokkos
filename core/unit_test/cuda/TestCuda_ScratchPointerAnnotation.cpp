// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <cstdint>
#include <type_traits>

#include <cuda/memory>

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif
#include <TestCuda_Category.hpp>

namespace Test {
namespace {

struct CudaScratchPointerAnnotationFunctor {
  using execution_space = TEST_EXECSPACE;
  using memory_space    = typename execution_space::memory_space;
  using member_type = typename Kokkos::TeamPolicy<execution_space>::member_type;
  using pointer_view_type = Kokkos::View<std::uintptr_t[8], memory_space>;
  using result_view_type  = Kokkos::View<int[8], memory_space>;

  pointer_view_type m_pointers;
  result_view_type m_results;

  KOKKOS_FUNCTION void operator()(const member_type &team) const {
    constexpr int allocation_size = 16;
    constexpr int alignment       = 8;

    // Round-trip the pointers through global memory and another thread so the
    // predicates validate the runtime address spaces independently of
    // annotations.
    if (team.team_rank() == 0) {
      const auto &scratch = team.team_shmem();
      m_pointers(0)       = reinterpret_cast<std::uintptr_t>(
          scratch.get_shmem(allocation_size, std::integral_constant<int, 0>{}));
      m_pointers(1) = reinterpret_cast<std::uintptr_t>(
          scratch.get_shmem<0>(allocation_size));
      m_pointers(2) =
          reinterpret_cast<std::uintptr_t>(scratch.get_shmem_aligned(
              allocation_size, alignment, std::integral_constant<int, 0>{}));
      m_pointers(3) = reinterpret_cast<std::uintptr_t>(
          scratch.get_shmem_aligned<0>(allocation_size, alignment));
      m_pointers(4) = reinterpret_cast<std::uintptr_t>(
          scratch.get_shmem(allocation_size, std::integral_constant<int, 1>{}));
      m_pointers(5) = reinterpret_cast<std::uintptr_t>(
          scratch.get_shmem<1>(allocation_size));
      m_pointers(6) =
          reinterpret_cast<std::uintptr_t>(scratch.get_shmem_aligned(
              allocation_size, alignment, std::integral_constant<int, 1>{}));
      m_pointers(7) = reinterpret_cast<std::uintptr_t>(
          scratch.get_shmem_aligned<1>(allocation_size, alignment));
    }

    team.team_barrier();

    if (team.team_rank() == 1) {
      const auto *level_0_tag = reinterpret_cast<const void *>(m_pointers(0));
      const auto *level_0_explicit =
          reinterpret_cast<const void *>(m_pointers(1));
      const auto *level_0_aligned_tag =
          reinterpret_cast<const void *>(m_pointers(2));
      const auto *level_0_aligned_explicit =
          reinterpret_cast<const void *>(m_pointers(3));
      const auto *level_1_tag = reinterpret_cast<const void *>(m_pointers(4));
      const auto *level_1_explicit =
          reinterpret_cast<const void *>(m_pointers(5));
      const auto *level_1_aligned_tag =
          reinterpret_cast<const void *>(m_pointers(6));
      const auto *level_1_aligned_explicit =
          reinterpret_cast<const void *>(m_pointers(7));

      KOKKOS_IF_ON_DEVICE(
          (m_results(0) = cuda::device::is_address_from(
               level_0_tag, cuda::device::address_space::shared);
           m_results(1) = cuda::device::is_address_from(
               level_0_explicit, cuda::device::address_space::shared);
           m_results(2) = cuda::device::is_address_from(
               level_0_aligned_tag, cuda::device::address_space::shared);
           m_results(3) = cuda::device::is_address_from(
               level_0_aligned_explicit, cuda::device::address_space::shared);
           m_results(4) = cuda::device::is_address_from(
               level_1_tag, cuda::device::address_space::global);
           m_results(5) = cuda::device::is_address_from(
               level_1_explicit, cuda::device::address_space::global);
           m_results(6) = cuda::device::is_address_from(
               level_1_aligned_tag, cuda::device::address_space::global);
           m_results(7) = cuda::device::is_address_from(
               level_1_aligned_explicit, cuda::device::address_space::global);))
    }
  }
};

}  // namespace

TEST(TEST_CATEGORY, scratch_pointer_address_spaces) {
  using execution_space   = TEST_EXECSPACE;
  using memory_space      = typename execution_space::memory_space;
  using pointer_view_type = Kokkos::View<std::uintptr_t[8], memory_space>;
  using result_view_type  = Kokkos::View<int[8], memory_space>;

  pointer_view_type pointers("scratch pointers");
  result_view_type results("address space results");

  Kokkos::TeamPolicy<execution_space> policy(1, 2);
  policy.set_scratch_size(0, Kokkos::PerTeam(64));
  policy.set_scratch_size(1, Kokkos::PerTeam(64));

  Kokkos::parallel_for("scratch_pointer_address_spaces", policy,
                       CudaScratchPointerAnnotationFunctor{pointers, results});
  Kokkos::fence();

  auto results_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results);
  EXPECT_EQ(results_host(0), 1) << "tag-dispatched level 0 pointer";
  EXPECT_EQ(results_host(1), 1) << "explicit level 0 pointer";
  EXPECT_EQ(results_host(2), 1) << "tag-dispatched aligned level 0 pointer";
  EXPECT_EQ(results_host(3), 1) << "explicit aligned level 0 pointer";
  EXPECT_EQ(results_host(4), 1) << "tag-dispatched level 1 pointer";
  EXPECT_EQ(results_host(5), 1) << "explicit level 1 pointer";
  EXPECT_EQ(results_host(6), 1) << "tag-dispatched aligned level 1 pointer";
  EXPECT_EQ(results_host(7), 1) << "explicit aligned level 1 pointer";
}

}  // namespace Test
