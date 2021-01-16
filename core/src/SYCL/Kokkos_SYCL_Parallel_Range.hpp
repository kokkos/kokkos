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

#ifndef KOKKOS_SYCL_PARALLEL_RANGE_HPP_
#define KOKKOS_SYCL_PARALLEL_RANGE_HPP_

#include <impl/KokkosExp_IterateTileGPU.hpp>

template <class FunctorType, class ExecPolicy>
class Kokkos::Impl::ParallelFor<FunctorType, ExecPolicy,
                                Kokkos::Experimental::SYCL> {
 public:
  using Policy = ExecPolicy;

 private:
  using Member       = typename Policy::member_type;
  using WorkTag      = typename Policy::work_tag;
  using LaunchBounds = typename Policy::launch_bounds;

  const FunctorType m_functor;
  const Policy m_policy;

  template <typename Functor>
  static void sycl_direct_launch(const Policy& policy, const Functor& functor) {
    // Convenience references
    const Kokkos::Experimental::SYCL& space = policy.space();
    Kokkos::Experimental::Impl::SYCLInternal& instance =
        *space.impl_internal_space_instance();
    sycl::queue& q = *instance.m_queue;

    space.fence();

    q.submit([functor, policy](sycl::handler& cgh) {
      sycl::range<1> range(policy.end() - policy.begin());

      cgh.parallel_for(range, [=](sycl::item<1> item) {
        const typename Policy::index_type id =
            static_cast<typename Policy::index_type>(item.get_linear_id()) +
            policy.begin();
        if constexpr (std::is_same<WorkTag, void>::value)
          functor(id);
        else
          functor(WorkTag(), id);
      });
    });

    space.fence();
  }

  // Indirectly launch a functor by explicitly creating it in USM shared memory
  void sycl_indirect_launch() const {
    // Convenience references
    const Kokkos::Experimental::SYCL& space = m_policy.space();
    Kokkos::Experimental::Impl::SYCLInternal& instance =
        *space.impl_internal_space_instance();
    using IndirectKernelMem =
        Kokkos::Experimental::Impl::SYCLInternal::IndirectKernelMem;
    IndirectKernelMem& indirectKernelMem = instance.m_indirectKernelMem;

    // Copy the functor into USM Shared Memory
    using KernelFunctorPtr =
        std::unique_ptr<FunctorType, IndirectKernelMem::Deleter>;
    KernelFunctorPtr kernelFunctorPtr = indirectKernelMem.copy_from(m_functor);

    // Use reference_wrapper (because it is both trivially copyable and
    // invocable) and launch it
    sycl_direct_launch(m_policy, std::reference_wrapper(*kernelFunctorPtr));
  }

 public:
  using functor_type = FunctorType;

  void execute() const {
    if (m_policy.begin() == m_policy.end()) return;

    // if the functor is trivially copyable, we can launch it directly;
    // otherwise, we will launch it indirectly via explicitly creating
    // it in USM shared memory.
    if constexpr (std::is_trivially_copyable_v<decltype(m_functor)>)
      sycl_direct_launch(m_policy, m_functor);
    else
      sycl_indirect_launch();
  }

  ParallelFor(const ParallelFor&) = delete;
  ParallelFor(ParallelFor&&)      = delete;
  ParallelFor& operator=(const ParallelFor&) = delete;
  ParallelFor& operator=(ParallelFor&&) = delete;
  ~ParallelFor()                        = default;

  ParallelFor(const FunctorType& arg_functor, const Policy& arg_policy)
      : m_functor(arg_functor), m_policy(arg_policy) {}
};

// ParallelFor
template <class FunctorType, class... Traits>
class Kokkos::Impl::ParallelFor<FunctorType, Kokkos::MDRangePolicy<Traits...>,
                                Kokkos::Experimental::SYCL> {
 public:
  using Policy = Kokkos::MDRangePolicy<Traits...>;

 private:
  using array_index_type = typename Policy::array_index_type;
  using index_type       = typename Policy::index_type;
  using LaunchBounds     = typename Policy::launch_bounds;
  using WorkTag          = typename Policy::work_tag;

  const FunctorType m_functor;
  const Policy m_policy;

  sycl::nd_range<3> compute_ranges() const {
    const auto& m_tile     = m_policy.m_tile;
    const auto& m_tile_end = m_policy.m_tile_end;

    if constexpr (Policy::rank == 2) {
      sycl::range<3> local_sizes(m_tile[0], m_tile[1], 1);
      sycl::range<3> global_sizes(m_tile_end[0] * m_tile[0],
                                  m_tile_end[1] * m_tile[1], 1);
      return {global_sizes, local_sizes};
    }
    if constexpr (Policy::rank == 3) {
      sycl::range<3> local_sizes(m_tile[0], m_tile[1], m_tile[2]);
      sycl::range<3> global_sizes(m_tile_end[0] * m_tile[0],
                                  m_tile_end[1] * m_tile[1],
                                  m_tile_end[2] * m_tile[2]);
      return {global_sizes, local_sizes};
    }
    if constexpr (Policy::rank == 4) {
      // id0,id1 encoded within first index; id2 to second index; id3 to third
      // index
      sycl::range<3> local_sizes(m_tile[0] * m_tile[1], m_tile[2], m_tile[3]);
      sycl::range<3> global_sizes(
          m_tile_end[0] * m_tile[0] * m_tile_end[1] * m_tile[1],
          m_tile_end[2] * m_tile[2], m_tile_end[3] * m_tile[3]);
      return {global_sizes, local_sizes};
    }
    if constexpr (Policy::rank == 5) {
      // id0,id1 encoded within first index; id2,id3 to second index; id4 to
      // third index
      sycl::range<3> local_sizes(m_tile[0] * m_tile[1], m_tile[2] * m_tile[3],
                                 m_tile[4]);
      sycl::range<3> global_sizes(
          m_tile_end[0] * m_tile[0] * m_tile_end[1] * m_tile[1],
          m_tile_end[2] * m_tile[2] * m_tile_end[3] * m_tile[3],
          m_tile_end[4] * m_tile[4]);
      return {global_sizes, local_sizes};
    }
    if constexpr (Policy::rank == 6) {
      // id0,id1 encoded within first index; id2,id3 to second index; id4,id5 to
      // third index
      sycl::range<3> local_sizes(m_tile[0] * m_tile[1], m_tile[2] * m_tile[3],
                                 m_tile[4] * m_tile[5]);
      sycl::range<3> global_sizes(
          m_tile_end[0] * m_tile[0] * m_tile_end[1] * m_tile[1],
          m_tile_end[2] * m_tile[2] * m_tile_end[3] * m_tile[3],
          m_tile_end[4] * m_tile[4] * m_tile_end[5] * m_tile[5]);
      return {global_sizes, local_sizes};
    }
    static_assert(Policy::rank > 1 && Policy::rank < 7,
                  "Kokkos::MDRange Error: Exceeded rank bounds with SYCL\n");
  }

  template <typename Functor>
  void sycl_direct_launch(const Functor& functor) const {
    // Convenience references
    const Kokkos::Experimental::SYCL& space = m_policy.space();
    Kokkos::Experimental::Impl::SYCLInternal& instance =
        *space.impl_internal_space_instance();
    sycl::queue& q = *instance.m_queue;

    space.fence();

    if (m_policy.m_num_tiles == 0) return;

    auto& policy = m_policy;

    q.submit([functor, this, policy](sycl::handler& cgh) {
      const auto range = compute_ranges();

      cgh.parallel_for(range, [functor, policy](sycl::nd_item<3> item) {
        const index_type local_x    = item.get_local_id(0);
        const index_type local_y    = item.get_local_id(1);
        const index_type local_z    = item.get_local_id(2);
        const index_type global_x   = item.get_group(0);
        const index_type global_y   = item.get_group(1);
        const index_type global_z   = item.get_group(2);
        const index_type n_global_x = item.get_group_range(0);
        const index_type n_global_y = item.get_group_range(1);
        const index_type n_global_z = item.get_group_range(2);

        Kokkos::Impl::DeviceIterateTile<Policy::rank, Policy, Functor,
                                        typename Policy::work_tag>(
            policy, functor, {n_global_x, n_global_y, n_global_z},
            {global_x, global_y, global_z}, {local_x, local_y, local_z})
            .exec_range();
      });
    });

    space.fence();
  }

  // Indirectly launch a functor by explicitly creating it in USM shared memory
  void sycl_indirect_launch() const {
    // Convenience references
    const Kokkos::Experimental::SYCL& space = m_policy.space();
    Kokkos::Experimental::Impl::SYCLInternal& instance =
        *space.impl_internal_space_instance();
    using IndirectKernelMem =
        Kokkos::Experimental::Impl::SYCLInternal::IndirectKernelMem;
    IndirectKernelMem& indirectKernelMem = instance.m_indirectKernelMem;

    // Copy the functor into USM Shared Memory
    using KernelFunctorPtr =
        std::unique_ptr<FunctorType, IndirectKernelMem::Deleter>;
    KernelFunctorPtr kernelFunctorPtr = indirectKernelMem.copy_from(m_functor);

    // Use reference_wrapper (because it is both trivially copyable and
    // invocable) and launch it
    sycl_direct_launch(std::reference_wrapper(*kernelFunctorPtr));
  }

 public:
  using functor_type = FunctorType;

  void execute() const {
    // if the functor is trivially copyable, we can launch it directly;
    // otherwise, we will launch it indirectly via explicitly creating
    // it in USM shared memory.
    if constexpr (std::is_trivially_copyable_v<decltype(m_functor)>)
      sycl_direct_launch(m_functor);
    else
      sycl_indirect_launch();
  }

  ParallelFor(const ParallelFor&) = delete;
  ParallelFor(ParallelFor&&)      = delete;
  ParallelFor& operator=(const ParallelFor&) = delete;
  ParallelFor& operator=(ParallelFor&&) = delete;
  ~ParallelFor()                        = default;

  ParallelFor(const FunctorType& arg_functor, const Policy& arg_policy)
      : m_functor(arg_functor), m_policy(arg_policy) {}
};

#endif  // KOKKOS_SYCL_PARALLEL_RANGE_HPP_
