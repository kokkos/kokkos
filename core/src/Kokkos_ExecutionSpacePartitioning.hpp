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

#ifndef KOKKOS_KOKKOS_EXECUTIONSPACEPARTITIONING_HPP
#define KOKKOS_KOKKOS_EXECUTIONSPACEPARTITIONING_HPP

#include <vector>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Core_fwd.hpp>

namespace Kokkos {
// tag class, signifying that the concurrency function
// returns the amount of resources available to each
// kernel (e.g. Cuda). In such cases, execution spaces
// ability to split aren't truly bound by "concurrency"
class ConcurrencyImpliesPerKernelResources {};

// tag class, signifying that the concurrency function
// returns the amount of resources available on the
// device. In such cases, total ability to split is
// determined by "concurrency"
class ConcurrencyImpliesDeviceResources {};

// tag class, signifying that the concurrency function
// doesn't really matter, different instances always submit
// to the same resources
class ConcurrencyTrivial {};

// enum class FencingSyncsSubInstances { yes, no };

template <class ExecutionSpace>
class ExecutionSpacePartitionerBase {
 public:
  using resource_count_type             = int32_t;
  using execution_space_collection_type = std::vector<ExecutionSpace>;
  execution_space_collection_type sub_instances;
  Kokkos::FencingSyncsSubInstances fences_forward_to_subinstances() const {
    return Kokkos::FencingSyncsSubInstances::no;
  }
  void fence_sub_instances() const {
    if (fences_forward_to_subinstances() == FencingSyncsSubInstances::yes) {
      for (auto instance : sub_instances) {
        instance.fence();
      }
    }
  }
  execution_space_collection_type partition_instances_impl(
      const ExecutionSpace& space, int num_subspaces) const {
    execution_space_collection_type subspaces;
    // auto max_concurrency      = get_max_partition_size_impl(space);
    auto max_concurrency      = space.get_max_partition_size();
    auto subspace_concurrency = max_concurrency / num_subspaces;
    KOKKOS_ASSERT(subspace_concurrency > 0);
    for (int x = 0; x < num_subspaces; ++x) {
      subspaces.emplace_back(Kokkos::create_execspace_instance<ExecutionSpace>(
          subspace_concurrency));
    }
    return subspaces;
  }
};

template <class ExecutionSpace, class PartitioningScheme>
class ExecutionSpacePartitioner;

template <class ExecutionSpace>
class ExecutionSpacePartitioner<ExecutionSpace,
                                ConcurrencyImpliesPerKernelResources>
    : public ExecutionSpacePartitionerBase<ExecutionSpace> {
 public:
  using base_t              = ExecutionSpacePartitionerBase<ExecutionSpace>;
  using resource_count_type = typename base_t::resource_count_type;
  using execution_space_collection_type =
      typename base_t::execution_space_collection_type;
  constexpr static resource_count_type max_instances = 9999;  // do better

  template <class... Floats>
  execution_space_collection_type partition_instances_with_weights_impl(
      ExecutionSpace& space, Floats...) const {
    /**
     * Note: this function is a little funky. On the one hand,
     * it's properly generic, you can split any kind of exec_space
     * using weights. On the other hand, for cases where we're not
     * actually partitioning threads (the streams case), users
     * aren't actually getting a split on weights.
     *
     * Worth discussing as a group whether we should *force*
     * people to do the right thing, and not split stream-like
     * ExecutionSpaces with weights.
     */
    return partition_instances_impl(space, sizeof...(Floats));
  }

 protected:
  resource_count_type get_max_partition_size_impl(
      const ExecutionSpace& space) const {
    return max_instances;
  }
};

template <class ExecutionSpace>
class ExecutionSpacePartitioner<ExecutionSpace,
                                ConcurrencyImpliesDeviceResources>
    : public ExecutionSpacePartitionerBase<ExecutionSpace> {
 public:
  using base_t              = ExecutionSpacePartitionerBase<ExecutionSpace>;
  using resource_count_type = typename base_t::resource_count_type;
  using execution_space_collection_type =
      typename base_t::execution_space_collection_type;

 private:
  void process_weights(resource_count_type root_concurrency,
                       execution_space_collection_type&) {}
  template <class Head, class... Cons>
  void process_weights(resource_count_type root_concurrency,
                       execution_space_collection_type& subspaces, Head head,
                       Cons... cons) {
    auto subspace_concurrency = root_concurrency * head;
    subspaces.emplace_back(Kokkos::create_execspace_instance<ExecutionSpace>(
        subspace_concurrency));
    process_weights(root_concurrency, subspaces, cons...);
  }

 public:
  template <class... Floats>
  execution_space_collection_type partition_instances_with_weights_impl(
      const ExecutionSpace& space, Floats... weights) const {
    execution_space_collection_type subspaces;
    process_weights(get_max_partition_size_impl(space), subspaces, weights...);
  }

 protected:
  resource_count_type get_max_partition_size_impl(
      const ExecutionSpace& space) const {
    return space.concurrency();
  }
};

template <class ExecutionSpace>
class ExecutionSpacePartitioner<ExecutionSpace, ConcurrencyTrivial> {};

}  // end namespace Kokkos
#endif  // KOKKOS_KOKKOS_EXECUTIONSPACEPARTITIONING_HPP
