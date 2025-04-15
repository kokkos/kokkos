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

#ifndef KOKKOS_HIP_GRAPHNODEKERNEL_HPP
#define KOKKOS_HIP_GRAPHNODEKERNEL_HPP

#include <Kokkos_Graph_fwd.hpp>

#include <impl/Kokkos_GraphImpl.hpp>

#include <Kokkos_Parallel.hpp>
#include <Kokkos_Parallel_Reduce.hpp>
#include <Kokkos_PointerOwnership.hpp>

#include <HIP/Kokkos_HIP_GraphNode_Impl.hpp>

namespace Kokkos {
namespace Impl {

template <typename PolicyType, typename Functor, typename PatternTag,
          typename... Args>
class GraphNodeKernelImpl<Kokkos::HIP, PolicyType, Functor, PatternTag, Args...>
    : public PatternImplSpecializationFromTag<PatternTag, Functor, PolicyType,
                                              Args..., Kokkos::HIP>::type {
 public:
  using Policy       = PolicyType;
  using graph_kernel = GraphNodeKernelImpl;
  using base_t =
      typename PatternImplSpecializationFromTag<PatternTag, Functor, Policy,
                                                Args..., Kokkos::HIP>::type;

  // TODO use the name and executionspace
  template <typename PolicyDeduced, typename... ArgsDeduced>
  GraphNodeKernelImpl(std::string label_, HIP const&, Functor arg_functor,
                      PolicyDeduced&& arg_policy, ArgsDeduced&&... args)
      : base_t(std::move(arg_functor), (PolicyDeduced&&)arg_policy,
               (ArgsDeduced&&)args...),
        label(std::move(label_)) {}

  template <typename PolicyDeduced>
  GraphNodeKernelImpl(Kokkos::HIP const& exec_space, Functor arg_functor,
                      PolicyDeduced&& arg_policy)
      : GraphNodeKernelImpl("[unlabeled]", exec_space, std::move(arg_functor),
                            (PolicyDeduced&&)arg_policy) {}

  void set_hip_graph_ptr(hipGraph_t* arg_graph_ptr) {
    m_graph_ptr = arg_graph_ptr;
  }

  void set_hip_graph_node_ptr(hipGraphNode_t* arg_node_ptr) {
    m_graph_node_ptr = arg_node_ptr;
  }

  hipGraphNode_t* get_hip_graph_node_ptr() const { return m_graph_node_ptr; }

  hipGraph_t const* get_hip_graph_ptr() const { return m_graph_ptr; }

  Kokkos::ObservingRawPtr<base_t> allocate_driver_memory_buffer(
      const HIP& exec) const {
    KOKKOS_EXPECTS(m_driver_storage == nullptr);
    std::string alloc_label =
        label + " - GraphNodeKernel global memory functor storage";
    m_driver_storage = std::shared_ptr<base_t>(
        static_cast<base_t*>(
            HIPSpace().allocate(exec, alloc_label.c_str(), sizeof(base_t))),
        // FIXME_HIP Custom deletor should use same 'exec' as for allocation.
        [alloc_label](base_t* ptr) {
          HIPSpace().deallocate(alloc_label.c_str(), ptr, sizeof(base_t));
        });
    KOKKOS_ENSURES(m_driver_storage != nullptr);
    return m_driver_storage.get();
  }

  auto get_driver_storage() const { return m_driver_storage; }

 private:
  Kokkos::ObservingRawPtr<const hipGraph_t> m_graph_ptr    = nullptr;
  Kokkos::ObservingRawPtr<hipGraphNode_t> m_graph_node_ptr = nullptr;
  mutable std::shared_ptr<base_t> m_driver_storage         = nullptr;
  std::string label;
};

struct HIPGraphNodeAggregate {};

template <typename KernelType,
          typename Tag =
              typename PatternTagFromImplSpecialization<KernelType>::type>
struct get_graph_node_kernel_type
    : type_identity<
          GraphNodeKernelImpl<Kokkos::HIP, typename KernelType::Policy,
                              typename KernelType::functor_type, Tag>> {};

template <typename KernelType>
struct get_graph_node_kernel_type<KernelType, Kokkos::ParallelReduceTag>
    : type_identity<GraphNodeKernelImpl<
          Kokkos::HIP, typename KernelType::Policy,
          CombinedFunctorReducer<typename KernelType::functor_type,
                                 typename KernelType::reducer_type>,
          Kokkos::ParallelReduceTag>> {};

template <typename KernelType>
auto* allocate_driver_storage_for_kernel(const HIP& exec,
                                         KernelType const& kernel) {
  using graph_node_kernel_t =
      typename get_graph_node_kernel_type<KernelType>::type;
  auto const& kernel_as_graph_kernel =
      static_cast<graph_node_kernel_t const&>(kernel);

  return kernel_as_graph_kernel.allocate_driver_memory_buffer(exec);
}

template <typename KernelType>
auto const& get_hip_graph_from_kernel(KernelType const& kernel) {
  using graph_node_kernel_t =
      typename get_graph_node_kernel_type<KernelType>::type;
  auto const& kernel_as_graph_kernel =
      static_cast<graph_node_kernel_t const&>(kernel);
  hipGraph_t const* graph_ptr = kernel_as_graph_kernel.get_hip_graph_ptr();
  KOKKOS_EXPECTS(graph_ptr != nullptr);

  return *graph_ptr;
}

template <typename KernelType>
auto& get_hip_graph_node_from_kernel(KernelType const& kernel) {
  using graph_node_kernel_t =
      typename get_graph_node_kernel_type<KernelType>::type;
  auto const& kernel_as_graph_kernel =
      static_cast<graph_node_kernel_t const&>(kernel);
  auto* graph_node_ptr = kernel_as_graph_kernel.get_hip_graph_node_ptr();
  KOKKOS_EXPECTS(graph_node_ptr != nullptr);

  return *graph_node_ptr;
}
}  // namespace Impl
}  // namespace Kokkos

#endif
