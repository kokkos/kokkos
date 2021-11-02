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

#ifndef KOKKOS_IMPL_KOKKOS_TOOLS_GENERIC_HPP
#define KOKKOS_IMPL_KOKKOS_TOOLS_GENERIC_HPP

#ifdef KOKKOS_ENABLE_CUDA

#include <Kokkos_Cuda.hpp>
#include <Cuda/Kokkos_Cuda_BlockSize_Deduction.hpp>
#endif

#include <impl/Kokkos_Profiling.hpp>

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_ExecPolicy.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Tuners.hpp>
namespace Kokkos {

class Cuda;  // forward declaration

namespace Experimental {
class HIP;  // forward declaration
class SYCL;
}  // namespace Experimental

namespace Tools {

namespace Experimental {

namespace Impl {

template <class, class = void>
struct IsTunableRangePolicy : public std::false_type {};

template <class... Properties>
class IsTunableRangePolicy<
    Kokkos::RangePolicy<Properties...>,
    typename std::enable_if<std::is_same<typename Kokkos::RangePolicy<
                                             Properties...>::execution_space,
                                         Kokkos::Cuda>::value,
                            void>::type> : public std::true_type {};

template <class... Properties>
class IsTunableRangePolicy<
    Kokkos::RangePolicy<Properties...>,
    typename std::enable_if<std::is_same<typename Kokkos::RangePolicy<
                                             Properties...>::execution_space,
                                         Kokkos::Experimental::HIP>::value,
                            void>::type> : public std::false_type {
};  // TODO: change to true when HIP supported

static std::map<std::string, Kokkos::Tools::Experimental::TeamSizeTuner>
    team_tuners;

template <int Rank>
using MDRangeTuningMap =
    std::map<std::string, Kokkos::Tools::Experimental::MDRangeTuner<Rank>>;

template <int Rank>
static MDRangeTuningMap<Rank> mdrange_tuners;
using BlockSizeTuner = Kokkos::Tools::Experimental::BlockSizeTuner;

template <class>
using BlockSizeTunerMap = std::map<std::string, BlockSizeTuner>;

template <class Space>
static BlockSizeTunerMap<Space> block_size_tuners;
// For any policies without a tuning implementation, with a reducer
template <class ReducerType, class ExecPolicy, class Functor, typename TagType>
void tune_policy(const size_t, const std::string&, ExecPolicy&, const Functor&,
                 TagType) {}

// For any policies without a tuning implementation, without a reducer
template <class ExecPolicy, class Functor, typename TagType>
void tune_policy(const size_t, const std::string&, ExecPolicy&, const Functor&,
                 const TagType&) {}

/**
 * Tuning for parallel_fors and parallel_scans is a fairly simple process.
 *
 * Tuning for a parallel_reduce turns out to be a little more complicated.
 *
 * If you're tuning a reducer, it might be a complex or a simple reducer
 * (an example of simple would be one where the join is just "+".
 *
 * Unfortunately these two paths are very different in terms of which classes
 * get instantiated. Thankfully, all of this complexity is encoded in the
 * ReducerType. If it's a "simple" reducer, this will be Kokkos::InvalidType,
 * otherwise it'll be something else.
 *
 * If the type is complex, for the code to be generally right you _must_
 * pass an instance of that ReducerType to functions that determine
 * eligible team sizes. If the type is simple, you can't construct one,
 * you use the simpler 2-arg formulation of team_size_recommended/max.
 */

namespace Impl {
template <class Tag>
struct DriverFor;
template <>
struct DriverFor<Kokkos::ParallelForTag> {
  template <class A, class B, class C>
  using type = Kokkos::Impl::ParallelFor<A, B, C>;
};
template <>
struct DriverFor<Kokkos::ParallelScanTag> {
  template <class A, class B, class C>
  using type = Kokkos::Impl::ParallelScan<A, B, C>;
};
template <>
struct DriverFor<Kokkos::ParallelReduceTag> {
  template <class A, class B, class C, class D = Kokkos::InvalidType>
  using type = Kokkos::Impl::ParallelReduce<A, B, D, C>;
};
struct SimpleTeamSizeCalculator {
  template <typename Policy, typename Functor, typename Tag>
  int get_max_team_size(const Policy& policy, const Functor& functor,
                        const Tag tag) {
    auto max = policy.team_size_max(functor, tag);
    return max;
  }
  template <typename Policy, typename Functor, typename Tag>
  int get_recommended_team_size(const Policy& policy, const Functor& functor,
                                const Tag tag) {
    auto max = policy.team_size_recommended(functor, tag);
    return max;
  }
  template <typename Policy, typename Functor>
  int get_mdrange_max_tile_size_product(const Policy& policy,
                                        const Functor& functor,
                                        const Kokkos::ParallelForTag&) {
    using exec_space = typename Policy::execution_space;
    using driver     = Kokkos::Impl::ParallelFor<Functor, Policy, exec_space>;
    return driver::max_tile_size_product(policy, functor);
  }
  template <typename Policy, typename Functor>
  int get_mdrange_max_tile_size_product(const Policy& policy,
                                        const Functor& functor,
                                        const Kokkos::ParallelReduceTag&) {
    using exec_space = typename Policy::execution_space;
    using driver =
        Kokkos::Impl::ParallelReduce<Functor, Policy, Kokkos::InvalidType,
                                     exec_space>;
    return driver::max_tile_size_product(policy, functor);
  }
  template <typename Functor, typename Tag, template <class...> class Policy,
            class... Traits>
  int64_t range_max_block_size(const Policy<Traits...>& policy,
                               const Functor& functor, const Tag&) const {
#ifdef KOKKOS_ENABLE_CUDA
    using traits = Kokkos::Impl::PolicyTraits<Traits...>;

    cudaFuncAttributes attr = Kokkos::Tools::Impl::get_cuda_func_attributes<
        typename DriverFor<Tag>::type<Functor, Policy<Traits...>, Kokkos::Cuda>,
        typename Policy<Traits...>::launch_bounds>();
    const int block_size =
        Kokkos::Impl::cuda_get_max_block_size<Functor,
                                              typename traits::launch_bounds>(
            policy.space().impl_internal_space_instance(), attr, functor, 1, 0,
            0);
    return block_size;
#elif defined(KOKKOS_ENABLE_HIP)
    (void)policy;
    (void)functor;
    return 1;
#endif
    (void)policy;
    (void)functor;
    return 1;
  }
  template <typename Functor, typename Tag, template <class...> class Policy,
            class... Traits>
  int64_t range_opt_block_size(const Policy<Traits...>& policy,
                               const Functor& functor, const Tag&) const {
#ifdef KOKKOS_ENABLE_CUDA
    using traits = Kokkos::Impl::PolicyTraits<Traits...>;

    cudaFuncAttributes attr = Kokkos::Tools::Impl::get_cuda_func_attributes<
        typename DriverFor<Tag>::type<Functor, Policy<Traits...>, Kokkos::Cuda>,
        typename Policy<Traits...>::launch_bounds>();
    const int block_size =
        Kokkos::Impl::cuda_get_opt_block_size<Functor,
                                              typename traits::launch_bounds>(
            policy.space().impl_internal_space_instance(), attr, functor, 1, 0,
            0);
    return block_size;
#elif defined(KOKKOS_ENABLE_HIP)
    (void)policy;
    (void)functor;
    return 1;
#endif
    (void)policy;
    (void)functor;
    return 1;
  }
};

// when we have a complex reducer, we need to pass an
// instance to team_size_recommended/max. Reducers
// aren't default constructible, but they are
// constructible from a reference to an
// instance of their value_type so we construct
// a value_type and temporary reducer here
template <typename ReducerType>
struct ComplexReducerSizeCalculator {
  template <typename Policy, typename Functor, typename Tag>
  int get_max_team_size(const Policy& policy, const Functor& functor,
                        const Tag tag) {
    using value_type = typename ReducerType::value_type;
    value_type value;
    ReducerType reducer_example = ReducerType(value);
    return policy.team_size_max(functor, reducer_example, tag);
  }
  template <typename Policy, typename Functor, typename Tag>
  int get_recommended_team_size(const Policy& policy, const Functor& functor,
                                const Tag tag) {
    using value_type = typename ReducerType::value_type;
    value_type value;
    ReducerType reducer_example = ReducerType(value);
    return policy.team_size_recommended(functor, reducer_example, tag);
  }
  template <typename Policy, typename Functor>
  int get_mdrange_max_tile_size_product(const Policy& policy,
                                        const Functor& functor,
                                        const Kokkos::ParallelReduceTag&) {
    using exec_space = typename Policy::execution_space;
    using driver =
        Kokkos::Impl::ParallelReduce<Functor, Policy, ReducerType, exec_space>;
    return driver::max_tile_size_product(policy, functor);
  }
  template <typename Functor, typename Tag, template <class...> class Policy,
            class... Traits>
  int64_t range_max_block_size(const Policy<Traits...>& policy,
                               const Functor& functor, const Tag&) const {
#ifdef KOKKOS_ENABLE_CUDA
    using traits = Kokkos::Impl::PolicyTraits<Traits...>;

    cudaFuncAttributes attr = Kokkos::Tools::Impl::get_cuda_func_attributes<
        typename DriverFor<Tag>::type<Functor, Policy<Traits...>, Kokkos::Cuda,
                                      ReducerType>,
        typename Policy<Traits...>::launch_bounds>();
    const int block_size =
        Kokkos::Impl::cuda_get_max_block_size<Functor,
                                              typename traits::launch_bounds>(
            policy.space().impl_internal_space_instance(), attr, functor, 1, 0,
            0);
    return block_size;
#elif defined(KOKKOS_ENABLE_HIP)
    (void)policy;
    (void)functor;
    return 1;  // TODO better
#endif
    (void)policy;
    (void)functor;
    return 1;
  }
  template <typename Functor, typename Tag, template <class...> class Policy,
            class... Traits>
  int64_t range_opt_block_size(const Policy<Traits...>& policy,
                               const Functor& functor, const Tag&) const {
#ifdef KOKKOS_ENABLE_CUDA
    using traits = Kokkos::Impl::PolicyTraits<Traits...>;

    cudaFuncAttributes attr = Kokkos::Tools::Impl::get_cuda_func_attributes<
        typename DriverFor<Tag>::type<Functor, Policy<Traits...>, Kokkos::Cuda,
                                      ReducerType>,
        typename Policy<Traits...>::launch_bounds>();
    const int block_size =
        Kokkos::Impl::cuda_get_opt_block_size<Functor,
                                              typename traits::launch_bounds>(
            policy.space().impl_internal_space_instance(), attr, functor, 1, 0,
            0);
    return block_size;
#elif defined(KOKKOS_ENABLE_HIP)
    (void)policy;
    (void)functor;
    return 1;  // TODO better
#endif
    (void)policy;
    (void)functor;
    return 1;
  }
};

}  // namespace Impl

template <class Tuner, class Functor, class TagType,
          class TuningPermissionFunctor, class Map, class Policy>
void generic_tune_policy(const std::string& label_in, Map& map, Policy& policy,
                         const Functor& functor, const TagType& tag,
                         const TuningPermissionFunctor& should_tune) {
  if (should_tune(policy)) {
    std::string label = label_in;
    if (label_in.empty()) {
      using policy_type =
          typename std::remove_reference<decltype(policy)>::type;
      using work_tag = typename policy_type::work_tag;
      Kokkos::Impl::ParallelConstructName<Functor, work_tag> name(label);
      label = name.get();
    }
    auto tuner_iter = [&]() {
      auto my_tuner = map.find(label);
      if (my_tuner == map.end()) {
        return (map.emplace(label, Tuner(label, policy, functor, tag,
                                         Impl::SimpleTeamSizeCalculator{}))
                    .first);
      }
      return my_tuner;
    }();
    tuner_iter->second.tune(policy);
  }
}
template <class Tuner, class ReducerType, class Functor, class TagType,
          class TuningPermissionFunctor, class Map, class Policy>
void generic_tune_policy(const std::string& label_in, Map& map, Policy& policy,
                         const Functor& functor, const TagType& tag,
                         const TuningPermissionFunctor& should_tune) {
  if (should_tune(policy)) {
    std::string label = label_in;
    if (label_in.empty()) {
      using policy_type =
          typename std::remove_reference<decltype(policy)>::type;
      using work_tag = typename policy_type::work_tag;
      Kokkos::Impl::ParallelConstructName<Functor, work_tag> name(label);
      label = name.get();
    }
    auto tuner_iter = [&]() {
      auto my_tuner = map.find(label);
      if (my_tuner == map.end()) {
        return (map.emplace(
                       label,
                       Tuner(label, policy, functor, tag,
                             Impl::ComplexReducerSizeCalculator<ReducerType>{}))
                    .first);
      }
      return my_tuner;
    }();
    tuner_iter->second.tune(policy);
  }
}

// tune a TeamPolicy, without reducer
template <class Functor, class TagType, class... Properties>
void tune_policy(const size_t /**tuning_context*/, const std::string& label_in,
                 Kokkos::TeamPolicy<Properties...>& policy,
                 const Functor& functor, const TagType& tag) {
  generic_tune_policy<Experimental::TeamSizeTuner>(
      label_in, team_tuners, policy, functor, tag,
      [](const Kokkos::TeamPolicy<Properties...>& candidate_policy) {
        return (candidate_policy.impl_auto_team_size() ||
                candidate_policy.impl_auto_vector_length());
      });
}

// tune a TeamPolicy, with reducer
template <class ReducerType, class Functor, class TagType, class... Properties>
void tune_policy(const size_t /**tuning_context*/, const std::string& label_in,
                 Kokkos::TeamPolicy<Properties...>& policy,
                 const Functor& functor, const TagType& tag) {
  generic_tune_policy<Experimental::TeamSizeTuner, ReducerType>(
      label_in, team_tuners, policy, functor, tag,
      [](const Kokkos::TeamPolicy<Properties...>& candidate_policy) {
        return (candidate_policy.impl_auto_team_size() ||
                candidate_policy.impl_auto_vector_length());
      });
}

template <class ReducerType, class Functor, class TagType, class... Properties>
void tune_range_policy(const size_t /**tuning_context*/,
                       const std::string& label_in,
                       Kokkos::RangePolicy<Properties...>& policy,
                       const Functor& functor, const TagType& tag,
                       std::true_type) {
  generic_tune_policy<Experimental::BlockSizeTuner, ReducerType>(
      label_in, block_size_tuners<Kokkos::Cuda>, policy, functor, tag,
      [](const Kokkos::RangePolicy<Properties...>&) { return true; });
}
template <class Functor, class TagType, class... Properties>
void tune_range_policy(const size_t /**tuning_context*/,
                       const std::string& label_in,
                       Kokkos::RangePolicy<Properties...>& policy,
                       const Functor& functor, const TagType& tag,
                       std::true_type) {
  generic_tune_policy<Experimental::BlockSizeTuner>(
      label_in, block_size_tuners<Kokkos::Cuda>, policy, functor, tag,
      [](const Kokkos::RangePolicy<Properties...>&) { return true; });
}
template <class ReducerType, class Functor, class TagType, class... Properties>
void tune_range_policy(const size_t, const std::string&,
                       Kokkos::RangePolicy<Properties...>&, const Functor&,
                       const TagType&, std::false_type) {}
template <class Functor, class TagType, class... Properties>
void tune_range_policy(const size_t, const std::string&,
                       Kokkos::RangePolicy<Properties...>&, const Functor&,
                       const TagType&, std::false_type) {}
// tune a RangePolicy, with reducer
template <class ReducerType, class Functor, class TagType, class... Properties>
void tune_policy(const size_t tuning_context, const std::string& label_in,
                 Kokkos::RangePolicy<Properties...>& policy,
                 const Functor& functor, const TagType& tag) {
  using is_tunable =
      typename IsTunableRangePolicy<Kokkos::RangePolicy<Properties...>>::type;
  tune_range_policy<ReducerType>(tuning_context, label_in, policy, functor, tag,
                                 is_tunable{});
}
// RangePolicy, without reducer
template <class Functor, class TagType, class... Properties>
void tune_policy(const size_t tuning_context, const std::string& label_in,
                 Kokkos::RangePolicy<Properties...>& policy,
                 const Functor& functor, const TagType& tag

) {
  using is_tunable =
      typename IsTunableRangePolicy<Kokkos::RangePolicy<Properties...>>::type;
  tune_range_policy(tuning_context, label_in, policy, functor, tag,
                    is_tunable{});
}
// tune a MDRangePolicy, without reducer
template <class Functor, class TagType, class... Properties>
void tune_policy(const size_t /**tuning_context*/, const std::string& label_in,
                 Kokkos::MDRangePolicy<Properties...>& policy,
                 const Functor& functor, const TagType& tag) {
  using Policy              = Kokkos::MDRangePolicy<Properties...>;
  static constexpr int rank = Policy::rank;
  generic_tune_policy<Experimental::MDRangeTuner<rank>>(
      label_in, mdrange_tuners<rank>, policy, functor, tag,
      [](const Policy& candidate_policy) {
        return candidate_policy.impl_tune_tile_size();
      });
}

// tune a MDRangePolicy, with reducer
template <class ReducerType, class Functor, class TagType, class... Properties>
void tune_policy(const size_t /**tuning_context*/, const std::string& label_in,
                 Kokkos::MDRangePolicy<Properties...>& policy,
                 const Functor& functor, const TagType& tag) {
  using Policy              = Kokkos::MDRangePolicy<Properties...>;
  static constexpr int rank = Policy::rank;
  generic_tune_policy<Experimental::MDRangeTuner<rank>, ReducerType>(
      label_in, mdrange_tuners<rank>, policy, functor, tag,
      [](const Policy& candidate_policy) {
        return candidate_policy.impl_tune_tile_size();
      });
}

template <class ReducerType>
struct ReductionSwitcher {
  template <class Functor, class TagType, class ExecPolicy>
  static void tune(const size_t tuning_context, const std::string& label,
                   ExecPolicy& policy, const Functor& functor,
                   const TagType& tag) {
    if (Kokkos::tune_internals()) {
      tune_policy<ReducerType>(tuning_context, label, policy, functor, tag);
    }
  }
};

template <>
struct ReductionSwitcher<Kokkos::InvalidType> {
  template <class Functor, class TagType, class ExecPolicy>
  static void tune(const size_t tuning_context, const std::string& label,
                   ExecPolicy& policy, const Functor& functor,
                   const TagType& tag) {
    if (Kokkos::tune_internals()) {
      tune_policy(tuning_context, label, policy, functor, tag);
    }
  }
};

template <class Tuner, class Functor, class TagType,
          class TuningPermissionFunctor, class Map, class Policy>
void generic_report_results(const std::string& label_in, Map& map,
                            Policy& policy, const Functor&, const TagType&,
                            const TuningPermissionFunctor& should_tune) {
  if (should_tune(policy)) {
    std::string label = label_in;
    if (label_in.empty()) {
      using policy_type =
          typename std::remove_reference<decltype(policy)>::type;
      using work_tag = typename policy_type::work_tag;
      Kokkos::Impl::ParallelConstructName<Functor, work_tag> name(label);
      label = name.get();
    }
    auto tuner_iter = map[label];
    tuner_iter.end();
  }
}

// report results for a policy type we don't tune (do nothing)
template <class ExecPolicy, class Functor, typename TagType>
void report_policy_results(const size_t, const std::string&, ExecPolicy&,
                           const Functor&, const TagType&) {}

// report results for a TeamPolicy
template <class Functor, class TagType, class... Properties>
void report_policy_results(const size_t /**tuning_context*/,
                           const std::string& label_in,
                           Kokkos::TeamPolicy<Properties...>& policy,
                           const Functor& functor, const TagType& tag) {
  generic_report_results<Experimental::TeamSizeTuner>(
      label_in, team_tuners, policy, functor, tag,
      [](const Kokkos::TeamPolicy<Properties...>& candidate_policy) {
        return (candidate_policy.impl_auto_team_size() ||
                candidate_policy.impl_auto_vector_length());
      });
}
template <class ReducerType, class Functor, class TagType, class... Properties>
void report_range_results(const size_t /**tuning_context*/,
                          const std::string& label_in,
                          Kokkos::RangePolicy<Properties...>& policy,
                          const Functor& functor, const TagType& tag,
                          std::true_type) {
  generic_report_results<Experimental::BlockSizeTuner, ReducerType>(
      label_in, block_size_tuners<Kokkos::Cuda>, policy, functor, tag,
      [](const Kokkos::RangePolicy<Properties...>&) { return true; });
}
template <class Functor, class TagType, class... Properties>
void report_range_results(const size_t /**tuning_context*/,
                          const std::string& label_in,
                          Kokkos::RangePolicy<Properties...>& policy,
                          const Functor& functor, const TagType& tag,
                          std::true_type) {
  generic_report_results<Experimental::BlockSizeTuner>(
      label_in, block_size_tuners<Kokkos::Cuda>, policy, functor, tag,
      [](const Kokkos::RangePolicy<Properties...>&) { return true; });
}
template <class ReducerType, class Functor, class TagType, class... Properties>
void report_range_results(const size_t, const std::string&,
                          Kokkos::RangePolicy<Properties...>&, const Functor&,
                          const TagType&, std::false_type) {}
template <class Functor, class TagType, class... Properties>
void report_range_results(const size_t, const std::string&,
                          Kokkos::RangePolicy<Properties...>&, const Functor&,
                          const TagType&, std::false_type) {}
// tune a RangePolicy, with reducer
template <class ReducerType, class Functor, class TagType, class... Properties>
void report_policy_results(const size_t tuning_context,
                           const std::string& label_in,
                           Kokkos::RangePolicy<Properties...>& policy,
                           const Functor& functor, const TagType& tag) {
  using is_tunable =
      typename IsTunableRangePolicy<Kokkos::RangePolicy<Properties...>>::type;
  report_range_results<ReducerType>(tuning_context, label_in, policy, functor,
                                    tag, is_tunable{});
}
// RangePolicy, without reducer
template <class Functor, class TagType, class... Properties>
void report_policy_results(const size_t tuning_context,
                           const std::string& label_in,
                           Kokkos::RangePolicy<Properties...>& policy,
                           const Functor& functor, const TagType& tag

) {
  using is_tunable =
      typename IsTunableRangePolicy<Kokkos::RangePolicy<Properties...>>::type;
  report_range_results(tuning_context, label_in, policy, functor, tag,
                       is_tunable{});
}
// report results for an MDRangePolicy
template <class Functor, class TagType, class... Properties>
void report_policy_results(const size_t /**tuning_context*/,
                           const std::string& label_in,
                           Kokkos::MDRangePolicy<Properties...>& policy,
                           const Functor& functor, const TagType& tag) {
  using Policy              = Kokkos::MDRangePolicy<Properties...>;
  static constexpr int rank = Policy::rank;
  generic_report_results<Experimental::MDRangeTuner<rank>>(
      label_in, mdrange_tuners<rank>, policy, functor, tag,
      [](const Policy& candidate_policy) {
        return candidate_policy.impl_tune_tile_size();
      });
}

}  // namespace Impl

}  // namespace Experimental

namespace Impl {

template <class ExecPolicy, class FunctorType>
void begin_parallel_for(ExecPolicy& policy, FunctorType& functor,
                        const std::string& label, uint64_t& kpID) {
  if (Kokkos::Tools::profileLibraryLoaded()) {
    Kokkos::Impl::ParallelConstructName<FunctorType,
                                        typename ExecPolicy::work_tag>
        name(label);
    Kokkos::Tools::beginParallelFor(
        name.get(), Kokkos::Profiling::Experimental::device_id(policy.space()),
        &kpID);
  }
#ifdef KOKKOS_ENABLE_TUNING
  size_t context_id = Kokkos::Tools::Experimental::get_new_context_id();
  if (Kokkos::tune_internals()) {
    Experimental::Impl::tune_policy(context_id, label, policy, functor,
                                    Kokkos::ParallelForTag{});
  }
#else
  (void)functor;
#endif
}

template <class ExecPolicy, class FunctorType>
void end_parallel_for(ExecPolicy& policy, FunctorType& functor,
                      const std::string& label, uint64_t& kpID) {
  if (Kokkos::Tools::profileLibraryLoaded()) {
    Kokkos::Tools::endParallelFor(kpID);
  }
#ifdef KOKKOS_ENABLE_TUNING
  size_t context_id = Kokkos::Tools::Experimental::get_current_context_id();
  if (Kokkos::tune_internals()) {
    Experimental::Impl::report_policy_results(
        context_id, label, policy, functor, Kokkos::ParallelForTag{});
  }
#else
  (void)policy;
  (void)functor;
  (void)label;
#endif
}

template <class ExecPolicy, class FunctorType>
void begin_parallel_scan(ExecPolicy& policy, FunctorType& functor,
                         const std::string& label, uint64_t& kpID) {
  if (Kokkos::Tools::profileLibraryLoaded()) {
    Kokkos::Impl::ParallelConstructName<FunctorType,
                                        typename ExecPolicy::work_tag>
        name(label);
    Kokkos::Tools::beginParallelScan(
        name.get(), Kokkos::Profiling::Experimental::device_id(policy.space()),
        &kpID);
  }
#ifdef KOKKOS_ENABLE_TUNING
  size_t context_id = Kokkos::Tools::Experimental::get_new_context_id();
  if (Kokkos::tune_internals()) {
    Experimental::Impl::tune_policy(context_id, label, policy, functor,
                                    Kokkos::ParallelScanTag{});
  }
#else
  (void)functor;
#endif
}

template <class ExecPolicy, class FunctorType>
void end_parallel_scan(ExecPolicy& policy, FunctorType& functor,
                       const std::string& label, uint64_t& kpID) {
  if (Kokkos::Tools::profileLibraryLoaded()) {
    Kokkos::Tools::endParallelScan(kpID);
  }
#ifdef KOKKOS_ENABLE_TUNING
  size_t context_id = Kokkos::Tools::Experimental::get_current_context_id();
  if (Kokkos::tune_internals()) {
    Experimental::Impl::report_policy_results(
        context_id, label, policy, functor, Kokkos::ParallelScanTag{});
  }
#else
  (void)policy;
  (void)functor;
  (void)label;
#endif
}

template <class ReducerType, class ExecPolicy, class FunctorType>
void begin_parallel_reduce(ExecPolicy& policy, FunctorType& functor,
                           const std::string& label, uint64_t& kpID) {
  if (Kokkos::Tools::profileLibraryLoaded()) {
    Kokkos::Impl::ParallelConstructName<FunctorType,
                                        typename ExecPolicy::work_tag>
        name(label);
    Kokkos::Tools::beginParallelReduce(
        name.get(), Kokkos::Profiling::Experimental::device_id(policy.space()),
        &kpID);
  }
#ifdef KOKKOS_ENABLE_TUNING
  size_t context_id = Kokkos::Tools::Experimental::get_new_context_id();
  Experimental::Impl::ReductionSwitcher<ReducerType>::tune(
      context_id, label, policy, functor, Kokkos::ParallelReduceTag{});
#else
  (void)functor;
#endif
}

template <class ReducerType, class ExecPolicy, class FunctorType>
void end_parallel_reduce(ExecPolicy& policy, FunctorType& functor,
                         const std::string& label, uint64_t& kpID) {
  if (Kokkos::Tools::profileLibraryLoaded()) {
    Kokkos::Tools::endParallelReduce(kpID);
  }
#ifdef KOKKOS_ENABLE_TUNING
  size_t context_id = Kokkos::Tools::Experimental::get_current_context_id();
  if (Kokkos::tune_internals()) {
    Experimental::Impl::report_policy_results(
        context_id, label, policy, functor, Kokkos::ParallelReduceTag{});
  }
#else
  (void)policy;
  (void)functor;
  (void)label;
#endif
}

}  // end namespace Impl

}  // namespace Tools

}  // namespace Kokkos

#endif  // header guard
