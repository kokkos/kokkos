// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_PARTITION_SPACE_HPP
#define KOKKOS_PARTITION_SPACE_HPP

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_PARTITION_SPACE
#endif

#include <Kokkos_Concepts.hpp>

#include <algorithm>
#include <array>
#include <iterator>
#include <ranges>
#include <type_traits>
#include <vector>

namespace Kokkos::Experimental::Impl {

// Customization point for backends. Default behavior is to return the passed
// in instance, ignoring weights
template <class ExecSpace, std::ranges::input_range Weights,
          std::output_iterator<ExecSpace> OutIter>
  requires(is_execution_space_v<ExecSpace>)
void impl_partition_space(const ExecSpace& base_instance,
                          const Weights& weights, OutIter out) {
  std::ranges::generate_n(out, std::ranges::size(weights),
                          [&base_instance] { return base_instance; });
}

}  // namespace Kokkos::Experimental::Impl

namespace Kokkos::Experimental {

// Partitioning an Execution Space
// Input:
//   - Base execution space
//   - integer arguments for relative weight, either input per weight or vector
//   of weights
// Ouput:
//   - Array (or vector) of execution spaces partitioned based on weights
template <class ExecSpace, class... Args>
  requires(is_execution_space_v<ExecSpace> &&
           (std::is_arithmetic_v<Args> && ...))
std::array<ExecSpace, sizeof...(Args)> partition_space(
    ExecSpace const& base_instance, Args... args) {
  using weight_type  = std::common_type_t<Args...>;
  constexpr size_t N = sizeof...(Args);

  std::array<ExecSpace, N> instances;
  Impl::impl_partition_space(base_instance, std::array<weight_type, N>{args...},
                             instances.begin());
  return instances;
}

template <class ExecSpace, class T>
  requires(is_execution_space_v<ExecSpace> && std::is_arithmetic_v<T>)
std::vector<ExecSpace> partition_space(ExecSpace const& base_instance,
                                       std::vector<T> const& weights) {
  std::vector<ExecSpace> instances;
  instances.reserve(weights.size());
  Impl::impl_partition_space(base_instance, weights,
                             std::back_inserter(instances));
  return instances;
}

}  // namespace Kokkos::Experimental

#ifdef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_PARTITION_SPACE
#undef KOKKOS_IMPL_PUBLIC_INCLUDE
#undef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_PARTITION_SPACE
#endif

#endif
