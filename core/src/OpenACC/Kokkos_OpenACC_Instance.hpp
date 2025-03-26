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

#ifndef KOKKOS_OPENACC_INSTANCE_HPP
#define KOKKOS_OPENACC_INSTANCE_HPP

#include <impl/Kokkos_InitializationSettings.hpp>

#include <openacc.h>

#include <cstdint>
#include <iosfwd>
#include <string>

namespace Kokkos::Experimental::Impl {

class OpenACCInternal {
  bool m_is_initialized = false;

  OpenACCInternal(const OpenACCInternal&)            = default;
  OpenACCInternal& operator=(const OpenACCInternal&) = default;

 public:
  static int m_acc_device_num;
  static int m_concurrency;
  int m_async_arg = acc_async_noval;

  OpenACCInternal() = default;

  static OpenACCInternal& singleton();

  bool verify_is_initialized(const char* const label) const;

  void initialize(int async_arg = acc_async_noval);
  void finalize();
  bool is_initialized() const;

  void print_configuration(std::ostream& os, bool verbose = false) const;

  void fence(std::string const& name) const;

  uint32_t instance_id() const noexcept;
};

void create_OpenACC_instances(std::vector<OpenACC>& instances);

}  // namespace Kokkos::Experimental::Impl

namespace Kokkos::Experimental {
// Partitioning an Execution Space: expects space and integer arguments for
// relative weight
//   Customization point for backends
//   Default behavior is to return the passed in instance

template <class... Args>
std::vector<OpenACC> partition_space(const OpenACC&, Args...) {
  static_assert(
      (... && std::is_arithmetic_v<Args>),
      "Kokkos Error: partitioning arguments must be integers or floats");
  std::vector<OpenACC> instances(sizeof...(Args));
  Kokkos::Experimental::Impl::create_OpenACC_instances(instances);
  return instances;
}

template <class T>
std::vector<OpenACC> partition_space(const OpenACC&,
                                     std::vector<T> const& weights) {
  static_assert(
      std::is_arithmetic_v<T>,
      "Kokkos Error: partitioning arguments must be integers or floats");

  // We only care about the number of instances to create and ignore weights
  // otherwise.
  std::vector<OpenACC> instances(weights.size());
  Kokkos::Experimental::Impl::create_OpenACC_instances(instances);
  return instances;
}

}  // namespace Kokkos::Experimental

#endif
