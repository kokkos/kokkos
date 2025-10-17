// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_OPENACC_INSTANCE_HPP
#define KOKKOS_OPENACC_INSTANCE_HPP

#include <impl/Kokkos_InitializationSettings.hpp>

#include <openacc.h>

#include <algorithm>
#include <cstdint>
#include <iosfwd>
#include <iterator>
#include <ranges>
#include <string>

namespace Kokkos::Experimental::Impl {

class OpenACCInternal {
  bool m_is_initialized = false;

  OpenACCInternal(const OpenACCInternal&)            = default;
  OpenACCInternal& operator=(const OpenACCInternal&) = default;

 public:
  static int m_acc_device_num;
  static int m_concurrency;
  static int m_next_async;
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

// For each space in partition, assign a new async ID, ignoring weights
template <std::ranges::input_range Weights,
          std::output_iterator<OpenACC> OutIter>
void impl_partition_space(const OpenACC& base_instance, const Weights& weights,
                          OutIter out) {
  constexpr int KOKKOS_IMPL_ACC_ASYNC_RANGE_BEGIN  = 64;
  constexpr int KOKKOS_IMPL_ACC_ASYNC_RANGE_LENGTH = 128;
  std::ranges::generate_n(out, std::ranges::size(weights), [] {
    OpenACCInternal::m_next_async = (OpenACCInternal::m_next_async + 1) %
                                    KOKKOS_IMPL_ACC_ASYNC_RANGE_LENGTH;
    return OpenACC(OpenACCInternal::m_next_async +
                   KOKKOS_IMPL_ACC_ASYNC_RANGE_BEGIN);
  });
}

}  // namespace Kokkos::Experimental::Impl

#endif
