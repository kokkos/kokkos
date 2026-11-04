// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_EXPERIMENTAL_ERROR_REPORTER_HPP
#define KOKKOS_EXPERIMENTAL_ERROR_REPORTER_HPP
#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_ERRORREPORTER
#endif

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

#include <cstddef>
#include <vector>
#include <string>
#include <algorithm>

namespace Kokkos {
namespace Experimental {
template <typename LogType,
          typename DeviceType = typename DefaultExecutionSpace::device_type>
class Logger {
 public:
  using log_type        = LogType;
  using device_type     = DeviceType;
  using execution_space = typename device_type::execution_space;

  Logger(const std::string& label, int max_results)
      : m_insertionAttempted(label + "::m_insertionAttempted"),
        m_logs(label + "::m_logs", max_results) {
    clear();
  }

  Logger(int max_results) : Logger("Logger", max_results) {}

  int capacity() const { return m_logs.extent(0); }

  int size() const { return std::clamp(insertion_attempts(), 0, capacity()); }

  int insertion_attempts() const {
    int value;
    Kokkos::deep_copy(value, m_insertionAttempted);
    return value;
  }

  std::vector<log_type> get() const {
    int num_elements = size();
    std::vector<log_type> res(num_elements);

    if (num_elements > 0) {
      Kokkos::View<log_type*, Kokkos::HostSpace> h_logs(res.data(),
                                                        num_elements);

      Kokkos::deep_copy(h_logs,
                        Kokkos::subview(m_logs, Kokkos::pair{0, num_elements}));
    }
    return res;
  }

  bool full() const { return (insertion_attempts() >= capacity()); }

  void clear() const { Kokkos::deep_copy(m_insertionAttempted, 0); }

  // This function keeps logs up to new_size alive
  // It may lose the information on attempted logs
  void resize(const size_t new_size) {
    // We have to reset the attempts so we don't accidentally
    // report more stored logs than there actually are
    // after growing capacity.
    int attempts = insertion_attempts();
    if (new_size > static_cast<size_t>(capacity()) && attempts > capacity())
      Kokkos::deep_copy(m_insertionAttempted, size());

    Kokkos::resize(m_logs, new_size);
  }

  KOKKOS_INLINE_FUNCTION bool try_push(const LogType&& log) const {
    return try_emplace(log) != nullptr;
  }

  template <class... Args>
  KOKKOS_INLINE_FUNCTION log_type* try_emplace(Args&&... args) const {
    int idx = Kokkos::atomic_fetch_inc(&m_insertionAttempted());

    if (idx >= 0 && (idx < m_logs.extent_int(0))) {
      log_type* mem = &m_logs(idx);
      mem->~LogType();
      return new (mem) LogType{args...};
    } else {
      return nullptr;
    }
  }

 private:
  Kokkos::View<int, device_type> m_insertionAttempted;
  Kokkos::View<log_type*, device_type> m_logs;
};

template <typename ErrorType,
          typename DeviceType = typename DefaultExecutionSpace::device_type>
struct ErrorReport {
  int reporter_id;
  ErrorType error;
};

template <typename ErrorType,
          typename DeviceType = typename DefaultExecutionSpace::device_type>
using ErrorReporter = Logger<ErrorReport<ErrorType>, DeviceType>;

}  // namespace Experimental
}  // namespace Kokkos

#ifdef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_ERRORREPORTER
#undef KOKKOS_IMPL_PUBLIC_INCLUDE
#undef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_ERRORREPORTER
#endif
#endif
