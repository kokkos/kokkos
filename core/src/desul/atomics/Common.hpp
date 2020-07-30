/* 
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_COMMON_HPP_
#define DESUL_ATOMICS_COMMON_HPP_
#include "desul/atomics/Macros.hpp"
#include <cstdint>
#include <atomic>

namespace desul {
struct alignas(16) Dummy16ByteValue {
  int64_t value1;
  int64_t value2;
  bool operator!=(Dummy16ByteValue v) const {
    return (value1 != v.value1) || (value2 != v.value2);
  }
  bool operator==(Dummy16ByteValue v) const {
    return (value1 == v.value1) && (value2 == v.value2);
  }
};
}  // namespace desul

// MemoryOrder Tags

namespace desul {
// Memory order sequential consistent
struct MemoryOrderSeqCst {};
// Memory order acquire release
struct MemoryOrderAcqRel {};
// Memory order acquire
struct MemoryOrderAcquire {};
// Memory order release
struct MemoryOrderRelease {};
// Memory order relaxed
struct MemoryOrderRelaxed {};
}  // namespace desul

// Memory Scope Tags

namespace desul {
// Entire machine scope (e.g. for global arrays)
struct MemoryScopeSystem {};
// Node level
struct MemoryScopeNode {};
// Device or socket scope (i.e. a CPU socket, a single GPU)
struct MemoryScopeDevice {};
// Core scoped (i.e. a shared Level 1 cache)
struct MemoryScopeCore {};
}  // namespace desul

#ifndef __ATOMIC_RELAXED
#define __ATOMIC_RELAXED 0
#define __ATOMIC_CONSUME 1
#define __ATOMIC_ACQUIRE 2
#define __ATOMIC_RELEASE 3
#define __ATOMIC_ACQ_REL 4
#define __ATOMIC_SEQ_CST 5
#endif

namespace desul {
template <class MemoryOrderDesul>
struct GCCMemoryOrder;

template <>
struct GCCMemoryOrder<MemoryOrderRelaxed> {
  static constexpr int value = __ATOMIC_RELAXED;
};

template <>
struct GCCMemoryOrder<MemoryOrderAcquire> {
  static constexpr int value = __ATOMIC_ACQUIRE;
};

template <>
struct GCCMemoryOrder<MemoryOrderRelease> {
  static constexpr int value = __ATOMIC_RELEASE;
};

template <>
struct GCCMemoryOrder<MemoryOrderAcqRel> {
  static constexpr int value = __ATOMIC_ACQ_REL;
};

template <>
struct GCCMemoryOrder<MemoryOrderSeqCst> {
  static constexpr int value = __ATOMIC_SEQ_CST;
};

template <class MemoryOrderDesul>
struct CXXMemoryOrder;

template <>
struct CXXMemoryOrder<MemoryOrderRelaxed> {
  static constexpr std::memory_order value = std::memory_order_relaxed;
};

template <>
struct CXXMemoryOrder<MemoryOrderAcquire> {
  static constexpr std::memory_order value = std::memory_order_acquire;
};

template <>
struct CXXMemoryOrder<MemoryOrderRelease> {
  static constexpr std::memory_order value = std::memory_order_release;
};

template <>
struct CXXMemoryOrder<MemoryOrderAcqRel> {
  static constexpr std::memory_order value = std::memory_order_acq_rel;
};

template <>
struct CXXMemoryOrder<MemoryOrderSeqCst> {
  static constexpr std::memory_order value = std::memory_order_seq_cst;
};

namespace Impl {
template <typename MemoryOrder>
struct CmpExchFailureOrder {
  using memory_order = std::conditional_t<
      std::is_same<MemoryOrder, MemoryOrderAcqRel>{},
      MemoryOrderAcquire,
      std::conditional_t<std::is_same<MemoryOrder, MemoryOrderRelease>{},
                         MemoryOrderRelaxed,
                         MemoryOrder>>;
};
template <typename MemoryOrder>
using cmpexch_failure_memory_order =
    typename CmpExchFailureOrder<MemoryOrder>::memory_order;
}  // namespace Impl

}

// We should in principle use std::numeric_limits, but that requires constexpr function support on device
// Currently that is still considered experimetal on CUDA and sometimes not reliable.
namespace desul {
namespace Impl {
  template<class T>
  struct numeric_limits_max;

  template<>
  struct numeric_limits_max<uint32_t> {
    static constexpr uint32_t value = 0xffffffffu;
  };
  template<>
  struct numeric_limits_max<uint64_t> {
    static constexpr uint64_t value = 0xfffffffflu;
  };

}
}
#endif
