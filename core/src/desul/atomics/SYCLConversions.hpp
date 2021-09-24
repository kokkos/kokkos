/* 
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_SYCL_CONVERSIONS_HPP_
#define DESUL_ATOMICS_SYCL_CONVERSIONS_HPP_
#ifdef DESUL_HAVE_SYCL_ATOMICS
#include <CL/sycl.hpp>

#include "desul/atomics/Common.hpp"

namespace desul {
namespace Impl {

#ifdef __clang__
namespace sycl_sync_and_atomics = ::sycl::ONEAPI;
#else
namespace sycl_sync_and_atomics = ::sycl;
#endif

using sycl_memory_order = sycl_sync_and_atomics::memory_order;
using sycl_memory_scope = sycl_sync_and_atomics::memory_scope;

template <class MemoryOrder>
struct DesulToSYCLMemoryOrder;
template <>
struct DesulToSYCLMemoryOrder<MemoryOrderSeqCst> {
  static constexpr sycl_memory_order value = sycl_memory_order::seq_cst;
};
template <>
struct DesulToSYCLMemoryOrder<MemoryOrderAcquire> {
  static constexpr sycl_memory_order value = sycl_memory_order::acquire;
};
template <>
struct DesulToSYCLMemoryOrder<MemoryOrderRelease> {
  static constexpr sycl_memory_order value = sycl_memory_order::release;
};
template <>
struct DesulToSYCLMemoryOrder<MemoryOrderAcqRel> {
  static constexpr sycl_memory_order value = sycl_memory_order::acq_rel;
};
template <>
struct DesulToSYCLMemoryOrder<MemoryOrderRelaxed> {
  static constexpr sycl_memory_order value = sycl_memory_order::relaxed;
};

template <class MemoryScope>
struct DesulToSYCLMemoryScope;
template <>
struct DesulToSYCLMemoryScope<MemoryScopeCore> {
  static constexpr sycl_memory_scope value = sycl_memory_scope::work_group;
};
template <>
struct DesulToSYCLMemoryScope<MemoryScopeDevice> {
  static constexpr sycl_memory_scope value = sycl_memory_scope::device;
};
template <>
struct DesulToSYCLMemoryScope<MemoryScopeSystem> {
  static constexpr sycl_memory_scope value = sycl_memory_scope::system;
};

template <class T, class MemoryOrder, class MemoryScope>
using sycl_atomic_ref =
    sycl_sync_and_atomics::atomic_ref<T,
                                      DesulToSYCLMemoryOrder<MemoryOrder>::value,
                                      DesulToSYCLMemoryScope<MemoryScope>::value,
                                      sycl::access::address_space::global_device_space>;

}  // namespace Impl
}  // namespace desul

#endif
#endif
