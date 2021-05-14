/* 
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_SYCL_CONVERSIONS_HPP_
#define DESUL_ATOMICS_SYCL_CONVERSIONS_HPP_
#ifdef DESUL_HAVE_SYCL_ATOMICS
#include "desul/atomics/Common.hpp"
#include <CL/sycl.hpp>

namespace desul {

template<class MemoryOrder>
struct DesulToSYCLMemoryOrder;
template<>
struct DesulToSYCLMemoryOrder<MemoryOrderSeqCst> {
  static constexpr sycl::ONEAPI::memory_order value = sycl::ONEAPI::memory_order::seq_cst;
};
template<>
struct DesulToSYCLMemoryOrder<MemoryOrderAcquire> {
  static constexpr sycl::ONEAPI::memory_order value = sycl::ONEAPI::memory_order::acquire;
};
template<>
struct DesulToSYCLMemoryOrder<MemoryOrderRelease> {
  static constexpr sycl::ONEAPI::memory_order value = sycl::ONEAPI::memory_order::release;
};
template<>
struct DesulToSYCLMemoryOrder<MemoryOrderAcqRel> {
  static constexpr sycl::ONEAPI::memory_order value = sycl::ONEAPI::memory_order::acq_rel;
};
template<>
struct DesulToSYCLMemoryOrder<MemoryOrderRelaxed> {
  static constexpr sycl::ONEAPI::memory_order value = sycl::ONEAPI::memory_order::relaxed;
};

template<class MemoryScope>
struct DesulToSYCLMemoryScope;
template<>
struct DesulToSYCLMemoryScope<MemoryScopeCore> {
  static constexpr sycl::ONEAPI::memory_scope value = sycl::ONEAPI::memory_scope::work_group;
};
template<>
struct DesulToSYCLMemoryScope<MemoryScopeDevice> {
  static constexpr sycl::ONEAPI::memory_scope value = sycl::ONEAPI::memory_scope::device;
};
template<>
struct DesulToSYCLMemoryScope<MemoryScopeSystem> {
  static constexpr sycl::ONEAPI::memory_scope value = sycl::ONEAPI::memory_scope::system;
};

}

#endif
#endif
