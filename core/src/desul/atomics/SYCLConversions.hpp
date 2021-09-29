/* 
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_SYCL_CONVERSIONS_HPP_
#define DESUL_ATOMICS_SYCL_CONVERSIONS_HPP_
#ifdef DESUL_HAVE_SYCL_ATOMICS

// clang-format off
#include "desul/atomics/Common.hpp"

#include <CL/sycl.hpp>
// clang-format on

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
using sycl_atomic_ref = sycl_sync_and_atomics::atomic_ref<
    T,
    DesulToSYCLMemoryOrder<MemoryOrder>::value,
    DesulToSYCLMemoryScope<MemoryScope>::value,
    // FIXME In SYCL 2020 Specification (revision 3) the class template atomic_ref has
    // its trailing (non-type) template parameter defaulted to
    // access::address_space::generic_space, but in currently available implementations
    // a/ the template parameter has no default template argument, and b/ the
    // generic_space enumerator is not a valid address space (only global_space,
    // local_space, and global_space are).  Worse it is not yet defined as part of the
    // access::address_space enumerator list.
    // Here we arbitrarily elected to use global_space as a temporary workaround.
    sycl::access::address_space::global_space>;

}  // namespace Impl
}  // namespace desul

#endif
#endif
