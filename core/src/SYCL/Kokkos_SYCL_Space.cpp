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

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#endif

#include <Kokkos_Macros.hpp>

#include <Kokkos_Core.hpp>  // Kokkos::DefaultHostExecutionSpace
#include <Kokkos_HostSpace.hpp>
#include <SYCL/Kokkos_SYCL.hpp>
#include <SYCL/Kokkos_SYCL_Space.hpp>
#include <SYCL/Kokkos_SYCL_DeepCopy.hpp>
#include <SYCL/Kokkos_SYCL_Instance.hpp>
#include <impl/Kokkos_Profiling.hpp>

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/
namespace Kokkos {
namespace Impl {

void DeepCopySYCL(void* dst, const void* src, size_t n) {
  Impl::SYCLInternal::singleton().m_queue->memcpy(dst, src, n);
}

void DeepCopyAsyncSYCL(const Kokkos::SYCL& instance, void* dst, const void* src,
                       size_t n) {
  sycl::queue& q = *instance.impl_internal_space_instance()->m_queue;
  auto event     = q.memcpy(dst, src, n);
#ifndef KOKKOS_IMPL_SYCL_USE_IN_ORDER_QUEUES
  q.ext_oneapi_submit_barrier(std::vector<sycl::event>{event});
#endif
}

void DeepCopyAsyncSYCL(void* dst, const void* src, size_t n) {
  Impl::SYCLInternal::singleton().m_queue->memcpy(dst, src, n);
  SYCL().fence("Kokkos::Impl::DeepCopyAsyncSYCL: fence after memcpy");
}

}  // namespace Impl
}  // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/
namespace {

std::string_view get_memory_space_name(sycl::usm::alloc allocation_kind) {
  switch (allocation_kind) {
    case sycl::usm::alloc::host: return Kokkos::SYCLHostUSMSpace::name();
    case sycl::usm::alloc::device: return Kokkos::SYCLDeviceUSMSpace::name();
    case sycl::usm::alloc::shared: return Kokkos::SYCLSharedUSMSpace::name();
    case sycl::usm::alloc::unknown: return "sycl::usm::alloc::unknown";
    default:
      Kokkos::abort("bug: unknown sycl allocation type");
      return "unreachable";
  }
}

template <typename RequestedMemorySpace>
void check_memory_space(sycl::usm::alloc deduced_allocation_kind) {
  switch (deduced_allocation_kind) {
    case sycl::usm::alloc::host:
      if (!Kokkos::SpaceAccessibility<RequestedMemorySpace,
                                      Kokkos::SYCLHostUSMSpace>::assignable)
        Kokkos::abort(("Detected SYCLHostUSMSpace but requested incompatible " +
                       std::string(RequestedMemorySpace::name()))
                          .c_str());
      return;
    case sycl::usm::alloc::device:
      if (!Kokkos::SpaceAccessibility<RequestedMemorySpace,
                                      Kokkos::SYCLDeviceUSMSpace>::assignable)
        Kokkos::abort(
            ("Detected SYCLDeviceUSMSpace but requeste incompatible " +
             std::string(RequestedMemorySpace::name()))
                .c_str());
      return;
    case sycl::usm::alloc::shared:
      if (!Kokkos::SpaceAccessibility<RequestedMemorySpace,
                                      Kokkos::SYCLSharedUSMSpace>::assignable)
        Kokkos::abort(
            ("Detected SYCLSharedUSMSpace but requested incompatible " +
             std::string(RequestedMemorySpace::name()))
                .c_str());
      return;
    case sycl::usm::alloc::unknown:
      if (!Kokkos::SpaceAccessibility<RequestedMemorySpace,
                                      Kokkos::HostSpace>::assignable)
        Kokkos::abort(("Detected HostSpace but requested incompatible " +
                       std::string(RequestedMemorySpace::name()))
                          .c_str());
      return;
    default: Kokkos::abort("bug: unknown sycl allocation type");
  }
}

}  // namespace

template <>
void Kokkos::Impl::runtime_check_memory_space<Kokkos::SYCLHostUSMSpace>(
    const void* ptr, const Kokkos::SYCLHostUSMSpace& space) {
  check_memory_space<Kokkos::SYCLHostUSMSpace>(
      sycl::get_pointer_type(ptr, space.impl_get_queue().get_context()));
}

template <>
void Kokkos::Impl::runtime_check_memory_space<Kokkos::SYCLSharedUSMSpace>(
    const void* ptr, const Kokkos::SYCLSharedUSMSpace& space) {
  check_memory_space<Kokkos::SYCLSharedUSMSpace>(
      sycl::get_pointer_type(ptr, space.impl_get_queue().get_context()));
}

template <>
void Kokkos::Impl::runtime_check_memory_space<Kokkos::SYCLDeviceUSMSpace>(
    const void* ptr, const Kokkos::SYCLDeviceUSMSpace& space) {
  check_memory_space<Kokkos::SYCLDeviceUSMSpace>(
      sycl::get_pointer_type(ptr, space.impl_get_queue().get_context()));
}

template <>
void Kokkos::Impl::runtime_check_memory_space<Kokkos::HostSpace>(
    const void* ptr, const Kokkos::HostSpace&) {
  check_memory_space<Kokkos::HostSpace>(
      sycl::get_pointer_type(ptr, SYCL{}.sycl_queue().get_context()));
}

namespace Kokkos {

SYCLDeviceUSMSpace::SYCLDeviceUSMSpace()
    : m_queue(*SYCL().impl_internal_space_instance()->m_queue) {}
SYCLDeviceUSMSpace::SYCLDeviceUSMSpace(sycl::queue queue)
    : m_queue(std::move(queue)) {}

SYCLSharedUSMSpace::SYCLSharedUSMSpace()
    : m_queue(*SYCL().impl_internal_space_instance()->m_queue) {}
SYCLSharedUSMSpace::SYCLSharedUSMSpace(sycl::queue queue)
    : m_queue(std::move(queue)) {}

SYCLHostUSMSpace::SYCLHostUSMSpace()
    : m_queue(*SYCL().impl_internal_space_instance()->m_queue) {}
SYCLHostUSMSpace::SYCLHostUSMSpace(sycl::queue queue)
    : m_queue(std::move(queue)) {}

void* allocate_sycl(const char* arg_label, const size_t arg_alloc_size,
                    const size_t arg_logical_size,
                    const Kokkos::Tools::SpaceHandle arg_handle,
                    const sycl::usm::alloc allocation_kind,
                    const sycl::queue& queue) {
  void* const hostPtr = sycl::malloc(arg_alloc_size, queue, allocation_kind);

  if (hostPtr == nullptr) {
    Kokkos::Impl::throw_bad_alloc(get_memory_space_name(allocation_kind),
                                  arg_alloc_size, arg_label);
  }

  if (Kokkos::Profiling::profileLibraryLoaded()) {
    const size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    Kokkos::Profiling::allocateData(arg_handle, arg_label, hostPtr,
                                    reported_size);
  }

  return hostPtr;
}

void* SYCLDeviceUSMSpace::allocate(const Kokkos::SYCL& exec_space,
                                   const size_t arg_alloc_size) const {
  return allocate(exec_space, "[unlabeled]", arg_alloc_size);
}

void* SYCLDeviceUSMSpace::allocate(const Kokkos::SYCL& exec_space,
                                   const char* arg_label,
                                   const size_t arg_alloc_size,
                                   const size_t arg_logical_size) const {
  return allocate_sycl(arg_label, arg_alloc_size, arg_logical_size,
                       Kokkos::Tools::make_space_handle(name()),
                       sycl::usm::alloc::device,
                       *exec_space.impl_internal_space_instance()->m_queue);
}

void* SYCLDeviceUSMSpace::allocate(const size_t arg_alloc_size) const {
  return allocate("[unlabeled]", arg_alloc_size);
}

void* SYCLDeviceUSMSpace::allocate(const char* arg_label,
                                   const size_t arg_alloc_size,
                                   const size_t arg_logical_size) const {
  return allocate_sycl(arg_label, arg_alloc_size, arg_logical_size,
                       Kokkos::Tools::make_space_handle(name()),
                       sycl::usm::alloc::device, m_queue);
}

void* SYCLSharedUSMSpace::allocate(const SYCL& exec_space,
                                   const size_t arg_alloc_size) const {
  return allocate(exec_space, "[unlabeled]", arg_alloc_size);
}
void* SYCLSharedUSMSpace::allocate(const SYCL& exec_space,
                                   const char* arg_label,
                                   const size_t arg_alloc_size,
                                   const size_t arg_logical_size) const {
  return allocate_sycl(arg_label, arg_alloc_size, arg_logical_size,
                       Kokkos::Tools::make_space_handle(name()),
                       sycl::usm::alloc::shared,
                       *exec_space.impl_internal_space_instance()->m_queue);
}

void* SYCLSharedUSMSpace::allocate(const size_t arg_alloc_size) const {
  return allocate("[unlabeled]", arg_alloc_size);
}
void* SYCLSharedUSMSpace::allocate(const char* arg_label,
                                   const size_t arg_alloc_size,
                                   const size_t arg_logical_size) const {
  return allocate_sycl(arg_label, arg_alloc_size, arg_logical_size,
                       Kokkos::Tools::make_space_handle(name()),
                       sycl::usm::alloc::shared, m_queue);
}

void* SYCLHostUSMSpace::allocate(const SYCL& exec_space,
                                 const size_t arg_alloc_size) const {
  return allocate(exec_space, "[unlabeled]", arg_alloc_size);
}
void* SYCLHostUSMSpace::allocate(const SYCL& exec_space, const char* arg_label,
                                 const size_t arg_alloc_size,
                                 const size_t arg_logical_size) const {
  return allocate_sycl(arg_label, arg_alloc_size, arg_logical_size,
                       Kokkos::Tools::make_space_handle(name()),
                       sycl::usm::alloc::host,
                       *exec_space.impl_internal_space_instance()->m_queue);
}

void* SYCLHostUSMSpace::allocate(const size_t arg_alloc_size) const {
  return allocate("[unlabeled]", arg_alloc_size);
}
void* SYCLHostUSMSpace::allocate(const char* arg_label,
                                 const size_t arg_alloc_size,
                                 const size_t arg_logical_size) const {
  return allocate_sycl(arg_label, arg_alloc_size, arg_logical_size,
                       Kokkos::Tools::make_space_handle(name()),
                       sycl::usm::alloc::host, m_queue);
}

void sycl_deallocate(const char* arg_label, void* const arg_alloc_ptr,
                     const size_t arg_alloc_size, const size_t arg_logical_size,
                     const Kokkos::Tools::SpaceHandle arg_handle,
                     const sycl::queue& queue) {
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    const size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    Kokkos::Profiling::deallocateData(arg_handle, arg_label, arg_alloc_ptr,
                                      reported_size);
  }

  SYCL::impl_static_fence(
      "Kokkos::Impl::sycl_deallocate: fence before deallocate");
  sycl::free(arg_alloc_ptr, queue);
}

void SYCLDeviceUSMSpace::deallocate(void* const arg_alloc_ptr,
                                    const size_t arg_alloc_size) const {
  deallocate("[unlabeled]", arg_alloc_ptr, arg_alloc_size);
}
void SYCLDeviceUSMSpace::deallocate(const char* arg_label,
                                    void* const arg_alloc_ptr,
                                    const size_t arg_alloc_size,
                                    const size_t arg_logical_size) const {
  sycl_deallocate(arg_label, arg_alloc_ptr, arg_alloc_size, arg_logical_size,
                  Kokkos::Tools::make_space_handle(name()), m_queue);
}

void SYCLSharedUSMSpace::deallocate(void* const arg_alloc_ptr,
                                    const size_t arg_alloc_size) const {
  deallocate("[unlabeled]", arg_alloc_ptr, arg_alloc_size);
}

void SYCLSharedUSMSpace::deallocate(const char* arg_label,
                                    void* const arg_alloc_ptr,
                                    const size_t arg_alloc_size,
                                    const size_t arg_logical_size) const {
  sycl_deallocate(arg_label, arg_alloc_ptr, arg_alloc_size, arg_logical_size,
                  Kokkos::Tools::make_space_handle(name()), m_queue);
}

void SYCLHostUSMSpace::deallocate(void* const arg_alloc_ptr,
                                  const size_t arg_alloc_size) const {
  deallocate("[unlabeled]", arg_alloc_ptr, arg_alloc_size);
}

void SYCLHostUSMSpace::deallocate(const char* arg_label,
                                  void* const arg_alloc_ptr,
                                  const size_t arg_alloc_size,
                                  const size_t arg_logical_size) const {
  sycl_deallocate(arg_label, arg_alloc_ptr, arg_alloc_size, arg_logical_size,
                  Kokkos::Tools::make_space_handle(name()), m_queue);
}

}  // namespace Kokkos

//==============================================================================
// <editor-fold desc="Explicit instantiations of CRTP Base classes"> {{{1

#include <impl/Kokkos_SharedAlloc_timpl.hpp>

KOKKOS_IMPL_HOST_INACCESSIBLE_SHARED_ALLOCATION_RECORD_EXPLICIT_INSTANTIATION(
    Kokkos::SYCLDeviceUSMSpace);
KOKKOS_IMPL_SHARED_ALLOCATION_RECORD_EXPLICIT_INSTANTIATION(
    Kokkos::SYCLSharedUSMSpace);
KOKKOS_IMPL_SHARED_ALLOCATION_RECORD_EXPLICIT_INSTANTIATION(
    Kokkos::SYCLHostUSMSpace);

// </editor-fold> end Explicit instantiations of CRTP Base classes }}}1
//==============================================================================
