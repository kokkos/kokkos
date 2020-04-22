/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
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

#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <atomic>
#include <Kokkos_Macros.hpp>

/* only compile this file if SYCL is enabled for Kokkos */
#ifdef KOKKOS_ENABLE_SYCL

#include <Kokkos_Core.hpp>
#include <Kokkos_SYCL.hpp>
#include <Kokkos_SYCL_Space.hpp>

#include <impl/Kokkos_Error.hpp>

#if defined(KOKKOS_ENABLE_PROFILING)
#include <impl/Kokkos_Profiling_Interface.hpp>
#endif

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/
namespace Kokkos {
namespace Impl {
namespace {
void USM_memcpy(cl::sycl::queue& q, void* dst, const void* src, size_t n) {
  cl::sycl::event e = q.memcpy(dst, src, n);
  e.wait();
}

void USM_memcpy(Kokkos::Experimental::Impl::SYCLInternal& space, void* dst,
                const void* src, size_t n) {
  USM_memcpy(*space.m_queue, dst, src, n);
}

void USM_memcpy(void* dst, const void* src, size_t n) {
  USM_memcpy(Kokkos::Experimental::Impl::SYCLInternal::singleton(), dst, src,
             n);
}
}  // namespace

DeepCopy<Kokkos::Experimental::SYCLSpace, Kokkos::Experimental::SYCLSpace,
         Kokkos::Experimental::SYCL>::DeepCopy(void* dst, const void* src,
                                               size_t n) {
  //  syclMemcpy(dst , src , n , syclMemcpyDefault );
}

DeepCopy<HostSpace, Kokkos::Experimental::SYCLSpace,
         Kokkos::Experimental::SYCL>::DeepCopy(void* dst, const void* src,
                                               size_t n) {
  //  syclMemcpy(dst , src , n , syclMemcpyDefault );
}

DeepCopy<Kokkos::Experimental::SYCLSpace, HostSpace,
         Kokkos::Experimental::SYCL>::DeepCopy(void* dst, const void* src,
                                               size_t n) {
  //  syclMemcpy(dst , src , n , syclMemcpyDefault );
}

DeepCopy<Kokkos::Experimental::SYCLSpace, Kokkos::Experimental::SYCLSpace,
         Kokkos::Experimental::SYCL>::DeepCopy(const Kokkos::Experimental::SYCL&
                                                   instance,
                                               void* dst, const void* src,
                                               size_t n) {
  //  syclMemcpy(dst , src , n , syclMemcpyDefault );
}

DeepCopy<HostSpace, Kokkos::Experimental::SYCLSpace,
         Kokkos::Experimental::SYCL>::DeepCopy(const Kokkos::Experimental::SYCL&
                                                   instance,
                                               void* dst, const void* src,
                                               size_t n) {
  //  syclMemcpy(dst , src , n , syclMemcpyDefault );
}

DeepCopy<Kokkos::Experimental::SYCLSpace, HostSpace,
         Kokkos::Experimental::SYCL>::DeepCopy(const Kokkos::Experimental::SYCL&
                                                   instance,
                                               void* dst, const void* src,
                                               size_t n) {
  //  syclMemcpy(dst , src , n , syclMemcpyDefault );
}

DeepCopy<Kokkos::Experimental::SYCLHostPinnedSpace,
         Kokkos::Experimental::SYCLHostPinnedSpace,
         Kokkos::Experimental::SYCL>::DeepCopy(void* dst, const void* src,
                                               size_t n) {
  //  syclMemcpy(dst , src , n , syclMemcpyDefault );
}

DeepCopy<HostSpace, Kokkos::Experimental::SYCLHostPinnedSpace,
         Kokkos::Experimental::SYCL>::DeepCopy(void* dst, const void* src,
                                               size_t n) {
  //  syclMemcpy(dst , src , n , syclMemcpyDefault );
}

DeepCopy<Kokkos::Experimental::SYCLHostPinnedSpace, HostSpace,
         Kokkos::Experimental::SYCL>::DeepCopy(void* dst, const void* src,
                                               size_t n) {
  //  syclMemcpy(dst , src , n , syclMemcpyDefault );
}

DeepCopy<Kokkos::Experimental::SYCLHostPinnedSpace,
         Kokkos::Experimental::SYCLHostPinnedSpace,
         Kokkos::Experimental::SYCL>::DeepCopy(const Kokkos::Experimental::SYCL&
                                                   instance,
                                               void* dst, const void* src,
                                               size_t n) {
  //  syclMemcpy(dst , src , n , syclMemcpyDefault );
}

DeepCopy<HostSpace, Kokkos::Experimental::SYCLHostPinnedSpace,
         Kokkos::Experimental::SYCL>::DeepCopy(const Kokkos::Experimental::SYCL&
                                                   instance,
                                               void* dst, const void* src,
                                               size_t n) {
  //  syclMemcpy(dst , src , n , syclMemcpyDefault );
}

DeepCopy<Kokkos::Experimental::SYCLHostPinnedSpace, HostSpace,
         Kokkos::Experimental::SYCL>::DeepCopy(const Kokkos::Experimental::SYCL&
                                                   instance,
                                               void* dst, const void* src,
                                               size_t n) {
  //  syclMemcpy(dst , src , n , syclMemcpyDefault );
}

DeepCopy<Kokkos::Experimental::SYCLHostUSMSpace,
         Kokkos::Experimental::SYCLHostUSMSpace, Kokkos::Experimental::SYCL>::
    DeepCopy(const Kokkos::Experimental::SYCL& instance, void* dst,
             const void* src, size_t n) {
  USM_memcpy(*instance.impl_internal_space_instance(), dst, src, n);
}

DeepCopy<Kokkos::Experimental::SYCLHostUSMSpace,
         Kokkos::Experimental::SYCLHostUSMSpace,
         Kokkos::Experimental::SYCL>::DeepCopy(void* dst, const void* src,
                                               size_t n) {
  USM_memcpy(dst, src, n);
}

DeepCopy<Kokkos::HostSpace, Kokkos::Experimental::SYCLHostUSMSpace,
         Kokkos::Experimental::SYCL>::DeepCopy(void* dst, const void* src,
                                               size_t n) {
  USM_memcpy(dst, src, n);
}

DeepCopy<Kokkos::Experimental::SYCLHostUSMSpace, Kokkos::HostSpace,
         Kokkos::Experimental::SYCL>::DeepCopy(void* dst, const void* src,
                                               size_t n) {
  USM_memcpy(dst, src, n);
}


DeepCopy<Kokkos::Experimental::SYCLDeviceUSMSpace,
         Kokkos::Experimental::SYCLDeviceUSMSpace, Kokkos::Experimental::SYCL>::
    DeepCopy(const Kokkos::Experimental::SYCL& instance, void* dst,
             const void* src, size_t n) {
  USM_memcpy(*instance.impl_internal_space_instance(), dst, src, n);
}

DeepCopy<Kokkos::Experimental::SYCLDeviceUSMSpace,
         Kokkos::Experimental::SYCLDeviceUSMSpace,
         Kokkos::Experimental::SYCL>::DeepCopy(void* dst, const void* src,
                                               size_t n) {
  USM_memcpy(dst, src, n);
}

DeepCopy<Kokkos::HostSpace, Kokkos::Experimental::SYCLDeviceUSMSpace,
         Kokkos::Experimental::SYCL>::DeepCopy(void* dst, const void* src,
                                               size_t n) {
  USM_memcpy(dst, src, n);
}

DeepCopy<Kokkos::Experimental::SYCLDeviceUSMSpace, Kokkos::HostSpace,
         Kokkos::Experimental::SYCL>::DeepCopy(void* dst, const void* src,
                                               size_t n) {
  USM_memcpy(dst, src, n);
}


DeepCopy<Kokkos::Experimental::SYCLSharedUSMSpace,
         Kokkos::Experimental::SYCLSharedUSMSpace, Kokkos::Experimental::SYCL>::
    DeepCopy(const Kokkos::Experimental::SYCL& instance, void* dst,
             const void* src, size_t n) {
  USM_memcpy(*instance.impl_internal_space_instance(), dst, src, n);
}

DeepCopy<Kokkos::Experimental::SYCLSharedUSMSpace,
         Kokkos::Experimental::SYCLSharedUSMSpace,
         Kokkos::Experimental::SYCL>::DeepCopy(void* dst, const void* src,
                                               size_t n) {
  USM_memcpy(dst, src, n);
}

DeepCopy<Kokkos::HostSpace, Kokkos::Experimental::SYCLSharedUSMSpace,
         Kokkos::Experimental::SYCL>::DeepCopy(void* dst, const void* src,
                                               size_t n) {
  USM_memcpy(dst, src, n);
}

DeepCopy<Kokkos::Experimental::SYCLSharedUSMSpace, Kokkos::HostSpace,
         Kokkos::Experimental::SYCL>::DeepCopy(void* dst, const void* src,
                                               size_t n) {
  USM_memcpy(dst, src, n);
}

}  // namespace Impl
}  // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {

void Experimental::SYCLSpace::access_error() {
  const std::string msg(
      "Kokkos::Experimental::SYCLSpace::access_error attempt to execute "
      "Experimental::SYCL function from non-SYCL space");
  Kokkos::Impl::throw_runtime_exception(msg);
}

void Experimental::SYCLSpace::access_error(const void* const) {
  const std::string msg(
      "Kokkos::Experimental::SYCLSpace::access_error attempt to execute "
      "Experimental::SYCL function from non-SYCL space");
  Kokkos::Impl::throw_runtime_exception(msg);
}

}  // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Experimental {

SYCLSpace::SYCLSpace() : m_device(SYCL().sycl_device()) {}

SYCLHostPinnedSpace::SYCLHostPinnedSpace() {}

void* SYCLSpace::allocate(const size_t arg_alloc_size) const {
  void* ptr = NULL;
  // syclMalloc( &ptr, arg_alloc_size );
  return ptr;
}

void* Experimental::SYCLHostPinnedSpace::allocate(
    const size_t arg_alloc_size) const {
  void* ptr = NULL;
  // syclHostMalloc( &ptr, arg_alloc_size );
  return ptr;
}

void SYCLSpace::deallocate(void* const arg_alloc_ptr,
                           const size_t /* arg_alloc_size */) const {
  // syclFree(arg_alloc_ptr);
}

void Experimental::SYCLHostPinnedSpace::deallocate(
    void* const arg_alloc_ptr, const size_t /* arg_alloc_size */) const {
  // syclHostFree(arg_alloc_ptr);
}

SYCLHostUSMSpace::SYCLHostUSMSpace() : m_device(SYCL().sycl_device()) {}

void* SYCLHostUSMSpace::allocate(const size_t arg_alloc_size) const {
  const cl::sycl::queue& queue =
      *SYCL().impl_internal_space_instance()->m_queue;
  void* const hostPtr = cl::sycl::malloc_host(arg_alloc_size, queue);
  return hostPtr;
}

void SYCLHostUSMSpace::deallocate(void* const arg_alloc_ptr,
                                  const size_t arg_alloc_size) const {
  const cl::sycl::queue& queue =
      *SYCL().impl_internal_space_instance()->m_queue;
  cl::sycl::free(arg_alloc_ptr, queue);
}

}  // namespace Experimental
}  // namespace Kokkos

namespace Kokkos {
namespace Impl {

#ifdef KOKKOS_DEBUG
SharedAllocationRecord<void, void> SharedAllocationRecord<
    Kokkos::Experimental::SYCLHostUSMSpace, void>::s_root_record;
#endif

std::string SharedAllocationRecord<Kokkos::Experimental::SYCLHostUSMSpace,
                                   void>::get_label() const {
  return std::string(m_alloc_ptr->m_label);
}

SharedAllocationRecord<Kokkos::Experimental::SYCLHostUSMSpace, void>*
SharedAllocationRecord<Kokkos::Experimental::SYCLHostUSMSpace, void>::allocate(
    const Kokkos::Experimental::SYCLHostUSMSpace& space,
    const std::string& label, const size_t size) {
  return new SharedAllocationRecord(space, label, size);
}

void SharedAllocationRecord<Kokkos::Experimental::SYCLHostUSMSpace, void>::
    deallocate(SharedAllocationRecord<void, void>* rec) {
  delete static_cast<SharedAllocationRecord*>(rec);
}

SharedAllocationRecord<Kokkos::Experimental::SYCLHostUSMSpace,
                       void>::~SharedAllocationRecord() {
#if defined(KOKKOS_ENABLE_PROFILING)
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    SharedAllocationHeader header;
#ifdef NLIBER
    Kokkos::Impl::DeepCopy<Kokkos::Experimental::SYCLHostUSMSpace, HostSpace>(
        &header, SharedAllocationRecord<void, void>::m_alloc_ptr,
        sizeof(SharedAllocationHeader));
#else
    assert(false);
#endif

    Kokkos::Profiling::deallocateData(
        Kokkos::Profiling::SpaceHandle(
            Kokkos::Experimental::SYCLHostUSMSpace::name()),
        header.m_label, data(), size());
  }
#endif

  m_space.deallocate(SharedAllocationRecord<void, void>::m_alloc_ptr,
                     SharedAllocationRecord<void, void>::m_alloc_size);
}

SharedAllocationRecord<Kokkos::Experimental::SYCLHostUSMSpace, void>::
    SharedAllocationRecord(
        const Kokkos::Experimental::SYCLHostUSMSpace& space,
        const std::string& label, const size_t size,
        const SharedAllocationRecord<void, void>::function_type dealloc)
    // Pass through allocated [ SharedAllocationHeader , user_memory ]
    // Pass through deallocation function
    : SharedAllocationRecord<void, void>(
#ifdef KOKKOS_DEBUG
          &SharedAllocationRecord<Kokkos::Experimental::SYCLHostUSMSpace,
                                  void>::s_root_record,
#endif
          reinterpret_cast<SharedAllocationHeader*>(
              space.allocate(sizeof(SharedAllocationHeader) + size)),
          sizeof(SharedAllocationHeader) + size, dealloc),
      m_space(space) {
#if defined(KOKKOS_ENABLE_PROFILING)
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::allocateData(
        Kokkos::Profiling::SpaceHandle(space.name()), label, data(), size);
  }
#endif

  SharedAllocationHeader header;

  // Fill in the Header information
  header.m_record = static_cast<SharedAllocationRecord<void, void>*>(this);

  strncpy(header.m_label, label.c_str(),
          SharedAllocationHeader::maximum_label_length);
  // Set last element zero, in case c_str is too long
  header.m_label[SharedAllocationHeader::maximum_label_length - 1] = (char)0;

  memcpy(m_alloc_ptr, &header, sizeof(SharedAllocationHeader));
}

void* SharedAllocationRecord<Kokkos::Experimental::SYCLHostUSMSpace, void>::
    allocate_tracked(const Kokkos::Experimental::SYCLHostUSMSpace& space,
                     const std::string& label, const size_t size) {
  if (!size) return nullptr;

  SharedAllocationRecord* const r = allocate(space, label, size);

  SharedAllocationRecord<void, void>::increment(r);

  return r->data();
}

void SharedAllocationRecord<Kokkos::Experimental::SYCLHostUSMSpace,
                            void>::deallocate_tracked(void* const ptr) {
  if (ptr != 0) {
    SharedAllocationRecord* const r = get_record(ptr);

    SharedAllocationRecord<void, void>::decrement(r);
  }
}

void* SharedAllocationRecord<Kokkos::Experimental::SYCLHostUSMSpace,
                             void>::reallocate_tracked(void* const ptr,
                                                       const size_t size) {
  SharedAllocationRecord* const r_old = get_record(ptr);
  SharedAllocationRecord* const r_new =
      allocate(r_old->m_space, r_old->get_label(), size);

#if NLIBER
  Kokkos::Impl::DeepCopy<Kokkos::Experimental::SYCLHostUSMSpace,
                         Kokkos::Experimental::SYCLHostUSMSpace>(
      r_new->data(), r_old->data(), std::min(r_old->size(), r_new->size()));
#else
  assert(false);
#endif

  SharedAllocationRecord<void, void>::increment(r_new);
  SharedAllocationRecord<void, void>::decrement(r_old);

  return r_new->data();
}

SharedAllocationRecord<Kokkos::Experimental::SYCLHostUSMSpace, void>*
SharedAllocationRecord<Kokkos::Experimental::SYCLHostUSMSpace,
                       void>::get_record(void* alloc_ptr) {
  using Header     = SharedAllocationHeader;
  using RecordBase = SharedAllocationRecord<void, void>;
  using RecordROCm =
      SharedAllocationRecord<Kokkos::Experimental::SYCLHostUSMSpace, void>;

  // Copy the header from the allocation
  Header head;

  Header const* const head_rocm =
      alloc_ptr ? Header::get_header(alloc_ptr) : (Header*)0;

#define NLIBER 1
#if NLIBER
  if (alloc_ptr) {
    Kokkos::Impl::DeepCopy<HostSpace, Kokkos::Experimental::SYCLHostUSMSpace>(
        &head, head_rocm, sizeof(SharedAllocationHeader));
  }
#else
  assert(false);
#endif

  RecordROCm* const record =
      alloc_ptr ? static_cast<RecordROCm*>(head.m_record) : (RecordROCm*)0;

  if (!alloc_ptr || record->m_alloc_ptr != head_rocm) {
    Kokkos::Impl::throw_runtime_exception(std::string(
        "Kokkos::Impl::SharedAllocationRecord< Kokkos::Experimental::ROCmSpace "
        ", void >::get_record ERROR"));
  }

  return record;
}

}  // namespace Impl
}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

#ifdef KOKKOS_DEBUG
SharedAllocationRecord<void, void> SharedAllocationRecord<
    Kokkos::Experimental::SYCLSpace, void>::s_root_record;

SharedAllocationRecord<void, void> SharedAllocationRecord<
    Kokkos::Experimental::SYCLHostPinnedSpace, void>::s_root_record;
#endif

std::string SharedAllocationRecord<Kokkos::Experimental::SYCLSpace,
                                   void>::get_label() const {
  SharedAllocationHeader header;

  Kokkos::Impl::DeepCopy<Kokkos::HostSpace, Kokkos::Experimental::SYCLSpace>(
      &header, RecordBase::head(), sizeof(SharedAllocationHeader));

  return std::string(header.m_label);
}
std::string SharedAllocationRecord<Kokkos::Experimental::SYCLHostPinnedSpace,
                                   void>::get_label() const {
  return std::string(RecordBase::head()->m_label);
}

SharedAllocationRecord<Kokkos::Experimental::SYCLSpace, void>*
SharedAllocationRecord<Kokkos::Experimental::SYCLSpace, void>::allocate(
    const Kokkos::Experimental::SYCLSpace& arg_space,
    const std::string& arg_label, const size_t arg_alloc_size) {
  return new SharedAllocationRecord(arg_space, arg_label, arg_alloc_size);
}

SharedAllocationRecord<Kokkos::Experimental::SYCLHostPinnedSpace, void>*
SharedAllocationRecord<Kokkos::Experimental::SYCLHostPinnedSpace, void>::
    allocate(const Kokkos::Experimental::SYCLHostPinnedSpace& arg_space,
             const std::string& arg_label, const size_t arg_alloc_size) {
  return new SharedAllocationRecord(arg_space, arg_label, arg_alloc_size);
}

void SharedAllocationRecord<Kokkos::Experimental::SYCLSpace, void>::deallocate(
    SharedAllocationRecord<void, void>* arg_rec) {
  delete static_cast<SharedAllocationRecord*>(arg_rec);
}

void SharedAllocationRecord<Kokkos::Experimental::SYCLHostPinnedSpace, void>::
    deallocate(SharedAllocationRecord<void, void>* arg_rec) {
  delete static_cast<SharedAllocationRecord*>(arg_rec);
}

SharedAllocationRecord<Kokkos::Experimental::SYCLSpace,
                       void>::~SharedAllocationRecord() {
#if defined(KOKKOS_ENABLE_PROFILING)
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    SharedAllocationHeader header;
    Kokkos::Impl::DeepCopy<Kokkos::Experimental::SYCLSpace, HostSpace>(
        &header, RecordBase::m_alloc_ptr, sizeof(SharedAllocationHeader));

    Kokkos::Profiling::deallocateData(
        Kokkos::Profiling::SpaceHandle(Kokkos::Experimental::SYCLSpace::name()),
        header.m_label, data(), size());
  }
#endif

  m_space.deallocate(SharedAllocationRecord<void, void>::m_alloc_ptr,
                     SharedAllocationRecord<void, void>::m_alloc_size);
}

SharedAllocationRecord<Kokkos::Experimental::SYCLHostPinnedSpace,
                       void>::~SharedAllocationRecord() {
#if defined(KOKKOS_ENABLE_PROFILING)
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::deallocateData(
        Kokkos::Profiling::SpaceHandle(
            Kokkos::Experimental::SYCLHostPinnedSpace::name()),
        RecordBase::m_alloc_ptr->m_label, data(), size());
  }
#endif

  m_space.deallocate(SharedAllocationRecord<void, void>::m_alloc_ptr,
                     SharedAllocationRecord<void, void>::m_alloc_size);
}

SharedAllocationRecord<Kokkos::Experimental::SYCLSpace, void>::
    SharedAllocationRecord(
        const Kokkos::Experimental::SYCLSpace& arg_space,
        const std::string& arg_label, const size_t arg_alloc_size,
        const SharedAllocationRecord<void, void>::function_type arg_dealloc)
    // Pass through allocated [ SharedAllocationHeader , user_memory ]
    // Pass through deallocation function
    : SharedAllocationRecord<void, void>(
#ifdef KOKKOS_DEBUG
          &SharedAllocationRecord<Kokkos::Experimental::SYCLSpace,
                                  void>::s_root_record,
#endif
          reinterpret_cast<SharedAllocationHeader*>(arg_space.allocate(
              sizeof(SharedAllocationHeader) + arg_alloc_size)),
          sizeof(SharedAllocationHeader) + arg_alloc_size, arg_dealloc),
      m_space(arg_space) {
#if defined(KOKKOS_ENABLE_PROFILING)
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::allocateData(
        Kokkos::Profiling::SpaceHandle(arg_space.name()), arg_label, data(),
        arg_alloc_size);
  }
#endif

  SharedAllocationHeader header;

  // Fill in the Header information
  header.m_record = static_cast<SharedAllocationRecord<void, void>*>(this);

  strncpy(header.m_label, arg_label.c_str(),
          SharedAllocationHeader::maximum_label_length);
  // Set last element zero, in case c_str is too long
  header.m_label[SharedAllocationHeader::maximum_label_length - 1] = (char)0;

  // Copy to device memory
  Kokkos::Impl::DeepCopy<Kokkos::Experimental::SYCLSpace, HostSpace>(
      RecordBase::m_alloc_ptr, &header, sizeof(SharedAllocationHeader));
}

SharedAllocationRecord<Kokkos::Experimental::SYCLHostPinnedSpace, void>::
    SharedAllocationRecord(
        const Kokkos::Experimental::SYCLHostPinnedSpace& arg_space,
        const std::string& arg_label, const size_t arg_alloc_size,
        const SharedAllocationRecord<void, void>::function_type arg_dealloc)
    // Pass through allocated [ SharedAllocationHeader , user_memory ]
    // Pass through deallocation function
    : SharedAllocationRecord<void, void>(
#ifdef KOKKOS_DEBUG
          &SharedAllocationRecord<Kokkos::Experimental::SYCLHostPinnedSpace,
                                  void>::s_root_record,
#endif
          reinterpret_cast<SharedAllocationHeader*>(arg_space.allocate(
              sizeof(SharedAllocationHeader) + arg_alloc_size)),
          sizeof(SharedAllocationHeader) + arg_alloc_size, arg_dealloc),
      m_space(arg_space) {
#if defined(KOKKOS_ENABLE_PROFILING)
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::allocateData(
        Kokkos::Profiling::SpaceHandle(arg_space.name()), arg_label, data(),
        arg_alloc_size);
  }
#endif
  // Fill in the Header information, directly accessible via host pinned memory

  RecordBase::m_alloc_ptr->m_record = this;

  strncpy(RecordBase::m_alloc_ptr->m_label, arg_label.c_str(),
          SharedAllocationHeader::maximum_label_length);
  // Set last element zero, in case c_str is too long
  RecordBase::m_alloc_ptr
      ->m_label[SharedAllocationHeader::maximum_label_length - 1] = (char)0;
}

//----------------------------------------------------------------------------

void* SharedAllocationRecord<Kokkos::Experimental::SYCLSpace, void>::
    allocate_tracked(const Kokkos::Experimental::SYCLSpace& arg_space,
                     const std::string& arg_alloc_label,
                     const size_t arg_alloc_size) {
  if (!arg_alloc_size) return (void*)0;

  SharedAllocationRecord* const r =
      allocate(arg_space, arg_alloc_label, arg_alloc_size);

  RecordBase::increment(r);

  return r->data();
}

void SharedAllocationRecord<Kokkos::Experimental::SYCLSpace,
                            void>::deallocate_tracked(void* const
                                                          arg_alloc_ptr) {
  if (arg_alloc_ptr != 0) {
    SharedAllocationRecord* const r = get_record(arg_alloc_ptr);

    RecordBase::decrement(r);
  }
}

void* SharedAllocationRecord<Kokkos::Experimental::SYCLSpace, void>::
    reallocate_tracked(void* const arg_alloc_ptr, const size_t arg_alloc_size) {
  SharedAllocationRecord* const r_old = get_record(arg_alloc_ptr);
  SharedAllocationRecord* const r_new =
      allocate(r_old->m_space, r_old->get_label(), arg_alloc_size);

  Kokkos::Impl::DeepCopy<Kokkos::Experimental::SYCLSpace,
                         Kokkos::Experimental::SYCLSpace>(
      r_new->data(), r_old->data(), std::min(r_old->size(), r_new->size()));

  RecordBase::increment(r_new);
  RecordBase::decrement(r_old);

  return r_new->data();
}

//----------------------------------------------------------------------------

SharedAllocationRecord<Kokkos::Experimental::SYCLSpace, void>*
SharedAllocationRecord<Kokkos::Experimental::SYCLSpace, void>::get_record(
    void* alloc_ptr) {
  using Header     = SharedAllocationHeader;
  using RecordBase = SharedAllocationRecord<void, void>;
  using RecordSYCL =
      SharedAllocationRecord<Kokkos::Experimental::SYCLSpace, void>;

  // Copy the header from the allocation
  Header head;

  Header const* const head_sycl =
      alloc_ptr ? Header::get_header(alloc_ptr) : (Header*)0;

  if (alloc_ptr) {
    Kokkos::Impl::DeepCopy<HostSpace, Kokkos::Experimental::SYCLSpace>(
        &head, head_sycl, sizeof(SharedAllocationHeader));
  }

  RecordSYCL* const record =
      alloc_ptr ? static_cast<RecordSYCL*>(head.m_record) : (RecordSYCL*)0;

  if (!alloc_ptr || record->m_alloc_ptr != head_sycl) {
    Kokkos::Impl::throw_runtime_exception(std::string(
        "Kokkos::Impl::SharedAllocationRecord< Kokkos::Experimental::SYCLSpace "
        ", void >::get_record ERROR"));
  }

  return record;
}

// Iterate records to print orphaned memory ...
void SharedAllocationRecord<Kokkos::Experimental::SYCLSpace, void>::
    print_records(std::ostream& s, const Kokkos::Experimental::SYCLSpace& space,
                  bool detail) {
#ifdef KOKKOS_DEBUG
  SharedAllocationRecord<void, void>* r = &s_root_record;

  char buffer[256];

  SharedAllocationHeader head;

  if (detail) {
    do {
      if (r->m_alloc_ptr) {
        Kokkos::Impl::DeepCopy<HostSpace, Kokkos::Experimental::SYCLSpace>(
            &head, r->m_alloc_ptr, sizeof(SharedAllocationHeader));
      } else {
        head.m_label[0] = 0;
      }

      // Formatting dependent on sizeof(uintptr_t)
      const char* format_string;

      if (sizeof(uintptr_t) == sizeof(unsigned long)) {
        format_string =
            "SYCL addr( 0x%.12lx ) list( 0x%.12lx 0x%.12lx ) extent[ 0x%.12lx "
            "+ %.8ld ] count(%d) dealloc(0x%.12lx) %s\n";
      } else if (sizeof(uintptr_t) == sizeof(unsigned long long)) {
        format_string =
            "SYCL addr( 0x%.12llx ) list( 0x%.12llx 0x%.12llx ) extent[ "
            "0x%.12llx + %.8ld ] count(%d) dealloc(0x%.12llx) %s\n";
      }

      snprintf(buffer, 256, format_string, reinterpret_cast<uintptr_t>(r),
               reinterpret_cast<uintptr_t>(r->m_prev),
               reinterpret_cast<uintptr_t>(r->m_next),
               reinterpret_cast<uintptr_t>(r->m_alloc_ptr), r->m_alloc_size,
               r->m_count, reinterpret_cast<uintptr_t>(r->m_dealloc),
               head.m_label);
      std::cout << buffer;
      r = r->m_next;
    } while (r != &s_root_record);
  } else {
    do {
      if (r->m_alloc_ptr) {
        Kokkos::Impl::DeepCopy<HostSpace, Kokkos::Experimental::SYCLSpace>(
            &head, r->m_alloc_ptr, sizeof(SharedAllocationHeader));

        // Formatting dependent on sizeof(uintptr_t)
        const char* format_string;

        if (sizeof(uintptr_t) == sizeof(unsigned long)) {
          format_string = "SYCL [ 0x%.12lx + %ld ] %s\n";
        } else if (sizeof(uintptr_t) == sizeof(unsigned long long)) {
          format_string = "SYCL [ 0x%.12llx + %ld ] %s\n";
        }

        snprintf(buffer, 256, format_string,
                 reinterpret_cast<uintptr_t>(r->data()), r->size(),
                 head.m_label);
      } else {
        snprintf(buffer, 256, "SYCL [ 0 + 0 ]\n");
      }
      std::cout << buffer;
      r = r->m_next;
    } while (r != &s_root_record);
  }
#else
  throw_runtime_exception(
      "Kokkos::Impl::SharedAllocationRecord<SYCLSpace>::print_records"
      " only works with KOKKOS_DEBUG enabled");
#endif
}

}  // namespace Impl
}  // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/
namespace Kokkos {
namespace {

#ifdef NLIBER
void* sycl_resize_scratch_space(size_t bytes, bool force_shrink) {
  static void* ptr           = NULL;
  static size_t current_size = 0;
  if (current_size == 0) {
    current_size = bytes;
    ptr          = Kokkos::kokkos_malloc<Kokkos::Experimental::SYCLSpace>(
        "SYCLSpace::ScratchMemory", current_size);
  }
  if (bytes > current_size) {
    current_size = bytes;
    ptr          = Kokkos::kokkos_realloc<Kokkos::Experimental::SYCLSpace>(ptr,
                                                                  current_size);
  }
  if ((bytes < current_size) && (force_shrink)) {
    current_size = bytes;
    Kokkos::kokkos_free<Kokkos::Experimental::SYCLSpace>(ptr);
    ptr = Kokkos::kokkos_malloc<Kokkos::Experimental::SYCLSpace>(
        "SYCLSpace::ScratchMemory", current_size);
  }
  return ptr;
}
#endif /* NLIBER */

}  // namespace
}  // namespace Kokkos

#endif  // KOKKOS_ENABLE_SYCL

