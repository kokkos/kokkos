/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
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
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
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

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#endif

#include <HIP/Kokkos_HIP_SharedAllocationRecord.hpp>
#include <HIP/Kokkos_HIP_DeepCopy.hpp>
#include <HIP/Kokkos_HIP.hpp>

namespace Kokkos {
namespace Impl {

#ifdef KOKKOS_ENABLE_DEBUG
SharedAllocationRecord<void, void>
    SharedAllocationRecord<HIPSpace, void>::s_root_record;

SharedAllocationRecord<void, void>
    SharedAllocationRecord<HIPHostPinnedSpace, void>::s_root_record;

SharedAllocationRecord<void, void>
    SharedAllocationRecord<HIPManagedSpace, void>::s_root_record;
#endif

SharedAllocationRecord<HIPSpace, void>::~SharedAllocationRecord() {
  auto alloc_size = SharedAllocationRecord<void, void>::m_alloc_size;
  m_space.deallocate(m_label.c_str(),
                     SharedAllocationRecord<void, void>::m_alloc_ptr,
                     alloc_size, (alloc_size - sizeof(SharedAllocationHeader)));
}

SharedAllocationRecord<HIPHostPinnedSpace, void>::~SharedAllocationRecord() {
  m_space.deallocate(m_label.c_str(),
                     SharedAllocationRecord<void, void>::m_alloc_ptr,
                     SharedAllocationRecord<void, void>::m_alloc_size);
}

SharedAllocationRecord<HIPManagedSpace, void>::~SharedAllocationRecord() {
  m_space.deallocate(m_label.c_str(),
                     SharedAllocationRecord<void, void>::m_alloc_ptr,
                     SharedAllocationRecord<void, void>::m_alloc_size);
}

SharedAllocationRecord<HIPSpace, void>::SharedAllocationRecord(
    const HIPSpace& arg_space, const std::string& arg_label,
    const size_t arg_alloc_size,
    const SharedAllocationRecord<void, void>::function_type arg_dealloc)
    // Pass through allocated [ SharedAllocationHeader , user_memory ]
    // Pass through deallocation function
    : base_t(
#ifdef KOKKOS_ENABLE_DEBUG
          &SharedAllocationRecord<HIPSpace, void>::s_root_record,
#endif
          Kokkos::Impl::checked_allocation_with_header(arg_space, arg_label,
                                                       arg_alloc_size),
          sizeof(SharedAllocationHeader) + arg_alloc_size, arg_dealloc,
          arg_label),
      m_space(arg_space) {

  SharedAllocationHeader header;

  this->base_t::_fill_host_accessible_header_info(header, arg_label);

  // Copy to device memory
  HIP exec;
  Kokkos::Impl::DeepCopy<HIPSpace, HostSpace>(
      exec, RecordBase::m_alloc_ptr, &header, sizeof(SharedAllocationHeader));
  exec.fence(
      "SharedAllocationRecord<Kokkos::HIPSpace, "
      "void>::SharedAllocationRecord(): fence after copying header from "
      "HostSpace");
}

SharedAllocationRecord<HIPSpace, void>::SharedAllocationRecord(
    const HIP& arg_exec_space, const HIPSpace& arg_space,
    const std::string& arg_label, const size_t arg_alloc_size,
    const SharedAllocationRecord<void, void>::function_type arg_dealloc)
    // Pass through allocated [ SharedAllocationHeader , user_memory ]
    // Pass through deallocation function
    : base_t(
#ifdef KOKKOS_ENABLE_DEBUG
          &SharedAllocationRecord<HIPSpace, void>::s_root_record,
#endif
          Kokkos::Impl::checked_allocation_with_header(arg_space, arg_label,
                                                       arg_alloc_size),
          sizeof(SharedAllocationHeader) + arg_alloc_size, arg_dealloc,
          arg_label),
      m_space(arg_space) {

  SharedAllocationHeader header;

  this->base_t::_fill_host_accessible_header_info(header, arg_label);

  // Copy to device memory
  Kokkos::Impl::DeepCopy<HIPSpace, HostSpace>(arg_exec_space,
                                              RecordBase::m_alloc_ptr, &header,
                                              sizeof(SharedAllocationHeader));
}

SharedAllocationRecord<HIPHostPinnedSpace, void>::SharedAllocationRecord(
    const HIPHostPinnedSpace& arg_space, const std::string& arg_label,
    const size_t arg_alloc_size,
    const SharedAllocationRecord<void, void>::function_type arg_dealloc)
    // Pass through allocated [ SharedAllocationHeader , user_memory ]
    // Pass through deallocation function
    : base_t(
#ifdef KOKKOS_ENABLE_DEBUG
          &SharedAllocationRecord<HIPHostPinnedSpace, void>::s_root_record,
#endif
          Kokkos::Impl::checked_allocation_with_header(arg_space, arg_label,
                                                       arg_alloc_size),
          sizeof(SharedAllocationHeader) + arg_alloc_size, arg_dealloc,
          arg_label),
      m_space(arg_space) {
  // Fill in the Header information, directly accessible via host pinned memory
  this->base_t::_fill_host_accessible_header_info(*RecordBase::m_alloc_ptr,
                                                  arg_label);
}

SharedAllocationRecord<HIPManagedSpace, void>::SharedAllocationRecord(
    const HIPManagedSpace& arg_space, const std::string& arg_label,
    const size_t arg_alloc_size,
    const SharedAllocationRecord<void, void>::function_type arg_dealloc)
    // Pass through allocated [ SharedAllocationHeader , user_memory ]
    // Pass through deallocation function
    : base_t(
#ifdef KOKKOS_ENABLE_DEBUG
          &SharedAllocationRecord<HIPManagedSpace, void>::s_root_record,
#endif
          Kokkos::Impl::checked_allocation_with_header(arg_space, arg_label,
                                                       arg_alloc_size),
          sizeof(SharedAllocationHeader) + arg_alloc_size, arg_dealloc,
          arg_label),
      m_space(arg_space) {
  // Fill in the Header information, directly accessible via managed memory
  this->base_t::_fill_host_accessible_header_info(*RecordBase::m_alloc_ptr,
                                                  arg_label);
}

}  // namespace Impl
}  // namespace Kokkos
