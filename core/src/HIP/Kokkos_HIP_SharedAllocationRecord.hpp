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

#ifndef KOKKOS_HIP_SHARED_ALLOCATION_RECORD_HPP
#define KOKKOS_HIP_SHARED_ALLOCATION_RECORD_HPP

#include <HIP/Kokkos_HIP_Space.hpp>

namespace Kokkos {
namespace Impl {

template <>
class SharedAllocationRecord<HIPSpace, void>
    : public HostInaccessibleSharedAllocationRecordCommon<HIPSpace> {
 private:
  friend class SharedAllocationRecordCommon<HIPSpace>;
  friend class HostInaccessibleSharedAllocationRecordCommon<HIPSpace>;
  using base_t     = HostInaccessibleSharedAllocationRecordCommon<HIPSpace>;
  using RecordBase = SharedAllocationRecord<void, void>;

  SharedAllocationRecord(const SharedAllocationRecord&) = delete;
  SharedAllocationRecord& operator=(const SharedAllocationRecord&) = delete;

#ifdef KOKKOS_ENABLE_DEBUG
  static RecordBase s_root_record;
#endif

  const HIPSpace m_space;

 protected:
  ~SharedAllocationRecord();

  template <typename ExecutionSpace>
  SharedAllocationRecord(
      const ExecutionSpace& /*exec*/, const HIPSpace& arg_space,
      const std::string& arg_label, const size_t arg_alloc_size,
      const RecordBase::function_type arg_dealloc = &base_t::deallocate)
      : SharedAllocationRecord(arg_space, arg_label, arg_alloc_size,
                               arg_dealloc) {}

  SharedAllocationRecord(
      const HIP& exec_space, const HIPSpace& arg_space,
      const std::string& arg_label, const size_t arg_alloc_size,
      const RecordBase::function_type arg_dealloc = &base_t::deallocate);

  SharedAllocationRecord(
      const HIPSpace& arg_space, const std::string& arg_label,
      const size_t arg_alloc_size,
      const RecordBase::function_type arg_dealloc = &base_t::deallocate);
};

template <>
class SharedAllocationRecord<HIPHostPinnedSpace, void>
    : public SharedAllocationRecordCommon<HIPHostPinnedSpace> {
 private:
  friend class SharedAllocationRecordCommon<HIPHostPinnedSpace>;
  using base_t     = SharedAllocationRecordCommon<HIPHostPinnedSpace>;
  using RecordBase = SharedAllocationRecord<void, void>;

  SharedAllocationRecord(const SharedAllocationRecord&) = delete;
  SharedAllocationRecord& operator=(const SharedAllocationRecord&) = delete;

#ifdef KOKKOS_ENABLE_DEBUG
  static RecordBase s_root_record;
#endif

  const HIPHostPinnedSpace m_space;

 protected:
  ~SharedAllocationRecord();
  SharedAllocationRecord() = default;

  template <typename ExecutionSpace>
  SharedAllocationRecord(
      const ExecutionSpace& /*exec_space*/, const HIPHostPinnedSpace& arg_space,
      const std::string& arg_label, const size_t arg_alloc_size,
      const RecordBase::function_type arg_dealloc = &base_t::deallocate)
      : SharedAllocationRecord(arg_space, arg_label, arg_alloc_size,
                               arg_dealloc) {}

  SharedAllocationRecord(
      const HIPHostPinnedSpace& arg_space, const std::string& arg_label,
      const size_t arg_alloc_size,
      const RecordBase::function_type arg_dealloc = &base_t::deallocate);
};

template <>
class SharedAllocationRecord<HIPManagedSpace, void>
    : public SharedAllocationRecordCommon<HIPManagedSpace> {
 private:
  friend class SharedAllocationRecordCommon<HIPManagedSpace>;
  using base_t     = SharedAllocationRecordCommon<HIPManagedSpace>;
  using RecordBase = SharedAllocationRecord<void, void>;

  SharedAllocationRecord(const SharedAllocationRecord&) = delete;
  SharedAllocationRecord& operator=(const SharedAllocationRecord&) = delete;

#ifdef KOKKOS_ENABLE_DEBUG
  static RecordBase s_root_record;
#endif

  const HIPManagedSpace m_space;

 protected:
  ~SharedAllocationRecord();
  SharedAllocationRecord() = default;

  template <typename ExecutionSpace>
  SharedAllocationRecord(
      const ExecutionSpace& /*exec_space*/, const HIPManagedSpace& arg_space,
      const std::string& arg_label, const size_t arg_alloc_size,
      const RecordBase::function_type arg_dealloc = &base_t::deallocate)
      : SharedAllocationRecord(arg_space, arg_label, arg_alloc_size,
                               arg_dealloc) {}

  SharedAllocationRecord(
      const HIPManagedSpace& arg_space, const std::string& arg_label,
      const size_t arg_alloc_size,
      const RecordBase::function_type arg_dealloc = &base_t::deallocate);
};
}  // namespace Impl
}  // namespace Kokkos

#endif
