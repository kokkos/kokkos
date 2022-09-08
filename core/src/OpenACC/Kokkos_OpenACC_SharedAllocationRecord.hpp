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

#ifndef KOKKOS_OPENACC_SHARED_ALLOCATION_RECORD_HPP
#define KOKKOS_OPENACC_SHARED_ALLOCATION_RECORD_HPP

#include <OpenACC/Kokkos_OpenACCSpace.hpp>
#include <OpenACC/Kokkos_OpenACC_DeepCopy.hpp>
#include <impl/Kokkos_SharedAlloc.hpp>

#include <openacc.h>

template <>
class Kokkos::Impl::SharedAllocationRecord<Kokkos::Experimental::OpenACCSpace,
                                           void>
    : public HostInaccessibleSharedAllocationRecordCommon<
          Kokkos::Experimental::OpenACCSpace> {
 private:
  friend class HostInaccessibleSharedAllocationRecordCommon<
      Kokkos::Experimental::OpenACCSpace>;
  friend class SharedAllocationRecordCommon<Kokkos::Experimental::OpenACCSpace>;
  friend Kokkos::Experimental::OpenACCSpace;

  using base_t = HostInaccessibleSharedAllocationRecordCommon<
      Kokkos::Experimental::OpenACCSpace>;
  using RecordBase = SharedAllocationRecord<void, void>;

  SharedAllocationRecord(const SharedAllocationRecord&) = delete;
  SharedAllocationRecord& operator=(const SharedAllocationRecord&) = delete;

  /**\brief  Root record for tracked allocations from this OpenACCSpace
   * instance */
  static RecordBase s_root_record;

  const Kokkos::Experimental::OpenACCSpace m_space;

 protected:
  ~SharedAllocationRecord();
  SharedAllocationRecord() = default;

  template <typename ExecutionSpace>
  SharedAllocationRecord(
      const ExecutionSpace& /*exec_space*/,
      const Kokkos::Experimental::OpenACCSpace& arg_space,
      const std::string& arg_label, const size_t arg_alloc_size,
      const RecordBase::function_type arg_dealloc = &deallocate)
      : SharedAllocationRecord(arg_space, arg_label, arg_alloc_size,
                               arg_dealloc) {}

  SharedAllocationRecord(
      const Kokkos::Experimental::OpenACC& exec_space,
      const Kokkos::Experimental::OpenACCSpace& arg_space,
      const std::string& arg_label, const size_t arg_alloc_size,
      const RecordBase::function_type arg_dealloc = &deallocate);

  SharedAllocationRecord(
      const Kokkos::Experimental::OpenACCSpace& arg_space,
      const std::string& arg_label, const size_t arg_alloc_size,
      const RecordBase::function_type arg_dealloc = &deallocate);

 public:
  KOKKOS_INLINE_FUNCTION static SharedAllocationRecord* allocate(
      const Kokkos::Experimental::OpenACCSpace& arg_space,
      const std::string& arg_label, const size_t arg_alloc_size) {
    if (acc_on_device(acc_device_host)) {
      return new SharedAllocationRecord(arg_space, arg_label, arg_alloc_size);
    } else {
      return nullptr;
    }
  }
};

#endif
