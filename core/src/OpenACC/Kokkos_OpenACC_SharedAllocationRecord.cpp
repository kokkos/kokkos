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

#define KOKKOS_IMPL_PUBLIC_INCLUDE

#include <OpenACC/Kokkos_OpenACC_SharedAllocationRecord.hpp>
#include <OpenACC/Kokkos_OpenACC_DeepCopy.hpp>
#include <impl/Kokkos_MemorySpace.hpp>
#include <Kokkos_HostSpace.hpp>

#ifdef KOKKOS_ENABLE_DEBUG
Kokkos::Impl::SharedAllocationRecord<void, void> SharedAllocationRecord<
    Kokkos::Experimental::OpenACCSpace, void>::s_root_record;
#endif

Kokkos::Impl::SharedAllocationRecord<Kokkos::Experimental::OpenACCSpace,
                                     void>::~SharedAllocationRecord() {
  m_space.deallocate(m_label.c_str(),
                     SharedAllocationRecord<void, void>::m_alloc_ptr,
                     (SharedAllocationRecord<void, void>::m_alloc_size -
                      sizeof(SharedAllocationHeader)));
}

Kokkos::Impl::SharedAllocationRecord<Kokkos::Experimental::OpenACCSpace, void>::
    SharedAllocationRecord(
        const Kokkos::Experimental::OpenACCSpace &arg_space,
        const std::string &arg_label, const size_t arg_alloc_size,
        const SharedAllocationRecord<void, void>::function_type arg_dealloc)
    // Pass through allocated [ SharedAllocationHeader , user_memory ]
    // Pass through deallocation function
    : base_t(
#ifdef KOKKOS_ENABLE_DEBUG
          &SharedAllocationRecord<Kokkos::Experimental::OpenACCSpace,
                                  void>::s_root_record,
#endif
          Impl::checked_allocation_with_header(arg_space, arg_label,
                                               arg_alloc_size),
          sizeof(SharedAllocationHeader) + arg_alloc_size, arg_dealloc,
          arg_label),
      m_space(arg_space) {
  SharedAllocationHeader header;

  this->base_t::_fill_host_accessible_header_info(header, arg_label);

  Kokkos::Impl::DeepCopy<Experimental::OpenACCSpace, HostSpace>(
      RecordBase::m_alloc_ptr, &header, sizeof(SharedAllocationHeader));
  Kokkos::fence(
      "SharedAllocationRecord<Kokkos::Experimental::OpenACCSpace, "
      "void>::SharedAllocationRecord(): fence after copying header from "
      "HostSpace");
}

Kokkos::Impl::SharedAllocationRecord<Kokkos::Experimental::OpenACCSpace, void>::
    SharedAllocationRecord(
        const Kokkos::Experimental::OpenACC &arg_exec_space,
        const Kokkos::Experimental::OpenACCSpace &arg_space,
        const std::string &arg_label, const size_t arg_alloc_size,
        const SharedAllocationRecord<void, void>::function_type arg_dealloc)
    // Pass through allocated [ SharedAllocationHeader , user_memory ]
    // Pass through deallocation function
    : base_t(
#ifdef KOKKOS_ENABLE_DEBUG
          &SharedAllocationRecord<Kokkos::Experimental::OpenACCSpace,
                                  void>::s_root_record,
#endif
          Impl::checked_allocation_with_header(arg_exec_space, arg_space,
                                               arg_label, arg_alloc_size),
          sizeof(SharedAllocationHeader) + arg_alloc_size, arg_dealloc,
          arg_label),
      m_space(arg_space) {
  SharedAllocationHeader header;

  this->base_t::_fill_host_accessible_header_info(header, arg_label);

  Kokkos::Impl::DeepCopy<Experimental::OpenACCSpace, HostSpace>(
      arg_exec_space, RecordBase::m_alloc_ptr, &header,
      sizeof(SharedAllocationHeader));
}

//==============================================================================
// <editor-fold desc="Explicit instantiations of CRTP Base classes"> {{{1

#include <impl/Kokkos_SharedAlloc_timpl.hpp>

// To avoid additional compilation cost for something that's (mostly?) not
// performance sensitive, we explicitly instantiate these CRTP base classes
// here, where we have access to the associated *_timpl.hpp header files.
template class Kokkos::Impl::HostInaccessibleSharedAllocationRecordCommon<
    Kokkos::Experimental::OpenACCSpace>;
template class Kokkos::Impl::SharedAllocationRecordCommon<
    Kokkos::Experimental::OpenACCSpace>;

// </editor-fold> end Explicit instantiations of CRTP Base classes }}}1
//==============================================================================
