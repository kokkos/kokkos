/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
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

/** @file Kokkos_MemorySpace.hpp
 *
 *  Operations common to memory space instances, or at least default
 *  implementations thereof.
 */

#ifndef KOKKOS_IMPL_MEMORYSPACE_HPP
#define KOKKOS_IMPL_MEMORYSPACE_HPP

#include <Kokkos_Macros.hpp>
#include <impl/Kokkos_SharedAlloc.hpp>
#include <impl/Kokkos_Error.hpp>

#include <string>
#include <iostream>

namespace Kokkos {
namespace Impl {

template <class MemorySpace>
SharedAllocationHeader *checked_allocation_with_header(MemorySpace const &space,
                                                  std::string const &label,
                                                  size_t alloc_size) {
  try {
    return reinterpret_cast<SharedAllocationHeader *>(
        space.allocate(alloc_size + sizeof(SharedAllocationHeader)));
  } catch (Kokkos::Experimental::RawMemoryAllocationFailure const &failure) {
    std::cerr << "Kokkos failed to allocate memory for label \"" << label
              << "\".  Allocation using MemorySpace named \"" << space.name()
              << " failed with the following error:  ";
    failure.print_error_message(std::cerr);
    if (failure.failure_mode() == Kokkos::Experimental::RawMemoryAllocationFailure::
                                      FailureMode::AllocationNotAligned) {
      // TODO: delete the misaligned memory?
      std::cerr << "Warning: Allocation failed due to misalignment; memory may "
                   "be leaked." << std::endl;
    }
    std::cerr.flush();
    Kokkos::Impl::throw_runtime_exception("Memory allocation failure");
  }
  return nullptr;  // unreachable
}

}  // end namespace Impl
}  // end namespace Kokkos

#endif  // KOKKOS_IMPL_MEMORYSPACE_HPP
