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

#include <stdio.h>
#include <limits>
#include <iostream>
#include <vector>
#include <Kokkos_Core.hpp>
#include <impl/Kokkos_Error.hpp>
#include <iostream>
#include <impl/Kokkos_CPUDiscovery.hpp>
#include <impl/Kokkos_Tools.hpp>

#ifdef KOKKOS_ENABLE_OPENMPTARGET

// FIXME_OPENMPTARGET currently unused
/*
namespace Kokkos {
namespace Impl {
namespace {

KOKKOS_INLINE_FUNCTION
int kokkos_omp_in_parallel();

KOKKOS_INLINE_FUNCTION
int kokkos_omp_in_parallel() { return omp_in_parallel(); }

bool s_using_hwloc = false;

}  // namespace
}  // namespace Impl
}  // namespace Kokkos
*/

namespace Kokkos {
namespace Impl {

void OpenMPTargetExec::verify_is_process(const char* const label) {
  if (omp_in_parallel()) {
    std::string msg(label);
    msg.append(" ERROR: in parallel");
    Kokkos::Impl::throw_runtime_exception(msg);
  }
}

void OpenMPTargetExec::verify_initialized(const char* const label) {
  if (0 == Kokkos::Experimental::OpenMPTarget().impl_is_initialized()) {
    std::string msg(label);
    msg.append(" ERROR: not initialized");
    Kokkos::Impl::throw_runtime_exception(msg);
  }
}

void* OpenMPTargetExec::m_scratch_ptr    = nullptr;
int64_t OpenMPTargetExec::m_scratch_size = 0;

void OpenMPTargetExec::clear_scratch() {
  Kokkos::Experimental::OpenMPTargetSpace space;
  space.deallocate(m_scratch_ptr, m_scratch_size);
  m_scratch_ptr  = nullptr;
  m_scratch_size = 0;
}

void* OpenMPTargetExec::get_scratch_ptr() { return m_scratch_ptr; }

void OpenMPTargetExec::resize_scratch(int64_t team_size, int64_t shmem_size_L0,
                                      int64_t shmem_size_L1) {
  Kokkos::Experimental::OpenMPTargetSpace space;
  const int64_t shmem_size =
      shmem_size_L0 + shmem_size_L1;  // L0 + L1 scratch memory per team.
  const int64_t padding = shmem_size * 10 / 100;  // Padding per team.
  // FIXME_OPENMPTARGET - Total amount of scratch memory allocated is depenedent
  // on the maximum number of teams possible. Currently the maximum number of
  // teams possible is calculated based on NVIDIA's Volta GPU. In
  // future this value should be based on the chosen architecture for the
  // OpenMPTarget backend.
  int64_t total_size = (shmem_size + 16 + padding) * ((2048 * 80) / team_size);

  if (total_size > m_scratch_size) {
    space.deallocate(m_scratch_ptr, m_scratch_size);
    m_scratch_size = total_size;
    m_scratch_ptr  = space.allocate(total_size);
  }
}
}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_ENABLE_OPENMPTARGET
