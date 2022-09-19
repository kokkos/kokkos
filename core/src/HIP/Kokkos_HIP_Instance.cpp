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

/*--------------------------------------------------------------------------*/
/* Kokkos interfaces */

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#endif

#include <Kokkos_Core.hpp>

#include <HIP/Kokkos_HIP_Instance.hpp>
#include <HIP/Kokkos_HIP.hpp>
#include <HIP/Kokkos_HIP_Space.hpp>
#include <impl/Kokkos_Error.hpp>

/*--------------------------------------------------------------------------*/
/* Standard 'C' libraries */
#include <stdlib.h>

/* Standard 'C++' libraries */
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef KOKKOS_ENABLE_HIP_RELOCATABLE_DEVICE_CODE
__device__ __constant__ unsigned long kokkos_impl_hip_constant_memory_buffer
    [Kokkos::Impl::HIPTraits::ConstantMemoryUsage / sizeof(unsigned long)];
#endif

namespace Kokkos {
namespace Impl {
Kokkos::View<uint32_t *, HIPSpace> hip_global_unique_token_locks(
    bool deallocate) {
  static Kokkos::View<uint32_t *, HIPSpace> locks =
      Kokkos::View<uint32_t *, HIPSpace>();
  if (!deallocate && locks.extent(0) == 0)
    locks = Kokkos::View<uint32_t *, HIPSpace>(
        "Kokkos::UniqueToken<HIP>::m_locks", HIP().concurrency());
  if (deallocate) locks = Kokkos::View<uint32_t *, HIPSpace>();
  return locks;
}
}  // namespace Impl
}  // namespace Kokkos

namespace Kokkos {

int Impl::HIPInternal::m_hipDev                                           = -1;
int Impl::HIPInternal::m_hipArch                                          = -1;
unsigned Impl::HIPInternal::m_multiProcCount                              = 0;
unsigned Impl::HIPInternal::m_maxWarpCount                                = 0;
std::array<Impl::HIPInternal::size_type, 3> Impl::HIPInternal::m_maxBlock = {
    0, 0, 0};
unsigned Impl::HIPInternal::m_maxWavesPerCU  = 0;
unsigned Impl::HIPInternal::m_maxSharedWords = 0;
int Impl::HIPInternal::m_shmemPerSM          = 0;
int Impl::HIPInternal::m_maxShmemPerBlock    = 0;
int Impl::HIPInternal::m_maxThreadsPerSM     = 0;
hipDeviceProp_t Impl::HIPInternal::m_deviceProp;
unsigned long *Impl::HIPInternal::constantMemHostStaging = nullptr;
hipEvent_t Impl::HIPInternal::constantMemReusable        = nullptr;
std::mutex Impl::HIPInternal::constantMemMutex;

namespace Impl {

//----------------------------------------------------------------------------

void HIPInternal::print_configuration(std::ostream &s) const {
  s << "macro  KOKKOS_ENABLE_HIP : defined" << '\n';
#if defined(HIP_VERSION)
  s << "macro  HIP_VERSION = " << HIP_VERSION << " = version "
    << HIP_VERSION_MAJOR << '.' << HIP_VERSION_MINOR << '.' << HIP_VERSION_PATCH
    << '\n';
#endif

  int hipDevCount;
  KOKKOS_IMPL_HIP_SAFE_CALL(hipGetDeviceCount(&hipDevCount));

  for (int i = 0; i < hipDevCount; ++i) {
    hipDeviceProp_t hipProp;
    KOKKOS_IMPL_HIP_SAFE_CALL(hipGetDeviceProperties(&hipProp, i));

    s << "Kokkos::HIP[ " << i << " ] "
      << "gcnArch " << hipProp.gcnArch << ", Total Global Memory: "
      << ::Kokkos::Impl::human_memory_size(hipProp.totalGlobalMem)
      << ", Shared Memory per Block: "
      << ::Kokkos::Impl::human_memory_size(hipProp.sharedMemPerBlock);
    if (m_hipDev == i) s << " : Selected";
    s << '\n';
  }
}

//----------------------------------------------------------------------------

HIPInternal::~HIPInternal() {
  if (m_scratchSpace || m_scratchFlags) {
    std::cerr << "Kokkos::HIP ERROR: Failed to call "
                 "Kokkos::HIP::finalize()"
              << std::endl;
    std::cerr.flush();
  }

  m_scratchSpaceCount = 0;
  m_scratchFlagsCount = 0;
  m_scratchSpace      = nullptr;
  m_scratchFlags      = nullptr;
  m_stream            = nullptr;
}

int HIPInternal::verify_is_initialized(const char *const label) const {
  if (m_hipDev < 0) {
    Kokkos::abort((std::string("Kokkos::HIP::") + label +
                   " : ERROR device not initialized\n")
                      .c_str());
  }
  return 0 <= m_hipDev;
}

uint32_t HIPInternal::impl_get_instance_id() const noexcept {
  return m_instance_id;
}
HIPInternal &HIPInternal::singleton() {
  static HIPInternal *self = nullptr;
  if (!self) {
    self = new HIPInternal();
  }
  return *self;
}

void HIPInternal::fence() const {
  fence("Kokkos::HIPInternal::fence: Unnamed Internal Fence");
}
void HIPInternal::fence(const std::string &name) const {
  Kokkos::Tools::Experimental::Impl::profile_fence_event<Kokkos::HIP>(
      name,
      Kokkos::Tools::Experimental::Impl::DirectFenceIDHandle{
          impl_get_instance_id()},
      [&]() { KOKKOS_IMPL_HIP_SAFE_CALL(hipStreamSynchronize(m_stream)); });
}

void HIPInternal::initialize(hipStream_t stream, bool manage_stream) {
  if (was_finalized)
    Kokkos::abort("Calling HIP::initialize after HIP::finalize is illegal\n");

  if (is_initialized()) return;

  if (!HostSpace::execution_space::impl_is_initialized()) {
    const std::string msg(
        "HIP::initialize ERROR : HostSpace::execution_space "
        "is not initialized");
    Kokkos::Impl::throw_runtime_exception(msg);
  }

  const bool ok_init = nullptr == m_scratchSpace || nullptr == m_scratchFlags;

  if (ok_init) {
    m_stream        = stream;
    m_manage_stream = manage_stream;

    //----------------------------------
    // Multiblock reduction uses scratch flags for counters
    // and scratch space for partial reduction values.
    // Allocate some initial space.  This will grow as needed.
    {
      const unsigned reduce_block_count =
          m_maxWarpCount * Impl::HIPTraits::WarpSize;

      (void)scratch_flags(reduce_block_count * 2 * sizeof(size_type));
      (void)scratch_space(reduce_block_count * 16 * sizeof(size_type));
    }
  } else {
    std::ostringstream msg;
    msg << "Kokkos::HIP::initialize(" << m_hipDev
        << ") FAILED : Already initialized";
    Kokkos::Impl::throw_runtime_exception(msg.str());
  }

  KOKKOS_IMPL_HIP_SAFE_CALL(
      hipMalloc(&m_scratch_locks, sizeof(int32_t) * HIP::concurrency()));
  KOKKOS_IMPL_HIP_SAFE_CALL(
      hipMemset(m_scratch_locks, 0, sizeof(int32_t) * HIP::concurrency()));
}

//----------------------------------------------------------------------------

using ScratchGrain = Kokkos::HIP::size_type[Impl::HIPTraits::WarpSize];
enum { sizeScratchGrain = sizeof(ScratchGrain) };

Kokkos::HIP::size_type *HIPInternal::scratch_space(const std::size_t size) {
  if (verify_is_initialized("scratch_space") &&
      m_scratchSpaceCount * sizeScratchGrain < size) {
    m_scratchSpaceCount = (size + sizeScratchGrain - 1) / sizeScratchGrain;

    using Record = Kokkos::Impl::SharedAllocationRecord<Kokkos::HIPSpace, void>;

    if (m_scratchSpace) Record::decrement(Record::get_record(m_scratchSpace));

    Record *const r =
        Record::allocate(Kokkos::HIPSpace(), "Kokkos::InternalScratchSpace",
                         (sizeScratchGrain * m_scratchSpaceCount));

    Record::increment(r);

    m_scratchSpace = reinterpret_cast<size_type *>(r->data());
  }

  return m_scratchSpace;
}

Kokkos::HIP::size_type *HIPInternal::scratch_flags(const std::size_t size) {
  if (verify_is_initialized("scratch_flags") &&
      m_scratchFlagsCount * sizeScratchGrain < size) {
    m_scratchFlagsCount = (size + sizeScratchGrain - 1) / sizeScratchGrain;

    using Record = Kokkos::Impl::SharedAllocationRecord<Kokkos::HIPSpace, void>;

    if (m_scratchFlags) Record::decrement(Record::get_record(m_scratchFlags));

    Record *const r =
        Record::allocate(Kokkos::HIPSpace(), "Kokkos::InternalScratchFlags",
                         (sizeScratchGrain * m_scratchFlagsCount));

    Record::increment(r);

    m_scratchFlags = reinterpret_cast<size_type *>(r->data());

    KOKKOS_IMPL_HIP_SAFE_CALL(
        hipMemset(m_scratchFlags, 0, m_scratchFlagsCount * sizeScratchGrain));
  }

  return m_scratchFlags;
}

std::pair<void *, int> HIPInternal::resize_team_scratch_space(
    std::int64_t bytes, bool force_shrink) {
  // Multiple ParallelFor/Reduce Teams can call this function at the same time
  // and invalidate the m_team_scratch_ptr. We use a pool to avoid any race
  // condition.

  int current_team_scratch = 0;
  int zero                 = 0;
  while (!m_team_scratch_pool[current_team_scratch].compare_exchange_weak(
      zero, 1, std::memory_order_release, std::memory_order_relaxed)) {
    current_team_scratch = (current_team_scratch + 1) % m_n_team_scratch;
  }
  if (m_team_scratch_current_size[current_team_scratch] == 0) {
    m_team_scratch_current_size[current_team_scratch] = bytes;
    m_team_scratch_ptr[current_team_scratch] =
        Kokkos::kokkos_malloc<Kokkos::HIPSpace>(
            "Kokkos::HIPSpace::TeamScratchMemory",
            m_team_scratch_current_size[current_team_scratch]);
  }
  if ((bytes > m_team_scratch_current_size[current_team_scratch]) ||
      ((bytes < m_team_scratch_current_size[current_team_scratch]) &&
       (force_shrink))) {
    m_team_scratch_current_size[current_team_scratch] = bytes;
    m_team_scratch_ptr[current_team_scratch] =
        Kokkos::kokkos_realloc<Kokkos::HIPSpace>(
            m_team_scratch_ptr[current_team_scratch],
            m_team_scratch_current_size[current_team_scratch]);
  }
  return std::make_pair(m_team_scratch_ptr[current_team_scratch],
                        current_team_scratch);
}

void HIPInternal::release_team_scratch_pool(int scratch_pool_id) {
  m_team_scratch_pool[scratch_pool_id] = 0;
}

//----------------------------------------------------------------------------

void HIPInternal::finalize() {
  this->fence("Kokkos::HIPInternal::finalize: fence on finalization");
  was_finalized = true;

  if (this == &singleton()) {
    (void)Kokkos::Impl::hip_global_unique_token_locks(true);
    KOKKOS_IMPL_HIP_SAFE_CALL(hipHostFree(constantMemHostStaging));
    KOKKOS_IMPL_HIP_SAFE_CALL(hipEventDestroy(constantMemReusable));
  }

  if (nullptr != m_scratchSpace || nullptr != m_scratchFlags) {
    using RecordHIP = Kokkos::Impl::SharedAllocationRecord<Kokkos::HIPSpace>;

    RecordHIP::decrement(RecordHIP::get_record(m_scratchFlags));
    RecordHIP::decrement(RecordHIP::get_record(m_scratchSpace));

    for (int i = 0; i < m_n_team_scratch; ++i) {
      if (m_team_scratch_current_size[i] > 0)
        Kokkos::kokkos_free<Kokkos::HIPSpace>(m_team_scratch_ptr[i]);
    }

    if (m_manage_stream && m_stream != nullptr)
      KOKKOS_IMPL_HIP_SAFE_CALL(hipStreamDestroy(m_stream));
  }

  m_scratchSpaceCount = 0;
  m_scratchFlagsCount = 0;
  m_scratchSpace      = nullptr;
  m_scratchFlags      = nullptr;
  m_stream            = nullptr;
  for (int i = 0; i < m_n_team_scratch; ++i) {
    m_team_scratch_current_size[i] = 0;
    m_team_scratch_ptr[i]          = nullptr;
  }

  KOKKOS_IMPL_HIP_SAFE_CALL(hipFree(m_scratch_locks));
  m_scratch_locks = nullptr;
}

//----------------------------------------------------------------------------

Kokkos::HIP::size_type hip_internal_multiprocessor_count() {
  return HIPInternal::singleton().m_multiProcCount;
}

Kokkos::HIP::size_type hip_internal_maximum_warp_count() {
  return HIPInternal::singleton().m_maxWarpCount;
}

std::array<Kokkos::HIP::size_type, 3> hip_internal_maximum_grid_count() {
  return HIPInternal::singleton().m_maxBlock;
}

Kokkos::HIP::size_type *hip_internal_scratch_space(const HIP &instance,
                                                   const std::size_t size) {
  return instance.impl_internal_space_instance()->scratch_space(size);
}

Kokkos::HIP::size_type *hip_internal_scratch_flags(const HIP &instance,
                                                   const std::size_t size) {
  return instance.impl_internal_space_instance()->scratch_flags(size);
}

}  // namespace Impl
}  // namespace Kokkos

//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {
void hip_device_synchronize(const std::string &name) {
  Kokkos::Tools::Experimental::Impl::profile_fence_event<Kokkos::HIP>(
      name,
      Kokkos::Tools::Experimental::SpecialSynchronizationCases::
          GlobalDeviceSynchronization,
      [&]() { KOKKOS_IMPL_HIP_SAFE_CALL(hipDeviceSynchronize()); });
}

void hip_internal_error_throw(hipError_t e, const char *name, const char *file,
                              const int line) {
  std::ostringstream out;
  out << name << " error( " << hipGetErrorName(e)
      << "): " << hipGetErrorString(e);
  if (file) {
    out << " " << file << ":" << line;
  }
  throw_runtime_exception(out.str());
}
}  // namespace Impl
}  // namespace Kokkos

//----------------------------------------------------------------------------

namespace Kokkos {
HIP::size_type HIP::detect_device_count() {
  int hipDevCount;
  KOKKOS_IMPL_HIP_SAFE_CALL(hipGetDeviceCount(&hipDevCount));
  return hipDevCount;
}
}  // namespace Kokkos
