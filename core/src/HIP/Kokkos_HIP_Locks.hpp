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

#ifndef KOKKOS_HIP_LOCKS_HPP
#define KOKKOS_HIP_LOCKS_HPP

#include <Kokkos_Macros.hpp>

#include <cstdint>

#include <HIP/Kokkos_HIP_Error.hpp>

#ifdef KOKKOS_ENABLE_IMPL_DESUL_ATOMICS
#include <desul/atomics/Lock_Array_HIP.hpp>
#endif

namespace Kokkos {
namespace Impl {

struct HIPLockArrays {
  std::int32_t* atomic;
  std::int32_t* scratch;
  std::int32_t n;
};

/// \brief This global variable in Host space is the central definition
///        of these arrays.
extern HIPLockArrays g_host_hip_lock_arrays;

/// \brief After this call, the g_host_hip_lock_arrays variable has
///        valid, initialized arrays.
///
/// This call is idempotent.
void initialize_host_hip_lock_arrays();

/// \brief After this call, the g_host_hip_lock_arrays variable has
///        all null pointers, and all array memory has been freed.
///
/// This call is idempotent.
void finalize_host_hip_lock_arrays();

#if defined(__HIPCC__)

/// \brief This global variable in HIP space is what kernels use
///        to get access to the lock arrays.
///
/// When relocatable device code is enabled, there can be one single
/// instance of this global variable for the entire executable,
/// whose definition will be in Kokkos_HIP_Locks.cpp (and whose declaration
/// here must then be extern).
/// This one instance will be initialized by initialize_host_HIP_lock_arrays
/// and need not be modified afterwards.
///
/// When relocatable device code is disabled, an instance of this variable
/// will be created in every translation unit that sees this header file.
/// Since the Kokkos_HIP_Locks.cpp translation unit cannot initialize the
/// instances in other translation units, we must update this HIP global
/// variable based on the Host global variable prior to running any kernels
/// that will use it.
/// That is the purpose of the KOKKOS_ENSURE_HIP_LOCK_ARRAYS_ON_DEVICE macro.
__device__
#ifdef KOKKOS_ENABLE_HIP_RELOCATABLE_DEVICE_CODE
    __constant__ extern
#endif
    HIPLockArrays g_device_hip_lock_arrays;

#define KOKKOS_IMPL_HIP_SPACE_ATOMIC_MASK 0x1FFFF

/// \brief Acquire a lock for the address
///
/// This function tries to acquire the lock for the hash value derived
/// from the provided ptr. If the lock is successfully acquired the
/// function returns true. Otherwise it returns false.
__device__ inline bool lock_address_hip_space(void* ptr) {
  auto offset = reinterpret_cast<size_t>(ptr);
  offset      = offset >> 2;
  offset      = offset & KOKKOS_IMPL_HIP_SPACE_ATOMIC_MASK;
  return (0 == atomicCAS(&g_device_hip_lock_arrays.atomic[offset], 0, 1));
}

/// \brief Release lock for the address
///
/// This function releases the lock for the hash value derived
/// from the provided ptr. This function should only be called
/// after previously successfully aquiring a lock with
/// lock_address.
__device__ inline void unlock_address_hip_space(void* ptr) {
  auto offset = reinterpret_cast<size_t>(ptr);
  offset      = offset >> 2;
  offset      = offset & KOKKOS_IMPL_HIP_SPACE_ATOMIC_MASK;
  atomicExch(&g_device_hip_lock_arrays.atomic[offset], 0);
}

}  // namespace Impl
}  // namespace Kokkos

// Make lock_array_copied an explicit translation unit scope thingy
namespace Kokkos {
namespace Impl {
namespace {
static int lock_array_copied = 0;
inline int eliminate_warning_for_lock_array() { return lock_array_copied; }
}  // namespace
}  // namespace Impl
}  // namespace Kokkos

/* Dan Ibanez: it is critical that this code be a macro, so that it will
   capture the right address for g_device_hip_lock_arrays!
   putting this in an inline function will NOT do the right thing! */
#define KOKKOS_COPY_HIP_LOCK_ARRAYS_TO_DEVICE()                 \
  {                                                             \
    if (::Kokkos::Impl::lock_array_copied == 0) {               \
      KOKKOS_IMPL_HIP_SAFE_CALL(hipMemcpyToSymbol(              \
          HIP_SYMBOL(::Kokkos::Impl::g_device_hip_lock_arrays), \
          &::Kokkos::Impl::g_host_hip_lock_arrays,              \
          sizeof(::Kokkos::Impl::HIPLockArrays)));              \
    }                                                           \
    ::Kokkos::Impl::lock_array_copied = 1;                      \
  }

#ifndef KOKKOS_ENABLE_IMPL_DESUL_ATOMICS

#ifdef KOKKOS_ENABLE_HIP_RELOCATABLE_DEVICE_CODE
#define KOKKOS_ENSURE_HIP_LOCK_ARRAYS_ON_DEVICE()
#else
#define KOKKOS_ENSURE_HIP_LOCK_ARRAYS_ON_DEVICE() \
  KOKKOS_COPY_HIP_LOCK_ARRAYS_TO_DEVICE()
#endif

#else

#ifdef KOKKOS_ENABLE_HIP_RELOCATABLE_DEVICE_CODE
#define KOKKOS_ENSURE_HIP_LOCK_ARRAYS_ON_DEVICE()
#else
// Still Need COPY_CUDA_LOCK_ARRAYS for team scratch etc.
#define KOKKOS_ENSURE_HIP_LOCK_ARRAYS_ON_DEVICE() \
  KOKKOS_COPY_HIP_LOCK_ARRAYS_TO_DEVICE()         \
  DESUL_ENSURE_HIP_LOCK_ARRAYS_ON_DEVICE()
#endif

#endif /* defined( KOKKOS_ENABLE_IMPL_DESUL_ATOMICS ) */

#endif /* defined( __HIPCC__ ) */

#endif /* #ifndef KOKKOS_HIP_LOCKS_HPP */
