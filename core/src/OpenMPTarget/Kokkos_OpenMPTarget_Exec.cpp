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
  // Fails if the current task is in a parallel region or is not on the host.
  if (omp_in_parallel() && (!omp_is_initial_device())) {
    std::string msg(label);
    msg.append(" ERROR: in parallel or on device");
    Kokkos::Impl::throw_runtime_exception(msg);
  }
}

void OpenMPTargetExec::verify_initialized(const char* const label) {
  if (0 == Kokkos::Experimental::OpenMPTarget().impl_is_initialized()) {
    std::string msg(label);
    msg.append(" ERROR: not initialized");
    Kokkos::Impl::throw_runtime_exception(msg);
  }
  OpenMPTargetExec::MAX_ACTIVE_THREADS =
      Kokkos::Experimental::OpenMPTarget().concurrency();
}

void* OpenMPTargetExec::m_scratch_ptr         = nullptr;
int64_t OpenMPTargetExec::m_scratch_size      = 0;
int* OpenMPTargetExec::m_lock_array           = nullptr;
uint64_t OpenMPTargetExec::m_lock_size        = 0;
uint32_t* OpenMPTargetExec::m_uniquetoken_ptr = nullptr;
int OpenMPTargetExec::MAX_ACTIVE_THREADS      = 0;

void OpenMPTargetExec::clear_scratch() {
  Kokkos::Experimental::OpenMPTargetSpace space;
  space.deallocate(m_scratch_ptr, m_scratch_size);
  m_scratch_ptr  = nullptr;
  m_scratch_size = 0;
}

void OpenMPTargetExec::clear_lock_array() {
  if (m_lock_array != nullptr) {
    Kokkos::Experimental::OpenMPTargetSpace space;
    space.deallocate(m_lock_array, m_lock_size);
    m_lock_array = nullptr;
    m_lock_size  = 0;
  }
}

void* OpenMPTargetExec::get_scratch_ptr() { return m_scratch_ptr; }

void OpenMPTargetExec::resize_scratch(int64_t team_size, int64_t shmem_size_L0,
                                      int64_t shmem_size_L1,
                                      int64_t league_size) {
  Kokkos::Experimental::OpenMPTargetSpace space;
  const int64_t shmem_size =
      shmem_size_L0 + shmem_size_L1;  // L0 + L1 scratch memory per team.
  const int64_t padding = shmem_size * 10 / 100;  // Padding per team.

  // Maximum active teams possible.
  // The number should not exceed the maximum in-flight teams possible or the
  // league_size.
  int max_active_teams =
      std::min(OpenMPTargetExec::MAX_ACTIVE_THREADS / team_size, league_size);

  // max_active_teams is the number of active teams on the given hardware.
  // We set the number of teams to be twice the number of max_active_teams for
  // the compiler to pick the right number in its case.
  // FIXME_OPENMPTARGET: Cray compiler did not yet implement omp_set_num_teams.
#if !defined(KOKKOS_COMPILER_CRAY_LLVM)
  omp_set_num_teams(max_active_teams * 2);
#endif

  // Total amount of scratch memory allocated is depenedent
  // on the maximum number of in-flight teams possible.
  int64_t total_size =
      (shmem_size + OpenMPTargetExecTeamMember::TEAM_REDUCE_SIZE + padding) *
      max_active_teams * 2;

  if (total_size > m_scratch_size) {
    space.deallocate(m_scratch_ptr, m_scratch_size);
    m_scratch_size = total_size;
    m_scratch_ptr  = space.allocate(total_size);
  }
}

int* OpenMPTargetExec::get_lock_array(int num_teams) {
  Kokkos::Experimental::OpenMPTargetSpace space;
  int max_active_league_size = MAX_ACTIVE_THREADS / 32;
  int lock_array_elem =
      (num_teams > max_active_league_size) ? num_teams : max_active_league_size;
  if (m_lock_size < (lock_array_elem * sizeof(int))) {
    space.deallocate(m_lock_array, m_lock_size);
    m_lock_size  = lock_array_elem * sizeof(int);
    m_lock_array = static_cast<int*>(space.allocate(m_lock_size));

    // FIXME_OPENMPTARGET - Creating a target region here to initialize the
    // lock_array with 0's fails. Hence creating an equivalent host array to
    // achieve the same. Value of host array are then copied to the lock_array.
    int* h_lock_array = static_cast<int*>(
        omp_target_alloc(m_lock_size, omp_get_initial_device()));

    for (int i = 0; i < lock_array_elem; ++i) h_lock_array[i] = 0;

    KOKKOS_IMPL_OMPT_SAFE_CALL(
        omp_target_memcpy(m_lock_array, h_lock_array, m_lock_size, 0, 0,
                          omp_get_default_device(), omp_get_initial_device()));

    omp_target_free(h_lock_array, omp_get_initial_device());
  }

  return m_lock_array;
}

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_ENABLE_OPENMPTARGET
