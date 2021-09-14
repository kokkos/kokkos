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

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_THREADS)

#include <Kokkos_Core_fwd.hpp>
/* Standard 'C' Linux libraries */

#include <pthread.h>
#include <sched.h>
#include <errno.h>

/* Standard C++ libraries */

#include <cstdlib>
#include <string>
#include <iostream>
#include <stdexcept>

#include <Kokkos_Threads.hpp>

//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {
namespace {

pthread_mutex_t host_internal_pthread_mutex = PTHREAD_MUTEX_INITIALIZER;

// Pthreads compatible driver.
// Recovery from an exception would require constant intra-thread health
// verification; which would negatively impact runtime.  As such simply
// abort the process.

void* internal_pthread_driver(void*) {
  try {
    ThreadsExec::driver();
  } catch (const std::exception& x) {
    std::cerr << "Exception thrown from worker thread: " << x.what()
              << std::endl;
    std::cerr.flush();
    std::abort();
  } catch (...) {
    std::cerr << "Exception thrown from worker thread" << std::endl;
    std::cerr.flush();
    std::abort();
  }
  return nullptr;
}

}  // namespace

//----------------------------------------------------------------------------
// Spawn a thread

bool ThreadsExec::spawn() {
  bool result = false;

  pthread_attr_t attr;

  if (0 == pthread_attr_init(&attr) &&
      0 == pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM) &&
      0 == pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED)) {
    pthread_t pt;

    result = 0 == pthread_create(&pt, &attr, internal_pthread_driver, nullptr);
  }

  pthread_attr_destroy(&attr);

  return result;
}

//----------------------------------------------------------------------------

bool ThreadsExec::is_process() {
  static const pthread_t master_pid = pthread_self();

  return pthread_equal(master_pid, pthread_self());
}

void ThreadsExec::global_lock() {
  pthread_mutex_lock(&host_internal_pthread_mutex);
}

void ThreadsExec::global_unlock() {
  pthread_mutex_unlock(&host_internal_pthread_mutex);
}

//----------------------------------------------------------------------------

void ThreadsExec::wait_yield(volatile int& flag, const int value) {
  while (value == flag) {
    sched_yield();
  }
}

}  // namespace Impl
}  // namespace Kokkos

#else
void KOKKOS_CORE_SRC_THREADS_EXEC_BASE_PREVENT_LINK_ERROR() {}
#endif /* end #if defined( KOKKOS_ENABLE_THREADS ) */
