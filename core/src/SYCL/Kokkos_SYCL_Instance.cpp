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

#include <Kokkos_Concepts.hpp>
#include <SYCL/Kokkos_SYCL_Instance.hpp>
#include <Kokkos_SYCL.hpp>
#include <Kokkos_HostSpace.hpp>
#include <Kokkos_Serial.hpp>
#include <impl/Kokkos_Error.hpp>

namespace Kokkos {
namespace Experimental {
namespace Impl {

std::vector<std::optional<sycl::queue>*> SYCLInternal::all_queues;
std::mutex SYCLInternal::mutex;

SYCLInternal::~SYCLInternal() {
  if (!was_finalized || m_scratchSpace || m_scratchFlags) {
    std::cerr << "Kokkos::Experimental::SYCL ERROR: Failed to call "
                 "Kokkos::Experimental::SYCL::finalize()"
              << std::endl;
    std::cerr.flush();
  }

  m_scratchSpace = nullptr;
  m_scratchFlags = nullptr;
}

int SYCLInternal::verify_is_initialized(const char* const label) const {
  if (!is_initialized()) {
    std::cerr << "Kokkos::Experimental::SYCL::" << label
              << " : ERROR device not initialized" << std::endl;
  }
  return is_initialized();
}
SYCLInternal& SYCLInternal::singleton() {
  static SYCLInternal self;
  return self;
}

void SYCLInternal::initialize(const sycl::device& d) {
  auto exception_handler = [](sycl::exception_list exceptions) {
    bool asynchronous_error = false;
    for (std::exception_ptr const& e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (sycl::exception const& e) {
        std::cerr << e.what() << '\n';
        asynchronous_error = true;
      }
    }
    if (asynchronous_error)
      Kokkos::Impl::throw_runtime_exception(
          "There was an asynchronous SYCL error!\n");
  };
  initialize(sycl::queue{d, exception_handler});
}

// FIXME_SYCL
void SYCLInternal::initialize(const sycl::queue& q) {
  if (was_finalized)
    Kokkos::abort("Calling SYCL::initialize after SYCL::finalize is illegal\n");

  if (is_initialized()) return;

  if (!HostSpace::execution_space::impl_is_initialized()) {
    const std::string msg(
        "SYCL::initialize ERROR : HostSpace::execution_space is not "
        "initialized");
    Kokkos::Impl::throw_runtime_exception(msg);
  }

  const bool ok_init = nullptr == m_scratchSpace || nullptr == m_scratchFlags;
  const bool ok_dev  = true;
  if (ok_init && ok_dev) {
    m_queue = q;
    // guard pushing to all_queues
    {
      std::lock_guard<std::mutex> lock(mutex);
      all_queues.push_back(&m_queue);
    }
    const sycl::device& d = m_queue->get_device();
    std::cout << SYCL::SYCLDevice(d) << '\n';

    m_maxThreadsPerSM =
        d.template get_info<sycl::info::device::max_work_group_size>();
    m_maxShmemPerBlock =
        d.template get_info<sycl::info::device::local_mem_size>();
    m_indirectKernelMem.reset(*m_queue);
    m_indirectReducerMem.reset(*m_queue);
  } else {
    std::ostringstream msg;
    msg << "Kokkos::Experimental::SYCL::initialize(...) FAILED";

    if (!ok_init) {
      msg << " : Already initialized";
    }
    Kokkos::Impl::throw_runtime_exception(msg.str());
  }
}

void SYCLInternal::finalize() {
  SYCL().fence();
  was_finalized = true;
  if (nullptr != m_scratchSpace || nullptr != m_scratchFlags) {
    // FIXME_SYCL
    std::abort();
  }

  m_indirectKernelMem.reset();
  m_indirectReducerMem.reset();
  // guard erasing from all_queues
  {
    std::lock_guard<std::mutex> lock(mutex);
    all_queues.erase(std::find(all_queues.begin(), all_queues.end(), &m_queue));
  }
  m_queue.reset();
}

}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos
