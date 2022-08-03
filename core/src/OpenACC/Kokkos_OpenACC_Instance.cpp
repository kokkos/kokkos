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

#include <OpenACC/Kokkos_OpenACC.hpp>
#include <OpenACC/Kokkos_OpenACC_Instance.hpp>
#include <OpenACC/Kokkos_OpenACC_Traits.hpp>
#include <impl/Kokkos_Profiling.hpp>
#include <impl/Kokkos_DeviceManagement.hpp>

#include <openacc.h>

#include <iostream>

namespace Kokkos {
bool show_warnings() noexcept;
}

int Kokkos::Experimental::Impl::OpenACCInternal::m_accDev = -1;

Kokkos::Experimental::Impl::OpenACCInternal&
Kokkos::Experimental::Impl::OpenACCInternal::singleton() {
  static OpenACCInternal self;
  return self;
}

bool Kokkos::Experimental::Impl::OpenACCInternal::verify_is_initialized(
    const char* const label) const {
  if (!m_is_initialized) {
    Kokkos::abort((std::string("Kokkos::Experimental::OpenACC::") + label +
                   " : ERROR device not initialized\n")
                      .c_str());
  }
  return m_is_initialized;
}

void Kokkos::Experimental::Impl::OpenACCInternal::initialize(int async_arg) {
  if ((async_arg < 0) && (async_arg != acc_async_sync) &&
      (async_arg != acc_async_noval)) {
    Kokkos::abort((std::string("Kokkos::Experimental::OpenACC::initialize()") +
                   " : ERROR async_arg should be a non-negative integer" +
                   " unless being a special value defined in OpenACC\n")
                      .c_str());
  }
  m_async_id       = async_arg;
  m_is_initialized = true;
}

void Kokkos::Experimental::Impl::OpenACCInternal::finalize() {
  m_is_initialized = false;
}

bool Kokkos::Experimental::Impl::OpenACCInternal::is_initialized() const {
  return m_is_initialized;
}

void Kokkos::Experimental::Impl::OpenACCInternal::print_configuration(
    std::ostream& os, bool /*verbose*/) const {
  os << "Using OpenACC\n";  // FIXME_OPENACC
}

void Kokkos::Experimental::Impl::OpenACCInternal::fence(
    std::string const& name) const {
  Kokkos::Tools::Experimental::Impl::profile_fence_event<
      Kokkos::Experimental::OpenACC>(
      name,
      Kokkos::Tools::Experimental::Impl::DirectFenceIDHandle{instance_id()},
      [&]() { acc_wait(m_async_id); });
}

uint32_t Kokkos::Experimental::Impl::OpenACCInternal::instance_id() const
    noexcept {
  return m_instance_id;
}

Kokkos::Experimental::Impl::OpenACCInternalDevices::OpenACCInternalDevices() {
  m_accDevCount = acc_get_num_devices(OpenACC_Traits::dev_type);
  if ((m_accDevCount == 0) && OpenACC_Traits::may_fallback_to_host) {
    if (show_warnings()) {
      std::cerr << "Warning: No GPU available for execution, falling back to"
                   " using the host!"
                << std::endl;
    }
    acc_set_device_type(acc_device_host);
    m_accDevCount = acc_get_num_devices(acc_device_host);
  }
}

const Kokkos::Experimental::Impl::OpenACCInternalDevices&
Kokkos::Experimental::Impl::OpenACCInternalDevices::singleton() {
  static OpenACCInternalDevices self;
  return self;
}
