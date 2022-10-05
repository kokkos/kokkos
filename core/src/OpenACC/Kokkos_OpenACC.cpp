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
#include <impl/Kokkos_ExecSpaceManager.hpp>
#include <impl/Kokkos_DeviceManagement.hpp>

#include <iostream>

Kokkos::Experimental::OpenACC::OpenACC()
    : m_space_instance(
          &Kokkos::Experimental::Impl::OpenACCInternal::singleton(),
          [](Impl::OpenACCInternal*) {}) {
  Impl::OpenACCInternal::singleton().verify_is_initialized(
      "OpenACC instance constructor");
}

Kokkos::Experimental::OpenACC::OpenACC(int async_arg)
    : m_space_instance(new Kokkos::Experimental::Impl::OpenACCInternal,
                       [](Impl::OpenACCInternal* ptr) {
                         ptr->finalize();
                         delete ptr;
                       }) {
  Impl::OpenACCInternal::singleton().verify_is_initialized(
      "OpenACC instance constructor");
  m_space_instance->initialize(async_arg);
}

void Kokkos::Experimental::OpenACC::impl_initialize(
    InitializationSettings const& settings) {
  if (Impl::OpenACC_Traits::may_fallback_to_host &&
      acc_get_num_devices(Impl::OpenACC_Traits::dev_type) == 0 &&
      !settings.has_device_id()) {
    if (show_warnings()) {
      std::cerr << "Warning: No GPU available for execution, falling back to"
                   " using the host!"
                << std::endl;
    }
    acc_set_device_type(acc_device_host);
    Impl::OpenACCInternal::m_acc_device_num =
        acc_get_device_num(acc_device_host);
  } else {
    using Kokkos::Impl::get_gpu;
    int const dev_num = get_gpu(settings);
    acc_set_device_num(dev_num, Impl::OpenACC_Traits::dev_type);
    Impl::OpenACCInternal::m_acc_device_num = dev_num;
  }
  Impl::OpenACCInternal::singleton().initialize();
}

void Kokkos::Experimental::OpenACC::impl_finalize() {
  Impl::OpenACCInternal::singleton().finalize();
}

bool Kokkos::Experimental::OpenACC::impl_is_initialized() {
  return Impl::OpenACCInternal::singleton().is_initialized();
}

void Kokkos::Experimental::OpenACC::print_configuration(std::ostream& os,
                                                        bool verbose) const {
  os << "Device Execution Space:\n";
  os << "  KOKKOS_ENABLE_OPENACC: yes\n";
  os << "OpenACC Options:\n";
  os << "  KOKKOS_ENABLE_OPENACC_COLLAPSE_HIERARCHICAL_CONSTRUCTS: ";
#ifdef KOKKOS_ENABLE_OPENACC_COLLAPSE_HIERARCHICAL_CONSTRUCTS
  os << "yes\n";
#else
  os << "no\n";
#endif
  m_space_instance->print_configuration(os, verbose);
}

void Kokkos::Experimental::OpenACC::fence(std::string const& name) const {
  m_space_instance->fence(name);
}

void Kokkos::Experimental::OpenACC::impl_static_fence(std::string const& name) {
  Kokkos::Tools::Experimental::Impl::profile_fence_event<
      Kokkos::Experimental::OpenACC>(
      name,
      Kokkos::Tools::Experimental::SpecialSynchronizationCases::
          GlobalDeviceSynchronization,
      [&]() { acc_wait_all(); });
}

uint32_t Kokkos::Experimental::OpenACC::impl_instance_id() const noexcept {
  return m_space_instance->instance_id();
}

int Kokkos::Experimental::OpenACC::acc_async_queue() const {
  return m_space_instance->m_async_arg;
}

int Kokkos::Experimental::OpenACC::acc_device_number() const {
  return Impl::OpenACCInternal::m_acc_device_num;
}

namespace Kokkos {
namespace Impl {
int g_openacc_space_factory_initialized =
    initialize_space_factory<Experimental::OpenACC>("170_OpenACC");
}  // namespace Impl
}  // Namespace Kokkos
