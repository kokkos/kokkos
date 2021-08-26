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

#include <Kokkos_Core.hpp>

#ifdef KOKKOS_ENABLE_HPX
#include <Kokkos_HPX.hpp>

#include <hpx/util/yield_while.hpp>

namespace Kokkos {
namespace Experimental {

bool HPX::m_hpx_initialized = false;
std::atomic<uint32_t> HPX::m_next_instance_id{1};
#if defined(KOKKOS_ENABLE_HPX_ASYNC_DISPATCH)
std::atomic<uint32_t> HPX::m_active_parallel_region_count{0};
HPX::instance_data HPX::m_global_instance_data;
#else
Kokkos::Impl::thread_buffer HPX::m_global_buffer;
#endif

int HPX::concurrency() {
  hpx::runtime *rt = hpx::get_runtime_ptr();
  if (rt == nullptr) {
    return hpx::threads::hardware_concurrency();
  } else {
    if (hpx::threads::get_self_ptr() == nullptr) {
      return hpx::resource::get_thread_pool(0).get_os_thread_count();
    } else {
      return hpx::this_thread::get_pool()->get_os_thread_count();
    }
  }
}

void HPX::impl_initialize(int thread_count) {
  hpx::runtime *rt = hpx::get_runtime_ptr();
  if (rt == nullptr) {
    std::vector<std::string> config = {
        "hpx.os_threads=" + std::to_string(thread_count),
#ifdef KOKKOS_ENABLE_DEBUG
        "--hpx:attach-debugger=exception",
#endif
    };
    int argc_hpx     = 1;
    char name[]      = "kokkos_hpx";
    char *argv_hpx[] = {name, nullptr};
    hpx::start(nullptr, argc_hpx, argv_hpx, config);

#if HPX_VERSION_FULL < 0x010400
    // This has been fixed in HPX 1.4.0.
    //
    // NOTE: Wait for runtime to start. hpx::start returns as soon as
    // possible, meaning some operations are not allowed immediately
    // after hpx::start. Notably, hpx::stop needs state_running. This
    // needs to be fixed in HPX itself.

    // Get runtime pointer again after it has been started.
    rt = hpx::get_runtime_ptr();
    hpx::util::yield_while(
        [rt]() { return rt->get_state() < hpx::state_running; });
#endif

    m_hpx_initialized = true;
  }
}

void HPX::impl_initialize() {
  hpx::runtime *rt = hpx::get_runtime_ptr();
  if (rt == nullptr) {
    std::vector<std::string> config = {
#ifdef KOKKOS_ENABLE_DEBUG
        "--hpx:attach-debugger=exception",
#endif
    };
    int argc_hpx     = 1;
    char name[]      = "kokkos_hpx";
    char *argv_hpx[] = {name, nullptr};
    hpx::start(nullptr, argc_hpx, argv_hpx, config);

    // NOTE: Wait for runtime to start. hpx::start returns as soon as
    // possible, meaning some operations are not allowed immediately
    // after hpx::start. Notably, hpx::stop needs state_running. This
    // needs to be fixed in HPX itself.

    // Get runtime pointer again after it has been started.
    rt = hpx::get_runtime_ptr();
    hpx::util::yield_while(
        [rt]() { return rt->get_state() < hpx::state_running; });

    m_hpx_initialized = true;
  }
}

bool HPX::impl_is_initialized() noexcept {
  hpx::runtime *rt = hpx::get_runtime_ptr();
  return rt != nullptr;
}

void HPX::impl_finalize() {
  if (m_hpx_initialized) {
    hpx::runtime *rt = hpx::get_runtime_ptr();
    if (rt != nullptr) {
      hpx::apply([]() { hpx::finalize(); });
      hpx::stop();
    } else {
      Kokkos::abort(
          "Kokkos::Experimental::HPX::impl_finalize: Kokkos started "
          "HPX but something else already stopped HPX\n");
    }
  }
}

}  // namespace Experimental

namespace Impl {

int g_hpx_space_factory_initialized =
    initialize_space_factory<HPXSpaceInitializer>("060_HPX");

void HPXSpaceInitializer::initialize(const InitArguments &args) {
  const int num_threads = args.num_threads;

  if (std::is_same<Kokkos::Experimental::HPX,
                   Kokkos::DefaultExecutionSpace>::value ||
      std::is_same<Kokkos::Experimental::HPX,
                   Kokkos::HostSpace::execution_space>::value) {
    if (num_threads > 0) {
      Kokkos::Experimental::HPX::impl_initialize(num_threads);
    } else {
      Kokkos::Experimental::HPX::impl_initialize();
    }
    // std::cout << "Kokkos::initialize() fyi: HPX enabled and initialized" <<
    // std::endl ;
  } else {
    // std::cout << "Kokkos::initialize() fyi: HPX enabled but not initialized"
    // << std::endl ;
  }
}

void HPXSpaceInitializer::finalize(const bool all_spaces) {
  if (std::is_same<Kokkos::Experimental::HPX,
                   Kokkos::DefaultExecutionSpace>::value ||
      std::is_same<Kokkos::Experimental::HPX,
                   Kokkos::HostSpace::execution_space>::value ||
      all_spaces) {
    if (Kokkos::Experimental::HPX::impl_is_initialized())
      Kokkos::Experimental::HPX::impl_finalize();
  }
}

void HPXSpaceInitializer::fence() { Kokkos::Experimental::HPX().fence(); }

void HPXSpaceInitializer::print_configuration(std::ostream &msg,
                                              const bool detail) {
  msg << "HPX Execution Space:" << std::endl;
  msg << "  KOKKOS_ENABLE_HPX: ";
  msg << "yes" << std::endl;

  msg << "\nHPX Runtime Configuration:" << std::endl;
  Kokkos::Experimental::HPX::print_configuration(msg, detail);
}

}  // namespace Impl
}  // namespace Kokkos

#else
void KOKKOS_CORE_SRC_IMPL_HPX_PREVENT_LINK_ERROR() {}
#endif  //#ifdef KOKKOS_ENABLE_HPX
