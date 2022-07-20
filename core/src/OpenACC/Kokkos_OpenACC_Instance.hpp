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

#ifndef KOKKOS_OPENACC_INSTANCE_HPP
#define KOKKOS_OPENACC_INSTANCE_HPP

// #include <Kokkos_Core_fwd.hpp>

#include <iosfwd>

namespace Kokkos {
namespace Experimental {
namespace Impl {

enum class openacc_fence_is_static { yes, no };

class OpenACCInternal {
 private:
  OpenACCInternal()                       = default;
  OpenACCInternal(const OpenACCInternal&) = default;
  OpenACCInternal& operator=(const OpenACCInternal&) = default;

 public:
  // FIXME_OPENACC - Currently the maximum number of
  // teams possible is calculated based on NVIDIA's Volta GPU. In
  // future this value should be based on the chosen architecture for the
  // OpenACC backend.
  constexpr static int MAX_ACTIVE_THREADS = 2080 * 80; 
  constexpr static int MAX_ACTIVE_TEAMS   = MAX_ACTIVE_THREADS / 32; 

  static void verify_is_process(const char* const);
  static void verify_initialized(const char* const);
  static int* get_lock_array(int num_teams);
  static void* get_scratch_ptr();
  static void clear_scratch();
  static void clear_lock_array();
  // static void resize_scratch(int64_t team_reduce_bytes,
  // int64_t team_shared_bytes,
  // int64_t thread_local_bytes);

  static void* m_scratch_ptr;
  static int64_t m_scratch_size;
  static int* m_lock_array;
  static uint64_t m_lock_size;
  static uint32_t* m_uniquetoken_ptr;

  void fence(const std::string& name,
             openacc_fence_is_static is_static = openacc_fence_is_static::no);

  /** \brief  Return the maximum amount of concurrency.  */
  int concurrency();

  //! Print configuration information to the given output stream.
  void print_configuration(std::ostream&, const bool detail = false);

  static const char* name();

  //! Free any resources being consumed by the device.
  void impl_finalize();

  //! Has been initialized
  int impl_is_initialized();
  uint32_t impl_get_instance_id() const noexcept;
  //! Initialize, telling the OpenACC run-time library which device to use.
  void impl_initialize();

  static OpenACCInternal* impl_singleton();

 private:
  bool m_is_initialized  = false;
  uint32_t m_instance_id = Kokkos::Tools::Experimental::Impl::idForInstance<
      Kokkos::Experimental::OpenACC>(reinterpret_cast<uintptr_t>(this));
};
}  // Namespace Impl
}  // Namespace Experimental
}  // Namespace Kokkos

#endif  // KOKKOS_OPENACC_INSTANCE_HPP
