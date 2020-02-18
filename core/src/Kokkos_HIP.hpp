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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
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

#ifndef KOKKOS_HIP_HPP
#define KOKKOS_HIP_HPP

#include <Kokkos_Core_fwd.hpp>

#if defined(KOKKOS_ENABLE_HIP)

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#include <Kokkos_HIP_Space.hpp>
#include <Kokkos_HostSpace.hpp>
#include <Kokkos_Layout.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_ScratchSpace.hpp>
#include <cstddef>
#include <impl/Kokkos_Tags.hpp>
#include <iosfwd>

#include <hip/hip_runtime_api.h>
/*--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Experimental {
namespace Impl {
class HIPInternal;
}
/// \class HIP
/// \brief Kokkos device for multicore processors in the host memory space.
class HIP {
 public:
  //------------------------------------
  //! \name Type declarations that all Kokkos devices must provide.
  //@{

  //! Tag this class as a kokkos execution space
  using execution_space = HIP;
  using memory_space    = HIPSpace;
  using device_type     = Kokkos::Device<execution_space, memory_space>;

  using array_layout = LayoutLeft;
  using size_type    = HIPSpace::size_type;

  using scratch_memory_space = ScratchMemorySpace<HIP>;

  ~HIP() = default;
  HIP();
  //  explicit HIP( const int instance_id );

  HIP(HIP &&)      = default;
  HIP(const HIP &) = default;
  HIP &operator=(HIP &&) = default;
  HIP &operator=(const HIP &) = default;

  //@}
  //------------------------------------
  //! \name Functions that all Kokkos devices must implement.
  //@{

  KOKKOS_INLINE_FUNCTION static int in_parallel() {
#if defined(__HIP_ARCH__)
    return true;
#else
    return false;
#endif
  }

  /** \brief Wait until all dispatched functors complete. A noop for OpenMP. */
  static void impl_static_fence();
  void fence() const;

  /// \brief Print configuration information to the given output stream.
  static void print_configuration(std::ostream &, const bool detail = false);

  /// \brief Free any resources being consumed by the device.
  static void impl_finalize();

  /** \brief  Initialize the device.
   *
   */
  struct SelectDevice {
    int hip_device_id;
    SelectDevice() : hip_device_id(0) {}
    explicit SelectDevice(int id) : hip_device_id(id) {}
  };

  int hip_device() const;

  static void impl_initialize(const SelectDevice = SelectDevice());

  static int impl_is_initialized();

  //  static size_type device_arch();

  //  static size_type detect_device_count();

  static int concurrency();
  static const char *name();

  inline Impl::HIPInternal *impl_internal_space_instance() const {
    return m_space_instance;
  }

 private:
  Impl::HIPInternal *m_space_instance;
};
}  // namespace Experimental
}  // namespace Kokkos

namespace Kokkos {
namespace Impl {

template <>
struct MemorySpaceAccess<Kokkos::Experimental::HIPSpace,
                         Kokkos::Experimental::HIP::scratch_memory_space> {
  enum { assignable = false };
  enum { accessible = true };
  enum { deepcopy = false };
};

template <>
struct VerifyExecutionCanAccessMemorySpace<
    Kokkos::Experimental::HIP::memory_space,
    Kokkos::Experimental::HIP::scratch_memory_space> {
  enum { value = true };
  KOKKOS_INLINE_FUNCTION static void verify(void) {}
  KOKKOS_INLINE_FUNCTION static void verify(const void *) {}
};

template <>
struct VerifyExecutionCanAccessMemorySpace<
    Kokkos::HostSpace, Kokkos::Experimental::HIP::scratch_memory_space> {
  enum { value = false };
  inline static void verify(void) {
    Kokkos::Experimental::HIPSpace::access_error();
  }
  inline static void verify(const void *p) {
    Kokkos::Experimental::HIPSpace::access_error(p);
  }
};

}  // namespace Impl
}  // namespace Kokkos

#include <HIP/Kokkos_HIP_Instance.hpp>
#include <HIP/Kokkos_HIP_Parallel.hpp>

#endif
#endif
