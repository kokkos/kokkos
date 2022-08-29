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

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#include <Kokkos_Macros.hpp>
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif
#ifndef KOKKOS_HIP_HPP
#define KOKKOS_HIP_HPP

#include <Kokkos_Core_fwd.hpp>

#include <Kokkos_Layout.hpp>
#include <HIP/Kokkos_HIP_Space.hpp>

#include <hip/hip_runtime_api.h>

namespace Kokkos {
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

  HIP();
  HIP(hipStream_t stream, bool manage_stream = false);

  //@}
  //------------------------------------
  //! \name Functions that all Kokkos devices must implement.
  //@{

  KOKKOS_INLINE_FUNCTION static int in_parallel() {
#if defined(__HIP_DEVICE_COMPILE__)
    return true;
#else
    return false;
#endif
  }

  /** \brief Wait until all dispatched functors complete.
   *
   * The parallel_for or parallel_reduce dispatch of a functor may return
   * asynchronously, before the functor completes. This method does not return
   * until all dispatched functors on this device have completed.
   */
  static void impl_static_fence(const std::string& name);

  void fence(const std::string& name =
                 "Kokkos::HIP::fence(): Unnamed Instance Fence") const;

  hipStream_t hip_stream() const;

  /// \brief Print configuration information to the given output stream.
  void print_configuration(std::ostream& os, bool verbose = false) const;

  /// \brief Free any resources being consumed by the device.
  static void impl_finalize();

  /** \brief  Initialize the device.
   *
   */
  int hip_device() const;
  static hipDeviceProp_t const& hip_device_prop();

  static void impl_initialize(InitializationSettings const&);

  static int impl_is_initialized();

  //  static size_type device_arch();

  static size_type detect_device_count();

  static int concurrency();
  static const char* name();

  inline Impl::HIPInternal* impl_internal_space_instance() const {
    return m_space_instance.get();
  }

  uint32_t impl_instance_id() const noexcept;

 private:
  friend bool operator==(HIP const& lhs, HIP const& rhs) {
    return lhs.impl_internal_space_instance() ==
           rhs.impl_internal_space_instance();
  }
  friend bool operator!=(HIP const& lhs, HIP const& rhs) {
    return !(lhs == rhs);
  }
  Kokkos::Impl::HostSharedPtr<Impl::HIPInternal> m_space_instance;
};

namespace Impl {
template <>
struct MemorySpaceAccess<HIPSpace, HIP::scratch_memory_space> {
  enum : bool { assignable = false };
  enum : bool { accessible = true };
  enum : bool { deepcopy = false };
};
}  // namespace Impl

namespace Tools {
namespace Experimental {
template <>
struct DeviceTypeTraits<HIP> {
  static constexpr DeviceType id = DeviceType::HIP;
  static int device_id(const HIP& exec) { return exec.hip_device(); }
};
}  // namespace Experimental
}  // namespace Tools

namespace Impl {
template <class DT, class... DP>
struct ZeroMemset<HIP, DT, DP...> {
  ZeroMemset(const HIP& exec_space, const View<DT, DP...>& dst,
             typename View<DT, DP...>::const_value_type&) {
    KOKKOS_IMPL_HIP_SAFE_CALL(hipMemsetAsync(
        dst.data(), 0,
        dst.size() * sizeof(typename View<DT, DP...>::value_type),
        exec_space.hip_stream()));
  }

  ZeroMemset(const View<DT, DP...>& dst,
             typename View<DT, DP...>::const_value_type&) {
    KOKKOS_IMPL_HIP_SAFE_CALL(
        hipMemset(dst.data(), 0,
                  dst.size() * sizeof(typename View<DT, DP...>::value_type)));
  }
};
}  // namespace Impl
}  // namespace Kokkos

#endif
