/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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

#ifndef KOKKOS_SYCL_HPP
#define KOKKOS_SYCL_HPP

#include <Kokkos_Core_fwd.hpp>

#if defined(KOKKOS_ENABLE_SYCL)

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#include <cstddef>
#include <iosfwd>
#include <Kokkos_HostSpace.hpp>
#include <Kokkos_SYCL_Space.hpp>
#include <Kokkos_ScratchSpace.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_Layout.hpp>
#include <impl/Kokkos_Tags.hpp>

#include <CL/sycl.hpp>

/*--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Experimental {
namespace Impl {
class SYCLInternal;
}

/// \class SYCL
/// \brief Kokkos device for multicore processors in the host memory space.
class SYCL {
 public:
  //------------------------------------
  //! \name Type declarations that all Kokkos devices must provide.
  //@{

  //! Tag this class as a kokkos execution space
  typedef SYCL execution_space;
  // typedef SYCLSpace             memory_space ;
  typedef SYCLHostUSMSpace memory_space;
  typedef Kokkos::Device<execution_space, memory_space> device_type;

  typedef LayoutLeft array_layout;
  typedef SYCLSpace::size_type size_type;

  typedef ScratchMemorySpace<SYCL> scratch_memory_space;

  ~SYCL() = default;
  SYCL();
  //  explicit SYCL( const int instance_id );

  SYCL(SYCL&&)      = default;
  SYCL(const SYCL&) = default;
  SYCL& operator=(SYCL&&) = default;
  SYCL& operator=(const SYCL&) = default;

  //@}
  //------------------------------------
  //! \name Functions that all Kokkos devices must implement.
  //@{

  KOKKOS_INLINE_FUNCTION static int in_parallel() {
#if defined(__HCC_ACCELERATOR__)
    return true;
#else
    return false;
#endif
  }

  /** \brief  Set the device in a "sleep" state. */
  static bool sleep();

  /** \brief Wake the device from the 'sleep' state. A noop for OpenMP. */
  static bool wake();

  /** \brief Wait until all dispatched functors complete. A noop for OpenMP. */
  static void impl_static_fence();
  void fence() const;

  /// \brief Print configuration information to the given output stream.
  static void print_configuration(std::ostream&, const bool detail = false);

  /// \brief Free any resources being consumed by the device.
  static void impl_finalize();

  /** \brief  Initialize the device.
   *
   */

  struct SelectDevice2 {
    SelectDevice2();
    explicit SelectDevice2(cl::sycl::device d);
    explicit SelectDevice2(const cl::sycl::device_selector& selector);
    explicit SelectDevice2(size_t id);
    explicit SelectDevice2(
        const std::function<bool(const sycl::device&)>& pred);

    cl::sycl::device get_device() const;

    friend std::ostream& operator<<(std::ostream& os,
                                    const SelectDevice2& that) {
      return that.info(os);
    }

    static std::ostream& list_devices(std::ostream& os);
    static void list_devices();

   private:
    std::ostream& info(std::ostream& os) const;

    cl::sycl::device m_device;
  };

  static void impl_initialize(SelectDevice2 = SelectDevice2());

  struct SelectDevice {
    int sycl_device_id;

    explicit SelectDevice(int selector) : sycl_device_id(selector) {
      auto devices = cl::sycl::device::get_devices();
      if (selector < 0 || devices.size() <= selector) {
        std::ostringstream oss;
        oss << "Cannot select SYCL device #" << selector << " out of "
            << devices.size() << " devices.";
        Kokkos::abort(oss.str().c_str());
      }
    }

#if defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_SYCL_CPU)
    SelectDevice() : SelectDevice(0) {}
#elif defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_SYCL_GPU)
    SelectDevice() : SelectDevice(0) {}
#else
    SelectDevice() {
      static_assert(
          false,
          "Neither KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_SYCL_CPU or "
          "KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_SYCL_GPU is defined.");
    }
#endif

   private:
    static int firstGPUDevice() {
      auto devices = cl::sycl::device::get_devices();
      auto found   = std::find_if(
          devices.begin(), devices.end(),
          [](const cl::sycl::device& device) { return device.is_gpu(); });

      // NLIBER
      std::cout << "devices.size(): " << devices.size()
                << "  GPU found: " << (found - devices.begin()) << '\n';
      if (found == devices.end()) {
        Kokkos::abort("No GPU device found.");
      }

      return found - devices.begin();
    }

    static int firstCPUDevice() {
      auto devices = cl::sycl::device::get_devices();
      auto found   = std::find_if(
          devices.begin(), devices.end(),
          [](const cl::sycl::device& device) { return device.is_cpu(); });

      // NLIBER
      std::cout << "devices.size(): " << devices.size()
                << "  CPU found: " << (found - devices.begin()) << '\n';
      if (found == devices.end()) {
        std::cerr << "No CPU device found.\n";
        for (auto& d : devices) std::cout << d.is_cpu() << '\n';
        Kokkos::abort("No CPU device found.");
      }

      return found - devices.begin();
    }
  };

  int sycl_device() const;

  // NLIBER static void impl_initialize(const SelectDevice = SelectDevice());

  static int impl_is_initialized();

  //  static size_type device_arch();

  //  static size_type detect_device_count();

  static int concurrency();
  static const char* name();

  inline Impl::SYCLInternal* impl_internal_space_instance() const {
    return m_space_instance;
  }

 private:
  Impl::SYCLInternal* m_space_instance;
};  // namespace Experimental
}  // namespace Experimental
}  // namespace Kokkos

namespace Kokkos {
namespace Impl {

template <>
struct MemorySpaceAccess<Kokkos::Experimental::SYCLSpace,
                         Kokkos::Experimental::SYCL::scratch_memory_space> {
  enum { assignable = false };
  enum { accessible = true };
  enum { deepcopy = false };
};

template <>
struct VerifyExecutionCanAccessMemorySpace<
    Kokkos::Experimental::SYCL::memory_space,
    Kokkos::Experimental::SYCL::scratch_memory_space> {
  enum { value = true };
  KOKKOS_INLINE_FUNCTION static void verify(void) {}
  KOKKOS_INLINE_FUNCTION static void verify(const void*) {}
};

template <>
struct VerifyExecutionCanAccessMemorySpace<
    Kokkos::HostSpace, Kokkos::Experimental::SYCL::scratch_memory_space> {
  enum { value = false };
  inline static void verify(void) {
    Kokkos::Experimental::SYCLSpace::access_error();
  }
  inline static void verify(const void* p) {
    Kokkos::Experimental::SYCLSpace::access_error(p);
  }
};

}  // namespace Impl
}  // namespace Kokkos

#include <SYCL/Kokkos_SYCL_Instance.hpp>
#include <SYCL/Kokkos_SYCL_Parallel_Range.hpp>

#endif
#endif

