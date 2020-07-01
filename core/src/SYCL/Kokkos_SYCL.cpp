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

/*--------------------------------------------------------------------------*/
/* Kokkos interfaces */

#include <Kokkos_Core.hpp>

/* only compile this file if SYCL is enabled for Kokkos */
#ifdef KOKKOS_ENABLE_SYCL

//#include <SYCL/Kokkos_SYCL_Internal.hpp>
#include <impl/Kokkos_Error.hpp>
#include <Kokkos_SYCL.hpp>
#include <Kokkos_SYCL_Space.hpp>
#include <SYCL/Kokkos_SYCL_Instance.hpp>
#include <impl/Kokkos_Error.hpp>

/*--------------------------------------------------------------------------*/
/* Standard 'C' libraries */
#include <stdlib.h>

/* Standard 'C++' libraries */
#include <vector>
#include <iostream>
#include <sstream>
#include <string>

namespace {
template <typename C>
struct Container {
  explicit Container(const C& c) : container(c) {}

  friend std::ostream& operator<<(std::ostream& os, const Container& that) {
    os << that.container.size();
    for (const auto& v : that.container) {
      os << "\n\t" << v;
    }
    return os;
  }

 private:
  const C& container;
};
}  // namespace

namespace Kokkos {
namespace Experimental {

//-----------------------------------------------------------------------------
SYCL::SYCLDevice::SYCLDevice(cl::sycl::device d) : m_device(std::move(d)) {}

SYCL::SYCLDevice::SYCLDevice(const cl::sycl::device_selector& selector)
    : m_device(selector.select_device()) {}

SYCL::SYCLDevice::SYCLDevice(size_t id) {
  std::vector<cl::sycl::device> devices = cl::sycl::device::get_devices();
  if (id >= devices.size()) {
    std::ostringstream oss;
    oss << "Cannot select SYCL device #" << id << " out of " << devices.size()
        << " devices";
    Kokkos::Impl::throw_runtime_exception(oss.str());
  }

  m_device = devices[id];
}

// Not in a hot path, otherwise should be templatized instead of std::function
SYCL::SYCLDevice::SYCLDevice(const std::function<bool(const cl::sycl::device&)>& pred)
{
  using Devices = std::vector<cl::sycl::device>;
  Devices devices = cl::sycl::device::get_devices();
  Devices::const_iterator found = std::find_if(devices.begin(), devices.end(), pred);
  if (found == devices.end())
  {
    std::ostringstream oss;
    oss << "None of the " << devices.size() << " SYCL devices match pred.";
    Kokkos::Impl::throw_runtime_exception(oss.str());
  }

  m_device = *found;
}

#if defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_SYCL_CPU)
SYCL::SYCLDevice::SYCLDevice()
    : SYCLDevice(cl::sycl::cpu_selector()) {}
#elif defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_SYCL_GPU)
SYCL::SYCLDevice::SYCLDevice()
    : SYCLDevice(cl::sycl::gpu_selector()) {}
#endif

cl::sycl::device SYCL::SYCLDevice::get_device() const { return m_device; }

std::ostream& SYCL::SYCLDevice::list_devices(std::ostream& os) {
  std::vector<cl::sycl::device> devices = cl::sycl::device::get_devices();
  os << "There are " << devices.size() << " SYCL devices\n";

  for (size_t d = 0; d != devices.size(); ++d) {
    os << "Device #" << d << '\n' << SYCLDevice(devices[d]) << '\n';
  }

  return os;
}

void SYCL::SYCLDevice::list_devices() { list_devices(std::cout); }

std::ostream& SYCL::SYCLDevice::info(std::ostream& os) const {
  using namespace cl::sycl::info;
  return os << "Name: " << m_device.get_info<device::name>()
            << "\nDriver Version: "
            << m_device.get_info<device::driver_version>()
            << "\nIs Host: " << m_device.is_host()
            << "\nIs CPU: " << m_device.is_cpu()
            << "\nIs GPU: " << m_device.is_gpu()
            << "\nIs Accelerator: " << m_device.is_accelerator()
            << "\nVendor Id: " << m_device.get_info<device::vendor_id>()
            << "\nMax Compute Units: "
            << m_device.get_info<device::max_compute_units>()
            << "\nMax Work Item Dimensions: "
            << m_device.get_info<device::max_work_item_dimensions>()
            << "\nMax Work Group Size: "
            << m_device.get_info<device::max_work_group_size>()
            << "\nPreferred Vector Width Char: "
            << m_device.get_info<device::preferred_vector_width_char>()
            << "\nPreferred Vector Width Short: "
            << m_device.get_info<device::preferred_vector_width_short>()
            << "\nPreferred Vector Width Int: "
            << m_device.get_info<device::preferred_vector_width_int>()
            << "\nPreferred Vector Width Long: "
            << m_device.get_info<device::preferred_vector_width_long>()
            << "\nPreferred Vector Width Float: "
            << m_device.get_info<device::preferred_vector_width_float>()
            << "\nPreferred Vector Width Double: "
            << m_device.get_info<device::preferred_vector_width_double>()
            << "\nPreferred Vector Width Half: "
            << m_device.get_info<device::preferred_vector_width_half>()
            << "\nNative Vector Width Char: "
            << m_device.get_info<device::native_vector_width_char>()
            << "\nNative Vector Width Short: "
            << m_device.get_info<device::native_vector_width_short>()
            << "\nNative Vector Width Int: "
            << m_device.get_info<device::native_vector_width_int>()
            << "\nNative Vector Width Long: "
            << m_device.get_info<device::native_vector_width_long>()
            << "\nNative Vector Width Float: "
            << m_device.get_info<device::native_vector_width_float>()
            << "\nNative Vector Width Double: "
            << m_device.get_info<device::native_vector_width_double>()
            << "\nNative Vector Width Half: "
            << m_device.get_info<device::native_vector_width_half>()
            << "\nAddress Bits: " << m_device.get_info<device::address_bits>()
            << "\nImage Support: " << m_device.get_info<device::image_support>()
            << "\nMax Mem Alloc Size: "
            << m_device.get_info<device::max_mem_alloc_size>()
            << "\nMax Read Image Args: "
            << m_device.get_info<device::max_read_image_args>()
            << "\nImage2d Max Width: "
            << m_device.get_info<device::image2d_max_width>()
            << "\nImage2d Max Height: "
            << m_device.get_info<device::image2d_max_height>()
            << "\nImage3d Max Width: "
            << m_device.get_info<device::image3d_max_width>()
            << "\nImage3d Max Height: "
            << m_device.get_info<device::image3d_max_height>()
            << "\nImage3d Max Depth: "
            << m_device.get_info<device::image3d_max_depth>()
            << "\nImage Max Buffer Size: "
            << m_device.get_info<device::image_max_buffer_size>()
            << "\nImage Max Array Size: "
            << m_device.get_info<device::image_max_array_size>()
            << "\nMax Samplers: " << m_device.get_info<device::max_samplers>()
            << "\nMax Parameter Size: "
            << m_device.get_info<device::max_parameter_size>()
            << "\nMem Base Addr Align: "
            << m_device.get_info<device::mem_base_addr_align>()
            << "\nGlobal Cache Mem Line Size: "
            << m_device.get_info<device::global_mem_cache_line_size>()
            << "\nGlobal Mem Cache Size: "
            << m_device.get_info<device::global_mem_cache_size>()
            << "\nGlobal Mem Size: "
            << m_device.get_info<device::global_mem_size>()
            << "\nMax Constant Buffer Size: "
            << m_device.get_info<device::max_constant_buffer_size>()
            << "\nMax Constant Args: "
            << m_device.get_info<device::max_constant_args>()
            << "\nLocal Mem Size: "
            << m_device.get_info<device::local_mem_size>()
            << "\nError Correction Support: "
            << m_device.get_info<device::error_correction_support>()
            << "\nHost Unified Memory: "
            << m_device.get_info<device::host_unified_memory>()
            << "\nProfiling Timer Resolution: "
            << m_device.get_info<device::profiling_timer_resolution>()
            << "\nIs Endian Little: "
            << m_device.get_info<device::is_endian_little>()
            << "\nIs Available: " << m_device.get_info<device::is_available>()
            << "\nIs Compiler Available: "
            << m_device.get_info<device::is_compiler_available>()
            << "\nIs Linker Available: "
            << m_device.get_info<device::is_linker_available>()
            << "\nQueue Profiling: "
            << m_device.get_info<device::queue_profiling>()
            << "\nBuilt In Kernels: "
            << Container(m_device.get_info<device::built_in_kernels>())
            << "\nVendor: " << m_device.get_info<device::vendor>()
            << "\nProfile: " << m_device.get_info<device::profile>()
            << "\nVersion: " << m_device.get_info<device::version>()
            << "\nExtensions: "
            << Container(m_device.get_info<device::extensions>())
            << "\nPrintf Buffer Size: "
            << m_device.get_info<device::printf_buffer_size>()
            << "\nPreferred Interop User Sync: "
            << m_device.get_info<device::preferred_interop_user_sync>()
            << "\nPartition Max Sub Devices: "
            << m_device.get_info<device::partition_max_sub_devices>()
            << "\nReference Count: "
            << m_device.get_info<device::reference_count>() << '\n';
}
//----------------------------------------------------------------------------

}  // namespace Experimental
}  // namespace Kokkos

#endif

