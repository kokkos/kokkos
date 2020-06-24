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

// KOKKOS_INLINE_FUNCTION
// Kokkos::Impl::SYCLLockArraysStruct kokkos_impl_sycl_lock_arrays ;

namespace Kokkos {
namespace Experimental {
namespace Impl {

int SYCLInternal::was_finalized = 0;

//----------------------------------------------------------------------------
void SYCLInternal::listDevices(std::ostream& out) const {
  auto devices = cl::sycl::device::get_devices();
  out << "The system contains " << devices.size() << " devices\n";

  for (size_t d = 0; d != devices.size(); ++d) {
    out << "Device: " << d << '\n' << SYCL::SelectDevice2(devices[d]) << '\n';
  }
}

void SYCLInternal::listDevices() const { return listDevices(std::cout); }

void SYCLInternal::print_configuration(std::ostream& s) const {
  // const SYCLInternalDevices & dev_info = SYCLInternalDevices::singleton();

#if defined(KOKKOS_ENABLE_SYCL)
  s << "macro  KOKKOS_ENABLE_SYCL      : defined" << std::endl;
#endif

  // for ( int i = 0 ; i < dev_info.m_syclDevCount ; ++i ) {
  //  s << "Kokkos::Experimental::SYCL[ " << i << " ] "
  //<< dev_info.m_syclProp[i].name
  //<< " version " << (dev_info.m_syclProp[i].major) << "." <<
  // dev_info.m_syclProp[i].minor
  //<< ", Total Global Memory: " <<
  // human_memory_size(dev_info.m_syclProp[i].totalGlobalMem)
  //<< ", Shared Memory per Wavefront: " <<
  // human_memory_size(dev_info.m_syclProp[i].sharedMemPerWavefront);
  // if ( m_syclDev == i ) s << " : Selected" ;
  // s << std::endl ;
  //}
}

//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
void SYCLInternal::print_configuration(std::ostream& s, const bool) const {
  print_configuration(s);
}

//----------------------------------------------------------------------------

SYCLInternal::SYCLInternal() = default;

SYCLInternal::~SYCLInternal() {
  if (m_scratchSpace || m_scratchFlags) {
    std::cerr << "Kokkos::Experimental::SYCL ERROR: Failed to call "
                 "Kokkos::Experimental::SYCL::finalize()"
              << std::endl;
    std::cerr.flush();
  }

  m_syclDev           = -1;
  m_syclArch          = -1;
  m_multiProcCount    = 0;
  m_maxWorkgroup      = 0;
  m_maxSharedWords    = 0;
  m_scratchSpaceCount = 0;
  m_scratchFlagsCount = 0;
  m_scratchSpace      = 0;
  m_scratchFlags      = 0;
}

int SYCLInternal::verify_is_initialized(const char* const label) const {
  if (!is_initialized()) {
    std::cerr << "Kokkos::Experimental::SYCL::" << label
              << " : ERROR device not initialized" << std::endl;
  }
  return true;
}

SYCLInternal& SYCLInternal::singleton() {
  static SYCLInternal* self = nullptr;
  if (!self) {
    self = new SYCLInternal();
  }
  return *self;
}

void SYCLInternal::initialize(cl::sycl::device d) {
  if (was_finalized)
    Kokkos::abort("Calling SYCL::initialize after SYCL::finalize is illegal\n");

  if (is_initialized()) return;

  enum { WordSize = sizeof(size_type) };

  if (!HostSpace::execution_space::impl_is_initialized()) {
    const std::string msg(
        "SYCL::initialize ERROR : HostSpace::execution_space is not "
        "initialized");
    Kokkos::Impl::throw_runtime_exception(msg);
  }

  // const SYCLInternalDevices & dev_info = SYCLInternalDevices::singleton();

  const bool ok_init = 0 == m_scratchSpace || 0 == m_scratchFlags;

  // const bool ok_id   = 1 <= sycl_device_id &&
  //                          sycl_device_id < dev_info.m_syclDevCount ;
  const bool ok_id = true;
  // Need at least a GPU device

  // const bool ok_dev = ok_id &&
  //  ( 1 <= dev_info.m_syclProp[ sycl_device_id ].major &&
  //    0 <= dev_info.m_syclProp[ sycl_device_id ].minor );
  const bool ok_dev = true;
  if (ok_init && ok_dev) {
    // const struct syclDeviceProp & syclProp =
    //  dev_info.m_syclProp[ sycl_device_id ];

    // m_syclDev = sycl_device_id;
    // listDevices();

    // Kokkos::Impl::sycl_device_synchronize();

    // auto devices = cl::sycl::device::get_devices();
    m_queue = std::make_unique<cl::sycl::queue>(d);
    std::cout << SYCL::SelectDevice2(d) << '\n';

    /*
        // Query what compute capability architecture a kernel executes:
        m_syclArch = sycl_kernel_arch();
        if ( m_syclArch != syclProp.major * 100 + syclProp.minor * 10 ) {
          std::cerr << "Kokkos::Experimental::SYCL::initialize WARNING:
       running kernels compiled for compute capability "
                    << ( m_syclArch / 100 ) << "." << ( ( m_syclArch % 100 ) /
       10 )
                    << " on device with compute capability "
                    << syclProp.major << "." << syclProp.minor
                    << " , this will likely reduce potential performance."
                    << std::endl ;
        }
    */
    // number of multiprocessors

    //    m_multiProcCount = syclProp.multiProcessorCount ;

    //----------------------------------
    // Maximum number of wavefronts,
    // at most one workgroup per thread in a workgroup for reduction.

    //    m_maxSharedWords = syclProp.sharedMemPerWavefront/ WordSize ;

    //----------------------------------
    // Maximum number of Workgroups:

    //    m_maxWorkgroup = 5*syclProp.multiProcessorCount;  //TODO: confirm
    //    usage and value

    //----------------------------------
    // Multiblock reduction uses scratch flags for counters
    // and scratch space for partial reduction values.
    // Allocate some initial space.  This will grow as needed.

    /*   {
         const unsigned reduce_block_count = m_maxWorkgroup *
       Impl::SYCLTraits::WorkgroupSize ;

         (void) scratch_flags( reduce_block_count * 2  * sizeof(size_type) );
         (void) scratch_space( reduce_block_count * 16 * sizeof(size_type) );
       }*/
    //----------------------------------

  } else {
    std::ostringstream msg;
    msg << "Kokkos::Experimental::SYCL::initialize(...) FAILED";

    if (!ok_init) {
      msg << " : Already initialized";
    }
    /*
    if ( ! ok_id ) {
      msg << " : Device identifier out of range "
          << "[0.." << (dev_info.m_syclDevCount-1) << "]" ;
    }
    else if ( ! ok_dev ) {
      msg << " : Device " ;
      msg << dev_info.m_syclProp[ sycl_device_id ].major ;
      msg << "." ;
      msg << dev_info.m_syclProp[ sycl_device_id ].minor ;
      msg << " Need at least a GPU" ;
      msg << std::endl;
    }
    */
    Kokkos::Impl::throw_runtime_exception(msg.str());
  }

  // Init the array for used for arbitrarily sized atomics
  // Kokkos::Impl::init_lock_arrays_sycl_space();

  //  Kokkos::Impl::SYCLLockArraysStruct locks;
  //  locks.atomic = atomic_lock_array_sycl_space_ptr(false);
  //  locks.scratch = scratch_lock_array_sycl_space_ptr(false);
  //  locks.threadid = threadid_lock_array_sycl_space_ptr(false);
  //  syclMemcpyToSymbol( kokkos_impl_sycl_lock_arrays , & locks ,
  //  sizeof(SYCLLockArraysStruct) );
}

void SYCLInternal::initialize(const cl::sycl::device_selector& s) {
  initialize(s.select_device());
}

void SYCLInternal::initialize(int sycl_device_id) {
  using Devices   = std::vector<cl::sycl::device>;
  Devices devices = cl::sycl::device::get_devices();
  if (sycl_device_id < 0 || sycl_device_id >= devices.size()) {
    std::ostringstream oss;
    oss << "SYCL::initialize ERROR : Device #" << sycl_device_id
        << " out of range (only " << devices.size()
        << " possible SYCL devices.";
    Kokkos::Impl::throw_runtime_exception(oss.str());
  }

  initialize(devices[sycl_device_id]);
}

// Initialize with the first GPU or accelerator
void SYCLInternal::initialize() {
  auto devices = cl::sycl::device::get_devices();
  auto found   = std::find_if(devices.begin(), devices.end(),
                            [](const cl::sycl::device& d) {
                              return d.is_accelerator() || d.is_gpu();
                            });
  // Didn't find a GPU or accelerator
  if (found == devices.end()) {
    Kokkos::Impl::throw_runtime_exception("SYCL::initialize ERROR: No GPU or Accelerator was found!");
  }

  return initialize(*found);
}

//----------------------------------------------------------------------------

// typedef Kokkos::Experimental::SYCL::size_type ScratchGrain[
// Impl::SYCLTraits::WorkgroupSize ] ; enum { sizeScratchGrain =
// sizeof(ScratchGrain) };

Kokkos::Experimental::SYCL::size_type* SYCLInternal::scratch_flags(
    const Kokkos::Experimental::SYCL::size_type size) {
  /*  if ( verify_is_initialized("scratch_flags") && m_scratchFlagsCount *
    sizeScratchGrain < size ) {


      m_scratchFlagsCount = ( size + sizeScratchGrain - 1 ) / sizeScratchGrain
    ;

      typedef Kokkos::Impl::SharedAllocationRecord<
    Kokkos::Experimental::SYCLSpace , void > Record ;

      Record * const r = Record::allocate( Kokkos::Experimental::SYCLSpace()
                                         , "InternalScratchFlags"
                                         , ( sizeScratchGrain  *
    m_scratchFlagsCount ) );

      Record::increment( r );

      m_scratchFlags = reinterpret_cast<size_type *>( r->data() );

      syclMemset( m_scratchFlags , 0 , m_scratchFlagsCount * sizeScratchGrain
    );
    }
  */
  return m_scratchFlags;
}

Kokkos::Experimental::SYCL::size_type* SYCLInternal::scratch_space(
    const Kokkos::Experimental::SYCL::size_type size) {
  /*  if ( verify_is_initialized("scratch_space") && m_scratchSpaceCount *
    sizeScratchGrain < size ) {

      m_scratchSpaceCount = ( size + sizeScratchGrain - 1 ) / sizeScratchGrain
    ;

       typedef Kokkos::Impl::SharedAllocationRecord<
    Kokkos::Experimental::SYCLSpace , void > Record ;

       static Record * const r = Record::allocate(
    Kokkos::Experimental::SYCLSpace() , "InternalScratchSpace" , (
    sizeScratchGrain  * m_scratchSpaceCount ) );

       Record::increment( r );

       m_scratchSpace = reinterpret_cast<size_type *>( r->data() );
    }
  */
  return m_scratchSpace;
}

//----------------------------------------------------------------------------

void SYCLInternal::finalize() {
  SYCL().fence();
  was_finalized = 1;
  if (0 != m_scratchSpace || 0 != m_scratchFlags) {
    //    atomic_lock_array_sycl_space_ptr(false);
    //    scratch_lock_array_sycl_space_ptr(false);
    //    threadid_lock_array_sycl_space_ptr(false);

    typedef Kokkos::Impl::SharedAllocationRecord<
        Kokkos::Experimental::SYCLSpace>
        RecordSYCL;
    typedef Kokkos::Impl::SharedAllocationRecord<
        Kokkos::Experimental::SYCLHostPinnedSpace>
        RecordHost;

    RecordSYCL::decrement(RecordSYCL::get_record(m_scratchFlags));
    RecordSYCL::decrement(RecordSYCL::get_record(m_scratchSpace));

    m_syclDev           = -1;
    m_multiProcCount    = 0;
    m_maxWorkgroup      = 0;
    m_maxSharedWords    = 0;
    m_scratchSpaceCount = 0;
    m_scratchFlagsCount = 0;
    m_scratchSpace      = 0;
    m_scratchFlags      = 0;
  }
  m_queue.reset();
}

//----------------------------------------------------------------------------

Kokkos::Experimental::SYCL::size_type sycl_internal_cu_count() {
  return SYCLInternal::singleton().m_multiProcCount;
}

Kokkos::Experimental::SYCL::size_type sycl_internal_maximum_extent_size() {
  return SYCLInternal::singleton().m_maxWorkgroup;
}

Kokkos::Experimental::SYCL::size_type sycl_internal_maximum_shared_words() {
  return SYCLInternal::singleton().m_maxSharedWords;
}

Kokkos::Experimental::SYCL::size_type* sycl_internal_scratch_space(
    const Kokkos::Experimental::SYCL::size_type size) {
  return SYCLInternal::singleton().scratch_space(size);
}

Kokkos::Experimental::SYCL::size_type* sycl_internal_scratch_flags(
    const Kokkos::Experimental::SYCL::size_type size) {
  return SYCLInternal::singleton().scratch_flags(size);
}

}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

//----------------------------------------------------------------------------

namespace Kokkos {
namespace Experimental {

// SYCL::size_type SYCL::detect_device_count()
//{ return Impl::SYCLInternalDevices::singleton().m_syclDevCount ; }

int SYCL::concurrency() {
#if defined(KOKKOS_ARCH_KAVERI)
  return 8 * 64 * 40;  // 20480 kaveri
#else
  return 32 * 8 * 40;  // 81920 fiji and hawaii
#endif
}
int SYCL::impl_is_initialized() {
  return Impl::SYCLInternal::singleton().is_initialized();
}

void SYCL::impl_initialize(SYCL::SelectDevice2 d) {
  Impl::SYCLInternal::singleton().initialize(d.get_device());
#if defined(KOKKOS_ENABLE_PROFILING)
  Kokkos::Profiling::initialize();
#endif
}

#if 0
std::vector<unsigned>
SYCL::detect_device_arch()
{
  const Impl::SYCLInternalDevices & s = Impl::SYCLInternalDevices::singleton();

  std::vector<unsigned> output( s.m_syclDevCount );

  for ( int i = 0 ; i < s.m_syclDevCount ; ++i ) {
    output[i] = s.m_syclProp[i].major * 100 + s.m_syclProp[i].minor ;
  }

  return output ;
}

SYCL::size_type SYCL::device_arch()
{
  return 1 ;
}
#endif

void SYCL::impl_finalize() {
  Impl::SYCLInternal::singleton().finalize();

#if defined(KOKKOS_ENABLE_PROFILING)
  Kokkos::Profiling::finalize();
#endif
}

SYCL::SYCL() : m_space_instance(&Impl::SYCLInternal::singleton()) {
  Impl::SYCLInternal::singleton().verify_is_initialized(
      "SYCL instance constructor");
}

// SYCL::SYCL( const int instance_id )
//  : m_device( Impl::SYCLInternal::singleton().m_syclDev )
//{}

void SYCL::print_configuration(std::ostream& s, const bool detail) {
  Impl::SYCLInternal::singleton().print_configuration(s, detail);
}

bool SYCL::sleep() { return false; }

bool SYCL::wake() { return true; }

void SYCL::fence() const {
  m_space_instance->m_queue->wait();
  // SYCL_SAFE_CALL( syclDeviceSynchronize() );
}

int SYCL::sycl_device() const {
  return impl_internal_space_instance()->m_syclDev;
}
const char* SYCL::name() { return "SYCL"; }

}  // namespace Experimental
/*
namespace Impl {
void sycl_device_synchronize()
{
  SYCL_SAFE_CALL( syclDeviceSynchronize() );
}

void sycl_internal_error_throw( syclError_t e , const char * name, const char *
file, const int line )
{
  std::ostringstream out ;
  out << name << " error( " << syclGetErrorName(e) << "): " <<
syclGetErrorString(e); if (file) { out << " " << file << ":" << line;
  }
  throw_runtime_exception( out.str() );
}
}*/
}  // namespace Kokkos

#endif  // KOKKOS_ENABLE_SYCL
//----------------------------------------------------------------------------

