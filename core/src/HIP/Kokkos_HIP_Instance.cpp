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

/* only compile this file if HIP is enabled for Kokkos */
#ifdef KOKKOS_ENABLE_HIP

//#include <HIP/Kokkos_HIP_Internal.hpp>
#include <impl/Kokkos_Error.hpp>
#include <Kokkos_HIP.hpp>
#include <Kokkos_HIP_Space.hpp>
#include <HIP/Kokkos_HIP_Instance.hpp>
#include <impl/Kokkos_Error.hpp>

/*--------------------------------------------------------------------------*/
/* Standard 'C' libraries */
#include <stdlib.h>

/* Standard 'C++' libraries */
#include <vector>
#include <iostream>
#include <sstream>
#include <string>



//KOKKOS_INLINE_FUNCTION
// Kokkos::Impl::HIPLockArraysStruct kokkos_impl_hip_lock_arrays ;

namespace Kokkos {
namespace Experimental {
namespace Impl {

int HIPInternal::was_finalized = 0;
//----------------------------------------------------------------------------


void HIPInternal::print_configuration( std::ostream & s ) const
{
  /*const HIPInternalDevices & dev_info = HIPInternalDevices::singleton();

#if defined( KOKKOS_ENABLE_HIP )
    s << "macro  KOKKOS_ENABLE_HIP      : defined" << std::endl ;
#endif
#if defined( __hcc_version__ )
    s << "macro  __hcc_version__          = " << __hcc_version__
      << std::endl ;
#endif

  for ( int i = 0 ; i < dev_info.m_hipDevCount ; ++i ) {
    s << "Kokkos::Experimental::HIP[ " << i << " ] "
      << dev_info.m_hipProp[i].name
      << " version " << (dev_info.m_hipProp[i].major) << "." << dev_info.m_hipProp[i].minor
      << ", Total Global Memory: " << human_memory_size(dev_info.m_hipProp[i].totalGlobalMem)
      << ", Shared Memory per Wavefront: " << human_memory_size(dev_info.m_hipProp[i].sharedMemPerWavefront);
    if ( m_hipDev == i ) s << " : Selected" ;
    s << std::endl ;
  }*/
}

//----------------------------------------------------------------------------

HIPInternal::~HIPInternal()
{
  if ( m_scratchSpace ||
       m_scratchFlags ) {
    std::cerr << "Kokkos::Experimental::HIP ERROR: Failed to call Kokkos::Experimental::HIP::finalize()"
              << std::endl ;
    std::cerr.flush();
  }

  m_hipDev                 = -1 ;
  m_hipArch                = -1 ;
  m_multiProcCount          = 0 ;
  m_maxWorkgroup            = 0 ;
  m_maxSharedWords          = 0 ;
  m_scratchSpaceCount       = 0 ;
  m_scratchFlagsCount       = 0 ;
  m_scratchSpace            = 0 ;
  m_scratchFlags            = 0 ;
}

int HIPInternal::verify_is_initialized( const char * const label ) const
{
  if ( m_hipDev < 0 ) {
    std::cerr << "Kokkos::Experimental::HIP::" << label << " : ERROR device not initialized" << std::endl ;
  }
  return 0 <= m_hipDev ;
}

HIPInternal & HIPInternal::singleton()
{
  static HIPInternal* self = nullptr ;
  if (!self) {
    self = new HIPInternal();
  }
  return *self ;

}

void HIPInternal::initialize( int hip_device_id  )
{
  printf("Initalize HIP\n");
  if ( was_finalized ) Kokkos::abort("Calling HIP::initialize after HIP::finalize is illegal\n");

  if ( is_initialized() ) return;

  enum { WordSize = sizeof(size_type) };

  if ( ! HostSpace::execution_space::impl_is_initialized() ) {
    const std::string msg("HIP::initialize ERROR : HostSpace::execution_space is not initialized");
    Kokkos::Impl::throw_runtime_exception( msg );
  }

  //const HIPInternalDevices & dev_info = HIPInternalDevices::singleton();

  const bool ok_init = 0 == m_scratchSpace || 0 == m_scratchFlags ;

  //const bool ok_id   = 1 <= hip_device_id &&
  //                          hip_device_id < dev_info.m_hipDevCount ;
  const bool ok_id = true;
  // Need at least a GPU device

  //const bool ok_dev = ok_id &&
  //  ( 1 <= dev_info.m_hipProp[ hip_device_id ].major &&
  //    0 <= dev_info.m_hipProp[ hip_device_id ].minor );
  const bool ok_dev = true;
  if ( ok_init && ok_dev ) {

    //const struct hipDeviceProp & hipProp =
    //  dev_info.m_hipProp[ hip_device_id ];

    m_hipDev = hip_device_id ;

    hipSetDevice( m_hipDev ) ;
    //Kokkos::Impl::hip_device_synchronize();

    m_stream = 0;
/*
    // Query what compute capability architecture a kernel executes:
    m_hipArch = hip_kernel_arch();
    if ( m_hipArch != hipProp.major * 100 + hipProp.minor * 10 ) {
      std::cerr << "Kokkos::Experimental::HIP::initialize WARNING: running kernels compiled for compute capability "
                << ( m_hipArch / 100 ) << "." << ( ( m_hipArch % 100 ) / 10 )
                << " on device with compute capability "
                << hipProp.major << "." << hipProp.minor
                << " , this will likely reduce potential performance."
                << std::endl ;
    }
*/
    // number of multiprocessors

//    m_multiProcCount = hipProp.multiProcessorCount ;

    //----------------------------------
    // Maximum number of wavefronts,
    // at most one workgroup per thread in a workgroup for reduction.


//    m_maxSharedWords = hipProp.sharedMemPerWavefront/ WordSize ;

    //----------------------------------
    // Maximum number of Workgroups:

//    m_maxWorkgroup = 5*hipProp.multiProcessorCount;  //TODO: confirm usage and value

    //----------------------------------
    // Multiblock reduction uses scratch flags for counters
    // and scratch space for partial reduction values.
    // Allocate some initial space.  This will grow as needed.

 /*   {
      const unsigned reduce_block_count = m_maxWorkgroup * Impl::HIPTraits::WorkgroupSize ;

      (void) scratch_flags( reduce_block_count * 2  * sizeof(size_type) );
      (void) scratch_space( reduce_block_count * 16 * sizeof(size_type) );
    }*/
    //----------------------------------

  }
  else {

    std::ostringstream msg ;
    msg << "Kokkos::Experimental::HIP::initialize(" << hip_device_id << ") FAILED" ;

    if ( ! ok_init ) {
      msg << " : Already initialized" ;
    }
    /*
    if ( ! ok_id ) {
      msg << " : Device identifier out of range "
          << "[0.." << (dev_info.m_hipDevCount-1) << "]" ;
    }
    else if ( ! ok_dev ) {
      msg << " : Device " ;
      msg << dev_info.m_hipProp[ hip_device_id ].major ;
      msg << "." ;
      msg << dev_info.m_hipProp[ hip_device_id ].minor ;
      msg << " Need at least a GPU" ;
      msg << std::endl;
    }
    */
    Kokkos::Impl::throw_runtime_exception( msg.str() );
  }


  // Init the array for used for arbitrarily sized atomics
  //Kokkos::Impl::init_lock_arrays_hip_space();

//  Kokkos::Impl::HIPLockArraysStruct locks;
//  locks.atomic = atomic_lock_array_hip_space_ptr(false);
//  locks.scratch = scratch_lock_array_hip_space_ptr(false);
//  locks.threadid = threadid_lock_array_hip_space_ptr(false);
//  hipMemcpyToSymbol( kokkos_impl_hip_lock_arrays , & locks , sizeof(HIPLockArraysStruct) );
}

//----------------------------------------------------------------------------

//typedef Kokkos::Experimental::HIP::size_type ScratchGrain[ Impl::HIPTraits::WorkgroupSize ] ;
//enum { sizeScratchGrain = sizeof(ScratchGrain) };

Kokkos::Experimental::HIP::size_type *
HIPInternal::scratch_flags( const Kokkos::Experimental::HIP::size_type size )
{
/*  if ( verify_is_initialized("scratch_flags") && m_scratchFlagsCount * sizeScratchGrain < size ) {


    m_scratchFlagsCount = ( size + sizeScratchGrain - 1 ) / sizeScratchGrain ;

    typedef Kokkos::Impl::SharedAllocationRecord< Kokkos::Experimental::HIPSpace , void > Record ;

    Record * const r = Record::allocate( Kokkos::Experimental::HIPSpace()
                                       , "InternalScratchFlags"
                                       , ( sizeScratchGrain  * m_scratchFlagsCount ) );

    Record::increment( r );

    m_scratchFlags = reinterpret_cast<size_type *>( r->data() );

    hipMemset( m_scratchFlags , 0 , m_scratchFlagsCount * sizeScratchGrain );
  }
*/
  return m_scratchFlags ;
}

Kokkos::Experimental::HIP::size_type *
HIPInternal::scratch_space( const Kokkos::Experimental::HIP::size_type size )
{
/*  if ( verify_is_initialized("scratch_space") && m_scratchSpaceCount * sizeScratchGrain < size ) {

    m_scratchSpaceCount = ( size + sizeScratchGrain - 1 ) / sizeScratchGrain ;

     typedef Kokkos::Impl::SharedAllocationRecord< Kokkos::Experimental::HIPSpace , void > Record ;

     static Record * const r = Record::allocate( Kokkos::Experimental::HIPSpace()
                                        , "InternalScratchSpace"
                                        , ( sizeScratchGrain  * m_scratchSpaceCount ) );

     Record::increment( r );

     m_scratchSpace = reinterpret_cast<size_type *>( r->data() );
  }
*/
  return m_scratchSpace ;
}

//----------------------------------------------------------------------------

void HIPInternal::finalize()
{
  HIP().fence();
  was_finalized = 1;
  if ( 0 != m_scratchSpace || 0 != m_scratchFlags ) {

//    atomic_lock_array_hip_space_ptr(false);
//    scratch_lock_array_hip_space_ptr(false);
//    threadid_lock_array_hip_space_ptr(false);

    typedef Kokkos::Impl::SharedAllocationRecord< Kokkos::Experimental::HIPSpace > RecordHIP ;
    typedef Kokkos::Impl::SharedAllocationRecord< Kokkos::Experimental::HIPHostPinnedSpace > RecordHost ;

    RecordHIP::decrement( RecordHIP::get_record( m_scratchFlags ) );
    RecordHIP::decrement( RecordHIP::get_record( m_scratchSpace ) );

    m_hipDev             = -1 ;
    m_multiProcCount      = 0 ;
    m_maxWorkgroup        = 0 ;
    m_maxSharedWords      = 0 ;
    m_scratchSpaceCount   = 0 ;
    m_scratchFlagsCount   = 0 ;
    m_scratchSpace        = 0 ;
    m_scratchFlags        = 0 ;
  }
}

//----------------------------------------------------------------------------

Kokkos::Experimental::HIP::size_type hip_internal_cu_count()
{ return HIPInternal::singleton().m_multiProcCount ; }

Kokkos::Experimental::HIP::size_type hip_internal_maximum_extent_size()
{ return HIPInternal::singleton().m_maxWorkgroup ; }

Kokkos::Experimental::HIP::size_type hip_internal_maximum_shared_words()
{ return HIPInternal::singleton().m_maxSharedWords ; }

Kokkos::Experimental::HIP::size_type * hip_internal_scratch_space( const Kokkos::Experimental::HIP::size_type size )
{ return HIPInternal::singleton().scratch_space( size ); }

Kokkos::Experimental::HIP::size_type * hip_internal_scratch_flags( const Kokkos::Experimental::HIP::size_type size )
{ return HIPInternal::singleton().scratch_flags( size ); }



} // namespace Impl
} // namespace Experimental
} // namespace Kokkos

//----------------------------------------------------------------------------

namespace Kokkos {
namespace Experimental {

//HIP::size_type HIP::detect_device_count()
//{ return Impl::HIPInternalDevices::singleton().m_hipDevCount ; }

int HIP::concurrency() {
#if defined(KOKKOS_ARCH_KAVERI) 
  return 8*64*40;  // 20480 kaveri
#else
  return 32*8*40;  // 81920 fiji and hawaii
#endif
}
int HIP::impl_is_initialized()
{ return Impl::HIPInternal::singleton().is_initialized(); }

void HIP::impl_initialize( const HIP::SelectDevice config )
{
  Impl::HIPInternal::singleton().initialize( config.hip_device_id );

  #if defined(KOKKOS_ENABLE_PROFILING)
    Kokkos::Profiling::initialize();
  #endif
}

#if 0
std::vector<unsigned>
HIP::detect_device_arch()
{
  const Impl::HIPInternalDevices & s = Impl::HIPInternalDevices::singleton();

  std::vector<unsigned> output( s.m_hipDevCount );

  for ( int i = 0 ; i < s.m_hipDevCount ; ++i ) {
    output[i] = s.m_hipProp[i].major * 100 + s.m_hipProp[i].minor ;
  }

  return output ;
}

HIP::size_type HIP::device_arch()
{
  return 1 ;
}
#endif

void HIP::impl_finalize()
{
  Impl::HIPInternal::singleton().finalize();

  #if defined(KOKKOS_ENABLE_PROFILING)
    Kokkos::Profiling::finalize();
  #endif
}

HIP::HIP()
  : m_space_instance(&Impl::HIPInternal::singleton())
{
  Impl::HIPInternal::singleton().verify_is_initialized( "HIP instance constructor" );
}

//HIP::HIP( const int instance_id )
//  : m_device( Impl::HIPInternal::singleton().m_hipDev )
//{}

void HIP::print_configuration( std::ostream & s , const bool )
{ Impl::HIPInternal::singleton().print_configuration( s ); }

bool HIP::sleep() { return false ; }

bool HIP::wake() { return true ; }

void HIP::fence()
{
  hipDeviceSynchronize();
}

int HIP::hip_device() const { return impl_internal_space_instance()->m_hipDev; }
const char* HIP::name() { return "HIP"; }

} // namespace Experimental
} // namespace Kokkos

#endif // KOKKOS_ENABLE_HIP
//----------------------------------------------------------------------------

