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
// Questions? Contact  H. Carter Edwards (hcedwar@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

/*--------------------------------------------------------------------------*/
/* Kokkos interfaces */

#include <Kokkos_Core.hpp>

/* only compile this file if CUDA is enabled for Kokkos */
#ifdef KOKKOS_ENABLE_CUDA

#include <Cuda/Kokkos_Cuda_Error.hpp>
#include <Cuda/Kokkos_Cuda_Internal.hpp>
#include <impl/Kokkos_Error.hpp>
#include <impl/Kokkos_Profiling_Interface.hpp>

/*--------------------------------------------------------------------------*/
/* Standard 'C' libraries */
#include <stdlib.h>

/* Standard 'C++' libraries */
#include <vector>
#include <iostream>
#include <sstream>
#include <string>

#ifdef KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE

__device__ __constant__
unsigned long kokkos_impl_cuda_constant_memory_buffer[ Kokkos::Impl::CudaTraits::ConstantMemoryUsage / sizeof(unsigned long) ] ;

__device__ __constant__
Kokkos::Impl::CudaLockArraysStruct kokkos_impl_cuda_lock_arrays ;

#endif

/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

namespace {

__global__
void query_cuda_kernel_arch( int * d_arch )
{
#if defined( __CUDA_ARCH__ )
  *d_arch = __CUDA_ARCH__ ;
#else
  *d_arch = 0 ;
#endif
}

/** Query what compute capability is actually launched to the device: */
int cuda_kernel_arch()
{
  int * d_arch = 0 ;
  cudaMalloc( (void **) & d_arch , sizeof(int) );
  query_cuda_kernel_arch<<<1,1>>>( d_arch );
  int arch = 0 ;
  cudaMemcpy( & arch , d_arch , sizeof(int) , cudaMemcpyDefault );
  cudaFree( d_arch );
  return arch ;
}

bool cuda_launch_blocking()
{
  const char * env = getenv("CUDA_LAUNCH_BLOCKING");

  if (env == 0) return false;

  return atoi(env);
}

}

void cuda_device_synchronize()
{
//  static const bool launch_blocking = cuda_launch_blocking();

//  if (!launch_blocking) {
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );
//  }
}

void cuda_internal_error_throw( cudaError e , const char * name, const char * file, const int line )
{
  std::ostringstream out ;
  out << name << " error( " << cudaGetErrorName(e) << "): " << cudaGetErrorString(e);
  if (file) {
    out << " " << file << ":" << line;
  }
  throw_runtime_exception( out.str() );
}

//----------------------------------------------------------------------------
// Some significant cuda device properties:
//
// cudaDeviceProp::name                : Text label for device
// cudaDeviceProp::major               : Device major number
// cudaDeviceProp::minor               : Device minor number
// cudaDeviceProp::warpSize            : number of threads per warp
// cudaDeviceProp::multiProcessorCount : number of multiprocessors
// cudaDeviceProp::sharedMemPerBlock   : capacity of shared memory per block
// cudaDeviceProp::totalConstMem       : capacity of constant memory
// cudaDeviceProp::totalGlobalMem      : capacity of global memory
// cudaDeviceProp::maxGridSize[3]      : maximum grid size

//
//  Section 4.4.2.4 of the CUDA Toolkit Reference Manual
//
// struct cudaDeviceProp {
//   char name[256];
//   size_t totalGlobalMem;
//   size_t sharedMemPerBlock;
//   int regsPerBlock;
//   int warpSize;
//   size_t memPitch;
//   int maxThreadsPerBlock;
//   int maxThreadsDim[3];
//   int maxGridSize[3];
//   size_t totalConstMem;
//   int major;
//   int minor;
//   int clockRate;
//   size_t textureAlignment;
//   int deviceOverlap;
//   int multiProcessorCount;
//   int kernelExecTimeoutEnabled;
//   int integrated;
//   int canMapHostMemory;
//   int computeMode;
//   int concurrentKernels;
//   int ECCEnabled;
//   int pciBusID;
//   int pciDeviceID;
//   int tccDriver;
//   int asyncEngineCount;
//   int unifiedAddressing;
//   int memoryClockRate;
//   int memoryBusWidth;
//   int l2CacheSize;
//   int maxThreadsPerMultiProcessor;
// };


namespace {



class CudaInternalDevices {
public:
  enum { MAXIMUM_DEVICE_COUNT = 64 };
  struct cudaDeviceProp  m_cudaProp[ MAXIMUM_DEVICE_COUNT ] ;
  int                    m_cudaDevCount ;

  CudaInternalDevices();

  static const CudaInternalDevices & singleton();
};

CudaInternalDevices::CudaInternalDevices()
{
  // See 'cudaSetDeviceFlags' for host-device thread interaction
  // Section 4.4.2.6 of the CUDA Toolkit Reference Manual

  CUDA_SAFE_CALL (cudaGetDeviceCount( & m_cudaDevCount ) );

  if(m_cudaDevCount > MAXIMUM_DEVICE_COUNT) {
    Kokkos::abort("Sorry, you have more GPUs per node than we thought anybody would ever have. Please report this to github.com/kokkos/kokkos.");
  }
  for ( int i = 0 ; i < m_cudaDevCount ; ++i ) {
    CUDA_SAFE_CALL( cudaGetDeviceProperties( m_cudaProp + i , i ) );
  }
}

const CudaInternalDevices & CudaInternalDevices::singleton()
{
  static CudaInternalDevices self ; return self ;
}

}

//----------------------------------------------------------------------------

class CudaInternal {
private:

  CudaInternal( const CudaInternal & );
  CudaInternal & operator = ( const CudaInternal & );


public:

  typedef Cuda::size_type size_type ;

  int         m_cudaDev ;
  int         m_cudaArch ;
  unsigned    m_multiProcCount ;
  unsigned    m_maxWarpCount ;
  unsigned    m_maxBlock ;
  unsigned    m_maxSharedWords ;
  size_type   m_scratchSpaceCount ;
  size_type   m_scratchFlagsCount ;
  size_type   m_scratchUnifiedCount ;
  size_type   m_scratchUnifiedSupported ;
  size_type   m_streamCount ;
  size_type * m_scratchSpace ;
  size_type * m_scratchFlags ;
  size_type * m_scratchUnified ;
  cudaStream_t * m_stream ;

  static int was_initialized;
  static int was_finalized;

  static CudaInternal & singleton();

  int verify_is_initialized( const char * const label ) const ;

  int is_initialized() const
    { return 0 != m_scratchSpace && 0 != m_scratchFlags ; }

  void initialize( int cuda_device_id , int stream_count );
  void finalize();

  void print_configuration( std::ostream & ) const ;

  ~CudaInternal();

  CudaInternal()
    : m_cudaDev( -1 )
    , m_cudaArch( -1 )
    , m_multiProcCount( 0 )
    , m_maxWarpCount( 0 )
    , m_maxBlock( 0 )
    , m_maxSharedWords( 0 )
    , m_scratchSpaceCount( 0 )
    , m_scratchFlagsCount( 0 )
    , m_scratchUnifiedCount( 0 )
    , m_scratchUnifiedSupported( 0 )
    , m_streamCount( 0 )
    , m_scratchSpace( 0 )
    , m_scratchFlags( 0 )
    , m_scratchUnified( 0 )
    , m_stream( 0 )
    {}

  size_type * scratch_space( const size_type size );
  size_type * scratch_flags( const size_type size );
  size_type * scratch_unified( const size_type size );
};

int CudaInternal::was_initialized = 0;
int CudaInternal::was_finalized = 0;
//----------------------------------------------------------------------------


void CudaInternal::print_configuration( std::ostream & s ) const
{
  const CudaInternalDevices & dev_info = CudaInternalDevices::singleton();

#if defined( KOKKOS_ENABLE_CUDA )
    s << "macro  KOKKOS_ENABLE_CUDA      : defined" << std::endl ;
#endif
#if defined( CUDA_VERSION )
    s << "macro  CUDA_VERSION          = " << CUDA_VERSION
      << " = version " << CUDA_VERSION / 1000
      << "." << ( CUDA_VERSION % 1000 ) / 10
      << std::endl ;
#endif

  for ( int i = 0 ; i < dev_info.m_cudaDevCount ; ++i ) {
    s << "Kokkos::Cuda[ " << i << " ] "
      << dev_info.m_cudaProp[i].name
      << " capability " << dev_info.m_cudaProp[i].major << "." << dev_info.m_cudaProp[i].minor
      << ", Total Global Memory: " << human_memory_size(dev_info.m_cudaProp[i].totalGlobalMem)
      << ", Shared Memory per Block: " << human_memory_size(dev_info.m_cudaProp[i].sharedMemPerBlock);
    if ( m_cudaDev == i ) s << " : Selected" ;
    s << std::endl ;
  }
}

//----------------------------------------------------------------------------

CudaInternal::~CudaInternal()
{
  if ( m_stream ||
       m_scratchSpace ||
       m_scratchFlags ||
       m_scratchUnified ) {
    std::cerr << "Kokkos::Cuda ERROR: Failed to call Kokkos::Cuda::finalize()"
              << std::endl ;
    std::cerr.flush();
  }

  m_cudaDev                 = -1 ;
  m_cudaArch                = -1 ;
  m_multiProcCount          = 0 ;
  m_maxWarpCount            = 0 ;
  m_maxBlock                = 0 ;
  m_maxSharedWords          = 0 ;
  m_scratchSpaceCount       = 0 ;
  m_scratchFlagsCount       = 0 ;
  m_scratchUnifiedCount     = 0 ;
  m_scratchUnifiedSupported = 0 ;
  m_streamCount             = 0 ;
  m_scratchSpace            = 0 ;
  m_scratchFlags            = 0 ;
  m_scratchUnified          = 0 ;
  m_stream                  = 0 ;
}

int CudaInternal::verify_is_initialized( const char * const label ) const
{
  if ( m_cudaDev < 0 ) {
    std::cerr << "Kokkos::Cuda::" << label << " : ERROR device not initialized" << std::endl ;
  }
  return 0 <= m_cudaDev ;
}

CudaInternal & CudaInternal::singleton()
{
  static CudaInternal self ;
  return self ;
}

void CudaInternal::initialize( int cuda_device_id , int stream_count )
{
  if ( was_finalized ) Kokkos::abort("Calling Cuda::initialize after Cuda::finalize is illegal\n");
  was_initialized = 1;
  if ( is_initialized() ) return;

  enum { WordSize = sizeof(size_type) };

  if ( ! HostSpace::execution_space::is_initialized() ) {
    const std::string msg("Cuda::initialize ERROR : HostSpace::execution_space is not initialized");
    throw_runtime_exception( msg );
  }

  const CudaInternalDevices & dev_info = CudaInternalDevices::singleton();

  const bool ok_init = 0 == m_scratchSpace || 0 == m_scratchFlags ;

  const bool ok_id   = 0 <= cuda_device_id &&
                            cuda_device_id < dev_info.m_cudaDevCount ;

  // Need device capability 3.0 or better

  const bool ok_dev = ok_id &&
    ( 3 <= dev_info.m_cudaProp[ cuda_device_id ].major &&
      0 <= dev_info.m_cudaProp[ cuda_device_id ].minor );

  if ( ok_init && ok_dev ) {

    const struct cudaDeviceProp & cudaProp =
      dev_info.m_cudaProp[ cuda_device_id ];

    m_cudaDev = cuda_device_id ;

    CUDA_SAFE_CALL( cudaSetDevice( m_cudaDev ) );
    CUDA_SAFE_CALL( cudaDeviceReset() );
    Kokkos::Impl::cuda_device_synchronize();

    // Query what compute capability architecture a kernel executes:
    m_cudaArch = cuda_kernel_arch();

    if ( m_cudaArch != cudaProp.major * 100 + cudaProp.minor * 10 ) {
      std::cerr << "Kokkos::Cuda::initialize WARNING: running kernels compiled for compute capability "
                << ( m_cudaArch / 100 ) << "." << ( ( m_cudaArch % 100 ) / 10 )
                << " on device with compute capability "
                << cudaProp.major << "." << cudaProp.minor
                << " , this will likely reduce potential performance."
                << std::endl ;
    }

    // number of multiprocessors

    m_multiProcCount = cudaProp.multiProcessorCount ;

    //----------------------------------
    // Maximum number of warps,
    // at most one warp per thread in a warp for reduction.

    // HCE 2012-February :
    // Found bug in CUDA 4.1 that sometimes a kernel launch would fail
    // if the thread count == 1024 and a functor is passed to the kernel.
    // Copying the kernel to constant memory and then launching with
    // thread count == 1024 would work fine.
    //
    // HCE 2012-October :
    // All compute capabilities support at least 16 warps (512 threads).
    // However, we have found that 8 warps typically gives better performance.

    m_maxWarpCount = 8 ;

    // m_maxWarpCount = cudaProp.maxThreadsPerBlock / Impl::CudaTraits::WarpSize ;

    if ( Impl::CudaTraits::WarpSize < m_maxWarpCount ) {
      m_maxWarpCount = Impl::CudaTraits::WarpSize ;
    }

    m_maxSharedWords = cudaProp.sharedMemPerBlock / WordSize ;

    //----------------------------------
    // Maximum number of blocks:

    m_maxBlock = cudaProp.maxGridSize[0] ;

    //----------------------------------

    m_scratchUnifiedSupported = cudaProp.unifiedAddressing ;

    if ( ! m_scratchUnifiedSupported ) {
      std::cout << "Kokkos::Cuda device "
                << cudaProp.name << " capability "
                << cudaProp.major << "." << cudaProp.minor
                << " does not support unified virtual address space"
                << std::endl ;
    }

    //----------------------------------
    // Multiblock reduction uses scratch flags for counters
    // and scratch space for partial reduction values.
    // Allocate some initial space.  This will grow as needed.

    {
      const unsigned reduce_block_count = m_maxWarpCount * Impl::CudaTraits::WarpSize ;

      (void) scratch_unified( 16 * sizeof(size_type) );
      (void) scratch_flags( reduce_block_count * 2  * sizeof(size_type) );
      (void) scratch_space( reduce_block_count * 16 * sizeof(size_type) );
    }
    //----------------------------------

    if ( stream_count ) {
      m_stream = (cudaStream_t*) ::malloc( stream_count * sizeof(cudaStream_t) );
      m_streamCount = stream_count ;
      for ( size_type i = 0 ; i < m_streamCount ; ++i ) m_stream[i] = 0 ;
    }
  }
  else {

    std::ostringstream msg ;
    msg << "Kokkos::Cuda::initialize(" << cuda_device_id << ") FAILED" ;

    if ( ! ok_init ) {
      msg << " : Already initialized" ;
    }
    if ( ! ok_id ) {
      msg << " : Device identifier out of range "
          << "[0.." << dev_info.m_cudaDevCount << "]" ;
    }
    else if ( ! ok_dev ) {
      msg << " : Device " ;
      msg << dev_info.m_cudaProp[ cuda_device_id ].major ;
      msg << "." ;
      msg << dev_info.m_cudaProp[ cuda_device_id ].minor ;
      msg << " has insufficient capability, required 3.0 or better" ;
    }
    Kokkos::Impl::throw_runtime_exception( msg.str() );
  }

  #ifdef KOKKOS_ENABLE_CUDA_UVM
    if(!cuda_launch_blocking()) {
      std::cout << "Kokkos::Cuda::initialize WARNING: Cuda is allocating into UVMSpace by default" << std::endl;
      std::cout << "                                  without setting CUDA_LAUNCH_BLOCKING=1." << std::endl;
      std::cout << "                                  The code must call Cuda::fence() after each kernel" << std::endl;
      std::cout << "                                  or will likely crash when accessing data on the host." << std::endl;
    }

    const char * env_force_device_alloc = getenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC");
    bool force_device_alloc;
    if (env_force_device_alloc == 0) force_device_alloc=false;
    else force_device_alloc=atoi(env_force_device_alloc)!=0;

    const char * env_visible_devices = getenv("CUDA_VISIBLE_DEVICES");
    bool visible_devices_one=true;
    if (env_visible_devices == 0) visible_devices_one=false;

    if(!visible_devices_one && !force_device_alloc) {
      std::cout << "Kokkos::Cuda::initialize WARNING: Cuda is allocating into UVMSpace by default" << std::endl;
      std::cout << "                                  without setting CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 or " << std::endl;
      std::cout << "                                  setting CUDA_VISIBLE_DEVICES." << std::endl;
      std::cout << "                                  This could on multi GPU systems lead to severe performance" << std::endl;
      std::cout << "                                  penalties." << std::endl;
    }
  #endif

  cudaThreadSetCacheConfig(cudaFuncCachePreferShared);

  // Init the array for used for arbitrarily sized atomics
  Impl::init_lock_arrays_cuda_space();

  #ifdef KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE
  Kokkos::Impl::CudaLockArraysStruct locks;
  locks.atomic = atomic_lock_array_cuda_space_ptr(false);
  locks.scratch = scratch_lock_array_cuda_space_ptr(false);
  locks.threadid = threadid_lock_array_cuda_space_ptr(false);
  locks.n = Kokkos::Cuda::concurrency();
  cudaMemcpyToSymbol( kokkos_impl_cuda_lock_arrays , & locks , sizeof(CudaLockArraysStruct) );
  #endif
}

//----------------------------------------------------------------------------

typedef Cuda::size_type ScratchGrain[ Impl::CudaTraits::WarpSize ] ;
enum { sizeScratchGrain = sizeof(ScratchGrain) };


Cuda::size_type *
CudaInternal::scratch_flags( const Cuda::size_type size )
{
  if ( verify_is_initialized("scratch_flags") && m_scratchFlagsCount * sizeScratchGrain < size ) {


    m_scratchFlagsCount = ( size + sizeScratchGrain - 1 ) / sizeScratchGrain ;

    typedef Kokkos::Experimental::Impl::SharedAllocationRecord< Kokkos::CudaSpace , void > Record ;

    Record * const r = Record::allocate( Kokkos::CudaSpace()
                                       , "InternalScratchFlags"
                                       , ( sizeof( ScratchGrain ) * m_scratchFlagsCount ) );

    Record::increment( r );

    m_scratchFlags = reinterpret_cast<size_type *>( r->data() );

    CUDA_SAFE_CALL( cudaMemset( m_scratchFlags , 0 , m_scratchFlagsCount * sizeScratchGrain ) );
  }

  return m_scratchFlags ;
}

Cuda::size_type *
CudaInternal::scratch_space( const Cuda::size_type size )
{
  if ( verify_is_initialized("scratch_space") && m_scratchSpaceCount * sizeScratchGrain < size ) {

    m_scratchSpaceCount = ( size + sizeScratchGrain - 1 ) / sizeScratchGrain ;

     typedef Kokkos::Experimental::Impl::SharedAllocationRecord< Kokkos::CudaSpace , void > Record ;

     Record * const r = Record::allocate( Kokkos::CudaSpace()
                                        , "InternalScratchSpace"
                                        , ( sizeof( ScratchGrain ) * m_scratchSpaceCount ) );

     Record::increment( r );

     m_scratchSpace = reinterpret_cast<size_type *>( r->data() );
  }

  return m_scratchSpace ;
}

Cuda::size_type *
CudaInternal::scratch_unified( const Cuda::size_type size )
{
  if ( verify_is_initialized("scratch_unified") &&
       m_scratchUnifiedSupported && m_scratchUnifiedCount * sizeScratchGrain < size ) {

    m_scratchUnifiedCount = ( size + sizeScratchGrain - 1 ) / sizeScratchGrain ;

    typedef Kokkos::Experimental::Impl::SharedAllocationRecord< Kokkos::CudaHostPinnedSpace , void > Record ;

    Record * const r = Record::allocate( Kokkos::CudaHostPinnedSpace()
                                       , "InternalScratchUnified"
                                       , ( sizeof( ScratchGrain ) * m_scratchUnifiedCount ) );

    Record::increment( r );

    m_scratchUnified = reinterpret_cast<size_type *>( r->data() );
  }

  return m_scratchUnified ;
}

//----------------------------------------------------------------------------

void CudaInternal::finalize()
{
  was_finalized = 1;
  if ( 0 != m_scratchSpace || 0 != m_scratchFlags ) {

    atomic_lock_array_cuda_space_ptr(true);
    scratch_lock_array_cuda_space_ptr(true);
    threadid_lock_array_cuda_space_ptr(true);

    if ( m_stream ) {
      for ( size_type i = 1 ; i < m_streamCount ; ++i ) {
        cudaStreamDestroy( m_stream[i] );
        m_stream[i] = 0 ;
      }
      ::free( m_stream );
    }

    typedef Kokkos::Experimental::Impl::SharedAllocationRecord< CudaSpace > RecordCuda ;
    typedef Kokkos::Experimental::Impl::SharedAllocationRecord< CudaHostPinnedSpace > RecordHost ;

    RecordCuda::decrement( RecordCuda::get_record( m_scratchFlags ) );
    RecordCuda::decrement( RecordCuda::get_record( m_scratchSpace ) );
    RecordHost::decrement( RecordHost::get_record( m_scratchUnified ) );

    m_cudaDev             = -1 ;
    m_multiProcCount      = 0 ;
    m_maxWarpCount        = 0 ;
    m_maxBlock            = 0 ;
    m_maxSharedWords      = 0 ;
    m_scratchSpaceCount   = 0 ;
    m_scratchFlagsCount   = 0 ;
    m_scratchUnifiedCount = 0 ;
    m_streamCount         = 0 ;
    m_scratchSpace        = 0 ;
    m_scratchFlags        = 0 ;
    m_scratchUnified      = 0 ;
    m_stream              = 0 ;
  }
}

//----------------------------------------------------------------------------

Cuda::size_type cuda_internal_multiprocessor_count()
{ return CudaInternal::singleton().m_multiProcCount ; }

Cuda::size_type cuda_internal_maximum_warp_count()
{ return CudaInternal::singleton().m_maxWarpCount ; }

Cuda::size_type cuda_internal_maximum_grid_count()
{ return CudaInternal::singleton().m_maxBlock ; }

Cuda::size_type cuda_internal_maximum_shared_words()
{ return CudaInternal::singleton().m_maxSharedWords ; }

Cuda::size_type * cuda_internal_scratch_space( const Cuda::size_type size )
{ return CudaInternal::singleton().scratch_space( size ); }

Cuda::size_type * cuda_internal_scratch_flags( const Cuda::size_type size )
{ return CudaInternal::singleton().scratch_flags( size ); }

Cuda::size_type * cuda_internal_scratch_unified( const Cuda::size_type size )
{ return CudaInternal::singleton().scratch_unified( size ); }


} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------

namespace Kokkos {

Cuda::size_type Cuda::detect_device_count()
{ return Impl::CudaInternalDevices::singleton().m_cudaDevCount ; }

int Cuda::concurrency() {
  return 131072;
}

int Cuda::is_initialized()
{ return Impl::CudaInternal::singleton().is_initialized(); }

void Cuda::initialize( const Cuda::SelectDevice config , size_t num_instances )
{
  Impl::CudaInternal::singleton().initialize( config.cuda_device_id , num_instances );

  #if defined(KOKKOS_ENABLE_PROFILING)
    Kokkos::Profiling::initialize();
  #endif
}

std::vector<unsigned>
Cuda::detect_device_arch()
{
  const Impl::CudaInternalDevices & s = Impl::CudaInternalDevices::singleton();

  std::vector<unsigned> output( s.m_cudaDevCount );

  for ( int i = 0 ; i < s.m_cudaDevCount ; ++i ) {
    output[i] = s.m_cudaProp[i].major * 100 + s.m_cudaProp[i].minor ;
  }

  return output ;
}

Cuda::size_type Cuda::device_arch()
{
  const int dev_id = Impl::CudaInternal::singleton().m_cudaDev ;

  int dev_arch = 0 ;

  if ( 0 <= dev_id ) {
    const struct cudaDeviceProp & cudaProp =
      Impl::CudaInternalDevices::singleton().m_cudaProp[ dev_id ] ;

    dev_arch = cudaProp.major * 100 + cudaProp.minor ;
  }

  return dev_arch ;
}

void Cuda::finalize()
{
  Impl::CudaInternal::singleton().finalize();

  #if defined(KOKKOS_ENABLE_PROFILING)
    Kokkos::Profiling::finalize();
  #endif
}

Cuda::Cuda()
  : m_device( Impl::CudaInternal::singleton().m_cudaDev )
  , m_stream( 0 )
{
  Impl::CudaInternal::singleton().verify_is_initialized( "Cuda instance constructor" );
}

Cuda::Cuda( const int instance_id )
  : m_device( Impl::CudaInternal::singleton().m_cudaDev )
  , m_stream(
      Impl::CudaInternal::singleton().verify_is_initialized( "Cuda instance constructor" )
        ? Impl::CudaInternal::singleton().m_stream[ instance_id % Impl::CudaInternal::singleton().m_streamCount ]
        : 0 )
{}

void Cuda::print_configuration( std::ostream & s , const bool )
{ Impl::CudaInternal::singleton().print_configuration( s ); }

bool Cuda::sleep() { return false ; }

bool Cuda::wake() { return true ; }

void Cuda::fence()
{
  Kokkos::Impl::cuda_device_synchronize();
}

} // namespace Kokkos

#endif // KOKKOS_ENABLE_CUDA
//----------------------------------------------------------------------------

