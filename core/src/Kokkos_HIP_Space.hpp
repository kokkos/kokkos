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

#ifndef KOKKOS_HIPSPACE_HPP
#define KOKKOS_HIPSPACE_HPP

#include <Kokkos_Core_fwd.hpp>

#if defined( KOKKOS_ENABLE_HIP )

#include <iosfwd>
#include <typeinfo>
#include <string>

#include <Kokkos_HostSpace.hpp>

#include <hip/hip_runtime_api.h>
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Experimental {
/** \brief  HIP on-device memory management */

class HIPSpace {
public:

  //! Tag this class as a kokkos memory space
  typedef HIPSpace             memory_space ;
  typedef Kokkos::Experimental::HIP          execution_space ;
  typedef Kokkos::Device<execution_space,memory_space> device_type;

  typedef unsigned int          size_type ;

  /*--------------------------------*/

  HIPSpace();
  HIPSpace( HIPSpace && rhs ) = default ;
  HIPSpace( const HIPSpace & rhs ) = default ;
  HIPSpace & operator = ( HIPSpace && rhs ) = default ;
  HIPSpace & operator = ( const HIPSpace & rhs ) = default ;
  ~HIPSpace() = default ;

  /**\brief  Allocate untracked memory in the rocm space */
  void * allocate( const size_t arg_alloc_size ) const ;

  /**\brief  Deallocate untracked memory in the rocm space */
  void deallocate( void * const arg_alloc_ptr
                 , const size_t arg_alloc_size ) const ;

  /**\brief Return Name of the MemorySpace */
  static constexpr const char* name() { return m_name; };

  /*--------------------------------*/
  /** \brief  Error reporting for HostSpace attempt to access HIPSpace */
  static void access_error();
  static void access_error( const void * const );

private:

  int  m_device ; ///< Which HIP device

  static constexpr const char* m_name = "HIP";
  friend class Kokkos::Impl::SharedAllocationRecord< Kokkos::Experimental::HIPSpace , void > ;
};

} // namespace Experimental

namespace Impl {


/// \brief Initialize lock array for arbitrary size atomics.
///
/// Arbitrary atomics are implemented using a hash table of locks
/// where the hash value is derived from the address of the
/// object for which an atomic operation is performed.
/// This function initializes the locks to zero (unset).
void init_lock_arrays_rocm_space();

/// \brief Retrieve the pointer to the lock array for arbitrary size atomics.
///
/// Arbitrary atomics are implemented using a hash table of locks
/// where the hash value is derived from the address of the
/// object for which an atomic operation is performed.
/// This function retrieves the lock array pointer.
/// If the array is not yet allocated it will do so.
int* atomic_lock_array_rocm_space_ptr(bool deallocate = false);

/// \brief Retrieve the pointer to the scratch array for team and thread private global memory.
///
/// Team and Thread private scratch allocations in
/// global memory are aquired via locks.
/// This function retrieves the lock array pointer.
/// If the array is not yet allocated it will do so.
int* scratch_lock_array_rocm_space_ptr(bool deallocate = false);

/// \brief Retrieve the pointer to the scratch array for unique identifiers.
///
/// Unique identifiers in the range 0-HIP::concurrency
/// are provided via locks.
/// This function retrieves the lock array pointer.
/// If the array is not yet allocated it will do so.
int* threadid_lock_array_rocm_space_ptr(bool deallocate = false);
}
} // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/


namespace Kokkos {
namespace Experimental {
/** \brief  Host memory that is accessible to HIP execution space
 *          through HIP's host-pinned memory allocation.
 */
class HIPHostPinnedSpace {
public:

  //! Tag this class as a kokkos memory space
  /** \brief  Memory is in HostSpace so use the HostSpace::execution_space */
  typedef HostSpace::execution_space  execution_space ;
  typedef HIPHostPinnedSpace         memory_space ;
  typedef Kokkos::Device<execution_space,memory_space> device_type;
  typedef unsigned int                size_type ;

  /*--------------------------------*/

  HIPHostPinnedSpace();
  HIPHostPinnedSpace( HIPHostPinnedSpace && rhs ) = default ;
  HIPHostPinnedSpace( const HIPHostPinnedSpace & rhs ) = default ;
  HIPHostPinnedSpace & operator = ( HIPHostPinnedSpace && rhs ) = default ;
  HIPHostPinnedSpace & operator = ( const HIPHostPinnedSpace & rhs ) = default ;
  ~HIPHostPinnedSpace() = default ;

  /**\brief  Allocate untracked memory in the space */
  void * allocate( const size_t arg_alloc_size ) const ;

  /**\brief  Deallocate untracked memory in the space */
  void deallocate( void * const arg_alloc_ptr
                 , const size_t arg_alloc_size ) const ;

  /**\brief Return Name of the MemorySpace */
  static constexpr const char* name() { return m_name; };

private:

  static constexpr const char* m_name = "HIPHostPinned";

  /*--------------------------------*/
};
} // namespace Experimental
} // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

static_assert( Kokkos::Impl::MemorySpaceAccess< Kokkos::Experimental::HIPSpace , Kokkos::Experimental::HIPSpace >::assignable , "" );

//----------------------------------------

template<>
struct MemorySpaceAccess< Kokkos::HostSpace , Kokkos::Experimental::HIPSpace > {
  enum { assignable = false };
  enum { accessible = false };
  enum { deepcopy   = true };
};

template<>
struct MemorySpaceAccess< Kokkos::HostSpace , Kokkos::Experimental::HIPHostPinnedSpace > {
  // HostSpace::execution_space == HIPHostPinnedSpace::execution_space
  enum { assignable = true };
  enum { accessible = true };
  enum { deepcopy   = true };
};

//----------------------------------------

template<>
struct MemorySpaceAccess< Kokkos::Experimental::HIPSpace , Kokkos::HostSpace > {
  enum { assignable = false };
  enum { accessible = false };
  enum { deepcopy   = true };
};

template<>
struct MemorySpaceAccess< Kokkos::Experimental::HIPSpace , Kokkos::Experimental::HIPHostPinnedSpace > {
  // HIPSpace::execution_space != HIPHostPinnedSpace::execution_space
  enum { assignable = false };
  enum { accessible = true }; // HIPSpace::execution_space
  enum { deepcopy   = true };
};


//----------------------------------------
// HIPHostPinnedSpace::execution_space == HostSpace::execution_space
// HIPHostPinnedSpace accessible to both HIP and Host

template<>
struct MemorySpaceAccess< Kokkos::Experimental::HIPHostPinnedSpace , Kokkos::HostSpace > {
  enum { assignable = false }; // Cannot access from HIP
  enum { accessible = true };  // HIPHostPinnedSpace::execution_space
  enum { deepcopy   = true };
};

template<>
struct MemorySpaceAccess< Kokkos::Experimental::HIPHostPinnedSpace , Kokkos::Experimental::HIPSpace > {
  enum { assignable = false }; // Cannot access from Host
  enum { accessible = false };
  enum { deepcopy   = true };
};

};
//----------------------------------------

} // namespace Kokkos::Impl

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

//hc::completion_future DeepCopyAsyncHIP( void * dst , const void * src , size_t n);

template<> struct DeepCopy< Kokkos::Experimental::HIPSpace , Kokkos::Experimental::HIPSpace , Kokkos::Experimental::HIP>
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const Kokkos::Experimental::HIP & , void * dst , const void * src , size_t );
};

template<> struct DeepCopy< Kokkos::Experimental::HIPSpace , HostSpace , Kokkos::Experimental::HIP >
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const Kokkos::Experimental::HIP & , void * dst , const void * src , size_t );
};

template<> struct DeepCopy< HostSpace , Kokkos::Experimental::HIPSpace , Kokkos::Experimental::HIP >
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const Kokkos::Experimental::HIP & , void * dst , const void * src , size_t );
};

template<class ExecutionSpace> struct DeepCopy< Kokkos::Experimental::HIPSpace , Kokkos::Experimental::HIPSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< Kokkos::Experimental::HIPSpace , Kokkos::Experimental::HIPSpace , Kokkos::Experimental::HIP >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
//    exec.fence();
//    hc::completion_future fut = DeepCopyAsyncHIP (dst,src,n);
//    fut.wait();
//    DeepCopy (dst,src,n);
  }
};

template<class ExecutionSpace> struct DeepCopy< Kokkos::Experimental::HIPSpace , HostSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< Kokkos::Experimental::HIPSpace , HostSpace , Kokkos::Experimental::HIP>( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopy (dst,src,n);
  }
};

template<class ExecutionSpace>
struct DeepCopy< HostSpace , Kokkos::Experimental::HIPSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< HostSpace , Kokkos::Experimental::HIPSpace , Kokkos::Experimental::HIP >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopy (dst,src,n);
  }
};

template<> struct DeepCopy< Kokkos::Experimental::HIPHostPinnedSpace , Kokkos::Experimental::HIPHostPinnedSpace , Kokkos::Experimental::HIP>
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const Kokkos::Experimental::HIP & , void * dst , const void * src , size_t );
};

template<> struct DeepCopy< Kokkos::Experimental::HIPHostPinnedSpace , HostSpace , Kokkos::Experimental::HIP >
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const Kokkos::Experimental::HIP & , void * dst , const void * src , size_t );
};

template<> struct DeepCopy< HostSpace , Kokkos::Experimental::HIPHostPinnedSpace , Kokkos::Experimental::HIP >
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const Kokkos::Experimental::HIP & , void * dst , const void * src , size_t );
};

template<class ExecutionSpace>
struct DeepCopy< Kokkos::Experimental::HIPSpace , Kokkos::Experimental::HIPHostPinnedSpace , ExecutionSpace>
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< Kokkos::Experimental::HIPSpace , HostSpace , Kokkos::Experimental::HIP >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
//    exec.fence();
//    hc::completion_future fut = DeepCopyAsyncHIP (dst,src,n);
//    fut.wait();
//    DeepCopyHIP (dst,src,n);
  }
};

template<class ExecutionSpace> struct DeepCopy< Kokkos::Experimental::HIPHostPinnedSpace , Kokkos::Experimental::HIPSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< HostSpace , Kokkos::Experimental::HIPSpace , Kokkos::Experimental::HIP >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
//    exec.fence();
//    hc::completion_future fut = DeepCopyAsyncHIP (dst,src,n);
//    fut.wait();
//    DeepCopyHIP (dst,src,n);
  }
};



template<class ExecutionSpace> struct DeepCopy< Kokkos::Experimental::HIPHostPinnedSpace , Kokkos::Experimental::HIPHostPinnedSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< Kokkos::Experimental::HIPHostPinnedSpace , Kokkos::Experimental::HIPHostPinnedSpace , Kokkos::Experimental::HIP >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
//    hc::completion_future fut = DeepCopyAsyncHIP (dst,src,n);
//    fut.wait();
//    DeepCopyAsyncHIP (dst,src,n);
    DeepCopy (dst,src,n);
  }
};

template<class ExecutionSpace> struct DeepCopy< Kokkos::Experimental::HIPHostPinnedSpace , HostSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< Kokkos::Experimental::HIPHostPinnedSpace , HostSpace , Kokkos::Experimental::HIP>( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopy (dst,src,n);
  }
};

template<class ExecutionSpace>
struct DeepCopy< HostSpace , Kokkos::Experimental::HIPHostPinnedSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< HostSpace , Kokkos::Experimental::HIPHostPinnedSpace , Kokkos::Experimental::HIP >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopy (dst,src,n);
  }
};
} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

/** Running in HIPSpace attempting to access HostSpace: error */
template<>
struct VerifyExecutionCanAccessMemorySpace< Kokkos::Experimental::HIPSpace , Kokkos::HostSpace >
{
  enum { value = false };
  KOKKOS_INLINE_FUNCTION static void verify( void )
    { Kokkos::abort("HIP code attempted to access HostSpace memory"); }

  KOKKOS_INLINE_FUNCTION static void verify( const void * )
    { Kokkos::abort("HIP code attempted to access HostSpace memory"); }
};

/** Running in HIPSpace accessing HIPHostPinnedSpace: ok */
template<>
struct VerifyExecutionCanAccessMemorySpace< Kokkos::Experimental::HIPSpace , Kokkos::Experimental::HIPHostPinnedSpace >
{
  enum { value = true };
  KOKKOS_INLINE_FUNCTION static void verify( void ) { }
  KOKKOS_INLINE_FUNCTION static void verify( const void * ) { }
};

/** Running in HIPSpace attempting to access an unknown space: error */
template< class OtherSpace >
struct VerifyExecutionCanAccessMemorySpace<
  typename enable_if< ! is_same<Kokkos::Experimental::HIPSpace,OtherSpace>::value , Kokkos::Experimental::HIPSpace >::type ,
  OtherSpace >
{
  enum { value = false };
  KOKKOS_INLINE_FUNCTION static void verify( void )
    { Kokkos::abort("HIP code attempted to access unknown Space memory"); }

  KOKKOS_INLINE_FUNCTION static void verify( const void * )
    { Kokkos::abort("HIP code attempted to access unknown Space memory"); }
};

//----------------------------------------------------------------------------
/** Running in HostSpace attempting to access HIPSpace */
template<>
struct VerifyExecutionCanAccessMemorySpace< Kokkos::HostSpace , Kokkos::Experimental::HIPSpace >
{
  enum { value = false };
  inline static void verify( void ) { Kokkos::Experimental::HIPSpace::access_error(); }
  inline static void verify( const void * p ) { Kokkos::Experimental::HIPSpace::access_error(p); }
};

/** Running in HostSpace accessing HIPHostPinnedSpace is OK */
template<>
struct VerifyExecutionCanAccessMemorySpace< Kokkos::HostSpace , Kokkos::Experimental::HIPHostPinnedSpace >
{
  enum { value = true };
  KOKKOS_INLINE_FUNCTION static void verify( void ) {}
  KOKKOS_INLINE_FUNCTION static void verify( const void * ) {}
};
} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template<>
class SharedAllocationRecord< Kokkos::Experimental::HIPSpace , void >
  : public SharedAllocationRecord< void , void >
{
private:


  typedef SharedAllocationRecord< void , void >  RecordBase ;

  SharedAllocationRecord( const SharedAllocationRecord & ) = delete ;
  SharedAllocationRecord & operator = ( const SharedAllocationRecord & ) = delete ;

  static void deallocate( RecordBase * );

#ifdef KOKKOS_DEBUG
  static RecordBase s_root_record ;
#endif

  const Kokkos::Experimental::HIPSpace m_space ;

protected:

  ~SharedAllocationRecord();

  SharedAllocationRecord( const Kokkos::Experimental::HIPSpace        & arg_space
                        , const std::string              & arg_label
                        , const size_t                     arg_alloc_size
                        , const RecordBase::function_type  arg_dealloc = & deallocate
                        );

public:

  std::string get_label() const ;

  static SharedAllocationRecord * allocate( const Kokkos::Experimental::HIPSpace &  arg_space
                                          , const std::string       &  arg_label
                                          , const size_t               arg_alloc_size );

  /**\brief  Allocate tracked memory in the space */
  static
  void * allocate_tracked( const Kokkos::Experimental::HIPSpace & arg_space
                         , const std::string & arg_label
                         , const size_t arg_alloc_size );

  /**\brief  Reallocate tracked memory in the space */
  static
  void * reallocate_tracked( void * const arg_alloc_ptr
                           , const size_t arg_alloc_size );

  /**\brief  Deallocate tracked memory in the space */
  static
  void deallocate_tracked( void * const arg_alloc_ptr );

  static SharedAllocationRecord * get_record( void * arg_alloc_ptr );

  static void print_records( std::ostream & , const Kokkos::Experimental::HIPSpace & , bool detail = false );
};

template<>
class SharedAllocationRecord< Kokkos::Experimental::HIPHostPinnedSpace , void >
  : public SharedAllocationRecord< void , void >
{
private:

  typedef SharedAllocationRecord< void , void >  RecordBase ;

  SharedAllocationRecord( const SharedAllocationRecord & ) = delete ;
  SharedAllocationRecord & operator = ( const SharedAllocationRecord & ) = delete ;

  static void deallocate( RecordBase * );

#ifdef KOKKOS_DEBUG
  static RecordBase s_root_record ;
#endif

  const Kokkos::Experimental::HIPHostPinnedSpace m_space ;

protected:

  ~SharedAllocationRecord();
  SharedAllocationRecord() : RecordBase(), m_space() {}

  SharedAllocationRecord( const Kokkos::Experimental::HIPHostPinnedSpace     & arg_space
                        , const std::string              & arg_label
                        , const size_t                     arg_alloc_size
                        , const RecordBase::function_type  arg_dealloc = & deallocate
                        );

public:

  std::string get_label() const ;

  static SharedAllocationRecord * allocate( const Kokkos::Experimental::HIPHostPinnedSpace &  arg_space
                                          , const std::string          &  arg_label
                                          , const size_t                  arg_alloc_size
                                          );
  /**\brief  Allocate tracked memory in the space */
  static
  void * allocate_tracked( const Kokkos::Experimental::HIPHostPinnedSpace & arg_space
                         , const std::string & arg_label
                         , const size_t arg_alloc_size );

  /**\brief  Reallocate tracked memory in the space */
  static
  void * reallocate_tracked( void * const arg_alloc_ptr
                           , const size_t arg_alloc_size );

  /**\brief  Deallocate tracked memory in the space */
  static
  void deallocate_tracked( void * const arg_alloc_ptr );


  static SharedAllocationRecord * get_record( void * arg_alloc_ptr );

  static void print_records( std::ostream & , const Kokkos::Experimental::HIPHostPinnedSpace & , bool detail = false );
};
} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #if defined( KOKKOS_ENABLE_HIP ) */
#endif /* #define KOKKOS_HIPSPACE_HPP */

