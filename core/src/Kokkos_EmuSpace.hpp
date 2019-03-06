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

#ifndef KOKKOS_EmuLocalSpace_HPP
#define KOKKOS_EmuLocalSpace_HPP

#include <Kokkos_Macros.hpp>
#if defined( KOKKOS_ENABLE_EMU )

#include <Kokkos_Core_fwd.hpp>
#include <memoryweb/memory.h>
#include <memoryweb/intrinsics.h>
#include <memoryweb/repl.h>
#include <Kokkos_HostSpace.hpp>

/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Experimental {

/** \brief  Emu on-device memory management */

class EmuLocalSpace {
public:

  //! Tag this class as a kokkos memory space
  typedef EmuLocalSpace                           memory_space ;
  typedef Kokkos::Experimental::CilkPlus          execution_space ;
  typedef Kokkos::Device<execution_space,memory_space> device_type;

  typedef unsigned int          size_type ;

  /*--------------------------------*/

  EmuLocalSpace();
  EmuLocalSpace( EmuLocalSpace && rhs ) = default ;
  EmuLocalSpace( const EmuLocalSpace & rhs ) = default ;
  EmuLocalSpace & operator = ( EmuLocalSpace && rhs ) = default ;
  EmuLocalSpace & operator = ( const EmuLocalSpace & rhs ) = default ;
  ~EmuLocalSpace() = default ;

  /**\brief  Allocate untracked memory in the emu local space */
  void * allocate( const size_t arg_alloc_size ) const ;

  /**\brief  Deallocate untracked memory in the emu local space */
  void deallocate( void * const arg_alloc_ptr
                 , const size_t arg_alloc_size ) const ;

  /**\brief Return Name of the MemorySpace */
  static constexpr const char* name() { return m_name; }

  /*--------------------------------*/
  /** \brief  Error reporting for attempt to access data not on current NODE */
  static void access_error();
  static void access_error( const void * const );
  static void * local_root_record;

private:
  static constexpr const char* m_name = "EmuLocalSpace";
  friend class Kokkos::Impl::SharedAllocationRecord< Kokkos::Experimental::EmuLocalSpace , void > ;
};

} // Experimental
} // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Experimental {

/** \brief  Replicated Emu memory that is accessible to Host execution space
              -- Generally this will be used with const views -- so that the 
                 data is consistent on each node.
 */
class EmuReplicatedSpace {
public:

  //! Tag this class as a kokkos memory space
  typedef EmuReplicatedSpace    memory_space ;
  typedef CilkPlus              execution_space ;
  typedef Kokkos::Device<execution_space,memory_space> device_type;
  typedef unsigned int          size_type ;

  /*--------------------------------*/


  /*--------------------------------*/

  EmuReplicatedSpace();
  EmuReplicatedSpace( EmuReplicatedSpace && rhs ) = default ;
  EmuReplicatedSpace( const EmuReplicatedSpace & rhs ) = default ;
  EmuReplicatedSpace & operator = ( EmuReplicatedSpace && rhs ) = default ;
  EmuReplicatedSpace & operator = ( const EmuReplicatedSpace & rhs ) = default ;
  ~EmuReplicatedSpace() = default ;

  /**\brief  Allocate untracked memory in the emu replicated space */
  void * allocate( const size_t arg_alloc_size ) const ;

  /**\brief  Deallocate untracked memory in the emu replicated space */
  void deallocate( void * const arg_alloc_ptr
                 , const size_t arg_alloc_size ) const ;

  /**\brief Return Name of the MemorySpace */
  static constexpr const char* name() { return m_name; }

  static void custom_increment( void * pRec );

  static void * custom_decrement( void * pRec );

  static long * getRefAddr();
  static int memory_zones();

  static void * ers;
  static void * repl_root_record;

  /*--------------------------------*/

private:
  static constexpr const char* m_name = "EmuReplicated";

};

} // Experimental
} // namespace Kokkos

  
/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Experimental {

/** \brief  Host memory that is accessible to emu/cilk execution space
 *          through emu stride allocation .   (1d, 2d, strided)
 */
class EmuStridedSpace {
public:

  //! Tag this class as a kokkos memory space
  /** \brief  Memory is in HostSpace so use the HostSpace::execution_space */
  typedef HostSpace::execution_space  execution_space ;
  typedef EmuStridedSpace         memory_space ;
  typedef Kokkos::Device<execution_space,memory_space> device_type;
  typedef unsigned int                size_type ;

  /*--------------------------------*/

  EmuStridedSpace();
  EmuStridedSpace( EmuStridedSpace && rhs ) = default ;
  EmuStridedSpace( const EmuStridedSpace & rhs ) = default ;
  EmuStridedSpace & operator = ( EmuStridedSpace && rhs ) = default ;
  EmuStridedSpace & operator = ( const EmuStridedSpace & rhs ) = default ;
  ~EmuStridedSpace() = default ;

  /**\brief  Allocate untracked memory in the space */
  void * allocate( const size_t arg_alloc_size ) const ;

  /**\brief  Deallocate untracked memory in the space */
  void deallocate( void * const arg_alloc_ptr
                 , const size_t arg_alloc_size ) const ;

  /**\brief Return Name of the MemorySpace */
  static constexpr const char* name() { return m_name; }

private:

  static constexpr const char* m_name = "EmuStridedSpace";

  /*--------------------------------*/
};

} // Experimental
} // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

static_assert( MemorySpaceAccess< Kokkos::Experimental::EmuLocalSpace , Kokkos::Experimental::EmuLocalSpace >::assignable , "" );
static_assert( MemorySpaceAccess< Kokkos::Experimental::EmuReplicatedSpace , Kokkos::Experimental::EmuReplicatedSpace >::assignable , "" );
static_assert( MemorySpaceAccess< Kokkos::Experimental::EmuStridedSpace , Kokkos::Experimental::EmuStridedSpace >::assignable , "" );

//----------------------------------------

template<>
struct MemorySpaceAccess< Kokkos::HostSpace , Kokkos::Experimental::EmuLocalSpace > {
  enum { assignable = false };
  enum { accessible = true };
  enum { deepcopy   = true };
};

template<>
struct MemorySpaceAccess< Kokkos::HostSpace , Kokkos::Experimental::EmuReplicatedSpace > {
  // HostSpace::execution_space != EmuReplicatedSpace::execution_space
  enum { assignable = false };
  enum { accessible = true };
  enum { deepcopy   = true };
};

template<>
struct MemorySpaceAccess< Kokkos::HostSpace , Kokkos::Experimental::EmuStridedSpace > {
  // HostSpace::execution_space == EmuStridedSpace::execution_space
  enum { assignable = true };
  enum { accessible = true };
  enum { deepcopy   = true };
};

//----------------------------------------

template<>
struct MemorySpaceAccess< Kokkos::Experimental::EmuLocalSpace , Kokkos::HostSpace > {
  enum { assignable = false };
  enum { accessible = false };
  enum { deepcopy   = true };
};

template<>
struct MemorySpaceAccess< Kokkos::Experimental::EmuLocalSpace , Kokkos::Experimental::EmuReplicatedSpace > {
  // EmuLocalSpace::execution_space == EmuReplicatedSpace::execution_space
  enum { assignable = true };
  enum { accessible = true };
  enum { deepcopy   = true };
};

template<>
struct MemorySpaceAccess< Kokkos::Experimental::EmuLocalSpace , Kokkos::Experimental::EmuStridedSpace > {
  // EmuLocalSpace::execution_space != EmuStridedSpace::execution_space
  enum { assignable = false };
  enum { accessible = true }; // EmuLocalSpace::execution_space
  enum { deepcopy   = true };
};

//----------------------------------------
// EmuReplicatedSpace::execution_space == CilkPlus
// EmuReplicatedSpace accessible to both Emu and Host

template<>
struct MemorySpaceAccess< Kokkos::Experimental::EmuReplicatedSpace , Kokkos::HostSpace > {
  enum { assignable = false };
  enum { accessible = true };
  enum { deepcopy   = true };
};

template<>
struct MemorySpaceAccess< Kokkos::Experimental::EmuReplicatedSpace , Kokkos::Experimental::EmuLocalSpace > {
  // EmuReplicatedSpace::execution_space == EmuLocalSpace::execution_space
  // Can access EmuReplicatedSpace from Host but cannot access EmuLocalSpace from Host
  enum { assignable = false };

  // EmuReplicatedSpace::execution_space can access EmuLocalSpace
  enum { accessible = true };
  enum { deepcopy   = true };
};

template<>
struct MemorySpaceAccess< Kokkos::Experimental::EmuReplicatedSpace , Kokkos::Experimental::EmuStridedSpace > {
  // EmuReplicatedSpace::execution_space != EmuStridedSpace::execution_space
  enum { assignable = false };
  enum { accessible = true }; // EmuReplicatedSpace::execution_space
  enum { deepcopy   = true };
};


//----------------------------------------
// EmuStridedSpace::execution_space == HostSpace::execution_space
// EmuStridedSpace accessible to both Emu and Host

template<>
struct MemorySpaceAccess< Kokkos::Experimental::EmuStridedSpace , Kokkos::HostSpace > {
  enum { assignable = true }; 
  enum { accessible = true };  // EmuStridedSpace::execution_space
  enum { deepcopy   = true };
};

template<>
struct MemorySpaceAccess< Kokkos::Experimental::EmuStridedSpace , Kokkos::Experimental::EmuLocalSpace > {
  enum { assignable = false }; // Cannot access from Host
  enum { accessible = false };
  enum { deepcopy   = true };
};

template<>
struct MemorySpaceAccess< Kokkos::Experimental::EmuStridedSpace , Kokkos::Experimental::EmuReplicatedSpace > {
  enum { assignable = false }; // different execution_space
  enum { accessible = true };  // same accessibility
  enum { deepcopy   = true };
};

//----------------------------------------

}} // namespace Kokkos::Impl

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

void DeepCopyAsyncEmu( void * dst , const void * src , size_t n);

template<> struct DeepCopy< Kokkos::Experimental::EmuLocalSpace , Kokkos::Experimental::EmuLocalSpace , Kokkos::Experimental::CilkPlus>
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const Kokkos::Experimental::CilkPlus & , void * dst , const void * src , size_t );
};

template<> struct DeepCopy< Kokkos::Experimental::EmuLocalSpace , HostSpace , Kokkos::Experimental::CilkPlus >
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const Kokkos::Experimental::CilkPlus & , void * dst , const void * src , size_t );
};

template<> struct DeepCopy< HostSpace , Kokkos::Experimental::EmuLocalSpace , Kokkos::Experimental::CilkPlus >
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const Kokkos::Experimental::CilkPlus & , void * dst , const void * src , size_t );
};

template<class ExecutionSpace> struct DeepCopy< Kokkos::Experimental::EmuLocalSpace , Kokkos::Experimental::EmuLocalSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< Kokkos::Experimental::EmuLocalSpace , Kokkos::Experimental::EmuLocalSpace , Kokkos::Experimental::CilkPlus >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncEmu (dst,src,n);
  }
};

template<class ExecutionSpace> struct DeepCopy< Kokkos::Experimental::EmuLocalSpace , HostSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< Kokkos::Experimental::EmuLocalSpace , HostSpace , Kokkos::Experimental::CilkPlus>( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncEmu (dst,src,n);
  }
};

template<class ExecutionSpace>
struct DeepCopy< HostSpace , Kokkos::Experimental::EmuLocalSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< HostSpace , Kokkos::Experimental::EmuLocalSpace , Kokkos::Experimental::CilkPlus >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncEmu (dst,src,n);
  }
};

template<class ExecutionSpace>
struct DeepCopy< Kokkos::Experimental::EmuLocalSpace , Kokkos::Experimental::EmuReplicatedSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< Kokkos::Experimental::EmuLocalSpace , Kokkos::Experimental::EmuReplicatedSpace , Kokkos::Experimental::CilkPlus >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncEmu (dst,src,n);
  }
};

template<class ExecutionSpace>
struct DeepCopy< Kokkos::Experimental::EmuLocalSpace , Kokkos::Experimental::EmuStridedSpace , ExecutionSpace>
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< Kokkos::Experimental::EmuLocalSpace , Kokkos::Experimental::EmuStridedSpace , Kokkos::Experimental::CilkPlus >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncEmu (dst,src,n);
  }
};


template<class ExecutionSpace>
struct DeepCopy< Kokkos::Experimental::EmuReplicatedSpace , Kokkos::Experimental::EmuLocalSpace , ExecutionSpace>
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< Kokkos::Experimental::EmuReplicatedSpace , Kokkos::Experimental::EmuLocalSpace , Kokkos::Experimental::CilkPlus >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncEmu (dst,src,n);
  }
};

template<class ExecutionSpace>
struct DeepCopy< Kokkos::Experimental::EmuReplicatedSpace , Kokkos::Experimental::EmuReplicatedSpace , ExecutionSpace>
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< Kokkos::Experimental::EmuReplicatedSpace , Kokkos::Experimental::EmuReplicatedSpace , Kokkos::Experimental::CilkPlus >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncEmu (dst,src,n);
  }
};

template<class ExecutionSpace>
struct DeepCopy< Kokkos::Experimental::EmuReplicatedSpace , Kokkos::Experimental::EmuStridedSpace , ExecutionSpace>
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< Kokkos::Experimental::EmuReplicatedSpace , Kokkos::Experimental::EmuStridedSpace , Kokkos::Experimental::CilkPlus >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncEmu (dst,src,n);
  }
};

template<class ExecutionSpace> struct DeepCopy< Kokkos::Experimental::EmuReplicatedSpace , HostSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< Kokkos::Experimental::EmuReplicatedSpace , HostSpace , Kokkos::Experimental::CilkPlus >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncEmu (dst,src,n);
  }
};


template<class ExecutionSpace> struct DeepCopy< Kokkos::Experimental::EmuStridedSpace , Kokkos::Experimental::EmuLocalSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< Kokkos::Experimental::EmuStridedSpace , Kokkos::Experimental::EmuLocalSpace , Kokkos::Experimental::CilkPlus >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncEmu (dst,src,n);
  }
};

template<class ExecutionSpace> struct DeepCopy< Kokkos::Experimental::EmuStridedSpace , Kokkos::Experimental::EmuReplicatedSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< Kokkos::Experimental::EmuStridedSpace , Kokkos::Experimental::EmuReplicatedSpace , Kokkos::Experimental::CilkPlus >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncEmu (dst,src,n);
  }
};

template<class ExecutionSpace> struct DeepCopy< Kokkos::Experimental::EmuStridedSpace , Kokkos::Experimental::EmuStridedSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< Kokkos::Experimental::EmuStridedSpace , Kokkos::Experimental::EmuStridedSpace , Kokkos::Experimental::CilkPlus >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncEmu (dst,src,n);
  }
};

template<class ExecutionSpace> struct DeepCopy< Kokkos::Experimental::EmuStridedSpace , HostSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< HostSpace , HostSpace , Kokkos::Experimental::CilkPlus >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncEmu (dst,src,n);
  }
};


template<class ExecutionSpace> struct DeepCopy< HostSpace , Kokkos::Experimental::EmuReplicatedSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< HostSpace , Kokkos::Experimental::EmuReplicatedSpace , Kokkos::Experimental::CilkPlus >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncEmu (dst,src,n);
  }
};

template<class ExecutionSpace> struct DeepCopy< HostSpace , Kokkos::Experimental::EmuStridedSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< HostSpace , Kokkos::Experimental::EmuStridedSpace , Kokkos::Experimental::CilkPlus >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncEmu (dst,src,n);
  }
};

} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

/** Running in EmuLocalSpace attempting to access HostSpace: error */
template<>
struct VerifyExecutionCanAccessMemorySpace< Kokkos::Experimental::EmuLocalSpace , Kokkos::HostSpace >
{
  enum { value = true };
  KOKKOS_INLINE_FUNCTION static void verify( void ) { }
  KOKKOS_INLINE_FUNCTION static void verify( const void * ) { }
};

/** Running in EmuLocalSpace accessing EmuReplicatedSpace: ok */
template<>
struct VerifyExecutionCanAccessMemorySpace< Kokkos::Experimental::EmuLocalSpace , Kokkos::Experimental::EmuReplicatedSpace >
{
  enum { value = true };
  KOKKOS_INLINE_FUNCTION static void verify( void ) { }
  KOKKOS_INLINE_FUNCTION static void verify( const void * ) { }
};

/** Running in EmuLocalSpace accessing EmuStridedSpace: ok */
template<>
struct VerifyExecutionCanAccessMemorySpace< Kokkos::Experimental::EmuLocalSpace , Kokkos::Experimental::EmuStridedSpace >
{
  enum { value = true };
  KOKKOS_INLINE_FUNCTION static void verify( void ) { }
  KOKKOS_INLINE_FUNCTION static void verify( const void * ) { }
};

/** Running in EmuLocalSpace attempting to access an unknown space: error */
template< class OtherSpace >
struct VerifyExecutionCanAccessMemorySpace<
  typename enable_if< ! is_same<Kokkos::Experimental::EmuLocalSpace,OtherSpace>::value , Kokkos::Experimental::EmuLocalSpace >::type ,
  OtherSpace >
{
  enum { value = true };
  KOKKOS_INLINE_FUNCTION static void verify( void ) { }
  KOKKOS_INLINE_FUNCTION static void verify( const void * ) { }
};

//----------------------------------------------------------------------------
/** Running in HostSpace attempting to access EmuLocalSpace */
template<>
struct VerifyExecutionCanAccessMemorySpace< Kokkos::HostSpace , Kokkos::Experimental::EmuLocalSpace >
{
  enum { value = true };
  KOKKOS_INLINE_FUNCTION static void verify( void ) { }
  KOKKOS_INLINE_FUNCTION static void verify( const void * ) { }
};

/** Running in HostSpace accessing EmuReplicatedSpace is OK */
template<>
struct VerifyExecutionCanAccessMemorySpace< Kokkos::HostSpace , Kokkos::Experimental::EmuReplicatedSpace >
{
  enum { value = true };
  inline static void verify( void ) { }
  inline static void verify( const void * ) { }
};

/** Running in HostSpace accessing EmuStridedSpace is OK */
template<>
struct VerifyExecutionCanAccessMemorySpace< Kokkos::HostSpace , Kokkos::Experimental::EmuStridedSpace >
{
  enum { value = true };
  KOKKOS_INLINE_FUNCTION static void verify( void ) {}
  KOKKOS_INLINE_FUNCTION static void verify( const void * ) {}
};

} // Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template<>
class SharedAllocationRecord< Kokkos::Experimental::EmuLocalSpace , void >
  : public SharedAllocationRecord< void , void >
{
public:
  typedef SharedAllocationRecord< void , void >  RecordBase ;
private:

  friend class SharedAllocationRecord< Kokkos::Experimental::EmuReplicatedSpace , void > ;

  SharedAllocationRecord( const SharedAllocationRecord & ) = delete ;
  SharedAllocationRecord & operator = ( const SharedAllocationRecord & ) = delete ;

  static void deallocate( RecordBase * );

  const Kokkos::Experimental::EmuLocalSpace * m_space ;

protected:

  ~SharedAllocationRecord();
  SharedAllocationRecord() : RecordBase(), m_space() {}

  SharedAllocationRecord( RecordBase*                      basePtr
                        , const char *                     arg_label
                        , const size_t                     arg_alloc_size
                        , const Kokkos::Experimental::EmuLocalSpace        * arg_space
                        , int node
                        , const RecordBase::function_type  arg_dealloc
                        );

  SharedAllocationRecord( RecordBase*                      basePtr
                        , const char *                     arg_label
                        , const size_t                     arg_alloc_size
                        , const Kokkos::Experimental::EmuLocalSpace        * arg_space
                        , int node
                        );

public:

  std::string get_label() const ;

  static SharedAllocationRecord * allocate( const Kokkos::Experimental::EmuLocalSpace &  arg_space
                                          , const std::string       &  arg_label
                                          , const size_t               arg_alloc_size );

  /**\brief  Allocate tracked memory in the space */
  static 
  void * allocate_tracked( const Kokkos::Experimental::EmuLocalSpace & arg_space
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

  static void print_records( std::ostream & , const Kokkos::Experimental::EmuLocalSpace & , bool detail = false );
};


template<>
class SharedAllocationRecord< Kokkos::Experimental::EmuReplicatedSpace , void >
  : public SharedAllocationRecord< void , void >
{
public:
  typedef SharedAllocationRecord< void , void >  RecordBase ;

private:
  SharedAllocationRecord( const SharedAllocationRecord & ) = delete ;
  SharedAllocationRecord & operator = ( const SharedAllocationRecord & ) = delete ;

  static void deallocate( RecordBase * );

  //const Kokkos::Experimental::EmuReplicatedSpace * m_space ;

protected:

  ~SharedAllocationRecord();
  SharedAllocationRecord() : RecordBase() {}

  SharedAllocationRecord( RecordBase*                      basePtr
                        , const char *                     arg_label
                        , const size_t                     arg_alloc_size
                        , void *                           arg_ptr
                        , int node
                        , const RecordBase::function_type  arg_dealloc
                        );

  SharedAllocationRecord( RecordBase*                      basePtr
                        , const char *                     arg_label
                        , const size_t                     arg_alloc_size
                        , void *                           arg_ptr
                        , int node
                        );


public:
  static void custom_increment( Kokkos::Impl::SharedAllocationRecord<void, void> * );

  static Kokkos::Impl::SharedAllocationRecord<void, void> * custom_decrement( Kokkos::Impl::SharedAllocationRecord<void, void> * );


  std::string get_label() const ;

  static SharedAllocationRecord * allocate( const char *                  arg_label
                                          , const size_t                  arg_alloc_size
                                          );

  /**\brief  Allocate tracked memory in the space */
  static 
  void * allocate_tracked( const Kokkos::Experimental::EmuReplicatedSpace & arg_space
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

  static void print_records( std::ostream & , const Kokkos::Experimental::EmuReplicatedSpace & , bool detail = false );
};

template< class DestroyFunctor >
class SharedAllocationRecord<Kokkos::Experimental::EmuLocalSpace, DestroyFunctor> : 
       public SharedAllocationRecord< Kokkos::Experimental::EmuLocalSpace , void >
{
public:
  typedef SharedAllocationRecord< void , void >  RecordBase ;

  SharedAllocationRecord( RecordBase*         basePtr
                        , const char *        arg_label
                        , const size_t        arg_alloc
                        , const Kokkos::Experimental::EmuLocalSpace * arg_space
                        , int node_id
                        )
    /*  Allocate user memory as [ SharedAllocationHeader , user_memory ] */
    : SharedAllocationRecord< Kokkos::Experimental::EmuLocalSpace , void >( basePtr , arg_label , arg_alloc , 
                                                       arg_space, node_id, 
                                                       & Kokkos::Impl::deallocate< Kokkos::Experimental::EmuLocalSpace , DestroyFunctor > )
    , m_destroy()
    {printf("custom destroy function constructor\n");}

private:
  SharedAllocationRecord() = delete ;
  SharedAllocationRecord( const SharedAllocationRecord & ) = delete ;
  SharedAllocationRecord & operator = ( const SharedAllocationRecord & ) = delete ;

public:

  DestroyFunctor  m_destroy ;

  // Allocate with a zero use count.  Incrementing the use count from zero to one
  // inserts the record into the tracking list.  Decrementing the count from one to zero
  // removes from the trakcing list and deallocates.
  KOKKOS_INLINE_FUNCTION static
  SharedAllocationRecord * allocate( const Kokkos::Experimental::EmuLocalSpace & arg_space
                                   , const std::string & arg_label
                                   , const size_t        arg_alloc
                                   );
};


template< class DestroyFunctor >
class SharedAllocationRecord<Kokkos::Experimental::EmuReplicatedSpace, DestroyFunctor> : 
       public SharedAllocationRecord< Kokkos::Experimental::EmuReplicatedSpace , void >
{
public:
  typedef SharedAllocationRecord< void , void >  RecordBase ;


  SharedAllocationRecord( RecordBase*         basePtr
                        , const char *        arg_label
                        , const size_t        arg_alloc
                        , void *              arg_ptr
                        , int node_id
                        )
    /*  Allocate user memory as [ SharedAllocationHeader , user_memory ] */
    : SharedAllocationRecord< Kokkos::Experimental::EmuReplicatedSpace , void >( basePtr , arg_label , arg_alloc , arg_ptr, node_id, 
                                                       & Kokkos::Impl::deallocate< Kokkos::Experimental::EmuReplicatedSpace , DestroyFunctor > )
    , m_destroy()
    {printf("custom destroy function constructor\n");}

private:
  SharedAllocationRecord() = delete ;
  SharedAllocationRecord( const SharedAllocationRecord & ) = delete ;
  SharedAllocationRecord & operator = ( const SharedAllocationRecord & ) = delete ;

public:

  DestroyFunctor  m_destroy ;

  // Allocate with a zero use count.  Incrementing the use count from zero to one
  // inserts the record into the tracking list.  Decrementing the count from one to zero
  // removes from the trakcing list and deallocates.
  KOKKOS_INLINE_FUNCTION static
  SharedAllocationRecord * allocate( const Kokkos::Experimental::EmuReplicatedSpace & arg_space
                                   , const std::string & arg_label
                                   , const size_t        arg_alloc
                                   );
};


template<>
class SharedAllocationRecord< Kokkos::Experimental::EmuStridedSpace , void >
  : public SharedAllocationRecord< void , void >
{
private:

  typedef SharedAllocationRecord< void , void >  RecordBase ;

  SharedAllocationRecord( const SharedAllocationRecord & ) = delete ;
  SharedAllocationRecord & operator = ( const SharedAllocationRecord & ) = delete ;

  static void deallocate( RecordBase * );

  static RecordBase s_root_record ;

  const Kokkos::Experimental::EmuStridedSpace m_space ;

protected:

  ~SharedAllocationRecord();
  SharedAllocationRecord() : RecordBase(), m_space() {}

  SharedAllocationRecord( const Kokkos::Experimental::EmuStridedSpace     & arg_space
                        , const std::string              & arg_label
                        , const size_t                     arg_alloc_size
                        , const RecordBase::function_type  arg_dealloc = & deallocate
                        );

public:

  std::string get_label() const ;

  static SharedAllocationRecord * allocate( const Kokkos::Experimental::EmuStridedSpace &  arg_space
                                          , const std::string          &  arg_label
                                          , const size_t                  arg_alloc_size
                                          );
  /**\brief  Allocate tracked memory in the space */
  static
  void * allocate_tracked( const Kokkos::Experimental::EmuStridedSpace & arg_space
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

  static void print_records( std::ostream & , const Kokkos::Experimental::EmuStridedSpace & , bool detail = false );
};


template< class DestroyFunctor >
SharedAllocationRecord< Kokkos::Experimental::EmuReplicatedSpace , DestroyFunctor > *
SharedAllocationRecord<Kokkos::Experimental::EmuReplicatedSpace, DestroyFunctor>::
allocate( const Kokkos::Experimental::EmuReplicatedSpace & arg_space
        , const std::string & arg_label
        , const size_t        arg_alloc
        )
{
   typedef SharedAllocationRecord< Kokkos::Experimental::EmuReplicatedSpace , DestroyFunctor > repl_shared_rec;

   long * lRef = (long*)Kokkos::Experimental::EmuReplicatedSpace::getRefAddr();
   Kokkos::Experimental::EmuReplicatedSpace* pMem = ((Kokkos::Experimental::EmuReplicatedSpace*)mw_ptr1to0(Kokkos::Experimental::EmuReplicatedSpace::ers));
   void *vr = pMem->allocate(sizeof(repl_shared_rec));
   void *vh = pMem->allocate(sizeof(SharedAllocationHeader) + arg_alloc);   
   const char * szLabel = mw_ptr1to0(arg_label.c_str());
                          
   for ( int i = 0; i < NODELETS(); i++) {
      repl_shared_rec * pRec = (repl_shared_rec*)mw_get_localto(vr,&lRef[i]);
      SharedAllocationHeader* pH = (SharedAllocationHeader*)mw_get_localto(vh, &lRef[i]);
      repl_shared_rec::RecordBase* rb = (repl_shared_rec::RecordBase*)mw_get_localto(Kokkos::Experimental::EmuReplicatedSpace::repl_root_record, &lRef[i]);
      new (pRec) repl_shared_rec( rb, szLabel, arg_alloc, pH, (int)NODE_ID() );
   }
   return (SharedAllocationRecord< Kokkos::Experimental::EmuReplicatedSpace , DestroyFunctor >*)vr;
}

template< class DestroyFunctor >
SharedAllocationRecord< Kokkos::Experimental::EmuLocalSpace , DestroyFunctor > *
SharedAllocationRecord<Kokkos::Experimental::EmuLocalSpace, DestroyFunctor>::
allocate( const Kokkos::Experimental::EmuLocalSpace & arg_space
        , const std::string & arg_label
        , const size_t        arg_alloc
        )
{
   typedef SharedAllocationRecord< Kokkos::Experimental::EmuLocalSpace , DestroyFunctor > local_shared_rec;
   const char * szLabel = mw_ptr1to0(arg_label.c_str());
                          
   local_shared_rec * pRec = (local_shared_rec*)arg_space.allocate(sizeof(local_shared_rec)); 
   local_shared_rec::RecordBase* rb = (local_shared_rec::RecordBase*)mw_get_localto(Kokkos::Experimental::EmuLocalSpace::local_root_record, pRec);
   new (pRec) local_shared_rec( rb, szLabel, arg_alloc, &arg_space, (int)NODE_ID() );
   return (SharedAllocationRecord< Kokkos::Experimental::EmuLocalSpace , DestroyFunctor >*)pRec;
}



} // Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #if defined( KOKKOS_ENABLE_EMU ) */
#endif /* #define KOKKOS_EmuLocalSpace_HPP */

