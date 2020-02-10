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

#ifndef KOKKOS_SICMSPACE_HPP
#define KOKKOS_SICMSPACE_HPP

#include <cstring>
#include <string>
#include <iosfwd>
#include <memory>
#include <typeinfo>

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Concepts.hpp>
#include <Kokkos_MemoryTraits.hpp>

#include <impl/Kokkos_Traits.hpp>
#include <impl/Kokkos_Error.hpp>
#include <impl/Kokkos_SharedAlloc.hpp>

#include <impl/Kokkos_HostSpace_deepcopy.hpp>

#include <sicm_low.h>

/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Experimental {

/// \class SICMSpace
/// \brief Memory management for SICM memory.
///
/// SICMSpace is a memory space that governs SICM memory.  "SICM"
/// memory means the usual CPU-accessible memory.
class SICMSpace {
public:
  //! Tag this class as a kokkos memory space
  typedef SICMSpace  memory_space;
  typedef size_t     size_type;

  /// \typedef execution_space
  /// \brief Default execution space for this memory space.
  ///
  /// Every memory space has a default execution space.  This is
  /// useful for things like initializing a View (which happens in
  /// parallel using the View's default execution space).
#if defined( KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_OPENMP )
  typedef Kokkos::OpenMP    execution_space;
#elif defined( KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_THREADS )
  typedef Kokkos::Threads   execution_space;
#elif defined( KOKKOS_ENABLE_OPENMP )
  typedef Kokkos::OpenMP    execution_space;
#elif defined( KOKKOS_ENABLE_THREADS )
  typedef Kokkos::Threads   execution_space;
#elif defined( KOKKOS_ENABLE_SERIAL )
  typedef Kokkos::Serial    execution_space;
#else
#  error "At least one of the following host execution spaces must be defined: Kokkos::OpenMP, Kokkos::Threads, or Kokkos::Serial.  You might be seeing this message if you disabled the Kokkos::Serial device explicitly using the Kokkos_ENABLE_Serial:BOOL=OFF CMake option, but did not enable any of the other host execution space devices."
#endif

  //! This memory space preferred device_type
  typedef Kokkos::Device< execution_space, memory_space > device_type;

  /**\brief  Default memory space instance */
  SICMSpace();
  SICMSpace( sicm_device_list * devices );
  SICMSpace( SICMSpace && rhs ) = default;
  SICMSpace( const SICMSpace & rhs ) = default;
  SICMSpace & operator = ( SICMSpace && ) = default;
  SICMSpace & operator = ( const SICMSpace & ) = default;
  ~SICMSpace() = default;

  /**\brief  Allocate untracked memory in the space */
  void * allocate( const size_t arg_alloc_size ) const;

  /**\brief  Deallocate untracked memory in the space */
  void deallocate( void * const arg_alloc_ptr
                 , const size_t arg_alloc_size ) const;

  /**\brief Return Name of the MemorySpace */
  static constexpr const char* name() { return "SICM"; }

private:
  std::shared_ptr<sicm_arena> arena;
  friend class Kokkos::Impl::SharedAllocationRecord< Kokkos::Experimental::SICMSpace, void >;
};

} // namespace Experimental
} // namespace Kokkos

//----------------------------------------------------------------------------

namespace Kokkos {

namespace Impl {

template <>
struct MemorySpaceAccess<
    Kokkos::Experimental::SICMSpace,
    Kokkos::HostSpace> {
  enum { assignable = true  };
  enum { accessible = true  };
  enum { deepcopy   = false };
};

template <>
struct MemorySpaceAccess<
    Kokkos::HostSpace,
    Kokkos::Experimental::SICMSpace> {
  enum { assignable = true  };
  enum { accessible = true  };
  enum { deepcopy   = false };
};

} // namespace Impl

} // namespace Kokkos

//----------------------------------------------------------------------------

namespace Kokkos {

namespace Impl {

template<>
class SharedAllocationRecord< Kokkos::Experimental::SICMSpace, void >
  : public SharedAllocationRecord< void, void >
{
private:
  friend Kokkos::Experimental::SICMSpace;

  typedef SharedAllocationRecord< void, void >  RecordBase;

  SharedAllocationRecord( const SharedAllocationRecord & ) = delete;
  SharedAllocationRecord & operator = ( const SharedAllocationRecord & ) = delete;

  static void deallocate( RecordBase * );

#ifdef KOKKOS_DEBUG
  /**\brief  Root record for tracked allocations from this SICMSpace instance */
  static RecordBase s_root_record;
#endif

  const Kokkos::Experimental::SICMSpace m_space;

protected:
  ~SharedAllocationRecord();
  SharedAllocationRecord() = default;

  SharedAllocationRecord( const Kokkos::Experimental::SICMSpace  & arg_space
                        , const std::string                      & arg_label
                        , const size_t                             arg_alloc_size
                        , const RecordBase::function_type          arg_dealloc = & deallocate
                        );

public:

  inline
  std::string get_label() const
  {
    return std::string( RecordBase::head()->m_label );
  }

  KOKKOS_INLINE_FUNCTION static
  SharedAllocationRecord * allocate( const Kokkos::Experimental::SICMSpace &  arg_space
                                   , const std::string                     &  arg_label
                                   , const size_t                             arg_alloc_size
                                   )
  {
#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
    return new SharedAllocationRecord( arg_space, arg_label, arg_alloc_size );
#else
    return (SharedAllocationRecord *) 0;
#endif
  }


  /**\brief  Allocate tracked memory in the space */
  static
  void * allocate_tracked( const Kokkos::Experimental::SICMSpace & arg_space
                         , const std::string                     & arg_label
                         , const size_t                            arg_alloc_size );

  /**\brief  Reallocate tracked memory in the space */
  static
  void * reallocate_tracked( void * const arg_alloc_ptr
                           , const size_t arg_alloc_size );

  /**\brief  Deallocate tracked memory in the space */
  static
  void deallocate_tracked( void * const arg_alloc_ptr );

  static SharedAllocationRecord * get_record( void * arg_alloc_ptr );

  static void print_records( std::ostream &, const Kokkos::Experimental::SICMSpace &, bool detail = false );
};

} // namespace Impl

} // namespace Kokkos

//----------------------------------------------------------------------------

namespace Kokkos {

namespace Impl {

template< class ExecutionSpace >
struct DeepCopy< Experimental::SICMSpace, Experimental::SICMSpace, ExecutionSpace > {
  DeepCopy( void * dst, const void * src, size_t n ) {
    hostspace_parallel_deepcopy(dst,src,n);
  }

  DeepCopy( const ExecutionSpace& exec, void * dst, const void * src, size_t n ) {
    exec.fence();
    hostspace_parallel_deepcopy(dst,src,n);
    exec.fence();
  }
};

template< class ExecutionSpace >
struct DeepCopy< HostSpace, Experimental::SICMSpace, ExecutionSpace > {
  DeepCopy( void * dst, const void * src, size_t n ) {
    hostspace_parallel_deepcopy(dst,src,n);
  }

  DeepCopy( const ExecutionSpace& exec, void * dst, const void * src, size_t n ) {
    exec.fence();
    hostspace_parallel_deepcopy(dst,src,n);
    exec.fence();
  }
};

template< class ExecutionSpace >
struct DeepCopy< Experimental::SICMSpace, HostSpace, ExecutionSpace > {
  DeepCopy( void * dst, const void * src, size_t n ) {
    hostspace_parallel_deepcopy(dst,src,n);
  }

  DeepCopy( const ExecutionSpace& exec, void * dst, const void * src, size_t n ) {
    exec.fence();
    hostspace_parallel_deepcopy(dst,src,n);
    exec.fence();
  }
};

} // namespace Impl

} // namespace Kokkos

#endif // #define KOKKOS_SICMSPACE_HPP
