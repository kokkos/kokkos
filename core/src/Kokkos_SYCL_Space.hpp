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

#ifndef KOKKOS_SYCLSPACE_HPP
#define KOKKOS_SYCLSPACE_HPP

#include <Kokkos_Core_fwd.hpp>

#if defined( KOKKOS_ENABLE_SYCL )

#include <iosfwd>
#include <typeinfo>
#include <string>

#include <Kokkos_HostSpace.hpp>

/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Experimental {
/** \brief  SYCL on-device memory management */

class SYCLSpace {
public:

  //! Tag this class as a kokkos memory space
  typedef SYCLSpace             memory_space ;
  typedef Kokkos::Experimental::SYCL          execution_space ;
  typedef Kokkos::Device<execution_space,memory_space> device_type;

  typedef unsigned int          size_type ;

  /*--------------------------------*/

  SYCLSpace();
  SYCLSpace( SYCLSpace && rhs ) = default ;
  SYCLSpace( const SYCLSpace & rhs ) = default ;
  SYCLSpace & operator = ( SYCLSpace && rhs ) = default ;
  SYCLSpace & operator = ( const SYCLSpace & rhs ) = default ;
  ~SYCLSpace() = default ;

  /**\brief  Allocate untracked memory in the rocm space */
  void * allocate( const size_t arg_alloc_size ) const ;

  /**\brief  Deallocate untracked memory in the rocm space */
  void deallocate( void * const arg_alloc_ptr
                 , const size_t arg_alloc_size ) const ;

  /**\brief Return Name of the MemorySpace */
  static constexpr const char* name() { return m_name; };

  /*--------------------------------*/
  /** \brief  Error reporting for HostSpace attempt to access SYCLSpace */
  static void access_error();
  static void access_error( const void * const );

private:

  int  m_device ; ///< Which SYCL device

  static constexpr const char* m_name = "SYCL";
  friend class Kokkos::Impl::SharedAllocationRecord< Kokkos::Experimental::SYCLSpace , void > ;
};

} // namespace Experimental

namespace Impl {


/// \brief Initialize lock array for arbitrary size atomics.
///
/// Arbitrary atomics are implemented using a hash table of locks
/// where the hash value is derived from the address of the
/// object for which an atomic operation is performed.
/// This function initializes the locks to zero (unset).
void init_lock_arrays_sycl_space();

/// \brief Retrieve the pointer to the lock array for arbitrary size atomics.
///
/// Arbitrary atomics are implemented using a hash table of locks
/// where the hash value is derived from the address of the
/// object for which an atomic operation is performed.
/// This function retrieves the lock array pointer.
/// If the array is not yet allocated it will do so.
int* atomic_lock_array_sycl_space_ptr(bool deallocate = false);

/// \brief Retrieve the pointer to the scratch array for team and thread private global memory.
///
/// Team and Thread private scratch allocations in
/// global memory are aquired via locks.
/// This function retrieves the lock array pointer.
/// If the array is not yet allocated it will do so.
int* scratch_lock_array_sycl_space_ptr(bool deallocate = false);

/// \brief Retrieve the pointer to the scratch array for unique identifiers.
///
/// Unique identifiers in the range 0-SYCL::concurrency
/// are provided via locks.
/// This function retrieves the lock array pointer.
/// If the array is not yet allocated it will do so.
int* threadid_lock_array_sycl_space_ptr(bool deallocate = false);
}
} // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Experimental {

class SYCLHostUSMSpace
{
public:
    typedef SYCL execution_space;
    typedef SYCLHostUSMSpace memory_space;
    typedef Kokkos::Device<execution_space, memory_space> device_type;
    typedef unsigned int size_type;

    SYCLHostUSMSpace();

    void* allocate(const size_t arg_alloc_size) const;
    void deallocate(void* const arg_alloc_ptr, const size_t arg_alloc_size) const;

    static constexpr const char* name() { return m_name; }

private:
  static constexpr const char* m_name = "SYCLHostUSM";
  int m_device;
};

class SYCLDeviceUSMSpace
{
public:
    typedef SYCL execution_space;
    typedef SYCLDeviceUSMSpace memory_space;
    typedef Kokkos::Device<execution_space, memory_space> device_type;
    typedef unsigned int size_type;

    SYCLDeviceUSMSpace();

    void* allocate(const size_t arg_alloc_size) const;
    void deallocate(void* const arg_alloc_ptr, const size_t arg_alloc_size) const;

    static constexpr const char* name() { return m_name; }

private:
  static constexpr const char* m_name = "SYCLDeviceUSM";
  int m_device;
};

class SYCLSharedUSMSpace
{
public:
    typedef SYCL execution_space;
    typedef SYCLSharedUSMSpace memory_space;
    typedef Kokkos::Device<execution_space, memory_space> device_type;
    typedef unsigned int size_type;

    SYCLSharedUSMSpace();

    void* allocate(const size_t arg_alloc_size) const;
    void deallocate(void* const arg_alloc_ptr, const size_t arg_alloc_size) const;

    static constexpr const char* name() { return m_name; }

private:
  static constexpr const char* m_name = "SYCLSharedUSM";
  int m_device;
};


}}

namespace Kokkos {
namespace Experimental {
/** \brief  Host memory that is accessible to SYCL execution space
 *          through SYCL's host-pinned memory allocation.
 */
//TODO NLIBER remove SYCLHostPinnedSpace
class SYCLHostPinnedSpace {
public:

  //! Tag this class as a kokkos memory space
  /** \brief  Memory is in HostSpace so use the HostSpace::execution_space */
  typedef HostSpace::execution_space  execution_space ;
  typedef SYCLHostPinnedSpace         memory_space ;
  typedef Kokkos::Device<execution_space,memory_space> device_type;
  typedef unsigned int                size_type ;

  /*--------------------------------*/

  SYCLHostPinnedSpace();
  SYCLHostPinnedSpace( SYCLHostPinnedSpace && rhs ) = default ;
  SYCLHostPinnedSpace( const SYCLHostPinnedSpace & rhs ) = default ;
  SYCLHostPinnedSpace & operator = ( SYCLHostPinnedSpace && rhs ) = default ;
  SYCLHostPinnedSpace & operator = ( const SYCLHostPinnedSpace & rhs ) = default ;
  ~SYCLHostPinnedSpace() = default ;

  /**\brief  Allocate untracked memory in the space */
  void * allocate( const size_t arg_alloc_size ) const ;

  /**\brief  Deallocate untracked memory in the space */
  void deallocate( void * const arg_alloc_ptr
                 , const size_t arg_alloc_size ) const ;

  /**\brief Return Name of the MemorySpace */
  static constexpr const char* name() { return m_name; };

private:

  static constexpr const char* m_name = "SYCLHostPinned";

  /*--------------------------------*/
};

} // namespace Experimental
} // namespace Kokkos

namespace Kokkos {
namespace Impl {

static_assert( Kokkos::Impl::MemorySpaceAccess< Kokkos::Experimental::SYCLSpace , Kokkos::Experimental::SYCLSpace >::assignable , "" );

template<>
struct MemorySpaceAccess<Kokkos::HostSpace, Kokkos::Experimental::SYCLHostUSMSpace> {
    enum { assignable = false };
    enum { accessible = true };
    enum { deepcopy   = false };
};

template<>
struct MemorySpaceAccess<Kokkos::HostSpace, Kokkos::Experimental::SYCLDeviceUSMSpace> {
    enum { assignable = false };
    enum { accessible = false };
    enum { deepcopy   = false };
};

template<>
struct MemorySpaceAccess<Kokkos::HostSpace, Kokkos::Experimental::SYCLSharedUSMSpace> {
    enum { assignable = true };
    enum { accessible = true };
    enum { deepcopy   = true };
};

template<>
struct MemorySpaceAccess<Kokkos::Experimental::SYCLHostUSMSpace, Kokkos::HostSpace> {
    enum { assignable = false };
    enum { accessible = true };
    enum { deepcopy   = false };
};

template<>
struct MemorySpaceAccess<Kokkos::Experimental::SYCLDeviceUSMSpace, Kokkos::HostSpace> {
    enum { assignable = false };
    enum { accessible = false };
    enum { deepcopy   = false };
};

template<>
struct MemorySpaceAccess<Kokkos::Experimental::SYCLSharedUSMSpace, Kokkos::HostSpace> {
    enum { assignable = true };
    enum { accessible = true };
    enum { deepcopy   = true };
};

// NLIBER TODO MemorySpaceAccess between Host, Device and Shared

template<>
struct SharedAllocationRecord<Kokkos::Experimental::SYCLHostUSMSpace, void>
: SharedAllocationRecord<void, void>
{
    SharedAllocationRecord(const SharedAllocationRecord&) = delete;
    SharedAllocationRecord(SharedAllocationRecord&&) = delete;
    SharedAllocationRecord& operator=(SharedAllocationRecord&&) = delete;
    SharedAllocationRecord& operator=(const SharedAllocationRecord&) = delete;
    
    static void deallocate(SharedAllocationRecord<void, void>*);
    
    #ifdef KOKKOS_DEBUG
      static SharedAllocationRecord<void, void> s_root_record;
    #endif

    const Kokkos::Experimental::SYCLHostUSMSpace m_space;
    
protected:
    ~SharedAllocationRecord();
    
    SharedAllocationRecord(
        const Kokkos::Experimental::SYCLHostUSMSpace& space,
        const std::string& label,
        const size_t size,
        const SharedAllocationRecord<void, void>::function_type dealloc = &deallocate);
    
public:
    std::string get_label() const;
    
    static SharedAllocationRecord* allocate(
        const Kokkos::Experimental::SYCLHostUSMSpace& space,
        const std::string& label,
        const size_t size);

    static void* allocate_tracked(
        const Kokkos::Experimental::SYCLHostUSMSpace& arg_space,
        const std::string& label,
        const size_t size);

    static void* reallocate_tracked(void* const ptr,
                                    const size_t size);

    static void deallocate_tracked(void* const ptr);

    static SharedAllocationRecord* get_record(void* ptr);

    static void print_records(std::ostream&,
                              const Kokkos::Experimental::SYCLHostUSMSpace&,
                              bool detail = false);

};

template<>
struct SharedAllocationRecord<Kokkos::Experimental::SYCLDeviceUSMSpace, void>
: SharedAllocationRecord<void, void>
{
    SharedAllocationRecord(const SharedAllocationRecord&) = delete;
    SharedAllocationRecord(SharedAllocationRecord&&) = delete;
    SharedAllocationRecord& operator=(SharedAllocationRecord&&) = delete;
    SharedAllocationRecord& operator=(const SharedAllocationRecord&) = delete;
    
    static void deallocate(SharedAllocationRecord<void, void>*);
    
    #ifdef KOKKOS_DEBUG
      static SharedAllocationRecord<void, void> s_root_record;
    #endif

    const Kokkos::Experimental::SYCLDeviceUSMSpace m_space;
    
protected:
    ~SharedAllocationRecord();
    
    SharedAllocationRecord(
        const Kokkos::Experimental::SYCLDeviceUSMSpace& space,
        const std::string& label,
        const size_t size,
        const SharedAllocationRecord<void, void>::function_type dealloc = &deallocate);
    
public:
    std::string get_label() const;
    
    static SharedAllocationRecord* allocate(
        const Kokkos::Experimental::SYCLDeviceUSMSpace& space,
        const std::string& label,
        const size_t size);

    static void* allocate_tracked(
        const Kokkos::Experimental::SYCLDeviceUSMSpace& arg_space,
        const std::string& label,
        const size_t size);

    static void* reallocate_tracked(void* const ptr,
                                    const size_t size);

    static void deallocate_tracked(void* const ptr);

    static SharedAllocationRecord* get_record(void* ptr);

    static void print_records(std::ostream&,
                              const Kokkos::Experimental::SYCLDeviceUSMSpace&,
                              bool detail = false);

};

template<>
struct SharedAllocationRecord<Kokkos::Experimental::SYCLSharedUSMSpace, void>
: SharedAllocationRecord<void, void>
{
    SharedAllocationRecord(const SharedAllocationRecord&) = delete;
    SharedAllocationRecord(SharedAllocationRecord&&) = delete;
    SharedAllocationRecord& operator=(SharedAllocationRecord&&) = delete;
    SharedAllocationRecord& operator=(const SharedAllocationRecord&) = delete;
    
    static void deallocate(SharedAllocationRecord<void, void>*);
    
    #ifdef KOKKOS_DEBUG
      static SharedAllocationRecord<void, void> s_root_record;
    #endif

    const Kokkos::Experimental::SYCLSharedUSMSpace m_space;
    
protected:
    ~SharedAllocationRecord();
    
    SharedAllocationRecord(
        const Kokkos::Experimental::SYCLSharedUSMSpace& space,
        const std::string& label,
        const size_t size,
        const SharedAllocationRecord<void, void>::function_type dealloc = &deallocate);
    
public:
    std::string get_label() const;
    
    static SharedAllocationRecord* allocate(
        const Kokkos::Experimental::SYCLSharedUSMSpace& space,
        const std::string& label,
        const size_t size);

    static void* allocate_tracked(
        const Kokkos::Experimental::SYCLSharedUSMSpace& arg_space,
        const std::string& label,
        const size_t size);

    static void* reallocate_tracked(void* const ptr,
                                    const size_t size);

    static void deallocate_tracked(void* const ptr);

    static SharedAllocationRecord* get_record(void* ptr);

    static void print_records(std::ostream&,
                              const Kokkos::Experimental::SYCLSharedUSMSpace&,
                              bool detail = false);

};

} // namespace Impl
} // namespace Kokkos

/*--------------------------------------------------------------------------*/ /*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

static_assert( Kokkos::Impl::MemorySpaceAccess< Kokkos::Experimental::SYCLSpace , Kokkos::Experimental::SYCLSpace >::assignable , "" );

//----------------------------------------

template<>
struct MemorySpaceAccess< Kokkos::HostSpace , Kokkos::Experimental::SYCLSpace > {
  enum { assignable = false };
  enum { accessible = false };
  enum { deepcopy   = true };
};

template<>
struct MemorySpaceAccess< Kokkos::HostSpace , Kokkos::Experimental::SYCLHostPinnedSpace > {
  // HostSpace::execution_space == SYCLHostPinnedSpace::execution_space
  enum { assignable = true };
  enum { accessible = true };
  enum { deepcopy   = true };
};

//----------------------------------------

template<>
struct MemorySpaceAccess< Kokkos::Experimental::SYCLSpace , Kokkos::HostSpace > {
  enum { assignable = false };
  enum { accessible = false };
  enum { deepcopy   = true };
};

template<>
struct MemorySpaceAccess< Kokkos::Experimental::SYCLSpace , Kokkos::Experimental::SYCLHostPinnedSpace > {
  // SYCLSpace::execution_space != SYCLHostPinnedSpace::execution_space
  enum { assignable = false };
  enum { accessible = true }; // SYCLSpace::execution_space
  enum { deepcopy   = true };
};


//----------------------------------------
// SYCLHostPinnedSpace::execution_space == HostSpace::execution_space
// SYCLHostPinnedSpace accessible to both SYCL and Host

template<>
struct MemorySpaceAccess< Kokkos::Experimental::SYCLHostPinnedSpace , Kokkos::HostSpace > {
  enum { assignable = false }; // Cannot access from SYCL
  enum { accessible = true };  // SYCLHostPinnedSpace::execution_space
  enum { deepcopy   = true };
};

template<>
struct MemorySpaceAccess< Kokkos::Experimental::SYCLHostPinnedSpace , Kokkos::Experimental::SYCLSpace > {
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

//hc::completion_future DeepCopyAsyncSYCL( void * dst , const void * src , size_t n);

template<> struct DeepCopy< Kokkos::Experimental::SYCLSpace , Kokkos::Experimental::SYCLSpace , Kokkos::Experimental::SYCL>
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const Kokkos::Experimental::SYCL & , void * dst , const void * src , size_t );
};

template<> struct DeepCopy< Kokkos::Experimental::SYCLSpace , HostSpace , Kokkos::Experimental::SYCL >
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const Kokkos::Experimental::SYCL & , void * dst , const void * src , size_t );
};

template<> struct DeepCopy< HostSpace , Kokkos::Experimental::SYCLSpace , Kokkos::Experimental::SYCL >
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const Kokkos::Experimental::SYCL & , void * dst , const void * src , size_t );
};

template<class ExecutionSpace> struct DeepCopy< Kokkos::Experimental::SYCLSpace , Kokkos::Experimental::SYCLSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< Kokkos::Experimental::SYCLSpace , Kokkos::Experimental::SYCLSpace , Kokkos::Experimental::SYCL >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
//    exec.fence();
//    hc::completion_future fut = DeepCopyAsyncSYCL (dst,src,n);
//    fut.wait();
//    DeepCopy (dst,src,n);
  }
};

template<class ExecutionSpace> struct DeepCopy< Kokkos::Experimental::SYCLSpace , HostSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< Kokkos::Experimental::SYCLSpace , HostSpace , Kokkos::Experimental::SYCL>( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopy (dst,src,n);
  }
};

template<class ExecutionSpace>
struct DeepCopy< HostSpace , Kokkos::Experimental::SYCLSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< HostSpace , Kokkos::Experimental::SYCLSpace , Kokkos::Experimental::SYCL >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopy (dst,src,n);
  }
};

//========================
template<> struct DeepCopy< Kokkos::Experimental::SYCLHostUSMSpace , Kokkos::Experimental::SYCLHostUSMSpace , Kokkos::Experimental::SYCL>
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const Kokkos::Experimental::SYCL & , void * dst , const void * src , size_t );
};

template<> struct DeepCopy< Kokkos::Experimental::SYCLHostUSMSpace , HostSpace , Kokkos::Experimental::SYCL >
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const Kokkos::Experimental::SYCL & , void * dst , const void * src , size_t );
};

template<> struct DeepCopy< HostSpace , Kokkos::Experimental::SYCLHostUSMSpace , Kokkos::Experimental::SYCL >
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const Kokkos::Experimental::SYCL & , void * dst , const void * src , size_t );
};

template<class ExecutionSpace> struct DeepCopy< Kokkos::Experimental::SYCLHostUSMSpace , Kokkos::Experimental::SYCLHostUSMSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< Kokkos::Experimental::SYCLHostUSMSpace , Kokkos::Experimental::SYCLHostUSMSpace , Kokkos::Experimental::SYCL >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
//    exec.fence();
//    hc::completion_future fut = DeepCopyAsyncSYCL (dst,src,n);
//    fut.wait();
//    DeepCopy (dst,src,n);
  }
};

template<class ExecutionSpace> struct DeepCopy< Kokkos::Experimental::SYCLHostUSMSpace , HostSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< Kokkos::Experimental::SYCLHostUSMSpace , HostSpace , Kokkos::Experimental::SYCL>( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopy (dst,src,n);
  }
};

template<class ExecutionSpace>
struct DeepCopy< HostSpace , Kokkos::Experimental::SYCLHostUSMSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< HostSpace , Kokkos::Experimental::SYCLHostUSMSpace , Kokkos::Experimental::SYCL >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopy (dst,src,n);
  }
};


//========================
//========================
template<> struct DeepCopy< Kokkos::Experimental::SYCLDeviceUSMSpace , Kokkos::Experimental::SYCLDeviceUSMSpace , Kokkos::Experimental::SYCL>
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const Kokkos::Experimental::SYCL & , void * dst , const void * src , size_t );
};

template<> struct DeepCopy< Kokkos::Experimental::SYCLDeviceUSMSpace , HostSpace , Kokkos::Experimental::SYCL >
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const Kokkos::Experimental::SYCL & , void * dst , const void * src , size_t );
};

template<> struct DeepCopy< HostSpace , Kokkos::Experimental::SYCLDeviceUSMSpace , Kokkos::Experimental::SYCL >
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const Kokkos::Experimental::SYCL & , void * dst , const void * src , size_t );
};

template<class ExecutionSpace> struct DeepCopy< Kokkos::Experimental::SYCLDeviceUSMSpace , Kokkos::Experimental::SYCLDeviceUSMSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< Kokkos::Experimental::SYCLDeviceUSMSpace , Kokkos::Experimental::SYCLDeviceUSMSpace , Kokkos::Experimental::SYCL >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
//    exec.fence();
//    hc::completion_future fut = DeepCopyAsyncSYCL (dst,src,n);
//    fut.wait();
//    DeepCopy (dst,src,n);
  }
};

template<class ExecutionSpace> struct DeepCopy< Kokkos::Experimental::SYCLDeviceUSMSpace , HostSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< Kokkos::Experimental::SYCLDeviceUSMSpace , HostSpace , Kokkos::Experimental::SYCL>( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopy (dst,src,n);
  }
};

template<class ExecutionSpace>
struct DeepCopy< HostSpace , Kokkos::Experimental::SYCLDeviceUSMSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< HostSpace , Kokkos::Experimental::SYCLDeviceUSMSpace , Kokkos::Experimental::SYCL >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopy (dst,src,n);
  }
};


//========================
//========================
template<> struct DeepCopy< Kokkos::Experimental::SYCLSharedUSMSpace , Kokkos::Experimental::SYCLSharedUSMSpace , Kokkos::Experimental::SYCL>
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const Kokkos::Experimental::SYCL & , void * dst , const void * src , size_t );
};

template<> struct DeepCopy< Kokkos::Experimental::SYCLSharedUSMSpace , HostSpace , Kokkos::Experimental::SYCL >
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const Kokkos::Experimental::SYCL & , void * dst , const void * src , size_t );
};

template<> struct DeepCopy< HostSpace , Kokkos::Experimental::SYCLSharedUSMSpace , Kokkos::Experimental::SYCL >
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const Kokkos::Experimental::SYCL & , void * dst , const void * src , size_t );
};

template<class ExecutionSpace> struct DeepCopy< Kokkos::Experimental::SYCLSharedUSMSpace , Kokkos::Experimental::SYCLSharedUSMSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< Kokkos::Experimental::SYCLSharedUSMSpace , Kokkos::Experimental::SYCLSharedUSMSpace , Kokkos::Experimental::SYCL >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
//    exec.fence();
//    hc::completion_future fut = DeepCopyAsyncSYCL (dst,src,n);
//    fut.wait();
//    DeepCopy (dst,src,n);
  }
};

template<class ExecutionSpace> struct DeepCopy< Kokkos::Experimental::SYCLSharedUSMSpace , HostSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< Kokkos::Experimental::SYCLSharedUSMSpace , HostSpace , Kokkos::Experimental::SYCL>( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopy (dst,src,n);
  }
};

template<class ExecutionSpace>
struct DeepCopy< HostSpace , Kokkos::Experimental::SYCLSharedUSMSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< HostSpace , Kokkos::Experimental::SYCLSharedUSMSpace , Kokkos::Experimental::SYCL >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopy (dst,src,n);
  }
};


//========================

template<> struct DeepCopy< Kokkos::Experimental::SYCLHostPinnedSpace , Kokkos::Experimental::SYCLHostPinnedSpace , Kokkos::Experimental::SYCL>
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const Kokkos::Experimental::SYCL & , void * dst , const void * src , size_t );
};

template<> struct DeepCopy< Kokkos::Experimental::SYCLHostPinnedSpace , HostSpace , Kokkos::Experimental::SYCL >
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const Kokkos::Experimental::SYCL & , void * dst , const void * src , size_t );
};

template<> struct DeepCopy< HostSpace , Kokkos::Experimental::SYCLHostPinnedSpace , Kokkos::Experimental::SYCL >
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const Kokkos::Experimental::SYCL & , void * dst , const void * src , size_t );
};

template<class ExecutionSpace>
struct DeepCopy< Kokkos::Experimental::SYCLSpace , Kokkos::Experimental::SYCLHostPinnedSpace , ExecutionSpace>
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< Kokkos::Experimental::SYCLSpace , HostSpace , Kokkos::Experimental::SYCL >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
//    exec.fence();
//    hc::completion_future fut = DeepCopyAsyncSYCL (dst,src,n);
//    fut.wait();
//    DeepCopySYCL (dst,src,n);
  }
};

template<class ExecutionSpace> struct DeepCopy< Kokkos::Experimental::SYCLHostPinnedSpace , Kokkos::Experimental::SYCLSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< HostSpace , Kokkos::Experimental::SYCLSpace , Kokkos::Experimental::SYCL >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
//    exec.fence();
//    hc::completion_future fut = DeepCopyAsyncSYCL (dst,src,n);
//    fut.wait();
//    DeepCopySYCL (dst,src,n);
  }
};



template<class ExecutionSpace> struct DeepCopy< Kokkos::Experimental::SYCLHostPinnedSpace , Kokkos::Experimental::SYCLHostPinnedSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< Kokkos::Experimental::SYCLHostPinnedSpace , Kokkos::Experimental::SYCLHostPinnedSpace , Kokkos::Experimental::SYCL >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
//    hc::completion_future fut = DeepCopyAsyncSYCL (dst,src,n);
//    fut.wait();
//    DeepCopyAsyncSYCL (dst,src,n);
    DeepCopy (dst,src,n);
  }
};

template<class ExecutionSpace> struct DeepCopy< Kokkos::Experimental::SYCLHostPinnedSpace , HostSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< Kokkos::Experimental::SYCLHostPinnedSpace , HostSpace , Kokkos::Experimental::SYCL>( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopy (dst,src,n);
  }
};

template<class ExecutionSpace>
struct DeepCopy< HostSpace , Kokkos::Experimental::SYCLHostPinnedSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< HostSpace , Kokkos::Experimental::SYCLHostPinnedSpace , Kokkos::Experimental::SYCL >( dst , src , n ); }

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

/** Running in SYCLSpace attempting to access HostSpace: error */
template<>
struct VerifyExecutionCanAccessMemorySpace< Kokkos::Experimental::SYCLSpace , Kokkos::HostSpace >
{
  enum { value = false };
  KOKKOS_INLINE_FUNCTION static void verify( void )
    { Kokkos::abort("SYCL code attempted to access HostSpace memory"); }

  KOKKOS_INLINE_FUNCTION static void verify( const void * )
    { Kokkos::abort("SYCL code attempted to access HostSpace memory"); }
};

/** Running in SYCLSpace accessing SYCLHostPinnedSpace: ok */
template<>
struct VerifyExecutionCanAccessMemorySpace< Kokkos::Experimental::SYCLSpace , Kokkos::Experimental::SYCLHostPinnedSpace >
{
  enum { value = true };
  KOKKOS_INLINE_FUNCTION static void verify( void ) { }
  KOKKOS_INLINE_FUNCTION static void verify( const void * ) { }
};

/** Running in SYCLSpace attempting to access an unknown space: error */
template< class OtherSpace >
struct VerifyExecutionCanAccessMemorySpace<
  typename enable_if< ! is_same<Kokkos::Experimental::SYCLSpace,OtherSpace>::value , Kokkos::Experimental::SYCLSpace >::type ,
  OtherSpace >
{
  enum { value = false };
  KOKKOS_INLINE_FUNCTION static void verify( void )
    { Kokkos::abort("SYCL code attempted to access unknown Space memory"); }

  KOKKOS_INLINE_FUNCTION static void verify( const void * )
    { Kokkos::abort("SYCL code attempted to access unknown Space memory"); }
};

//----------------------------------------------------------------------------
/** Running in HostSpace attempting to access SYCLSpace */
template<>
struct VerifyExecutionCanAccessMemorySpace< Kokkos::HostSpace , Kokkos::Experimental::SYCLSpace >
{
  enum { value = false };
  inline static void verify( void ) { Kokkos::Experimental::SYCLSpace::access_error(); }
  inline static void verify( const void * p ) { Kokkos::Experimental::SYCLSpace::access_error(p); }
};

/** Running in HostSpace accessing SYCLHostPinnedSpace is OK */
template<>
struct VerifyExecutionCanAccessMemorySpace< Kokkos::HostSpace , Kokkos::Experimental::SYCLHostPinnedSpace >
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
class SharedAllocationRecord< Kokkos::Experimental::SYCLSpace , void >
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

  const Kokkos::Experimental::SYCLSpace m_space ;

protected:

  ~SharedAllocationRecord();

  SharedAllocationRecord( const Kokkos::Experimental::SYCLSpace        & arg_space
                        , const std::string              & arg_label
                        , const size_t                     arg_alloc_size
                        , const RecordBase::function_type  arg_dealloc = & deallocate
                        );

public:

  std::string get_label() const ;

  static SharedAllocationRecord * allocate( const Kokkos::Experimental::SYCLSpace &  arg_space
                                          , const std::string       &  arg_label
                                          , const size_t               arg_alloc_size );

  /**\brief  Allocate tracked memory in the space */
  static
  void * allocate_tracked( const Kokkos::Experimental::SYCLSpace & arg_space
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

  static void print_records( std::ostream & , const Kokkos::Experimental::SYCLSpace & , bool detail = false );
};

template<>
class SharedAllocationRecord< Kokkos::Experimental::SYCLHostPinnedSpace , void >
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

  const Kokkos::Experimental::SYCLHostPinnedSpace m_space ;

protected:

  ~SharedAllocationRecord();
  SharedAllocationRecord() : RecordBase(), m_space() {}

  SharedAllocationRecord( const Kokkos::Experimental::SYCLHostPinnedSpace     & arg_space
                        , const std::string              & arg_label
                        , const size_t                     arg_alloc_size
                        , const RecordBase::function_type  arg_dealloc = & deallocate
                        );

public:

  std::string get_label() const ;

  static SharedAllocationRecord * allocate( const Kokkos::Experimental::SYCLHostPinnedSpace &  arg_space
                                          , const std::string          &  arg_label
                                          , const size_t                  arg_alloc_size
                                          );
  /**\brief  Allocate tracked memory in the space */
  static
  void * allocate_tracked( const Kokkos::Experimental::SYCLHostPinnedSpace & arg_space
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

  static void print_records( std::ostream & , const Kokkos::Experimental::SYCLHostPinnedSpace & , bool detail = false );
};
} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #if defined( KOKKOS_ENABLE_SYCL ) */
#endif /* #define KOKKOS_SYCLSPACE_HPP */

