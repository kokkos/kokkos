
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
#ifndef __KOKKOS_STD_FILE_SPACE_
#define __KOKKOS_STD_FILE_SPACE_

#include <cstring>
#include <string>
#include <iosfwd>
#include <typeinfo>

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Concepts.hpp>
#include <Kokkos_MemoryTraits.hpp>
#include <impl/Kokkos_SharedAlloc.hpp>
#include <impl/Kokkos_ExternalIOInterface.hpp>
#include <fstream>


namespace Kokkos {

namespace Experimental {

class KokkosStdFileAccessor : public KokkosIOAccessor {


public:
   size_t file_offset;
   std::fstream file_strm;

   enum { READ_FILE = 0,
          WRITE_FILE = 1 };

   KokkosStdFileAccessor() : KokkosIOAccessor(),
                             file_offset(0) {
   }
   KokkosStdFileAccessor(const size_t size, const std::string & path ) : KokkosIOAccessor(size, path, true),
                                                                         file_offset(0) {
   }

   KokkosStdFileAccessor( const KokkosStdFileAccessor & rhs ) = default;
   KokkosStdFileAccessor( KokkosStdFileAccessor && rhs ) = default;
   KokkosStdFileAccessor & operator = ( KokkosStdFileAccessor && ) = default;
   KokkosStdFileAccessor & operator = ( const KokkosStdFileAccessor & ) = default;
   KokkosStdFileAccessor( void* ptr ) {
      KokkosStdFileAccessor * pAcc = static_cast<KokkosStdFileAccessor*>(ptr);
      if (pAcc) {
         data_size = pAcc->data_size;
         file_path = pAcc->file_path;
         file_offset = pAcc->file_offset;
      }

   } 

   KokkosStdFileAccessor( void* ptr, const size_t offset ) {
      KokkosStdFileAccessor * pAcc = static_cast<KokkosStdFileAccessor*>(ptr);
      if (pAcc) {
         data_size = pAcc->data_size;
         file_path = pAcc->file_path;
         file_offset = offset;
      }
   }

   int initialize( const std::string & filepath );

   bool open_file(int read_write = KokkosStdFileAccessor::READ_FILE);
   void close_file();

   virtual size_t ReadFile_impl(void * dest, const size_t dest_size);
   
   virtual size_t WriteFile_impl(const void * src, const size_t src_size);

   void finalize();
   
   virtual ~KokkosStdFileAccessor() {
   }
};


/// \class StdFileSpace
/// \brief Memory management for StdFile 
///
/// StdFileSpace is a memory space that governs access to StdFile data.
/// 
class StdFileSpace {
public:
  //! Tag this class as a kokkos memory space
  typedef Kokkos::Experimental::StdFileSpace  file_space;   // used to uniquely identify file spaces
  typedef Kokkos::Experimental::StdFileSpace  memory_space;
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
//#elif defined( KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_QTHREADS )
//  typedef Kokkos::Qthreads  execution_space;
#elif defined( KOKKOS_ENABLE_OPENMP )
  typedef Kokkos::OpenMP    execution_space;
#elif defined( KOKKOS_ENABLE_THREADS )
  typedef Kokkos::Threads   execution_space;
//#elif defined( KOKKOS_ENABLE_QTHREADS )
//  typedef Kokkos::Qthreads  execution_space;
#elif defined( KOKKOS_ENABLE_SERIAL )
  typedef Kokkos::Serial    execution_space;
#else
#  error "At least one of the following host execution spaces must be defined: Kokkos::OpenMP, Kokkos::Threads, Kokkos::Qthreads, or Kokkos::Serial.  You might be seeing this message if you disabled the Kokkos::Serial device explicitly using the Kokkos_ENABLE_Serial:BOOL=OFF CMake option, but did not enable any of the other host execution space devices."
#endif

  //! This memory space preferred device_type
  typedef Kokkos::Device< execution_space, memory_space > device_type;

  /**\brief  Default memory space instance */
  StdFileSpace();
  StdFileSpace( StdFileSpace && rhs ) = default;
  StdFileSpace( const StdFileSpace & rhs ) = default;
  StdFileSpace & operator = ( StdFileSpace && ) = default;
  StdFileSpace & operator = ( const StdFileSpace & ) = default;
  ~StdFileSpace() = default;

  /**\brief  Allocate untracked memory in the space */
  void * allocate( const size_t arg_alloc_size, const std::string & path ) const;

  /**\brief  Deallocate untracked memory in the space */
  void deallocate( void * const arg_alloc_ptr
                 , const size_t arg_alloc_size ) const;

  /**\brief Return Name of the MemorySpace */
  static constexpr const char* name() { return m_name; }

  static void restore_all_views(); 
  static void restore_view(const std::string name);
  static void checkpoint_views();
  static void set_default_path( const std::string path );
  static std::string s_default_path;

private:
  static constexpr const char* m_name = "StdFile";
  friend class Kokkos::Impl::SharedAllocationRecord< Kokkos::Experimental::StdFileSpace, void >;
};

}
}

namespace Kokkos {

namespace Impl {

template<>
class SharedAllocationRecord< Kokkos::Experimental::StdFileSpace, void >
  : public SharedAllocationRecord< void, void >
{
private:
  friend Kokkos::Experimental::StdFileSpace;

  typedef SharedAllocationRecord< void, void >  RecordBase;

  SharedAllocationRecord( const SharedAllocationRecord & ) = delete;
  SharedAllocationRecord & operator = ( const SharedAllocationRecord & ) = delete;

  static void deallocate( RecordBase * );

#ifdef KOKKOS_DEBUG
  /**\brief  Root record for tracked allocations from this StdFileSpace instance */
  static RecordBase s_root_record;
#endif

  const Kokkos::Experimental::StdFileSpace m_space;

protected:
  ~SharedAllocationRecord();
  SharedAllocationRecord() = default;

  SharedAllocationRecord( const Kokkos::Experimental::StdFileSpace        & arg_space
                        , const std::string              & arg_label
                        , const size_t                     arg_alloc_size
                        , const RecordBase::function_type  arg_dealloc = & deallocate
                        );

public:

  inline
  std::string get_label() const
  {
    return std::string( RecordBase::head()->m_label );
  }

  KOKKOS_INLINE_FUNCTION static
  SharedAllocationRecord * allocate( const Kokkos::Experimental::StdFileSpace &  arg_space
                                   , const std::string       &  arg_label
                                   , const size_t               arg_alloc_size
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
  void * allocate_tracked( const Kokkos::Experimental::StdFileSpace & arg_space
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

  static void print_records( std::ostream &, const Kokkos::Experimental::StdFileSpace &, bool detail = false );
};


template<class ExecutionSpace> struct DeepCopy< Kokkos::Experimental::StdFileSpace , Kokkos::HostSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  {  
      Kokkos::Experimental::KokkosIOAccessor::transfer_from_host( dst, src, n );
  }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    Kokkos::Experimental::KokkosIOAccessor::transfer_from_host( dst, src, n );
  }
};

template<class ExecutionSpace> struct DeepCopy<  Kokkos::HostSpace , Kokkos::Experimental::StdFileSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  {       
     Kokkos::Experimental::KokkosIOAccessor::transfer_to_host( dst, src, n );
  }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    Kokkos::Experimental::KokkosIOAccessor::transfer_to_host( dst, src, n );
  }
};

} // Impl

} // Kokkos

#endif
