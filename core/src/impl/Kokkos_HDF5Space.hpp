
#ifndef __KOKKOS_HDF5_SPACE_
#define __KOKKOS_HDF5_SPACE_

#include <cstring>
#include <string>
#include <iosfwd>
#include <typeinfo>

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Concepts.hpp>
#include <Kokkos_MemoryTraits.hpp>
#include <impl/Kokkos_SharedAlloc.hpp>
#include <hdf5.h>


namespace Kokkos {

namespace Experimental {

class KokkosHDF5Accessor : public Kokkos::Impl::SharedAllocationHeader {


public:
   size_t data_size;
   std::string file_path;
   std::string data_set;
   size_t rank;
   size_t chunk_size;
   size_t file_offset;
   hid_t m_fid;    
   hid_t m_did;    
   hid_t m_mid;    

   KokkosHDF5Accessor() : data_size(0),
                          file_path(""),
                          data_set("default_dataset"),
                          m_fid(0),
                          m_did(0),
                          m_mid(0)  {
      rank = 1;
      chunk_size = 65536;
      file_offset = 0;
   }
   KokkosHDF5Accessor(const size_t size, const std::string & path ) : data_size(size),
                                                                      file_path(path),
                                                                      data_set("default_dataset"),
                                                                      m_fid(0),
                                                                      m_did(0),
                                                                      m_mid(0)  {
      rank = 1;
      //chunk_size = ( data_size / 20 );
      chunk_size = 131072;
      chunk_size = chunk_size < 65536 ? 65536 : chunk_size;
      chunk_size = chunk_size > 1048576 ? 1048576 : chunk_size;
      file_offset = 0;
   }

   KokkosHDF5Accessor( const KokkosHDF5Accessor & rhs ) = default;
   KokkosHDF5Accessor( KokkosHDF5Accessor && rhs ) = default;
   KokkosHDF5Accessor & operator = ( KokkosHDF5Accessor && ) = default;
   KokkosHDF5Accessor & operator = ( const KokkosHDF5Accessor & ) = default;
   KokkosHDF5Accessor( void* ptr ) {
      KokkosHDF5Accessor * pAcc = static_cast<KokkosHDF5Accessor*>(ptr);
      if (pAcc) {
         data_size = pAcc->data_size;
         file_path = pAcc->file_path;
         rank = pAcc->rank;
         chunk_size = pAcc->chunk_size;
      }

   } 

   KokkosHDF5Accessor( void* ptr, const size_t offset ) {
      KokkosHDF5Accessor * pAcc = static_cast<KokkosHDF5Accessor*>(ptr);
      if (pAcc) {
         data_size = pAcc->data_size;
         file_path = pAcc->file_path;
         rank = pAcc->rank;
         chunk_size = pAcc->chunk_size;
         file_offset = offset;
      }
   }

   int initialize( const std::string & filepath, 
               const std::string & dataset_name );

   int open_file();
   void close_file();

   size_t ReadFile(void * dest, const size_t dest_size);
   
   size_t WriteFile(const void * src, const size_t src_size);

   void finalize();
   
   ~KokkosHDF5Accessor() {
      finalize();
   }
};



/// \class HDF5Space
/// \brief Memory management for HDF5 
///
/// HDF5Space is a memory space that governs access to HDF5 data.
/// 
class HDF5Space {
public:
  //! Tag this class as a kokkos memory space
  typedef Kokkos::Experimental::HDF5Space  file_space;   // used to uniquely identify file spaces
  typedef Kokkos::Experimental::HDF5Space  memory_space;
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
  HDF5Space();
  HDF5Space( HDF5Space && rhs ) = default;
  HDF5Space( const HDF5Space & rhs ) = default;
  HDF5Space & operator = ( HDF5Space && ) = default;
  HDF5Space & operator = ( const HDF5Space & ) = default;
  ~HDF5Space() = default;

  /**\brief  Allocate untracked memory in the space */
  void * allocate( const size_t arg_alloc_size, const std::string & path ) const;

  /**\brief  Deallocate untracked memory in the space */
  void deallocate( void * const arg_alloc_ptr
                 , const size_t arg_alloc_size ) const;

  /**\brief Return Name of the MemorySpace */
  static constexpr const char* name() { return m_name; }

  static void track_check_point_mirror( const std::string label, void * dst, void * src, const size_t size );

  static void restore_all_views(); 
  static void restore_view(const std::string name);
  static void checkpoint_views();

  static void set_default_path( const std::string path );

private:
  static constexpr const char* m_name = "HDF5";
  friend class Kokkos::Impl::SharedAllocationRecord< Kokkos::Experimental::HDF5Space, void >;
};

}
}

namespace Kokkos {

namespace Impl {

template<>
class SharedAllocationRecord< Kokkos::Experimental::HDF5Space, void >
  : public SharedAllocationRecord< void, void >
{
private:
  friend Kokkos::Experimental::HDF5Space;

  typedef SharedAllocationRecord< void, void >  RecordBase;

  SharedAllocationRecord( const SharedAllocationRecord & ) = delete;
  SharedAllocationRecord & operator = ( const SharedAllocationRecord & ) = delete;

  static void deallocate( RecordBase * );

#ifdef KOKKOS_DEBUG
  /**\brief  Root record for tracked allocations from this HDF5Space instance */
  static RecordBase s_root_record;
#endif

  const Kokkos::Experimental::HDF5Space m_space;

protected:
  ~SharedAllocationRecord();
  SharedAllocationRecord() = default;

  SharedAllocationRecord( const Kokkos::Experimental::HDF5Space        & arg_space
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
  SharedAllocationRecord * allocate( const Kokkos::Experimental::HDF5Space &  arg_space
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
  void * allocate_tracked( const Kokkos::Experimental::HDF5Space & arg_space
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

  static void print_records( std::ostream &, const Kokkos::Experimental::HDF5Space &, bool detail = false );
};


template<class ExecutionSpace> struct DeepCopy< Kokkos::Experimental::HDF5Space , Kokkos::HostSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  {   
      Kokkos::Impl::SharedAllocationHeader * pData = (Kokkos::Impl::SharedAllocationHeader*)dst;
      Kokkos::Experimental::KokkosHDF5Accessor * pAcc = static_cast<Kokkos::Experimental::KokkosHDF5Accessor*>(pData-1);

      if (pAcc) {
         pAcc->WriteFile( src, n );
      }
  }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    Kokkos::Impl::SharedAllocationHeader * pData = (Kokkos::Impl::SharedAllocationHeader*)dst;
    Kokkos::Experimental::KokkosHDF5Accessor * pAcc = static_cast<Kokkos::Experimental::KokkosHDF5Accessor*>(pData-1);
    if (pAcc) {
       pAcc->WriteFile( src, n );
    }
  }
};

template<class ExecutionSpace> struct DeepCopy<  Kokkos::HostSpace , Kokkos::Experimental::HDF5Space , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  {       
      Kokkos::Impl::SharedAllocationHeader * pData = (Kokkos::Impl::SharedAllocationHeader*)src;
      Kokkos::Experimental::KokkosHDF5Accessor * pAcc = static_cast<Kokkos::Experimental::KokkosHDF5Accessor*>(pData-1);
      if (pAcc) {
         pAcc->ReadFile( dst, n );
      }
  }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    Kokkos::Impl::SharedAllocationHeader * pData = (Kokkos::Impl::SharedAllocationHeader*)src;
    Kokkos::Experimental::KokkosHDF5Accessor * pAcc = static_cast<Kokkos::Experimental::KokkosHDF5Accessor*>(pData - 1);
    if (pAcc) {
       pAcc->ReadFile( dst, n );
    }
  }
};

} // Impl

} // Kokkos

#endif
