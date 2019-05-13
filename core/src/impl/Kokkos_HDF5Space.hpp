
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
#include <impl/Kokkos_ExternalIOInterface.hpp>
#include <hdf5.h>


namespace Kokkos {

namespace Experimental {

class KokkosHDF5ConfigurationManager  {
public:
   class OperationPrimitive {
    public:
      enum { OPP_INVALID = 0,
             OPP_ADD = 1,
             OPP_SUB = 2,
             OPP_DIV = 3,
             OPP_MUL = 4,
             OPP_MOD = 5 };
      int type;
      size_t val;
      int operation;
      OperationPrimitive * m_left;
      OperationPrimitive * m_right;

      int which() { return type; }

      size_t get_val() { return val; }
      int get_opp() { return operation; }

      OperationPrimitive(  ) : type(0), val(0), operation(0), m_left(nullptr), m_right(nullptr) {}
      OperationPrimitive( size_t val_ ) : type(1), val(val_), operation(0), m_left(nullptr), m_right(nullptr) {}
      OperationPrimitive( int op_, OperationPrimitive * left_ ) : type(2), val(-1), operation(op_), m_left(left_)  {}

      OperationPrimitive( const OperationPrimitive & rhs ) = default;
      OperationPrimitive( OperationPrimitive && rhs ) = default;
      OperationPrimitive & operator = ( OperationPrimitive && ) = default;
      OperationPrimitive & operator = ( const OperationPrimitive & ) = default;

      OperationPrimitive( OperationPrimitive * ptr_ ) : type(ptr_->type), val(ptr_->val), operation(ptr_->operation), 
                                                              m_left(ptr_->m_left), m_right(ptr_->m_right) {}

      ~OperationPrimitive() {
         if (m_left != nullptr) delete m_left;
         if (m_right != nullptr) delete m_right;
      }

      void set_right_opp( OperationPrimitive * rhs) {
         m_right = rhs;
      }

      size_t evaluate () {
         switch ( which() ) {
            case 0:
               return 0;
            case 1:
               return val;
            case 2:
               return per_opp( m_left != nullptr ? m_left->evaluate() : 0 , 
                               m_right != nullptr ? m_right->evaluate() : 0 );
            default:
               return 0;
         }
         return 0;
      }

      size_t per_opp ( size_t left, size_t right ) {
          switch ( get_opp() ) {
             case 0:
                return 0;
             case 1:
                return left + right;
             case 2:
                return left - right;
             case 3:
                return left / right;
             case 4:
                return left * right;
             case 5:
                return left % right;
             default:
                return 0;
          }
          return 0;
      }

      static OperationPrimitive * parse_operator( const std::string & val, OperationPrimitive * left ) {
         if ( val == "+" ) {
            return new OperationPrimitive(1,left);
         } else if ( val == "-" ) {
            return new OperationPrimitive(2,left);
         } else if ( val == "/" ) {
            return new OperationPrimitive(3,left);
         } else if ( val == "*" ) {
            return new OperationPrimitive(4,left);
         } else if ( val == "%" ) {
            return new OperationPrimitive(5,left);
         }
         return new OperationPrimitive(0,left);
      }
   };

   enum { LAYOUT_DEFAULT = 0,
          LAYOUT_CONTIGUOUS = 1,
          LAYOUT_CHUNK = 2,
          LAYOUT_REGULAR = 3,
          LAYOUT_PATTERN = 4 };

   enum { VAR_OFFSET = 1,
          VAR_COUNT = 2,
          VAR_BLOCK = 3, 
          VAR_STRIDE =4 };

   int m_layout;
   boost::property_tree::ptree m_config;

   OperationPrimitive * resolve_variable( std::string data, std::map<const std::string, size_t> & var_map );
   OperationPrimitive * resolve_arithmetic( std::string data, std::map<const std::string, size_t> & var_map );

   int get_layout() { return m_layout; }

   boost::property_tree::ptree * get_config() { return &m_config; }

   void set_param_list( boost::property_tree::ptree l_config, int data_scope, std::string var_name, 
                        hsize_t var [], std::map<const std::string, size_t> & var_map );

   KokkosHDF5ConfigurationManager( const boost::property_tree::ptree & config_ ) : m_config(config_) { }
};

class KokkosHDF5Accessor : public KokkosIOAccessor {


public:
   std::string data_set;  // name of the dataset
   size_t rank;           // rank of the dataset
   hsize_t file_count[4];  // maximum dimensions = 4, default will be 1
   hsize_t file_offset[4]; // offset of this instance into the file/dataset
   hsize_t file_stride[4]; // stride defining data pattern
   hsize_t file_block[4];  // size of the blocks/chunks moved from source view to file
   hsize_t data_extents[4];// dimensions of the dataset
   hsize_t local_extents[4];// dimensions of the local memory set
   hid_t m_fid;           // file space handle
   hid_t m_did;           // data space handle
   hid_t m_mid;           // memory space handle
   size_t mpi_size;
   size_t mpi_rank;
   int m_layout;
   bool m_is_initialized;

   KokkosHDF5Accessor() : KokkosIOAccessor(),
                          data_set("default_dataset"),
                          rank(1),
                          file_count{1,1,1,1},
                          file_offset{0,0,0,0},
                          file_stride{1,0,0,0},
                          file_block{1,0,0,0},
                          m_fid(0),
                          m_did(0),
                          m_mid(0),
                          mpi_size(1),
                          mpi_rank(0),
                          m_layout(KokkosHDF5ConfigurationManager::LAYOUT_DEFAULT),
                          m_is_initialized(false)  { }

   KokkosHDF5Accessor(const size_t size, const std::string & path ) : KokkosIOAccessor(size, path, true),
                                                                      data_set("default_dataset"),
                                                                      rank(1),
                                                                      file_count{1,1,1,1},
                                                                      file_offset{0,0,0,0},
                                                                      file_stride{1,0,0,0},
                                                                      file_block{size,0,0,0},
                                                                      m_fid(0),
                                                                      m_did(0),
                                                                      m_mid(0),
                                                                      mpi_size(1),
                                                                      mpi_rank(0),
                                                                      m_layout(KokkosHDF5ConfigurationManager::LAYOUT_DEFAULT),
                                                                      m_is_initialized(true)  { }

   KokkosHDF5Accessor( const KokkosHDF5Accessor & rhs ) = default;
   KokkosHDF5Accessor( KokkosHDF5Accessor && rhs ) = default;
   KokkosHDF5Accessor & operator = ( KokkosHDF5Accessor && ) = default;
   KokkosHDF5Accessor & operator = ( const KokkosHDF5Accessor & ) = default;
   KokkosHDF5Accessor( const KokkosHDF5Accessor & cp_, const size_t size  ) :  m_fid(0), m_did(0), m_mid(0) {
      data_size = size;
      file_path = cp_.file_path;
      data_set = cp_.data_set;
      rank = cp_.rank;
      mpi_size = cp_.mpi_size;
      mpi_rank = cp_.mpi_rank;
      m_layout = cp_.m_layout;
      for (int i = 0; i < 4; i++) {
         file_count[i] = cp_.file_count[i];
         file_offset[i] = cp_.file_offset[i];
         file_stride[i] = cp_.file_stride[i];
         file_block[i] = cp_.file_block[i];
         data_extents[i] = cp_.data_extents[i];
         local_extents[i] = cp_.local_extents[i];
      }
      // need to re-initialize 
      if (data_size != cp_.data_size) {
         if (m_layout == KokkosHDF5ConfigurationManager::LAYOUT_DEFAULT) {
            initialize( size, file_path, data_set ); 
         } else {
            initialize( size, file_path, KokkosHDF5ConfigurationManager ( 
                                 KokkosIOConfigurationManager::get_instance()->get_config(file_path) ) );
         }
      }
      m_is_initialized = true;
   } 

   int initialize( const size_t size_,
                   const std::string & filepath, 
                   KokkosHDF5ConfigurationManager config_ );

   int initialize( const size_t size_,
                   const std::string & filepath, 
                   const std::string & dataset_name );

   int open_file();
   void close_file();
   bool is_initialized() { return m_is_initialized; }

   virtual size_t ReadFile_impl(void * dest, const size_t dest_size);
   
   virtual size_t WriteFile_impl(const void * src, const size_t src_size);

   void finalize();
   
   virtual ~KokkosHDF5Accessor() {
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

  static void restore_all_views(); 
  static void restore_view(const std::string name);
  static void checkpoint_views();

  static void set_default_path( const std::string path );
  static std::string s_default_path;

  static std::map<const std::string, KokkosHDF5Accessor> m_accessor_map;

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
     Kokkos::Experimental::KokkosIOAccessor::transfer_from_host( dst, src, n );
  }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    Kokkos::Experimental::KokkosIOAccessor::transfer_from_host( dst, src, n );
  }
};

template<class ExecutionSpace> struct DeepCopy<  Kokkos::HostSpace , Kokkos::Experimental::HDF5Space , ExecutionSpace >
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
