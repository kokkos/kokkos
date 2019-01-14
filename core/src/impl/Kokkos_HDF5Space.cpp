#include "Kokkos_Core.hpp"
#include "Kokkos_HDF5Space.hpp"

namespace Kokkos {

namespace Experimental {

   #define min(X,Y) (X > Y) ? Y : X

   int KokkosHDF5Accessor::initialize( const std::string & filepath, 
                                       const std::string & dataset_name ) { 

       file_path = filepath;
       data_set = dataset_name;

 //       printf("Initializing HDF5 properties: %s - %d\n", file_path.c_str(), data_size );


   }

   int KokkosHDF5Accessor::open_file( ) { 

       hsize_t dims[1];
       dims[0] = data_size;

       if (m_fid == 0  && !file_exists(file_path)) {
 //          printf("creating HDF5 file: %s - %d\n", file_path.c_str(), data_size );
          hid_t pid = H5Pcreate(H5P_FILE_ACCESS);
          m_fid = H5Fcreate( file_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, pid );
          H5Pclose(pid);

          if (m_fid == 0) {
              printf("Error creating HDF5 file\n");
              return -1;
          }

          hid_t fsid = H5Screate_simple(1, dims, NULL);
          pid = H5Pcreate(H5P_DATASET_CREATE);
          hsize_t chunk[1];
          chunk[0] = min(chunk_size, data_size);
          H5Pset_chunk(pid, 1, chunk);
          m_did = H5Dcreate(m_fid, data_set.c_str(), H5T_NATIVE_CHAR, fsid, 
                             H5P_DEFAULT, pid, H5P_DEFAULT );
          if (m_did == 0) {
              printf("Error creating dataset\n");
              return -1;
          }
          H5Pclose(pid);
          H5Sclose(fsid);
      } else if (m_fid == 0) {

 //          printf("opening HDF5 file: %s - %d\n", file_path.c_str(), data_size );
          m_fid = H5Fopen( file_path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT );
          if (m_fid == 0) {
             printf("Error opening HDF5 file\n");
             return -1;
          }
          m_did = H5Dopen2(m_fid, data_set.c_str(), H5P_DEFAULT );
          if (m_did == 0) {
             printf("Error creating dataset\n");
             return -1;
          } else {
              int nFileOk = 0;
              hid_t dtype  = H5Dget_type(m_did);
              hid_t dspace = H5Dget_space(m_did);
              int rank = H5Sget_simple_extent_ndims(dspace);
              if ( H5Tequal(dtype, H5T_NATIVE_CHAR) > 0 && rank == 1 ) {     
                 hsize_t test_dims[1];
                 herr_t status  = H5Sget_simple_extent_dims(dspace, test_dims, NULL);
                 if (status != 1 || test_dims[0] != data_size) {
                    printf("HDF5: Dims don't match: %d, %d \n", (int)status, (int)test_dims[0] );
                    nFileOk = -1;
                 }
              } else {
                 printf("HDF5: Datatype and rank don't match, %d, %d \n", (int)dtype, rank);
                 nFileOk = -1;
              }

              if (nFileOk != 0) {
                  printf("HDF5: Existing file does not match requested attributes, \n");
                  printf("HDF5: recreating file from scratch. \n");
                  close_file();
                  remove(file_path.c_str());
                  return open_file();                  
             }            
          }
      } else {
         printf("open_file: file handle already set .\n");
      }

      return 0;

   }

   size_t KokkosHDF5Accessor::ReadFile(void * dest, const size_t dest_size) {
      size_t dataRead = 0;
      char* ptr = (char*)dest;
      hsize_t stepSize = min(dest_size, chunk_size);      
      if (open_file() == 0 && m_fid != 0) {
         for (int i = 0; i < dest_size; i+=stepSize) {
            hsize_t offset[1];
            hsize_t doffset[1] = {0};
            hsize_t count[1];
            offset[0] = i;
            count[0] = min(stepSize, dest_size-i);
            m_mid = H5Screate_simple(1, count, NULL);
            hid_t  fsid = H5Dget_space(m_did);
  //            printf("reading %d, %d \n", offset[0], count[0]);
            herr_t status = H5Sselect_hyperslab(fsid, H5S_SELECT_SET, offset, NULL, count, NULL);
            status = H5Sselect_hyperslab(m_mid, H5S_SELECT_SET, doffset, NULL, count, NULL);
            status = H5Dread(m_did, H5T_NATIVE_CHAR, m_mid, fsid, H5P_DEFAULT, &ptr[i]);
            if (status == 0) {
               dataRead += min(stepSize, dest_size-i);
            } else {
               printf("Error with read: %d \n", status);
               return dataRead;
            }
 //            printf("read complete: %d, %d \n", status, dataRead);
            H5Sclose(m_mid);
            H5Sclose(fsid);
         }
      }
      close_file();
      return dataRead;

   }
   
   size_t KokkosHDF5Accessor::WriteFile(const void * src, const size_t src_size) {
      size_t m_written = 0;
      hsize_t stride[1];
      hsize_t block[1];
      stride[0] = 1;
      block[0] = 1;
      hsize_t stepSize = min(chunk_size, src_size);
      char* ptr = (char*)src;
 //     printf("write file: %s, %d, %d \n", file_path.c_str(),  m_fid, src_size);
      if (open_file() == 0 && m_fid != 0) {
         for (int i = 0; i < src_size; i+=stepSize) {
            hsize_t offset[1];
            offset[0] = i;
            hsize_t count[1];
            count[0] = min(stepSize, (src_size - i));
            m_mid = H5Screate_simple(1, count, NULL);
            hid_t fsid = H5Dget_space(m_did);
            herr_t status = H5Sselect_hyperslab(fsid, H5S_SELECT_SET, offset, stride, count, block);
            if (status != 0) {
               printf("Error with write(selecting hyperslab): %d \n", status);
               return m_written;
            }
            hid_t pid = H5Pcreate(H5P_DATASET_XFER);
            status = H5Dwrite(m_did, H5T_NATIVE_CHAR, m_mid, fsid, pid, &ptr[i]);
            if (status == 0) {
               m_written+= min(stepSize, (src_size - i));
            } else {
               printf("Error with write: %d \n", status);
               return m_written;
            }
 //            printf("write complete: %d, %d \n", status, m_written);
            H5Sclose(m_mid);
            H5Sclose(fsid);
            H5Pclose(pid);
         }
      }
      close_file();
      return m_written;
   }
   void KokkosHDF5Accessor::close_file() {
      if (m_did != 0) {
         H5Dclose(m_did);
         m_did = 0;
      }
      if (m_fid != 0) {
         H5Fclose(m_fid);
         m_fid = 0;
      }
   }

   void KokkosHDF5Accessor::finalize() {
      close_file();
   }


   HDF5Space::HDF5Space() {

   }

   /**\brief  Allocate untracked memory in the space */
   void * HDF5Space::allocate( const size_t arg_alloc_size, const std::string & path ) const {
      KokkosHDF5Accessor * pAcc = new KokkosHDF5Accessor( arg_alloc_size, path );
      pAcc->initialize( path, "default_dataset" );
      return (void*)pAcc;

   }

   /**\brief  Deallocate untracked memory in the space */
   void HDF5Space::deallocate( void * const arg_alloc_ptr
                             , const size_t arg_alloc_size ) const {
       KokkosHDF5Accessor * pAcc = static_cast<KokkosHDF5Accessor*>(arg_alloc_ptr);

       if (pAcc) {
          pAcc->finalize();
          delete pAcc;
       }

   }
  

} // Experimental

} // Kokkos



namespace Kokkos {
namespace Impl {

#ifdef KOKKOS_DEBUG
SharedAllocationRecord< void , void >
SharedAllocationRecord< Kokkos::Experimental::HDF5Space , void >::s_root_record ;
#endif

void
SharedAllocationRecord< Kokkos::Experimental::HDF5Space , void >::
deallocate( SharedAllocationRecord< void , void > * arg_rec )
{
  delete static_cast<SharedAllocationRecord*>(arg_rec);
}

SharedAllocationRecord< Kokkos::Experimental::HDF5Space , void >::
~SharedAllocationRecord()
{
  #if defined(KOKKOS_ENABLE_PROFILING)
  if(Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::deallocateData(
      Kokkos::Profiling::SpaceHandle(Kokkos::Experimental::HDF5Space::name()),RecordBase::m_alloc_ptr->m_label,
      data(),size());
  }
  #endif

  m_space.deallocate( SharedAllocationRecord< void , void >::m_alloc_ptr
                    , SharedAllocationRecord< void , void >::m_alloc_size
                    );
}

SharedAllocationRecord< Kokkos::Experimental::HDF5Space , void >::
SharedAllocationRecord( const Kokkos::Experimental::HDF5Space & arg_space
                      , const std::string       & arg_label
                      , const size_t              arg_alloc_size
                      , const SharedAllocationRecord< void , void >::function_type arg_dealloc
                      )
  // Pass through allocated [ SharedAllocationHeader , user_memory ]
  // Pass through deallocation function
  : SharedAllocationRecord< void , void >
      (
#ifdef KOKKOS_DEBUG
      & SharedAllocationRecord< Kokkos::Experimental::HDF5Space , void >::s_root_record,
#endif
        reinterpret_cast<SharedAllocationHeader*>( arg_space.allocate( arg_alloc_size, arg_label ) )
      , arg_alloc_size
      , arg_dealloc
      )
  , m_space( arg_space )
{
#if defined(KOKKOS_ENABLE_PROFILING)
  if(Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::allocateData(Kokkos::Profiling::SpaceHandle(arg_space.name()),arg_label,data(),arg_alloc_size);
   }
#endif
  // Fill in the Header information
  RecordBase::m_alloc_ptr->m_record = static_cast< SharedAllocationRecord< void , void > * >( this );

  strncpy( RecordBase::m_alloc_ptr->m_label
          , arg_label.c_str()
          , SharedAllocationHeader::maximum_label_length
          );
  // Set last element zero, in case c_str is too long
  RecordBase::m_alloc_ptr->m_label[SharedAllocationHeader::maximum_label_length - 1] = (char) 0;
}

//----------------------------------------------------------------------------

void * SharedAllocationRecord< Kokkos::Experimental::HDF5Space , void >::
allocate_tracked( const Kokkos::Experimental::HDF5Space & arg_space
                , const std::string & arg_alloc_label
                , const size_t arg_alloc_size )
{
  if ( ! arg_alloc_size ) return (void *) 0 ;

  SharedAllocationRecord * const r =
    allocate( arg_space , arg_alloc_label , arg_alloc_size );

  RecordBase::increment( r );

  return r->data();
}

void SharedAllocationRecord< Kokkos::Experimental::HDF5Space , void >::
deallocate_tracked( void * const arg_alloc_ptr )
{
  if ( arg_alloc_ptr != 0 ) {
    SharedAllocationRecord * const r = get_record( arg_alloc_ptr );

    RecordBase::decrement( r );
  }
}

void * SharedAllocationRecord< Kokkos::Experimental::HDF5Space , void >::
reallocate_tracked( void * const arg_alloc_ptr
                  , const size_t arg_alloc_size )
{
  SharedAllocationRecord * const r_old = get_record( arg_alloc_ptr );
  SharedAllocationRecord * const r_new = allocate( r_old->m_space , r_old->get_label() , arg_alloc_size );

  RecordBase::increment( r_new );
  RecordBase::decrement( r_old );

  return r_new->data();
}

SharedAllocationRecord< Kokkos::Experimental::HDF5Space , void > *
SharedAllocationRecord< Kokkos::Experimental::HDF5Space , void >::get_record( void * alloc_ptr )
{
  typedef SharedAllocationHeader  Header ;
  typedef SharedAllocationRecord< Kokkos::Experimental::HDF5Space , void >  RecordHost ;

  SharedAllocationHeader const * const head   = alloc_ptr ? Header::get_header( alloc_ptr ) : (SharedAllocationHeader *)0 ;
  RecordHost                   * const record = head ? static_cast< RecordHost * >( head->m_record ) : (RecordHost *) 0 ;

  if ( ! alloc_ptr || record->m_alloc_ptr != head ) {
    Kokkos::Impl::throw_runtime_exception( std::string("Kokkos::Impl::SharedAllocationRecord< Kokkos::Experimental::HDF5Space , void >::get_record ERROR" ) );
  }

  return record ;
}

// Iterate records to print orphaned memory ...
void SharedAllocationRecord< Kokkos::Experimental::HDF5Space , void >::
print_records( std::ostream & s , const Kokkos::Experimental::HDF5Space & , bool detail )
{
#ifdef KOKKOS_DEBUG
  SharedAllocationRecord< void , void >::print_host_accessible_records( s , "HDF5Space" , & s_root_record , detail );
#else
  throw_runtime_exception("SharedAllocationRecord<HDF5Space>::print_records only works with KOKKOS_DEBUG enabled");
#endif
}

} // namespace Experimental
} // namespace Kokkos

