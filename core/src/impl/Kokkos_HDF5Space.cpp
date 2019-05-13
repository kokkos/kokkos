#include "Kokkos_Core.hpp"
#include "Kokkos_HDF5Space.hpp"

#ifdef KOKKOS_HDF5_ENABLE_MPI
   #include "mpi.h"
#endif

namespace Kokkos {

namespace Experimental {


   KokkosHDF5ConfigurationManager::OperationPrimitive * KokkosHDF5ConfigurationManager::resolve_variable( 
                      std::string data, std::map<const std::string, size_t> & var_map ) {
   
      size_t val = var_map[data];
      printf("variable [%s] returned %d\n", data.c_str(), val);
      return new KokkosHDF5ConfigurationManager::OperationPrimitive(val);
   }

   KokkosHDF5ConfigurationManager::OperationPrimitive *
   KokkosHDF5ConfigurationManager::resolve_arithmetic( std::string data, 
                                                       std::map<const std::string, size_t> & var_map ) {
      KokkosHDF5ConfigurationManager::OperationPrimitive * lhs_op = nullptr;
      for ( size_t n = 0; n< data.length(); ) {
         printf(" resolve_arithmetic: %d \n", n);
         char c = data.at(n);
         if ( c >= 0x30 && c <= 0x39 ) {
            std::string cur_num = "";
            while ( c >= 0x30 && c <= 0x39 ) {
               cur_num += data.substr(n,1); n++;
               if (n < data.length()) {
                  c = data.at(n);
               } else {
                  break;
               }
            }
            lhs_op = new KokkosHDF5ConfigurationManager::OperationPrimitive((std::atoi(cur_num.c_str())));
         } else if (c == '(') {
            size_t end_pos = data.find_first_of(')',n);
            if (end_pos >= 0 && end_pos < data.length()) {
               printf("calling resolving arithmetic: %s \n", data.substr(n+1,end_pos-n-1).c_str());
               lhs_op = resolve_arithmetic ( data.substr(n+1,end_pos-n-1), var_map );
            } else {
               printf("syntax error in arithmetic: %s, at %d \n", data.c_str(), n);
               return nullptr;
            }
            n = end_pos+1;
         } else if (c == '{') {
            size_t end_pos = data.find_first_of('}',n);
            if (end_pos >= 0 && end_pos < data.length()) {
               printf("resolving variable: %s \n", data.substr(n+1,end_pos-n-1).c_str());
               lhs_op = resolve_variable( data.substr(n+1,end_pos-n-1), var_map );
            } else {
               printf("syntax error in variable name: %s, at %d \n", data.c_str(), n);
               return nullptr;
            }
            n = end_pos+1;
         } else {
            printf("parsing rhs: %s \n", data.substr(n,1).c_str());
            KokkosHDF5ConfigurationManager::OperationPrimitive * op = 
                        KokkosHDF5ConfigurationManager::OperationPrimitive::parse_operator(data.substr(n,1),lhs_op);
            n++;
            op->set_right_opp( resolve_arithmetic ( data.substr(n), var_map ) );
            return op;
         }
      
      }
      return lhs_op;
   
   }

   void KokkosHDF5ConfigurationManager::set_param_list( boost::property_tree::ptree l_config, int data_scope, 
                       std::string param_name, hsize_t output [], std::map<const std::string, size_t> & var_map ) {
      printf("set_param_list: %s \n", param_name.c_str());
      for ( auto & param_list : l_config ) {
          if ( param_list.first == param_name ) {
              int n = 0;    
              for (auto & param : param_list.second ) {
                 printf("processing param list: %s\n", param.second.get_value<std::string>().c_str());
                 KokkosHDF5ConfigurationManager::OperationPrimitive * opp = resolve_arithmetic( param.second.get_value<std::string>(), var_map ); 
                 if (opp != nullptr) {
                     output[n++] = opp->evaluate();
                     delete opp;
                 } else {
                     output[n++] = 0;
                 }  
              }
              break;
          }
      }
   }


   #define min(X,Y) (X > Y) ? Y : X

   int KokkosHDF5Accessor::initialize( const size_t size_, const std::string & filepath, const std::string & data_set_ ) { 

       file_path = filepath;
       data_set = data_set_;
       data_size = size_;
       file_block[0] = data_size;
       data_extents[0] = data_size;
       local_extents[0] = data_size;
   }

   int KokkosHDF5Accessor::initialize( const size_t size_, const std::string & filepath,
                                       KokkosHDF5ConfigurationManager config_ ) { 

       for (int i = 0; i < 4; i++) {
         file_count[i] = 0;
         file_offset[i] = 0;
         file_stride[i] = 0;
         file_block[i] = 0;
         data_extents[i] = 0;
         local_extents[i] = 0;
       }
       boost::property_tree::ptree l_config = config_.get_config()->get_child("Layout_Config");
       file_path = filepath;
       data_size = size_;
       std::map<const std::string, size_t> var_list;
       var_list["DATA_SIZE"] = data_size;
       var_list["MPI_SIZE"] = mpi_size;
       var_list["MPI_RANK"] = mpi_rank;
       config_.set_param_list( l_config, 0, "data_extents", data_extents, var_list );
       var_list["DATA_EXTENTS_1"] = (size_t)data_extents[0];
       var_list["DATA_EXTENTS_2"] = (size_t)data_extents[1];
       var_list["DATA_EXTENTS_3"] = (size_t)data_extents[2];
       var_list["DATA_EXTENTS_4"] = (size_t)data_extents[3];
       config_.set_param_list( l_config, 0, "local_extents", local_extents, var_list );
       var_list["LOCAL_EXTENTS_1"] = (size_t)local_extents[0];
       var_list["LOCAL_EXTENTS_2"] = (size_t)local_extents[1];
       var_list["LOCAL_EXTENTS_3"] = (size_t)local_extents[2];
       var_list["LOCAL_EXTENTS_4"] = (size_t)local_extents[3];
       config_.set_param_list( l_config, 0, "count", file_count, var_list );
       config_.set_param_list( l_config, 0, "offset", file_offset, var_list );
       config_.set_param_list( l_config, 0, "stride", file_stride, var_list );
       config_.set_param_list( l_config, 0, "block", file_block, var_list );
       rank = l_config.get<int>("rank");
       data_set = l_config.get<std::string>("data_set");
       m_layout = config_.get_layout();
       m_is_initialized = true;
   }

   int KokkosHDF5Accessor::open_file( ) { 

       std::string sFullPath = KokkosIOAccessor::resolve_path( file_path, Kokkos::Experimental::HDF5Space::s_default_path );
       if (m_fid == 0  && !file_exists(sFullPath)) {
          hid_t pid = H5Pcreate(H5P_FILE_ACCESS);
          printf("creating HDF5 file: %s \n", sFullPath.c_str() );
          m_fid = H5Fcreate( sFullPath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, pid );
          H5Pclose(pid);

          if (m_fid == 0) {
              printf("Error creating HDF5 file\n");
              return -1;
          }

          printf ("creating file set: %s, %d, %d, %d, %d \n", data_set.c_str(), rank, 
                    data_extents[0], data_extents[1], data_extents[2]);
          hid_t fsid = H5Screate_simple(rank, data_extents, NULL);
          pid = H5Pcreate(H5P_DATASET_CREATE);
          m_did = H5Dcreate(m_fid, data_set.c_str(), H5T_NATIVE_CHAR, fsid, 
                             H5P_DEFAULT, pid, H5P_DEFAULT );
          if (m_did == 0) {
              printf("Error creating file set\n");
              return -1;
          }
          H5Pclose(pid);
          H5Sclose(fsid);
      } else if (m_fid == 0) {
          m_fid = H5Fopen( sFullPath.c_str(), H5F_ACC_RDWR, H5P_DEFAULT );
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
              int rank_ = H5Sget_simple_extent_ndims(dspace);
              printf("file opened for read: %s, %s, %d \n", sFullPath.c_str(), data_set.c_str(), rank_);
              if ( H5Tequal(dtype, H5T_NATIVE_CHAR) > 0 && rank_ == rank ) {     
                 hsize_t test_dims[4] = {0,0,0,0};
                 herr_t status  = H5Sget_simple_extent_dims(dspace, test_dims, NULL);
                 if (status != rank || 
                     test_dims[0] != data_extents[0] && 
                     test_dims[1] != data_extents[1] && 
                     test_dims[2] != data_extents[2] && 
                     test_dims[3] != data_extents[3] ) {
                     printf("HDF5: Dims don't match: %d: %d,%d,%d,%d \n", (int)status, (int)test_dims[0],
                                                                                       (int)test_dims[1],
                                                                                       (int)test_dims[2],
                                                                                       (int)test_dims[3] );
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
                  remove(sFullPath.c_str());
                  return open_file();                  
             }            
          }
      } else {
         printf("open_file: file handle already set: %s.\n", sFullPath.c_str());
      }

      return 0;

   }

   size_t KokkosHDF5Accessor::ReadFile_impl(void * dest, const size_t dest_size) {
      size_t dataRead = 0;
      char* ptr = (char*)dest;
      hsize_t stepSize = dest_size;      
      //hsize_t stepSize = min(dest_size, chunk_size);      
      if (open_file() == 0 && m_fid != 0) {
           printf ("reading data set: %s, %d, %d, %d, %d \n", data_set.c_str(), rank, 
                     local_extents[0], local_extents[1], local_extents[2]);
            m_mid = H5Screate_simple(rank, local_extents, NULL);
            hid_t  fsid = H5Dget_space(m_did);
            herr_t status = H5Sselect_hyperslab(fsid, H5S_SELECT_SET, file_offset, file_stride, file_count, file_block);

         printf("[R] hyperslab: %d, %d, %d, %d \n", file_offset[0], file_stride[0], file_count[0], file_block[0] );
         // for this to work, the unused file count indicies have to be 1.
         for (int i = 0; i < file_count[0]; i++) {
            for (int j = 0; j < file_count[1]; j++) {
               for (int k = 0; k < file_count[2]; k++) {
                  for (int l = 0; l < file_count[3]; l++) {
                     size_t offset_ = i*file_block[0] + j*file_block[1] + k*file_block[2] + l*file_block[3];
                     hsize_t l_off[4] = {i*file_block[0],j*file_block[2],k*file_block[2],l*file_block[3]};
                     hsize_t l_stride[4] = {1,1,1,1};
                     hsize_t l_count[4] = {1,1,1,1};
                     
                     status = H5Sselect_hyperslab(m_mid, H5S_SELECT_SET, l_off, l_stride, l_count, file_block);
                     status = H5Dread(m_did, H5T_NATIVE_CHAR, m_mid, fsid, H5P_DEFAULT, &ptr[offset_]);
                     if (status == 0) {
                        int read_ = 1;
                        for (int r = 0; r < rank; r++) {
                           read_ = read_ * file_block[r];
                        }
                        dataRead += read_;
                     } else {
                        printf("Error with read: %d \n", status);
                        close_file();
                        return dataRead;
                     }
                 }
               }
            }
         }
         H5Sclose(m_mid);
         H5Sclose(fsid);
      }
      close_file();
      return dataRead;

   }
   
   size_t KokkosHDF5Accessor::WriteFile_impl(const void * src, const size_t src_size) {
      size_t m_written = 0;
      hsize_t stepSize = src_size;
      char* ptr = (char*)src;
      if (open_file() == 0 && m_fid != 0) {
         printf ("creating data set: %s, %d, %d, %d, %d \n", data_set.c_str(), rank, 
                   local_extents[0], local_extents[1], local_extents[2]);
         printf("[W] hyperslab: %d, %d, %d, %d \n", file_offset[0], file_stride[0], file_count[0], file_block[0] );
         m_mid = H5Screate_simple(rank, local_extents, NULL);
         hid_t fsid = H5Dget_space(m_did);
         herr_t status = H5Sselect_hyperslab(fsid, H5S_SELECT_SET, file_offset, file_stride, file_count, file_block);
         if (status != 0) {
             printf("Error with write(selecting hyperslab): %d \n", status);
             close_file();
             return 0;
         }
         // for this to work, the unused file count indicies have to be 1.
         for (int i = 0; i < file_count[0]; i++) {
            for (int j = 0; j < file_count[1]; j++) {
               for (int k = 0; k < file_count[2]; k++) {
                  for (int l = 0; l < file_count[3]; l++) {           
                     size_t offset_ = i*file_block[0] + j*file_block[1] + k*file_block[2] + l*file_block[3];
                     printf("write file: %d, %d, %d, %d -- %d \n", i, j, k, l, offset_);
                     hid_t pid = H5Pcreate(H5P_DATASET_XFER);
                     status = H5Dwrite(m_did, H5T_NATIVE_CHAR, m_mid, fsid, pid, &ptr[offset_]);
                     if (status == 0) {
                        int written_ = 1;
                        for (int r = 0; r < rank; r++) {
                           written_ = written_ * file_block[r];
                        }
                        m_written+= written_;
                    } else {
                       printf("Error with write: %d \n", status);
                       close_file();
                       return m_written;
                    }
                    H5Pclose(pid);
                 }         
              }
           }
         }
         H5Sclose(m_mid);
         H5Sclose(fsid);
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

   std::string HDF5Space::s_default_path = "./";

   HDF5Space::HDF5Space() {

   }
 
   std::map<const std::string, KokkosHDF5Accessor> HDF5Space::m_accessor_map;

   /**\brief  Allocate untracked memory in the space */
   void * HDF5Space::allocate( const size_t arg_alloc_size, const std::string & path ) const {

      printf("allocating HDF5 file accessor: %d, %s \n", arg_alloc_size, path.c_str());
      KokkosHDF5Accessor acc = m_accessor_map[path];
      if (!acc.is_initialized() ) {
         printf ("creating new accessor \n");
         boost::property_tree::ptree pConfig = KokkosIOConfigurationManager::get_instance()->get_config(path);
         if ( pConfig.size() > 0 ) {
             printf("creating accessor from ptree \n");
             acc.initialize( arg_alloc_size, path, KokkosHDF5ConfigurationManager ( pConfig ) );
         }
         else {
             printf("creating default accessor \n");
             acc.initialize( arg_alloc_size, path, "default_dataset" );
         }
         m_accessor_map[path] = acc;
      }
      KokkosHDF5Accessor * pAcc = new KokkosHDF5Accessor( acc, arg_alloc_size ); 
      
      KokkosIOInterface * pInt = new KokkosIOInterface;
      pInt->pAcc = static_cast<KokkosIOAccessor*>(pAcc);
      return reinterpret_cast<void*>(pInt);

   }

   /**\brief  Deallocate untracked memory in the space */
   void HDF5Space::deallocate( void * const arg_alloc_ptr
                             , const size_t arg_alloc_size ) const {
       const KokkosIOInterface * pInt = reinterpret_cast<KokkosIOInterface *>(arg_alloc_ptr);
       if (pInt) {
          KokkosHDF5Accessor * pAcc = static_cast<KokkosHDF5Accessor*>(pInt->pAcc);

          if (pAcc) {
             pAcc->finalize();
             delete pAcc;
          }
          delete pInt;
      }

   }
   
   void HDF5Space::restore_all_views() {
      typedef Kokkos::Impl::SharedAllocationRecord<void,void> base_record;
      Kokkos::Impl::MirrorTracker * pList = base_record::get_filtered_mirror_list( (std::string)name() );
      while (pList != nullptr) {
         Kokkos::Impl::DeepCopy< Kokkos::HostSpace, Kokkos::Experimental::HDF5Space, Kokkos::DefaultHostExecutionSpace >
                        (((base_record*)pList->src)->data(), ((base_record*)pList->dst)->data(), ((base_record*)pList->src)->size());
         // delete the records along the way...
         if (pList->pNext == nullptr) {
            delete pList;
            pList = nullptr;
         } else {
            pList = pList->pNext;
            delete pList->pPrev;
         }
      }
   }
   
   void HDF5Space::restore_view(const std::string lbl) {
      typedef Kokkos::Impl::SharedAllocationRecord<void,void> base_record;
      Kokkos::Impl::MirrorTracker * pRes = base_record::get_filtered_mirror_entry( (std::string)name(), lbl );
      if (pRes != nullptr) {
         Kokkos::Impl::DeepCopy< Kokkos::HostSpace, Kokkos::Experimental::HDF5Space, Kokkos::DefaultHostExecutionSpace >
                        (((base_record*)pRes->src)->data(), ((base_record*)pRes->dst)->data(), ((base_record*)pRes->src)->size());
         delete pRes;
      }
   }
  
   void HDF5Space::checkpoint_views() {
      typedef Kokkos::Impl::SharedAllocationRecord<void,void> base_record;
      Kokkos::Impl::MirrorTracker * pList = base_record::get_filtered_mirror_list( (std::string)name() );
      if (pList == nullptr) {
         printf("memspace %s returned empty list of checkpoint views \n", name());
      }
      while (pList != nullptr) {
         Kokkos::Impl::DeepCopy< Kokkos::Experimental::HDF5Space, Kokkos::HostSpace, Kokkos::DefaultHostExecutionSpace >
                        (((base_record*)pList->dst)->data(), ((base_record*)pList->src)->data(), ((base_record*)pList->src)->size());
         // delete the records along the way...
         if (pList->pNext == nullptr) {
            delete pList;
            pList = nullptr;
         } else {
            pList = pList->pNext;
            delete pList->pPrev;
         }
      }
   }
   void HDF5Space::set_default_path( const std::string path ) {
      HDF5Space::s_default_path = path;
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

