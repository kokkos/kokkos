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
#include "Kokkos_Core.hpp"
#include "Kokkos_StdFileSpace.hpp"
#include "sys/stat.h"

namespace Kokkos {

namespace Experimental {

   int KokkosStdFileAccessor::initialize( const std::string & filepath ) { 

       file_path = filepath;
 //       printf("Initializing StdFile properties: %s - %d\n", file_path.c_str(), data_size );

   }

   bool KokkosStdFileAccessor::open_file( int read_write ) { 

       if (file_strm.is_open())
          close_file();

       if ( read_write == KokkosStdFileAccessor::WRITE_FILE ) {
//            printf("open StdFile file for write: %s - %d\n", file_path.c_str(), data_size );
            file_strm.open( file_path.c_str(), std::ios::out | std::ios::trunc | std::ios::binary );
      } else if (read_write == KokkosStdFileAccessor::READ_FILE ) { 
//            printf("open StdFile file for reading: %s - %d\n", file_path.c_str(), data_size );
            file_strm.open( file_path.c_str(), std::ios::in | std::ios::binary );
      } else {
         printf("open_file: incorrect read write parameter specified .\n");
         return -1;
      }

      return file_strm.is_open();

   }

   size_t KokkosStdFileAccessor::ReadFile(void * dest, const size_t dest_size) {
      size_t dataRead = 0;
      char* ptr = (char*)dest;
      if (open_file(KokkosStdFileAccessor::READ_FILE)) {
         while ( !file_strm.eof() && dataRead < dest_size ) {
            file_strm.read( &ptr[dataRead], dest_size );
            dataRead += file_strm.gcount();
         }
      }
      close_file();
      if (dataRead < dest_size) {
         printf("StdFile: less data available than requested \n");
      }
      return dataRead;

   }
   
   size_t KokkosStdFileAccessor::WriteFile(const void * src, const size_t src_size) {
      size_t m_written = 0;
      char* ptr = (char*)src;
      if (open_file(KokkosStdFileAccessor::WRITE_FILE) ) {
          file_strm.write(&ptr[0], src_size);
          if (!file_strm.fail())
             m_written = src_size;
      }
      close_file();
      if (m_written != src_size) {
         printf("StdFile: write failed \n");
      }
      return m_written;
   }
   void KokkosStdFileAccessor::close_file() {
      if (file_strm.is_open()) {
         file_strm.close();
      }
   }

   void KokkosStdFileAccessor::finalize() {
      close_file();
   }

   std::string StdFileSpace::s_default_path = "./";

   StdFileSpace::StdFileSpace() {

   }

   /**\brief  Allocate untracked memory in the space */
   void * StdFileSpace::allocate( const size_t arg_alloc_size, const std::string & path ) const {
      std::string sFullPath = s_default_path;
      size_t pos = path.find("/");
      printf("adding file accessor: %s, %s, %d \n", s_default_path.c_str(), path.c_str(), (int)pos );
      if ( pos >= 0 && pos < path.length() ) {    // only use the default if there is no path info in the path...
         sFullPath = path;
      } else {
         sFullPath += (std::string)"/";
         sFullPath += path;
      }
      printf("final path: %s \n", sFullPath.c_str());
      KokkosStdFileAccessor * pAcc = new KokkosStdFileAccessor( arg_alloc_size, sFullPath );
      pAcc->initialize( sFullPath );
      return (void*)pAcc;

   }

   /**\brief  Deallocate untracked memory in the space */
   void StdFileSpace::deallocate( void * const arg_alloc_ptr
                             , const size_t arg_alloc_size ) const {
       KokkosStdFileAccessor * pAcc = static_cast<KokkosStdFileAccessor*>(arg_alloc_ptr);

       if (pAcc) {
          pAcc->finalize();
          delete pAcc;
       }

   }
  
   void StdFileSpace::restore_all_views() {
      typedef Kokkos::Impl::SharedAllocationRecord<void,void> base_record;
      Kokkos::Impl::MirrorTracker * pList = base_record::get_filtered_mirror_list( (std::string)name() );
      while (pList != nullptr) {
         Kokkos::Impl::DeepCopy< Kokkos::HostSpace, Kokkos::Experimental::StdFileSpace, Kokkos::DefaultHostExecutionSpace >
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
   
   void StdFileSpace::restore_view(const std::string lbl) {
      typedef Kokkos::Impl::SharedAllocationRecord<void,void> base_record;
      Kokkos::Impl::MirrorTracker * pRes = base_record::get_filtered_mirror_entry( (std::string)name(), lbl );
      if (pRes != nullptr) {
         Kokkos::Impl::DeepCopy< Kokkos::HostSpace, Kokkos::Experimental::StdFileSpace, Kokkos::DefaultHostExecutionSpace >
                        (((base_record*)pRes->src)->data(), ((base_record*)pRes->dst)->data(), ((base_record*)pRes->src)->size());
         delete pRes;
      }
   }
  
   void StdFileSpace::checkpoint_views() {
      typedef Kokkos::Impl::SharedAllocationRecord<void,void> base_record;
      Kokkos::Impl::MirrorTracker * pList = base_record::get_filtered_mirror_list( (std::string)name() );
      if (pList == nullptr) {
         printf("memspace %s returned empty list of checkpoint views \n", name());
      }
      while (pList != nullptr) {
      typedef Kokkos::Impl::SharedAllocationRecord<void,void> base_record;
         Kokkos::Impl::DeepCopy< Kokkos::Experimental::StdFileSpace, Kokkos::HostSpace, Kokkos::DefaultHostExecutionSpace >
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
   void StdFileSpace::set_default_path( const std::string path ) {

      StdFileSpace::s_default_path = path;

   }
  
} // Experimental

} // Kokkos



namespace Kokkos {
namespace Impl {

#ifdef KOKKOS_DEBUG
SharedAllocationRecord< void , void >
SharedAllocationRecord< Kokkos::Experimental::StdFileSpace , void >::s_root_record ;
#endif

void
SharedAllocationRecord< Kokkos::Experimental::StdFileSpace , void >::
deallocate( SharedAllocationRecord< void , void > * arg_rec )
{
  delete static_cast<SharedAllocationRecord*>(arg_rec);
}

SharedAllocationRecord< Kokkos::Experimental::StdFileSpace , void >::
~SharedAllocationRecord()
{
  #if defined(KOKKOS_ENABLE_PROFILING)
  if(Kokkos::Profiling::profileLibraryLoaded()) {
      Kokkos::Profiling::deallocateData(
      Kokkos::Profiling::SpaceHandle(Kokkos::Experimental::StdFileSpace::name()),RecordBase::m_alloc_ptr->m_label,
      data(),size());
  }
  #endif

  m_space.deallocate( SharedAllocationRecord< void , void >::m_alloc_ptr
                    , SharedAllocationRecord< void , void >::m_alloc_size
                    );
}

SharedAllocationRecord< Kokkos::Experimental::StdFileSpace , void >::
SharedAllocationRecord( const Kokkos::Experimental::StdFileSpace & arg_space
                      , const std::string       & arg_label
                      , const size_t              arg_alloc_size
                      , const SharedAllocationRecord< void , void >::function_type arg_dealloc
                      )
  // Pass through allocated [ SharedAllocationHeader , user_memory ]
  // Pass through deallocation function
  : SharedAllocationRecord< void , void >
      (
#ifdef KOKKOS_DEBUG
      & SharedAllocationRecord< Kokkos::Experimental::StdFileSpace , void >::s_root_record,
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

void * SharedAllocationRecord< Kokkos::Experimental::StdFileSpace , void >::
allocate_tracked( const Kokkos::Experimental::StdFileSpace & arg_space
                , const std::string & arg_alloc_label
                , const size_t arg_alloc_size )
{
  if ( ! arg_alloc_size ) return (void *) 0 ;

  SharedAllocationRecord * const r =
    allocate( arg_space , arg_alloc_label , arg_alloc_size );

  RecordBase::increment( r );

  return r->data();
}

void SharedAllocationRecord< Kokkos::Experimental::StdFileSpace , void >::
deallocate_tracked( void * const arg_alloc_ptr )
{
  if ( arg_alloc_ptr != 0 ) {
    SharedAllocationRecord * const r = get_record( arg_alloc_ptr );

    RecordBase::decrement( r );
  }
}

void * SharedAllocationRecord< Kokkos::Experimental::StdFileSpace , void >::
reallocate_tracked( void * const arg_alloc_ptr
                  , const size_t arg_alloc_size )
{
  SharedAllocationRecord * const r_old = get_record( arg_alloc_ptr );
  SharedAllocationRecord * const r_new = allocate( r_old->m_space , r_old->get_label() , arg_alloc_size );

  RecordBase::increment( r_new );
  RecordBase::decrement( r_old );

  return r_new->data();
}

SharedAllocationRecord< Kokkos::Experimental::StdFileSpace , void > *
SharedAllocationRecord< Kokkos::Experimental::StdFileSpace , void >::get_record( void * alloc_ptr )
{
  typedef SharedAllocationHeader  Header ;
  typedef SharedAllocationRecord< Kokkos::Experimental::StdFileSpace , void >  RecordHost ;

  SharedAllocationHeader const * const head   = alloc_ptr ? Header::get_header( alloc_ptr ) : (SharedAllocationHeader *)0 ;
  RecordHost                   * const record = head ? static_cast< RecordHost * >( head->m_record ) : (RecordHost *) 0 ;

  if ( ! alloc_ptr || record->m_alloc_ptr != head ) {
    Kokkos::Impl::throw_runtime_exception( std::string("Kokkos::Impl::SharedAllocationRecord< Kokkos::Experimental::StdFileSpace , void >::get_record ERROR" ) );
  }

  return record ;
}

// Iterate records to print orphaned memory ...
void SharedAllocationRecord< Kokkos::Experimental::StdFileSpace , void >::
print_records( std::ostream & s , const Kokkos::Experimental::StdFileSpace & , bool detail )
{
#ifdef KOKKOS_DEBUG
  SharedAllocationRecord< void , void >::print_host_accessible_records( s , "StdFileSpace" , & s_root_record , detail );
#else
  throw_runtime_exception("SharedAllocationRecord<StdFileSpace>::print_records only works with KOKKOS_DEBUG enabled");
#endif
}

} // namespace Experimental
} // namespace Kokkos

