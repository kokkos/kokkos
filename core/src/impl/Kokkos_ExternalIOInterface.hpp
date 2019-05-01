

#ifndef __KOKKOS_EXTERNAL_IO_
#define __KOKKOS_EXTERNAL_IO_

#include <cstring>
#include <string>
#include <iosfwd>
#include <typeinfo>

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Concepts.hpp>
#include <Kokkos_MemoryTraits.hpp>
#include <impl/Kokkos_SharedAlloc.hpp>

namespace Kokkos {

namespace Experimental {

class KokkosIOAccessor  {

public:
   size_t data_size;   
   bool is_contiguous;
   std::string file_path;

   KokkosIOAccessor() : data_size(0),
                        is_contiguous(true),
                        file_path("") {
   }
   KokkosIOAccessor(const size_t size, const std::string & path, bool cont_ = true ) : data_size(size),
                                                                    is_contiguous(cont_),
                                                                    file_path(path) {
   }

   KokkosIOAccessor( const KokkosIOAccessor & rhs ) = default;
   KokkosIOAccessor( KokkosIOAccessor && rhs ) = default;
   KokkosIOAccessor & operator = ( KokkosIOAccessor && ) = default;
   KokkosIOAccessor & operator = ( const KokkosIOAccessor & ) = default;

   size_t ReadFile(void * dest, const size_t dest_size) {
      return ReadFile_impl( dest, dest_size );
   }
   
   size_t WriteFile(const void * src, const size_t src_size) {
      return WriteFile_impl( src, src_size );
   }

   virtual size_t ReadFile_impl(void * dest, const size_t dest_size) = 0;
      
   virtual size_t WriteFile_impl(const void * src, const size_t src_size) = 0;
      
   virtual ~KokkosIOAccessor() {
   }

   static void transfer_from_host ( void * dst, const void * src, size_t size_ );
   static void transfer_to_host ( void * dst, const void * src, size_t size_ );
};

struct KokkosIOInterface : Kokkos::Impl::SharedAllocationHeader {
   KokkosIOAccessor * pAcc;
};

} // Experimental

} // Kokkos


#endif // __KOKKOS_EXTERNAL_IO
