
#include "Kokkos_Core.hpp"
#include "Kokkos_ExternalIOInterface.hpp"

namespace Kokkos {

namespace Experimental {


   // Copy from host memory space to designated IO buffer (dst is an instance of KokkosIOAccessor offset by SharedAllocationHeader)
   //                                                      src is the data() pointer from the souce view.
   void KokkosIOAccessor::transfer_from_host ( void * dst, const void * src, size_t size_ )  {

      Kokkos::Impl::SharedAllocationHeader * pData = reinterpret_cast<Kokkos::Impl::SharedAllocationHeader*>(dst);
      KokkosIOInterface * pDataII = reinterpret_cast<KokkosIOInterface*>(pData-1);
      Kokkos::Experimental::KokkosIOAccessor * pAcc = pDataII->pAcc;

      if (pAcc) {
         pAcc->WriteFile( src, size_ );   // virtual method implemented by specific IO interface
      }
   }
   

   // Copy from IO buffer to host memory space  (dst is the data() pointer from the target view
   //                                            src is an instance of KokkosIOAccessor offset by SharedAllocationHeader)
   void KokkosIOAccessor::transfer_to_host ( void * dst, const void * src, size_t size_ ) {

      const Kokkos::Impl::SharedAllocationHeader * pData = reinterpret_cast<const Kokkos::Impl::SharedAllocationHeader*>(src);
      if (pData != nullptr) {
         const KokkosIOInterface * pDataII = reinterpret_cast<const KokkosIOInterface*>(pData-1);
         if ( pDataII != nullptr) {
            Kokkos::Experimental::KokkosIOAccessor * pAcc = pDataII->pAcc;
            if (pAcc) {
               // printf("calling read: %08x, %08x, %ld \n", (unsigned long)pAcc,  (unsigned long)dst, size_ );
               pAcc->ReadFile( dst, size_ );
            }
         } else {
            printf("IOInterface src is empty!!!\n");
         }
      } else {
         printf("Allocation header for src is empty!!!\n");
      }
   }


} // Experimental

} // Kokkos

