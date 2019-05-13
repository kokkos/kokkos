
#include "Kokkos_Core.hpp"
#include "Kokkos_ExternalIOInterface.hpp"

namespace Kokkos {

namespace Experimental {

   std::string KokkosIOAccessor::resolve_path( std::string path, std::string default_ ) {

      std::string sFullPath = default_;
      size_t pos = path.find("/");
      if ( pos >= 0 && pos < path.length() ) {    // only use the default if there is no path info in the path...
         sFullPath = path;
      } else {
         sFullPath += (std::string)"/";
         sFullPath += path;
      }

      return sFullPath;

   }

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
      const KokkosIOInterface * pDataII = reinterpret_cast<const KokkosIOInterface*>(pData-1);
      Kokkos::Experimental::KokkosIOAccessor * pAcc = pDataII->pAcc;
      if (pAcc) {
         pAcc->ReadFile( dst, size_ );
      }
   }

   void KokkosIOConfigurationManager::load_configuration ( std::string path ) {

      if (path.length() == 0 ) {
         printf("WARNING:KOKKOS_IO_CONFIG not set. loading default setting for HDF5 files access. \n");
         return;
      }

      boost::property_tree::ptree pt;
      boost::property_tree::json_parser::read_json( path, pt );

      for (auto & ar: pt) {         
         boost::property_tree::ptree ptII = ar.second;
         std::string name = ptII.get<std::string>("name");
         m_config_list[name] = ptII;
      }

   }

   KokkosIOConfigurationManager * KokkosIOConfigurationManager::get_instance() {
      if (KokkosIOConfigurationManager::m_Inst == nullptr) {
         KokkosIOConfigurationManager::m_Inst = new KokkosIOConfigurationManager;
         std::string path;
         char * config = std::getenv( "KOKKOS_IO_CONFIG" );
         if (config != nullptr)
            path = config;
         printf("loading IOConfigurationManager: %s\n", path.c_str());
         KokkosIOConfigurationManager::m_Inst->load_configuration(path);
      }
      return KokkosIOConfigurationManager::m_Inst;
   }

   KokkosIOConfigurationManager * KokkosIOConfigurationManager::m_Inst = nullptr;


} // Experimental

} // Kokkos

