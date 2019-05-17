#ifndef __KOKKOS_DIRECTORY_MANAGER__
#define __KOKKOS_DIRECTORY_MANAGER__

#include <errno.h>
#include <string>
#include <sys/stat.h>
#include <sstream>

namespace Kokkos {
namespace Experimental {

template<class MemorySpace>
struct DirectoryManager {

   template<typename D>
   inline static constexpr std::string ensure_directory_exists( const std::string dir, D d ) {
      printf("last call creating dir: %s \n", dir.c_str());
      mkdir(dir.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
      if( errno == EEXIST || errno == 0 ) {
         std::string path = dir;
         std::stringstream iter_num;
         iter_num << d << "/";
         path += iter_num.str();
         printf("final dir: %s \n", path.c_str());
         mkdir(path.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
         if( errno == EEXIST || errno == 0 ) {
            return path;
         } else {
            printf("WARNING: Error creating path: %s, %d \n", path.c_str(), errno);
            return "";
         }
      } else {
         printf("WARNING: Error creating path: %s, %d \n", dir.c_str(), errno);
         return "";
      }
   }

   template<typename D, typename ...Dargs>
   inline static constexpr std::string ensure_directory_exists( const std::string dir, D d, Dargs... dargs) {
      printf("recursive dir call: %s \n", dir.c_str());
      mkdir(dir.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
      if( errno == EEXIST || errno == 0 ) {
         std::string path = dir;
         std::stringstream iter_num;
         iter_num << "/" << d << "/";
         path += iter_num.str();
         return ensure_directory_exists( path, dargs... );
      } else {
         printf("WARNING: Error creating path: %s, %d \n", dir.c_str(), errno);
         return "";
      }
   }

   template<class ... Dargs>
   inline static constexpr int set_checkpoint_directory(std::string dir, Dargs ...dargs ) {
      std::string path = ensure_directory_exists( dir, dargs... );
      if ( path.length() > 0 ) { 
          MemorySpace::set_default_path(path);
          return 0;
      } else {
         return -1;
      }
   }
};

} // Kokkos

} // EXPERIMENTAL
#endif
