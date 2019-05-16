#include <hip/hip_runtime.h>
namespace Kokkos {
namespace Impl {

void hip_internal_error_throw( hipError_t e , const char * name, const char * file = NULL, const int line = 0 );

inline void hip_internal_safe_call( hipError_t e , const char * name, const char * file = NULL, const int line = 0)
{
  if ( hipSuccess != e ) { hip_internal_error_throw( e , name, file, line ); }
}

}
}
#define HIP_SAFE_CALL( call )  \
        Kokkos::Impl::hip_internal_safe_call( call , #call, __FILE__, __LINE__ )
