namespace Kokkos {
namespace Impl {
/*
void sycl_internal_error_throw( syclError_t e , const char * name, const char * file = NULL, const int line = 0 );

inline void sycl_internal_safe_call( syclError_t e , const char * name, const char * file = NULL, const int line = 0)
{
  if ( syclSuccess != e ) { sycl_internal_error_throw( e , name, file, line ); }
}
*/
}
}
#define SYCL_SAFE_CALL( call )  \
        call
        //Kokkos::Impl::sycl_internal_safe_call( call , #call, __FILE__, __LINE__ )
