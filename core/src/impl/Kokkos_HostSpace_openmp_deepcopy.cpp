
// this will define KOKKOS_ENABLE_OPENMP
#include <impl/Kokkos_HostSpace_openmp_deepcopy.hpp>

#if defined( KOKKOS_ENABLE_OPENMP )
// I use the OpenMP API
#include <omp.h>

// required tools
#include <cstring> // memcpy
#include <cstdint> // unitptr_t
namespace Kokkos {

namespace Impl {

#if defined (KOKKOS_IMPL_ENABLE_OMP_TARGET_COPY)
void hostspace_openmp_deepcopy(void * dst, const void * src, size_t n) {
  int devid = omp_get_initial_device();
  omp_target_memcpy(dst, const_cast<void*>(src), n,
                    0,
                    0,
                    devid, devid);
}
#elif defined (KOKKOS_IMPL_ENABLE_OMP_LOOP_COPY)
void hostspace_openmp_deepcopy(void *dst, const void * src, size_t n) {

  #warning("Using omp loops")
  if(n%8==0) {
    double* dst_p = (double*)dst;
    const double* src_p = (double*)src;
    #pragma omp parallel for
    for (size_t i = 0; i < n/8; ++i)
    {
      dst_p[i] = src_p[i];
    }
  } else if(n%4==0) {
    int32_t* dst_p = (int32_t*)dst;
    const int32_t* src_p = (int32_t*)src;
    #pragma omp  parallel for
    for (size_t i = 0; i < n/4; ++i)
    {
      dst_p[i] = src_p[i];
    }
  } else {
    char* dst_p = (char*)dst;
    const char* src_p = (char*)src;
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i)
    {
      dst_p[i] = src_p[i];
    }
  }
}
#else
/*
 * TODO: change casting to whatever C++ likes (reinterpret_cast)
 *       adjust logic with nthreads==1 to test if inside a parallel region
 *       the reason for arranging the logic this way, is because
 *       calling the omp_* API can incur some latency.
 */

void hostspace_openmp_deepcopy(void * dst, const void * src, size_t n) {
  // this can be whatever, but you you really want
  // bytes_per_chunk to be small enough so that two arrays fit within 32KB
  constexpr size_t page_sz_ = 4*1024;

  // on KNL avoid having each thread requesting whole pages of memcpy
  // on HSW, this may be equal to bytes_per_chunk
  constexpr size_t bytes_per_chunk = page_sz_/2;

  // make sure this is atleast 1 page, so that the calculation
  // in the remainder below is valid unisgned arithmetic
  if( n < 3*page_sz_)  {
    std::memcpy( dst, src, n );
    return;
  }

  // if 1 thread or parallel then just memcpy
  if(omp_get_max_threads() == 1 ||  omp_in_parallel()  ) {
    std::memcpy( dst, src, n );
    return;
  }

  #pragma omp parallel
  {
    // otherwise, execute a parallel loop over memcpy,
    // copying pages at a time
  
    // This line tests if dst is page aligned. We don't 
    // care about src, since it is readonly.
    // Effectively, mask out the lower bits that account for page_sz-1.
    // if the resulting address is the same as dst, then dst is page 
    // aligned to begin with we also are promised that dst + page_sz 
    // will always be valid, because we enforce that 2*page_sz the smallest
    // work allowed in this region of code.
    unsigned char * pg_aligned_start = ( (((uintptr_t) dst) & ~(page_sz_-1)) == ((uintptr_t) dst) )
                                              ? (unsigned char *) dst
                                              : (unsigned char *)(  (((uintptr_t) dst) & ~(page_sz_-1)) + page_sz_);
    // handle the remainder. How many bytes are there from dst to pg_aligned_start
    const size_t remainder = (size_t ) (pg_aligned_start - ((unsigned char *) dst) );
    const unsigned char * corrected_src_from_pg_aligned_dst = (unsigned char *) (((uintptr_t) src) + remainder);
    const size_t new_n =  (n-remainder);
    const size_t chunks = (new_n / bytes_per_chunk ) + 1;
    const int tid = omp_get_thread_num();
 
    // first thread out handles the remainder
    #pragma omp single nowait
    {
      // the initial dst and src are correct, only copy remainder bytes
      std::memcpy(dst,src,remainder);
    }

    // per dan sunderland, try nonmonotonic dynamic rather than the default static
    // ideally, if KNL, then use dynamic
    // if not use static
    #if defined (KOKKOS_COMPILER_GNU) && ( 485 > KOKKOS_COMPILER_GNU )
    // gcc 4.8.4 does not support OpenMP 4.5
    #pragma omp for nowait
    #else
    #pragma omp for nowait schedule(nonmonotonic: dynamic)
    #endif
    for(size_t i=0; i < chunks; ++i)
    {
      // global offset
      const size_t my_offset = i * bytes_per_chunk;

      // nothing to do if our block starts past the end
      if (my_offset < new_n)
      {
        unsigned char * my_dst = pg_aligned_start + ((uintptr_t) my_offset);
        const unsigned char * my_src = (unsigned char *) (((uintptr_t) corrected_src_from_pg_aligned_dst) + ((uintptr_t) my_offset));

        // esnure we do not go past the end new_n is greater than my_offset, so new_n - my_offset is safe
        if ((my_offset + bytes_per_chunk) < new_n) {
          const size_t my_n = bytes_per_chunk;

          if (tid%4 == 0) {
            uint64_t * my_ddst = reinterpret_cast<uint64_t*>(my_dst);
            const uint64_t * my_dsrc = reinterpret_cast<const uint64_t*>(my_src);
            
            for(uint32_t kk=0; kk < my_n/8; kk+=64) {
              my_ddst[kk] = my_dsrc[kk];
              my_ddst[kk+1] = my_dsrc[kk+1];
              my_ddst[kk+2] = my_dsrc[kk+2];
              my_ddst[kk+3] = my_dsrc[kk+3];
              my_ddst[kk+4] = my_dsrc[kk+4];
              my_ddst[kk+5] = my_dsrc[kk+5];
              my_ddst[kk+7] = my_dsrc[kk+6];
              my_ddst[kk+7] = my_dsrc[kk+7];
            }
          } else if (tid%4 == 1) {
            double * my_ddst = reinterpret_cast<double*>(my_dst);
            const double * my_dsrc = reinterpret_cast<const double*>(my_src);

            for(uint32_t kk=0; kk < my_n/8; kk+=64) {
              my_ddst[kk] = my_dsrc[kk];
              my_ddst[kk+1] = my_dsrc[kk+1];
              my_ddst[kk+2] = my_dsrc[kk+2];
              my_ddst[kk+3] = my_dsrc[kk+3];
              my_ddst[kk+4] = my_dsrc[kk+4];
              my_ddst[kk+5] = my_dsrc[kk+5];
              my_ddst[kk+7] = my_dsrc[kk+6];
              my_ddst[kk+7] = my_dsrc[kk+7];
            }
          } else {
            std::memcpy( (void*) my_dst, (void*) my_src, my_n );
          }
        }
        else {
          const size_t my_n = new_n - my_offset;
          std::memcpy( (void*) my_dst, (void*) my_src, my_n );
        }

     }
    }
  }
}
#endif

} // namespace Impl

} // namespace Kokkos


#endif // if defined( KOKKOS_ENABLE_OPENMP )
