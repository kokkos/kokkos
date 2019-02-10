#ifdef JJE_DEBUG_PAR_DEEPCOPY
  #include <cstdio>
#endif

// for memcpy
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

#define PAR_DEEP_COPY_USE_MEMCPY

namespace Kokkos {

namespace Impl {

void hostspace_parallel_deepcopy(void * dst, const void * src, size_t n) {
  constexpr size_t page_sz_ = 4*1024;

  constexpr size_t bytes_per_chunk = page_sz_/2;
  // on KNL avoid having each thread requesting whole pages of memcpy
  // on HSW, this may be equal to bytes_per_chunk
  constexpr size_t thread_bytes_per_block = bytes_per_chunk; //page_sz_/2;

  if( n < bytes_per_chunk)  {
    std::memcpy( dst, src, n );
    return;
  }

  // I can't use exec_space::concurrency here (or can I?)
  // this may require specializing for each execSpace...
  #ifdef _OPENMP
    const int nthreads = omp_get_max_threads();
  #else
    const int nthreads = 1;
  #endif

  if(nthreads == 1) {
    std::memcpy( dst, src, n );
    return;
  }

  // otherwise, execute a parallel loop over memcpy,
  // copying pages at a time

  // This line tests if dst is page aligned. We don't 
  // care about src, since it is readonly.
  // Effectively, mask out the lower bits that account for page_sz-1.
  // if the resulting address is the same as dst, then dst is page 
  // aligned to begin with we also are promised that dst + page_sz 
  // will always be valid, because we enforce that 2*page_sz the smallest
  // work allowed in this region of code.
  const unsigned char * pg_aligned_start = ( (((uintptr_t) dst) & ~(page_sz_-1)) == ((uintptr_t) dst) )
                                            ? (unsigned char *) dst
                                            : (unsigned char *)(  (((uintptr_t) dst) & ~(page_sz_-1)) + page_sz_);
  // handle the remainder. How many bytes are there from dst to pg_aligned_start
  const size_t remainder = (size_t ) (pg_aligned_start - ((unsigned char *) dst) );
  const unsigned char * corrected_src_from_pg_aligned_dst = (unsigned char *) (((uintptr_t) src) + remainder);
  const size_t new_n =  (n-remainder);
  const size_t chunks = (new_n / bytes_per_chunk ) + 1;

  #ifdef JJE_DEBUG_PAR_DEEPCOPY
  fprintf(stderr, "pg_aligned_start %p\n", pg_aligned_start);
  fprintf(stderr, "dest             %p\n", dst);
  fprintf(stderr, "src              %p\n", src);
  fprintf(stderr, "remainder        %d\n", (int) remainder);
  fprintf(stderr, "new_n            %d\n", (int) new_n);
  fprintf(stderr, "chunks           %d\n", (int) chunks);
  fprintf(stderr, "n                %d\n", (int) n);
  #endif

  #pragma omp parallel
  {

    // first thread out handles the remainder
    #pragma omp single nowait
    {
      // the initial dst and src are correct, only copy remainder bytes
      std::memcpy(dst,src,remainder);
      #ifdef JJE_DEBUG_PAR_DEEPCOPY
      for(uint32_t i = 0; i < (remainder/8); ++i)
        fprintf(stderr, "src[%d] = %d, dst[%d] =  %d\n", i, (int) (((int64_t*)src)[i]), i, (int) (((int64_t*)dst)[i]));
      #endif
    }

    #pragma omp for nowait
    for(size_t i=0; i < chunks; ++i)
    {
      // global offset
      //for( size_t my_offset = i * bytes_per_chunk; my_offset < new_n; my_offset += thread_bytes_per_block)

      const size_t my_offset = i * bytes_per_chunk;
      if (my_offset < new_n)
      {
        const unsigned char * my_dst = pg_aligned_start + ((uintptr_t) my_offset);
        const unsigned char * my_src = (unsigned char *) (((uintptr_t) corrected_src_from_pg_aligned_dst) + ((uintptr_t) my_offset));
        const size_t my_n = (my_offset + thread_bytes_per_block) < new_n ? thread_bytes_per_block : new_n - my_offset;
        #ifdef JJE_DEBUG_PAR_DEEPCOPY
        #pragma omp critical
        {
          fprintf(stderr, "-------------------------------------------\n");
          fprintf(stderr, "chunk                 %d\n", (int) i);
          fprintf(stderr, "my_offset             %d\n", (int) my_offset);
          fprintf(stderr, "my_dst                %p\n", my_dst);
          fprintf(stderr, "my_src                %p\n", my_src);
          fprintf(stderr, "my_n                  %d\n", (int) my_n);
        

          std::memcpy( (void*) my_dst, (void*) my_src, my_n );
 
          for(uint32_t i = 0; i < (my_n/8); ++i)
            fprintf(stderr, "src[%d] = %d, dst[%d] =  %d\n", i, (int) (((int64_t*)my_src)[i]), i, (int) (((int64_t*)my_dst)[i]));
        }
        #else
        #ifdef PAR_DEEP_COPY_USE_MEMCPY
        std::memcpy( (void*) my_dst, (void*) my_src, my_n );
        #else
          #pragma message("using for copy")
          for(int k=0; k < my_n; ++k) my_dst[k] = my_src[k];
        #endif
        #endif
     }
    }
  }
}

} // namespace Impl

} // namespace Kokkos

