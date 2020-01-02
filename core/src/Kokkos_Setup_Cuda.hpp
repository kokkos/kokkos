
#if defined(KOKKOS_ENABLE_CUDA) && defined(__CUDACC__)
// Compiling with a CUDA compiler.
//
//  Include <cuda.h> to pick up the CUDA_VERSION macro defined as:
//    CUDA_VERSION = ( MAJOR_VERSION * 1000 ) + ( MINOR_VERSION * 10 )
//
//  When generating device code the __CUDA_ARCH__ macro is defined as:
//    __CUDA_ARCH__ = ( MAJOR_CAPABILITY * 100 ) + ( MINOR_CAPABILITY * 10 )

#include <cuda_runtime.h>
#include <cuda.h>

#if !defined(CUDA_VERSION)
#error "#include <cuda.h> did not define CUDA_VERSION."
#endif

#if (CUDA_VERSION < 7000)
// CUDA supports C++11 in device code starting with version 7.0.
// This includes auto type and device code internal lambdas.
#error "Cuda version 7.0 or greater required."
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 300)
// Compiling with CUDA compiler for device code.
#error "Cuda device capability >= 3.0 is required."
#endif

#ifdef KOKKOS_ENABLE_CUDA_LAMBDA
#if (CUDA_VERSION < 7050)
// CUDA supports C++11 lambdas generated in host code to be given
// to the device starting with version 7.5. But the release candidate (7.5.6)
// still identifies as 7.0.
#error "Cuda version 7.5 or greater required for host-to-device Lambda support."
#endif

#if (CUDA_VERSION < 8000) && defined(__NVCC__)
#define KOKKOS_LAMBDA [=] __device__
#if defined(KOKKOS_INTERNAL_ENABLE_NON_CUDA_BACKEND)
#undef KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA
#endif
#else
#define KOKKOS_LAMBDA [=] __host__ __device__

#if defined(KOKKOS_ENABLE_CXX17) || defined(KOKKOS_ENABLE_CXX20)
#define KOKKOS_CLASS_LAMBDA [ =, *this ] __host__ __device__
#endif
#endif

#if defined(__NVCC__)
#define KOKKOS_IMPL_NEED_FUNCTOR_WRAPPER
#endif
#else  // !defined(KOKKOS_ENABLE_CUDA_LAMBDA)
#undef KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA
#endif  // !defined(KOKKOS_ENABLE_CUDA_LAMBDA)

#if (9000 <= CUDA_VERSION) && (CUDA_VERSION < 10000)
// CUDA 9 introduced an incorrect warning,
// see https://github.com/kokkos/kokkos/issues/1470
#define KOKKOS_CUDA_9_DEFAULTED_BUG_WORKAROUND
#endif

#if (10000 > CUDA_VERSION)
#define KOKKOS_ENABLE_PRE_CUDA_10_DEPRECATION_API
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
// PTX atomics with memory order semantics are only available on volta and later
#if !defined(KOKKOS_DISABLE_CUDA_ASM)
#if !defined(KOKKOS_ENABLE_CUDA_ASM)
#define KOKKOS_ENABLE_CUDA_ASM
#if !defined(KOKKOS_DISABLE_CUDA_ASM_ATOMICS)
#define KOKKOS_ENABLE_CUDA_ASM_ATOMICS
#endif
#endif
#endif
#endif


#endif  // #if defined( KOKKOS_ENABLE_CUDA ) && defined( __CUDACC__ )

#if defined(KOKKOS_ENABLE_CUDA)

#define KOKKOS_FORCEINLINE_FUNCTION __device__ __host__ __forceinline__
#define KOKKOS_IMPL_FORCEINLINE __forceinline__
#define KOKKOS_INLINE_FUNCTION __device__ __host__ inline
#define KOKKOS_FUNCTION __device__ __host__

#if defined(KOKKOS_COMPILER_NVCC)
#define KOKKOS_INLINE_FUNCTION_DELETED inline
#else
#define KOKKOS_INLINE_FUNCTION_DELETED __device__ __host__ inline
#endif

#endif // KOKKOS_ENABLE_CUDA
