#include <SYCL/Kokkos_SYCL_Error.hpp>
namespace Kokkos {
namespace Experimental {
namespace Impl {
/*
template< class DriverType>
__global__
static void sycl_parallel_launch_local_memory( const DriverType driver )
{
  driver();
}

template< class DriverType, unsigned int maxTperB, unsigned int minBperSM >
__global__
__launch_bounds__(maxTperB, minBperSM)
static void sycl_parallel_launch_local_memory( const DriverType driver )
{
  driver();
}

enum class SYCLLaunchMechanism:unsigned{Default=0,ConstantMemory=1,GlobalMemory=2,LocalMemory=4};

constexpr inline SYCLLaunchMechanism operator | (SYCLLaunchMechanism p1, SYCLLaunchMechanism p2) {
  return static_cast<SYCLLaunchMechanism>(static_cast<unsigned>(p1) |  static_cast<unsigned>(p2));
}
constexpr inline SYCLLaunchMechanism operator & (SYCLLaunchMechanism p1, SYCLLaunchMechanism p2) {
  return static_cast<SYCLLaunchMechanism>(static_cast<unsigned>(p1) &  static_cast<unsigned>(p2));
}

template<SYCLLaunchMechanism l>
struct SYCLDispatchProperties {
  SYCLLaunchMechanism launch_mechanism = l;
};

template < class DriverType
         , class LaunchBounds = Kokkos::LaunchBounds<>
         , SYCLLaunchMechanism LaunchMechanism =
             SYCLLaunchMechanism::LocalMemory >
struct SYCLParallelLaunch ;

template < class DriverType
         , unsigned int MaxThreadsPerBlock
         , unsigned int MinBlocksPerSM >
struct SYCLParallelLaunch< DriverType
                         , Kokkos::LaunchBounds< MaxThreadsPerBlock
                                               , MinBlocksPerSM >
                         , SYCLLaunchMechanism::LocalMemory >
{
  //static_assert(sizeof(DriverType)<SYCLTraits::KernelArgumentLimit,"Kokkos Error: Requested SYCLLaunchLocalMemory with a Functor larger than 4096 bytes.");
  inline
  SYCLParallelLaunch( const DriverType & driver
                    , const dim3       & grid
                    , const dim3       & block
                    , const int          shmem
                    , const SYCLInternal* sycl_instance
                    , const bool prefer_shmem )
  {
    if ( (grid.x != 0) && ( ( block.x * block.y * block.z ) != 0 ) ) {

      // Invoke the driver function on the device
      printf("%i %i %i | %i %i %i | %i\n",grid.x,grid.y,grid.z,block.x,block.y,block.z,shmem);
      printf("Pre Launch Error: %s\n",syclGetErrorName(syclGetLastError()));
      sycl_parallel_launch_local_memory
        < DriverType, MaxThreadsPerBlock, MinBlocksPerSM >
          <<< grid , block , shmem , sycl_instance->m_stream >>>( driver );

      Kokkos::Experimental::SYCL().fence();
      printf("Post Launch Error: %s\n",syclGetErrorName(syclGetLastError()));
#if defined( KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK )
      //CUDA_SAFE_CALL( syclGetLastError() );
      Kokkos::SYCL().fence();
#endif
    }
  }
};

template < class DriverType>
struct SYCLParallelLaunch< DriverType
                         , Kokkos::LaunchBounds<0,0>
                         , SYCLLaunchMechanism::LocalMemory >
{
  //static_assert(sizeof(DriverType)<SYCLTraits::KernelArgumentLimit,"Kokkos Error: Requested SYCLLaunchLocalMemory with a Functor larger than 4096 bytes.");
  inline
  SYCLParallelLaunch( const DriverType & driver
                    , const dim3       & grid
                    , const dim3       & block
                    , const int          shmem
                    , const SYCLInternal* sycl_instance
                    , const bool prefer_shmem)
  {
    if ( (grid.x != 0) && ( ( block.x * block.y * block.z ) != 0 ) ) {
      // Invoke the driver function on the device
      sycl_parallel_launch_local_memory< DriverType >
          <<< grid , block , shmem , sycl_instance->m_stream >>>( driver );

      Kokkos::Experimental::SYCL().fence();
#if defined( KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK )
      SYCL_SAFE_CALL( syclGetLastError() );
      Kokkos::SYCL().fence();
#endif
    }
  }
};*/
}
}
}
