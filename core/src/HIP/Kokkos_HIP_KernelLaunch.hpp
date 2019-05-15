namespace Kokkos {
namespace Experimental {
namespace Impl {

template< class DriverType>
__global__
static void hip_parallel_launch_local_memory( const DriverType driver )
{
  driver();
}

template< class DriverType, unsigned int maxTperB, unsigned int minBperSM >
__global__
__launch_bounds__(maxTperB, minBperSM)
static void hip_parallel_launch_local_memory( const DriverType driver )
{
  driver();
}

enum class HIPLaunchMechanism:unsigned{Default=0,ConstantMemory=1,GlobalMemory=2,LocalMemory=4};

constexpr inline HIPLaunchMechanism operator | (HIPLaunchMechanism p1, HIPLaunchMechanism p2) {
  return static_cast<HIPLaunchMechanism>(static_cast<unsigned>(p1) |  static_cast<unsigned>(p2));
}
constexpr inline HIPLaunchMechanism operator & (HIPLaunchMechanism p1, HIPLaunchMechanism p2) {
  return static_cast<HIPLaunchMechanism>(static_cast<unsigned>(p1) &  static_cast<unsigned>(p2));
}

template<HIPLaunchMechanism l>
struct HIPDispatchProperties {
  HIPLaunchMechanism launch_mechanism = l;
};

template < class DriverType
         , class LaunchBounds = Kokkos::LaunchBounds<>
         , HIPLaunchMechanism LaunchMechanism =
             HIPLaunchMechanism::LocalMemory >
struct HIPParallelLaunch ;

template < class DriverType
         , unsigned int MaxThreadsPerBlock
         , unsigned int MinBlocksPerSM >
struct HIPParallelLaunch< DriverType
                         , Kokkos::LaunchBounds< MaxThreadsPerBlock
                                               , MinBlocksPerSM >
                         , HIPLaunchMechanism::LocalMemory >
{
  //static_assert(sizeof(DriverType)<HIPTraits::KernelArgumentLimit,"Kokkos Error: Requested HIPLaunchLocalMemory with a Functor larger than 4096 bytes.");
  inline
  HIPParallelLaunch( const DriverType & driver
                    , const dim3       & grid
                    , const dim3       & block
                    , const int          shmem
                    , const HIPInternal* hip_instance
                    , const bool prefer_shmem )
  {
    if ( (grid.x != 0) && ( ( block.x * block.y * block.z ) != 0 ) ) {
/*
      if ( hip_instance->m_maxShmemPerBlock < shmem ) {
        Kokkos::Impl::throw_runtime_exception( std::string("HIPParallelLaunch FAILED: shared memory request is too large") );
      }
      #ifndef KOKKOS_ARCH_KEPLER
      // On Kepler the L1 has no benefit since it doesn't cache reads
      else {
        CUDA_SAFE_CALL(
          hipFuncSetCacheConfig
            ( hip_parallel_launch_local_memory
                < DriverType, MaxThreadsPerBlock, MinBlocksPerSM >
            , ( prefer_shmem ? hipFuncCachePreferShared : hipFuncCachePreferL1 )
            ) );
      }
      #endif

      KOKKOS_ENSURE_CUDA_LOCK_ARRAYS_ON_DEVICE();*/

      // Invoke the driver function on the device
      printf("%i %i %i | %i %i %i | %i\n",grid.x,grid.y,grid.z,block.x,block.y,block.z,shmem);
      printf("Pre Launch Error: %s\n",hipGetErrorName(hipGetLastError()));
      hip_parallel_launch_local_memory
        < DriverType, MaxThreadsPerBlock, MinBlocksPerSM >
          <<< grid , block , shmem , hip_instance->m_stream >>>( driver );

      Kokkos::Experimental::HIP().fence();
      printf("Post Launch Error: %s\n",hipGetErrorName(hipGetLastError()));
#if defined( KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK )
      //CUDA_SAFE_CALL( hipGetLastError() );
      Kokkos::HIP().fence();
#endif
    }
  }
/*
  static hipFuncAttributes get_hip_func_attributes() {
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr,hip_parallel_launch_local_memory
            < DriverType, MaxThreadsPerBlock, MinBlocksPerSM >);
    return attr;
  }*/
};

template < class DriverType>
struct HIPParallelLaunch< DriverType
                         , Kokkos::LaunchBounds<0,0>
                         , HIPLaunchMechanism::LocalMemory >
{
  //static_assert(sizeof(DriverType)<HIPTraits::KernelArgumentLimit,"Kokkos Error: Requested HIPLaunchLocalMemory with a Functor larger than 4096 bytes.");
  inline
  HIPParallelLaunch( const DriverType & driver
                    , const dim3       & grid
                    , const dim3       & block
                    , const int          shmem
                    , const HIPInternal* hip_instance
                    , const bool prefer_shmem)
  {
    if ( (grid.x != 0) && ( ( block.x * block.y * block.z ) != 0 ) ) {
/**
      if ( hip_instance->m_maxShmemPerBlock < shmem ) {
        Kokkos::Impl::throw_runtime_exception( std::string("HIPParallelLaunch FAILED: shared memory request is too large") );
      }

      KOKKOS_ENSURE_CUDA_LOCK_ARRAYS_ON_DEVICE();
*/
      // Invoke the driver function on the device
      printf("%i %i %i | %i %i %i | %i\n",grid.x,grid.y,grid.z,block.x,block.y,block.z,shmem);
      printf("Pre Launch Error: %s\n",hipGetErrorName(hipGetLastError()));
      hip_parallel_launch_local_memory< DriverType >
          ///<<< grid , block , shmem , hip_instance->m_stream >>>( driver );
          <<<40,256>>>(driver);

      Kokkos::Experimental::HIP().fence();
      printf("Post Launch Error: %s\n",hipGetErrorName(hipGetLastError()));
#if defined( KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK )
//      CUDA_SAFE_CALL( hipGetLastError() );
      Kokkos::HIP().fence();
#endif
    }
  }
/*
  static hipFuncAttributes get_hip_func_attributes() {
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr,hip_parallel_launch_local_memory
            < DriverType >);
    return attr;
  }*/
};
}
}
}
