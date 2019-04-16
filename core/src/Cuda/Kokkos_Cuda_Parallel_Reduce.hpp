/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_CUDA_PARALLEL_HPP
#define KOKKOS_CUDA_PARALLEL_HPP

#include <Kokkos_Macros.hpp>
// TODO!!! put this back to the way it was
#include <device_functions.h>
#if defined( KOKKOS_ENABLE_CUDA ) // && defined( __CUDACC__ )

#include <iostream>
#include <algorithm>
#include <cstdio>
#include <cstdint>

#include <utility>
#include <Kokkos_Parallel.hpp>

#include <Patterns/ParallelReduce/impl/Kokkos_ReducerStorage.hpp>
#include <Concepts/Reducer/Kokkos_Reducer_Concept.hpp>
#include <Concepts/Functor/Kokkos_Functor_Concept.hpp>
#include <Concepts/Functor/Kokkos_ReductionFunctor_Concept.hpp>

#include <Cuda/Kokkos_Cuda_KernelLaunch.hpp>
#include <Cuda/Kokkos_Cuda_ReduceScan.hpp>
#include <Cuda/Kokkos_Cuda_BlockSize_Deduction.hpp>
#include <Cuda/Kokkos_Cuda_Locks.hpp>
#include <Kokkos_Vectorization.hpp>
#include <Cuda/Kokkos_Cuda_Version_9_8_Compatibility.hpp>

#if defined(KOKKOS_ENABLE_PROFILING)
#include <impl/Kokkos_Profiling_Interface.hpp>
#include <typeinfo>
#endif

#include <KokkosExp_MDRangePolicy.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template <
  class Derived,
  class ReducerType
>
class CudaIntraWarpReduceImpl
{
private:
  // CRTP boilerplate:
  KOKKOS_FORCEINLINE_FUNCTION
    Derived& self() noexcept { return *static_cast<Derived*>(this); }
  KOKKOS_FORCEINLINE_FUNCTION
    Derived const& self() const noexcept { return *static_cast<Derived const*>(this); }

  template <class, class>
  friend class CudaIntraBlockReduceImpl;

  using value_type = Concepts::reducer_value_type_t<ReducerType>;
  using reference_type = Concepts::reducer_value_type_t<ReducerType>;

protected:

  inline __device__
  void
  intra_warp_reduce(
    value_type value,
    const bool skip_vector, // Skip threads if Kokkos vector lanes are not part of the reduction
    const int width, // How much of the warp participates
    reference_type result
  )
  {
    const auto value_size = Concepts::reducer_value_size(self().reducer());

    unsigned mask =
      width==32
      ? 0xffffffff
      : ((1<<width)-1)<<((threadIdx.y*blockDim.x+threadIdx.x)/width)*width;

    reference_type result_ref =
      Concepts::reducer_bind_reference(self().reducer(), &value);
    for(int delta = skip_vector ? blockDim.x : 1; delta < width; delta*=2) {
      value_type tmp;
      cuda_shfl_down(tmp, value, delta, width, mask);
      reference_type tmp_ref =
        Concepts::reducer_bind_reference(self().reducer(), &tmp);
      Concepts::reducer_join(self().reducer(), result_ref, tmp_ref);
    }

    cuda_shfl(result, value, 0,width, mask);
  }


};

template <
  class Derived,
  class ReducerType
>
class CudaIntraBlockReduceImpl
{
private:
  // CRTP boilerplate:
  KOKKOS_FORCEINLINE_FUNCTION
  Derived& self() noexcept { return *static_cast<Derived*>(this); }
  KOKKOS_FORCEINLINE_FUNCTION
  Derived const& self() const noexcept { return *static_cast<Derived const*>(this); }

  template <class, class>
  friend class CudaInterBlockReduceImpl;

  using value_type = Concepts::reducer_value_type_t<ReducerType>;
  using reference_type = Concepts::reducer_reference_type_t<ReducerType>;

protected:

  inline __device__
  void
  intra_block_reduce(
    reference_type value,
    reference_type my_global_team_buffer_element,
    const int shared_elements,
    Cuda::size_type* shared_team_buffer_element,
    bool skip = false
  ) const
  {
    const auto value_size = Concepts::reducer_value_size(self().reducer());
    const auto word_count =
      value_size / sizeof(Cuda::size_type) + int(value_size % sizeof(Cuda::size_type) != 0);

    const int warp_id = (threadIdx.y * blockDim.x) / 32; // TODO shouldn't this be not 32 as a constant?
    reference_type my_shared_team_buffer_element =
      Concepts::reducer_bind_reference(
        self().reducer(),
        shared_team_buffer_element + ((warp_id % shared_elements) * word_count)
      );

    // Warp Level Reduction, ignoring Kokkos vector entries
    self().intra_warp_reduce(value, skip, 32, value);

    // If we're participating, put the value into our spot in the team buffer
    if(warp_id < shared_elements) {
      Concepts::reducer_assign_reference(
        self().reducer(),
        my_shared_team_buffer_element,
        value
      );
    }
    // Wait for every warp to be done before using one warp to do final cross warp reduction
    __syncthreads();

    // TODO this is shfl-specific code
    reference_type result_ref =
      Concepts::reducer_bind_reference(self().reducer(), my_shared_team_buffer_element);

    const int num_warps = blockDim.x * blockDim.y / 32;
    for(int w = shared_elements; w < num_warps; w += shared_elements) {
      if(warp_id >= w && warp_id < w + shared_elements) {
        if((threadIdx.y*blockDim.x + threadIdx.x) % 32 == 0) {
          Concepts::reducer_join(self().reducer(), result_ref, value);
        }
      }
      __syncthreads();
    }
  }

};

template <
  class Derived,
  class ReducerType
>
class CudaInterBlockReduceImpl
{
private:
  // CRTP boilerplate:
  KOKKOS_FORCEINLINE_FUNCTION
  Derived& self() noexcept { return *static_cast<Derived*>(this); }
  KOKKOS_FORCEINLINE_FUNCTION
  Derived const& self() const noexcept { return *static_cast<Derived const*>(this); }

protected:

  inline __device__
  bool
  inter_block_reduce(
    Cuda::size_type block_id,
    Cuda::size_type block_count,
    Cuda::size_type* shared_data,
    Cuda::size_type* global_data,
    Cuda::size_type* global_flags
  ) const
  {
    using reference_type = Concepts::reducer_reference_type_t<ReducerType>;

    const auto value_size = Concepts::reducer_value_size(self().reducer());
    const auto word_count =
      value_size / sizeof(Cuda::size_type) + int(value_size % sizeof(Cuda::size_type) != 0);

    reference_type global_team_buffer_element =
      Concepts::reducer_bind_reference(self().reducer(), (void*)global_data);
    reference_type my_global_team_buffer_element =
      Concepts::reducer_bind_reference(
        self().reducer(),
        global_team_buffer_element + (blockIdx.x * word_count)
      );
    reference_type value_ref =
      Concepts::reducer_bind_reference(
        self().reducer(),
        shared_data + (threadIdx.y * word_count)
      );

    int shared_elements = blockDim.x * blockDim.y / 32;
    int global_elements = block_count;
    __syncthreads(); // is this necessary?

    // Do an intra-block reduction first
    self().intra_block_reduce(
      value_ref, my_global_team_buffer_element, shared_elements, shared_data,
      true // skip vector entries not participating in the reduction
    );
    __syncthreads();

    // If we're the block leader, update the number of intra-block reductions that
    // that have completed
    unsigned int num_teams_done = 0;
    if(threadIdx.x + threadIdx.y == 0) {
      __threadfence();
      num_teams_done = Kokkos::atomic_fetch_add(global_flags, 1) + 1;
    }

    // Perform the join on the last block to finish
    bool is_last_block = false;
    if(__syncthreads_or(num_teams_done == gridDim.x)) {
      // update the return value
      is_last_block = true;

      // ????
      *global_flags = 0;

      // Initialize a value to reduce into on each thread in this block
      Concepts::reducer_init(self().reducer(), value_ref);

      // Invoke the join function for each block result
      for(
        int i = threadIdx.y * blockDim.x + threadIdx.x; // start at our index in the block
        i < global_elements;
        i += blockDim.x * blockDim.y // stride by the size of the block
      ) {
        auto global_element_ref =
          Concepts::reducer_bind_reference(self().reducer(), &global_team_buffer_element[i]);
        Concepts::reducer_join(self().reducer(), value_ref, global_element_ref);
      }

      // Then reduce across the threads in this block
      self().intra_block_reduce(
        value_ref,
        shared_data + ((blockDim.y-1) * word_count),
        shared_elements,
        shared_data
      );
    }

    // Return whether or not we're the block that has the answer
    return is_last_block;
  }

};


template <
  class Derived,
  class FunctorType,
  class PolicyType,
  class ReducerType
>
class CudaParallelReduceCommon
  : CudaInterBlockReduceImpl<Derived, ReducerType>
{
private:
  // CRTP boilerplate:
  KOKKOS_FORCEINLINE_FUNCTION
  Derived& self() noexcept { return *static_cast<Derived*>(this); }
  KOKKOS_FORCEINLINE_FUNCTION
  Derived const& self() const noexcept { return *static_cast<Derived const*>(this); }
};


template <
  class Derived,
  class FunctorType,
  class PolicyType,
  class ReducerType
>
class CudaParallelReduceCommonStorage
  : public CudaParallelReduceCommon<Derived, FunctorType, PolicyType, ReducerType>
{
private:
  // CRTP boilerplate:
  KOKKOS_FORCEINLINE_FUNCTION
  Derived& self() noexcept { return *static_cast<Derived*>(this); }
  KOKKOS_FORCEINLINE_FUNCTION
  Derived const& self() const noexcept { return *static_cast<Derived const*>(this); }

  FunctorType const m_functor;
  PolicyType const m_policy;
  ReducerStorage<ReducerType> m_reducer_storage;

  template <class, class, class, class>
  friend class CudaParallelReduceCommon;

protected:

  KOKKOS_FORCEINLINE_FUNCTION
  FunctorType const& functor() const noexcept { return m_functor; }

  KOKKOS_FORCEINLINE_FUNCTION
  PolicyType const& policy() const noexcept { return m_policy; }

  KOKKOS_FORCEINLINE_FUNCTION
  ReducerType const& reducer() const noexcept { return m_reducer_storage.get_reducer(); }

  template <class ViewType>
  inline
  CudaParallelReduceCommonStorage(
    FunctorType arg_functor,
    PolicyType arg_policy,
    ViewType const& arg_view,
    typename std::enable_if<
      Kokkos::is_view<ViewType>::value
        && !Kokkos::is_reducer_type<ViewType>::value,
      void const**
    >::type = nullptr
  ) : m_functor(arg_functor),
      m_policy(arg_policy),
      m_reducer_storage(m_functor, m_policy, arg_view)
  { }

  inline
  CudaParallelReduceCommonStorage(
    FunctorType arg_functor,
    PolicyType arg_policy,
    ReducerType const& reducer
  ) : m_functor(std::move(arg_functor)),
      m_policy(std::move(arg_policy)),
      m_reducer_storage(reducer)
  { }

};


template< class FunctorType , class ReducerType, class ... Traits >
class ParallelReduce<
  FunctorType,
  Kokkos::RangePolicy<Traits...>,
  ReducerType,
  Kokkos::Cuda
> : public
      CudaParallelReduceCommonStorage<
        ParallelReduce<FunctorType, Kokkos::RangePolicy<Traits...>, ReducerType, Kokkos::Cuda>,
        FunctorType,
        Kokkos::RangePolicy<Traits...>,
        ReducerType
      >
{
public:
  typedef Kokkos::RangePolicy< Traits ... >         Policy ;
private:

  using storage_base_t =
    CudaParallelReduceCommonStorage<
      ParallelReduce<FunctorType, Kokkos::RangePolicy<Traits...>, ReducerType, Kokkos::Cuda>,
      FunctorType,
      Kokkos::RangePolicy<Traits...>,
      ReducerType
    >;

  template <class, class, class, class>
  friend class CudaParallelReduceCommon;

  using Member = typename Policy::member_type;
  using LaunchBounds = typename Policy::launch_bounds;
  using WorkRange = typename Policy::WorkRange;

public:

  using size_type = Kokkos::Cuda::size_type;
  using index_type = typename Policy::index_type;

  // Algorithmic constraints: blockSize is a power of two AND blockDim.y == blockDim.z == 1

  size_type* m_scratch_space = nullptr;
  size_type* m_scratch_flags = nullptr;
  size_type* m_unified_space = nullptr;

  // Shall we use the shfl based reduction or not (only use it for static sized types of more than 128bit)
  enum { UseShflReduction = false };//((sizeof(value_type)>2*sizeof(double)) && ValueTraits::StaticValueSize) };

public:
  // Make the exec_range calls call to Reduce::DeviceIterateTile
  __device__ inline
  void
  exec_range(Member i, Concepts::reducer_reference_type_t<ReducerType> update) const
  {
    Concepts::functor_invoke_with_policy(
      this->policy(), this->functor(), i, update
    );
  }


  __device__ inline
  void operator() () const {
    const auto value_size = Concepts::reducer_value_size(this->reducer());
    const auto word_count = value_size / sizeof(size_type) + int(value_size % sizeof(size_type) != 0);

    //const integral_nonzero_constant< size_type , ValueTraits::StaticValueSize / sizeof(size_type) >
    //  word_count( ValueTraits::value_size( ReducerConditional::select(m_functor , m_reducer) ) / sizeof(size_type) );

    {
      Concepts::reducer_reference_type_t<ReducerType> update =
        Concepts::reducer_bind_reference(
          this->reducer(),
          kokkos_impl_cuda_shared_memory<size_type>() + threadIdx.y * word_count.value
        );
      Concepts::reducer_init(this->reducer(), update);
      //reference_type value =
      //  ValueInit::init( ReducerConditional::select(m_functor , m_reducer) , kokkos_impl_cuda_shared_memory<size_type>() + threadIdx.y * word_count.value );

      // Number of blocks is bounded so that the reduction can be limited to two passes.
      // Each thread block is given an approximately equal amount of work to perform.
      // Accumulate the values for this block.
      // The accumulation ordering does not match the final pass, but is arithmatically equivalent.

      const WorkRange range(this->policy(), blockIdx.x, gridDim.x);

      for(
        Member iwork = range.begin() + threadIdx.y, iwork_end = range.end();
        iwork < iwork_end;
        iwork += blockDim.y
      ) {
        exec_range(iwork, update);
      }
    }

    // Reduce with final value at blockDim.y - 1 location.
    if(
      cuda_single_inter_block_reduce_scan<false, ReducerTypeFwd, WorkTagFwd>(
           ReducerConditional::select(m_functor , m_reducer) , blockIdx.x , gridDim.x ,
           kokkos_impl_cuda_shared_memory<size_type>() , m_scratch_space , m_scratch_flags ) ) {

      // This is the final block with the final result at the final threads' location

      size_type * const shared = kokkos_impl_cuda_shared_memory<size_type>() + ( blockDim.y - 1 ) * word_count.value ;
      size_type * const global = m_result_ptr_device_accessible? reinterpret_cast<size_type*>(m_result_ptr) : 
                                 ( m_unified_space ? m_unified_space : m_scratch_space );

      if ( threadIdx.y == 0 ) {
        Kokkos::Impl::FunctorFinal< ReducerTypeFwd , WorkTagFwd >::final( ReducerConditional::select(m_functor , m_reducer) , shared );
      }

      if ( CudaTraits::WarpSize < word_count.value ) { __syncthreads(); }

      for ( unsigned i = threadIdx.y ; i < word_count.value ; i += blockDim.y ) { global[i] = shared[i]; }
    }
  }

  // Determine block size constrained by shared memory:
  static inline
  unsigned local_block_size( const FunctorType & f )
    {
      unsigned n = CudaTraits::WarpSize * 8 ;
      int shmem_size = cuda_single_inter_block_reduce_scan_shmem<false,FunctorType,WorkTag>( f , n );
      while ( (n && (CudaTraits::SharedMemoryCapacity < shmem_size)) ||
          (n > static_cast<unsigned>(Kokkos::Impl::cuda_get_max_block_size< ParallelReduce, LaunchBounds>( f , 1, shmem_size , 0 )))) {
        n >>= 1 ;
        shmem_size = cuda_single_inter_block_reduce_scan_shmem<false,FunctorType,WorkTag>( f , n );
      }
      return n ;
    }

  inline
  void execute()
    {
      const index_type nwork = m_policy.end() - m_policy.begin();
      if ( nwork ) {
        const int block_size = local_block_size( m_functor );

        m_scratch_space = cuda_internal_scratch_space( ValueTraits::value_size( ReducerConditional::select(m_functor , m_reducer) ) * block_size /* block_size == max block_count */ );
        m_scratch_flags = cuda_internal_scratch_flags( sizeof(size_type) );
        m_unified_space = cuda_internal_scratch_unified( ValueTraits::value_size( ReducerConditional::select(m_functor , m_reducer) ) );

        // REQUIRED ( 1 , N , 1 )
        const dim3 block( 1 , block_size , 1 );
        // Required grid.x <= block.y
        const dim3 grid( std::min( int(block.y) , int( ( nwork + block.y - 1 ) / block.y ) ) , 1 , 1 );

      const int shmem = UseShflReduction?0:cuda_single_inter_block_reduce_scan_shmem<false,FunctorType,WorkTag>( m_functor , block.y );

      CudaParallelLaunch< ParallelReduce, LaunchBounds >( *this, grid, block, shmem , m_policy.space().impl_internal_space_instance() ); // copy to device and execute

      if(!m_result_ptr_device_accessible) {
        Cuda::fence();

        if ( m_result_ptr ) {
          if ( m_unified_space ) {
            const int count = ValueTraits::value_count( ReducerConditional::select(m_functor , m_reducer)  );
            for ( int i = 0 ; i < count ; ++i ) { m_result_ptr[i] = pointer_type(m_unified_space)[i] ; }
          }
          else {
            const int size = ValueTraits::value_size( ReducerConditional::select(m_functor , m_reducer)  );
            DeepCopy<HostSpace,CudaSpace>( m_result_ptr , m_scratch_space , size );
          }
        }
      }
    }
    else {
      if (m_result_ptr) {
        ValueInit::init( ReducerConditional::select(m_functor , m_reducer) , m_result_ptr );
      }
    }
  }

  template <class ViewType>
  ParallelReduce(
    FunctorType const& arg_functor,
    Policy const& arg_policy,
    ViewType const& arg_result_view,
    typename std::enable_if<
      Kokkos::is_view<ViewType>::value, void const**
    >::type = nullptr
  ) : storage_base_t(arg_functor, arg_policy, arg_result_view)
  { }

  ParallelReduce(
    FunctorType const & arg_functor,
    Policy const & arg_policy,
    ReducerType const & arg_reducer
  ) : storage_base_t(arg_functor, arg_policy, arg_reducer)
  { }

//  , m_result_ptr_device_accessible(MemorySpaceAccess< Kokkos::CudaSpace , typename ReducerType::result_view_type::memory_space>::accessible )

};


// MDRangePolicy impl
template< class FunctorType , class ReducerType, class ... Traits >
class ParallelReduce< FunctorType
                    , Kokkos::MDRangePolicy< Traits ... >
                    , ReducerType
                    , Kokkos::Cuda
                    >
{
public:
  typedef Kokkos::MDRangePolicy< Traits ... > Policy ;
private:

  typedef typename Policy::array_index_type                 array_index_type;
  typedef typename Policy::index_type                       index_type;

  typedef typename Policy::work_tag     WorkTag ;
  typedef typename Policy::member_type  Member ;
  typedef typename Policy::launch_bounds LaunchBounds;

  typedef Kokkos::Impl::if_c< std::is_same<InvalidType,ReducerType>::value, FunctorType, ReducerType> ReducerConditional;
  typedef typename ReducerConditional::type ReducerTypeFwd;
  typedef typename Kokkos::Impl::if_c< std::is_same<InvalidType,ReducerType>::value, WorkTag, void>::type WorkTagFwd;

  typedef Kokkos::Impl::FunctorValueTraits< ReducerTypeFwd, WorkTagFwd > ValueTraits ;
  typedef Kokkos::Impl::FunctorValueInit<   ReducerTypeFwd, WorkTagFwd > ValueInit ;
  typedef Kokkos::Impl::FunctorValueJoin<   ReducerTypeFwd, WorkTagFwd > ValueJoin ;

public:

  typedef typename ValueTraits::pointer_type    pointer_type ;
  typedef typename ValueTraits::value_type      value_type ;
  typedef typename ValueTraits::reference_type  reference_type ;
  typedef FunctorType                           functor_type ;
  typedef Cuda::size_type                       size_type ;

  // Algorithmic constraints: blockSize is a power of two AND blockDim.y == blockDim.z == 1

  const FunctorType   m_functor ;
  const Policy        m_policy ; // used for workrange and nwork
  const ReducerType   m_reducer ;
  const pointer_type  m_result_ptr ;
  const bool          m_result_ptr_device_accessible ;
  size_type *         m_scratch_space ;
  size_type *         m_scratch_flags ;
  size_type *         m_unified_space ;

  typedef typename Kokkos::Impl::Reduce::DeviceIterateTile<Policy::rank, Policy, FunctorType, typename Policy::work_tag, reference_type> DeviceIteratePattern;

  // Shall we use the shfl based reduction or not (only use it for static sized types of more than 128bit
  enum { UseShflReduction = ((sizeof(value_type)>2*sizeof(double)) && (ValueTraits::StaticValueSize!=0)) };
  // Some crutch to do function overloading
private:
  typedef double DummyShflReductionType;
  typedef int DummySHMEMReductionType;

public:
  inline
  __device__
  void
  exec_range( reference_type update ) const
  {
    Kokkos::Impl::Reduce::DeviceIterateTile<Policy::rank,Policy,FunctorType,typename Policy::work_tag, reference_type>(m_policy, m_functor, update).exec_range();
  }

  inline
  __device__
  void operator() (void) const {
/*    run(Kokkos::Impl::if_c<UseShflReduction, DummyShflReductionType, DummySHMEMReductionType>::select(1,1.0) );
  }

  __device__ inline
  void run(const DummySHMEMReductionType& ) const
  {*/
    const integral_nonzero_constant< size_type , ValueTraits::StaticValueSize / sizeof(size_type) >
      word_count( ValueTraits::value_size( ReducerConditional::select(m_functor , m_reducer) ) / sizeof(size_type) );

    {
      reference_type value =
        ValueInit::init( ReducerConditional::select(m_functor , m_reducer) , kokkos_impl_cuda_shared_memory<size_type>() + threadIdx.y * word_count.value );

      // Number of blocks is bounded so that the reduction can be limited to two passes.
      // Each thread block is given an approximately equal amount of work to perform.
      // Accumulate the values for this block.
      // The accumulation ordering does not match the final pass, but is arithmatically equivalent.

      this-> exec_range( value );
    }

    // Reduce with final value at blockDim.y - 1 location.
    // Problem: non power-of-two blockDim
    if ( cuda_single_inter_block_reduce_scan<false,ReducerTypeFwd,WorkTagFwd>(
           ReducerConditional::select(m_functor , m_reducer) , blockIdx.x , gridDim.x ,
           kokkos_impl_cuda_shared_memory<size_type>() , m_scratch_space , m_scratch_flags ) ) {

      // This is the final block with the final result at the final threads' location
      size_type * const shared = kokkos_impl_cuda_shared_memory<size_type>() + ( blockDim.y - 1 ) * word_count.value ;
      size_type * const global = m_result_ptr_device_accessible? reinterpret_cast<size_type*>(m_result_ptr) :
                                 ( m_unified_space ? m_unified_space : m_scratch_space );

      if ( threadIdx.y == 0 ) {
        Kokkos::Impl::FunctorFinal< ReducerTypeFwd , WorkTagFwd >::final( ReducerConditional::select(m_functor , m_reducer) , shared );
      }

      if ( CudaTraits::WarpSize < word_count.value ) { __syncthreads(); }

      for ( unsigned i = threadIdx.y ; i < word_count.value ; i += blockDim.y ) { global[i] = shared[i]; }
    }
  }

/*  __device__ inline
   void run(const DummyShflReductionType&) const
   {

     value_type value;
     ValueInit::init( ReducerConditional::select(m_functor , m_reducer) , &value);
     // Number of blocks is bounded so that the reduction can be limited to two passes.
     // Each thread block is given an approximately equal amount of work to perform.
     // Accumulate the values for this block.
     // The accumulation ordering does not match the final pass, but is arithmatically equivalent.

     const Member work_part =
       ( ( m_policy.m_num_tiles + ( gridDim.x - 1 ) ) / gridDim.x ); //portion of tiles handled by each block

     this-> exec_range( value );

     pointer_type const result = (pointer_type) (m_unified_space ? m_unified_space : m_scratch_space) ;

     int max_active_thread = work_part < blockDim.y ? work_part:blockDim.y;
     max_active_thread = (max_active_thread == 0)?blockDim.y:max_active_thread;

     value_type init;
     ValueInit::init( ReducerConditional::select(m_functor , m_reducer) , &init);
     if(Impl::cuda_inter_block_reduction<ReducerTypeFwd,ValueJoin,WorkTagFwd>
         (value,init,ValueJoin(ReducerConditional::select(m_functor , m_reducer)),m_scratch_space,result,m_scratch_flags,max_active_thread)) {
       const unsigned id = threadIdx.y*blockDim.x + threadIdx.x;
       if(id==0) {
         Kokkos::Impl::FunctorFinal< ReducerTypeFwd , WorkTagFwd >::final( ReducerConditional::select(m_functor , m_reducer) , (void*) &value );
         *result = value;
       }
     }
   }
*/
  // Determine block size constrained by shared memory:
  static inline
  unsigned local_block_size( const FunctorType & f )
    {
      unsigned n = CudaTraits::WarpSize * 8 ;
      int shmem_size = cuda_single_inter_block_reduce_scan_shmem<false,FunctorType,WorkTag>( f , n );
      while ( (n && (CudaTraits::SharedMemoryCapacity < shmem_size)) ||
          (n > static_cast<unsigned>(Kokkos::Impl::cuda_get_max_block_size< ParallelReduce, LaunchBounds>( f , 1, shmem_size , 0 )))) {
        n >>= 1 ;
        shmem_size = cuda_single_inter_block_reduce_scan_shmem<false,FunctorType,WorkTag>( f , n );
      }
      return n ;
    }

  inline
  void execute()
    {
      const int nwork = m_policy.m_num_tiles;
      if ( nwork ) {
        int block_size = m_policy.m_prod_tile_dims;
        // CONSTRAINT: Algorithm requires block_size >= product of tile dimensions
        // Nearest power of two
        int exponent_pow_two = std::ceil( std::log2(block_size) );
        block_size = std::pow(2, exponent_pow_two);
        int suggested_blocksize = local_block_size( m_functor );

        block_size = (block_size > suggested_blocksize) ? block_size : suggested_blocksize ; //Note: block_size must be less than or equal to 512


        m_scratch_space = cuda_internal_scratch_space( ValueTraits::value_size( ReducerConditional::select(m_functor , m_reducer) ) * block_size /* block_size == max block_count */ );
        m_scratch_flags = cuda_internal_scratch_flags( sizeof(size_type) );
        m_unified_space = cuda_internal_scratch_unified( ValueTraits::value_size( ReducerConditional::select(m_functor , m_reducer) ) );

        // REQUIRED ( 1 , N , 1 )
        const dim3 block( 1 , block_size , 1 );
        // Required grid.x <= block.y
        const dim3 grid( std::min( int(block.y) , int( nwork ) ) , 1 , 1 );

      const int shmem = UseShflReduction?0:cuda_single_inter_block_reduce_scan_shmem<false,FunctorType,WorkTag>( m_functor , block.y );

      CudaParallelLaunch< ParallelReduce, LaunchBounds >( *this, grid, block, shmem , m_policy.space().impl_internal_space_instance() ); // copy to device and execute

      if(!m_result_ptr_device_accessible) {
        Cuda::fence();

        if ( m_result_ptr ) {
          if ( m_unified_space ) {
            const int count = ValueTraits::value_count( ReducerConditional::select(m_functor , m_reducer)  );
            for ( int i = 0 ; i < count ; ++i ) { m_result_ptr[i] = pointer_type(m_unified_space)[i] ; }
          }
          else {
            const int size = ValueTraits::value_size( ReducerConditional::select(m_functor , m_reducer)  );
            DeepCopy<HostSpace,CudaSpace>( m_result_ptr , m_scratch_space , size );
          }
        }
      }
    }
    else {
      if (m_result_ptr) {
        ValueInit::init( ReducerConditional::select(m_functor , m_reducer) , m_result_ptr );
      }
    }
  }

  template< class ViewType >
  ParallelReduce( const FunctorType  & arg_functor
                , const Policy       & arg_policy
                , const ViewType & arg_result
                , typename std::enable_if<
                   Kokkos::is_view< ViewType >::value
                ,void*>::type = NULL)
  : m_functor( arg_functor )
  , m_policy(  arg_policy )
  , m_reducer( InvalidType() )
  , m_result_ptr( arg_result.data() )
  , m_result_ptr_device_accessible(MemorySpaceAccess< Kokkos::CudaSpace , typename ViewType::memory_space>::accessible )
  , m_scratch_space( 0 )
  , m_scratch_flags( 0 )
  , m_unified_space( 0 )
  {}

  ParallelReduce( const FunctorType  & arg_functor
                , const Policy       & arg_policy
                , const ReducerType & reducer)
  : m_functor( arg_functor )
  , m_policy(  arg_policy )
  , m_reducer( reducer )
  , m_result_ptr( reducer.view().data() )
  , m_result_ptr_device_accessible(MemorySpaceAccess< Kokkos::CudaSpace , typename ReducerType::result_view_type::memory_space>::accessible )
  , m_scratch_space( 0 )
  , m_scratch_flags( 0 )
  , m_unified_space( 0 )
  {}
};


//----------------------------------------------------------------------------

template< class FunctorType , class ReducerType, class ... Properties >
class ParallelReduce< FunctorType
                    , Kokkos::TeamPolicy< Properties ... >
                    , ReducerType
                    , Kokkos::Cuda
                    >
{
public:
  typedef TeamPolicyInternal< Kokkos::Cuda, Properties ... >  Policy ;
private:

  typedef typename Policy::member_type  Member ;
  typedef typename Policy::work_tag     WorkTag ;
  typedef typename Policy::launch_bounds     LaunchBounds ;

  typedef Kokkos::Impl::if_c< std::is_same<InvalidType,ReducerType>::value, FunctorType, ReducerType> ReducerConditional;
  typedef typename ReducerConditional::type ReducerTypeFwd;
  typedef typename Kokkos::Impl::if_c< std::is_same<InvalidType,ReducerType>::value, WorkTag, void>::type WorkTagFwd;

  typedef Kokkos::Impl::FunctorValueTraits< ReducerTypeFwd, WorkTagFwd > ValueTraits ;
  typedef Kokkos::Impl::FunctorValueInit<   ReducerTypeFwd, WorkTagFwd > ValueInit ;
  typedef Kokkos::Impl::FunctorValueJoin<   ReducerTypeFwd, WorkTagFwd > ValueJoin ;

  typedef typename ValueTraits::pointer_type    pointer_type ;
  typedef typename ValueTraits::reference_type  reference_type ;
  typedef typename ValueTraits::value_type      value_type ;

public:

  typedef FunctorType      functor_type ;
  typedef Cuda::size_type  size_type ;

  enum { UseShflReduction = (true && (ValueTraits::StaticValueSize!=0)) };

private:
  typedef double DummyShflReductionType;
  typedef int DummySHMEMReductionType;

  // Algorithmic constraints: blockDim.y is a power of two AND blockDim.y == blockDim.z == 1
  // shared memory utilization:
  //
  //  [ global reduce space ]
  //  [ team   reduce space ]
  //  [ team   shared space ]
  //

  const FunctorType   m_functor ;
  const Policy        m_policy ;
  const ReducerType   m_reducer ;
  const pointer_type  m_result_ptr ;
  const bool          m_result_ptr_device_accessible ;
  size_type *         m_scratch_space ;
  size_type *         m_scratch_flags ;
  size_type *         m_unified_space ;
  size_type           m_team_begin ;
  size_type           m_shmem_begin ;
  size_type           m_shmem_size ;
  void*               m_scratch_ptr[2] ;
  int                 m_scratch_size[2] ;
  const size_type     m_league_size ;
  const size_type     m_team_size ;
  const size_type     m_vector_size ;

  template< class TagType >
  __device__ inline
  typename std::enable_if< std::is_same< TagType , void >::value >::type
  exec_team( const Member & member , reference_type update ) const
    { m_functor( member , update ); }

  template< class TagType >
  __device__ inline
  typename std::enable_if< ! std::is_same< TagType , void >::value >::type
  exec_team( const Member & member , reference_type update ) const
    { m_functor( TagType() , member , update ); }

public:

  __device__ inline
  void operator() () const {
    int64_t threadid = 0;
    if ( m_scratch_size[1]>0 ) {
      __shared__ int64_t base_thread_id;
      if (threadIdx.x==0 && threadIdx.y==0 ) {
        threadid = (blockIdx.x*blockDim.z + threadIdx.z) %
          (Kokkos::Impl::g_device_cuda_lock_arrays.n / (blockDim.x * blockDim.y));
        threadid *= blockDim.x * blockDim.y;
        int done = 0;
        while (!done) {
          done = (0 == atomicCAS(&Kokkos::Impl::g_device_cuda_lock_arrays.scratch[threadid],0,1));
          if(!done) {
            threadid += blockDim.x * blockDim.y;
            if(int64_t(threadid + blockDim.x * blockDim.y) >= int64_t(Kokkos::Impl::g_device_cuda_lock_arrays.n)) threadid = 0;
          }
        }
        base_thread_id = threadid;
      }
      __syncthreads();
      threadid = base_thread_id;
    }

    run(Kokkos::Impl::if_c<UseShflReduction, DummyShflReductionType, DummySHMEMReductionType>::select(1,1.0), threadid );
    if ( m_scratch_size[1]>0 ) {
      __syncthreads();
      if (threadIdx.x==0 && threadIdx.y==0 )
        Kokkos::Impl::g_device_cuda_lock_arrays.scratch[threadid]=0;
    }
  }

  __device__ inline
  void run(const DummySHMEMReductionType&, const int& threadid) const
  {
    const integral_nonzero_constant< size_type , ValueTraits::StaticValueSize / sizeof(size_type) >
      word_count( ValueTraits::value_size( ReducerConditional::select(m_functor , m_reducer) ) / sizeof(size_type) );

    reference_type value =
      ValueInit::init( ReducerConditional::select(m_functor , m_reducer) , kokkos_impl_cuda_shared_memory<size_type>() + threadIdx.y * word_count.value );

    // Iterate this block through the league
    const int int_league_size = (int)m_league_size;
    for ( int league_rank = blockIdx.x ; league_rank < int_league_size ; league_rank += gridDim.x ) {
      this-> template exec_team< WorkTag >
        ( Member( kokkos_impl_cuda_shared_memory<char>() + m_team_begin
                                        , m_shmem_begin
                                        , m_shmem_size
                                        , (void*) ( ((char*)m_scratch_ptr[1]) + ptrdiff_t(threadid/(blockDim.x*blockDim.y)) * m_scratch_size[1])
                                        , m_scratch_size[1]
                                        , league_rank
                                        , m_league_size )
        , value );
    }

    // Reduce with final value at blockDim.y - 1 location.
    if ( cuda_single_inter_block_reduce_scan<false,FunctorType,WorkTag>(
           ReducerConditional::select(m_functor , m_reducer) , blockIdx.x , gridDim.x ,
           kokkos_impl_cuda_shared_memory<size_type>() , m_scratch_space , m_scratch_flags ) ) {

      // This is the final block with the final result at the final threads' location

      size_type * const shared = kokkos_impl_cuda_shared_memory<size_type>() + ( blockDim.y - 1 ) * word_count.value ;
      size_type * const global = m_result_ptr_device_accessible? reinterpret_cast<size_type*>(m_result_ptr) :
                                 ( m_unified_space ? m_unified_space : m_scratch_space );

      if ( threadIdx.y == 0 ) {
        Kokkos::Impl::FunctorFinal< ReducerTypeFwd , WorkTagFwd >::final( ReducerConditional::select(m_functor , m_reducer) , shared );
      }

      if ( CudaTraits::WarpSize < word_count.value ) { __syncthreads(); }

      for ( unsigned i = threadIdx.y ; i < word_count.value ; i += blockDim.y ) { global[i] = shared[i]; }
    }

  }

  __device__ inline
  void run(const DummyShflReductionType&, const int& threadid) const
  {
    value_type value;
    ValueInit::init( ReducerConditional::select(m_functor , m_reducer) , &value);

    // Iterate this block through the league
    const int int_league_size = (int)m_league_size;
    for ( int league_rank = blockIdx.x ; league_rank < int_league_size ; league_rank += gridDim.x ) {
      this-> template exec_team< WorkTag >
        ( Member( kokkos_impl_cuda_shared_memory<char>() + m_team_begin
                                        , m_shmem_begin
                                        , m_shmem_size
                                        , (void*) ( ((char*)m_scratch_ptr[1]) + ptrdiff_t(threadid/(blockDim.x*blockDim.y)) * m_scratch_size[1])
                                        , m_scratch_size[1]
                                        , league_rank
                                        , m_league_size )
        , value );
    }

    pointer_type const result = m_result_ptr_device_accessible? m_result_ptr :
                                (pointer_type) ( m_unified_space ? m_unified_space : m_scratch_space );

    value_type init;
    ValueInit::init( ReducerConditional::select(m_functor , m_reducer) , &init);
    if(
        Impl::cuda_inter_block_reduction<FunctorType,ValueJoin,WorkTag>
           (value,init,ValueJoin(ReducerConditional::select(m_functor , m_reducer)),m_scratch_space,result,m_scratch_flags,blockDim.y)
        //This breaks a test
        //   Kokkos::Impl::CudaReductionsFunctor<FunctorType,WorkTag,false,true>::scalar_inter_block_reduction(ReducerConditional::select(m_functor , m_reducer) , blockIdx.x , gridDim.x ,
        //              kokkos_impl_cuda_shared_memory<size_type>() , m_scratch_space , m_scratch_flags)
    ) {
      const unsigned id = threadIdx.y*blockDim.x + threadIdx.x;
      if(id==0) {
        Kokkos::Impl::FunctorFinal< ReducerTypeFwd , WorkTagFwd >::final( ReducerConditional::select(m_functor , m_reducer) , (void*) &value );
        *result = value;
      }
    }
  }

  inline
  void execute()
    {
      const int nwork = m_league_size * m_team_size ;
      if ( nwork ) {
        const int block_count = UseShflReduction? std::min( m_league_size , size_type(1024*32) )
          :std::min( m_league_size , m_team_size );

        m_scratch_space = cuda_internal_scratch_space( ValueTraits::value_size( ReducerConditional::select(m_functor , m_reducer) ) * block_count );
        m_scratch_flags = cuda_internal_scratch_flags( sizeof(size_type) );
        m_unified_space = cuda_internal_scratch_unified( ValueTraits::value_size( ReducerConditional::select(m_functor , m_reducer) ) );

        const dim3 block( m_vector_size , m_team_size , 1 );
        const dim3 grid( block_count , 1 , 1 );
        const int shmem_size_total = m_team_begin + m_shmem_begin + m_shmem_size ;

        CudaParallelLaunch< ParallelReduce, LaunchBounds >( *this, grid, block, shmem_size_total , m_policy.space().impl_internal_space_instance() ); // copy to device and execute

        if(!m_result_ptr_device_accessible) {
          Cuda::fence();

          if ( m_result_ptr ) {
            if ( m_unified_space ) {
              const int count = ValueTraits::value_count( ReducerConditional::select(m_functor , m_reducer) );
              for ( int i = 0 ; i < count ; ++i ) { m_result_ptr[i] = pointer_type(m_unified_space)[i] ; }
            }
            else {
              const int size = ValueTraits::value_size( ReducerConditional::select(m_functor , m_reducer) );
              DeepCopy<HostSpace,CudaSpace>( m_result_ptr, m_scratch_space, size );
            }
          }
        }
      }
      else {
        if (m_result_ptr) {
          ValueInit::init( ReducerConditional::select(m_functor , m_reducer) , m_result_ptr );
        }
      }
    }

  template< class ViewType >
  ParallelReduce( const FunctorType  & arg_functor
                , const Policy       & arg_policy
                , const ViewType & arg_result
                , typename std::enable_if<
                                   Kokkos::is_view< ViewType >::value
                                ,void*>::type = NULL)
  : m_functor( arg_functor )
  , m_policy ( arg_policy )
  , m_reducer( InvalidType() )
  , m_result_ptr( arg_result.data() )
  , m_result_ptr_device_accessible(MemorySpaceAccess< Kokkos::CudaSpace , typename ViewType::memory_space>::accessible )
  , m_scratch_space( 0 )
  , m_scratch_flags( 0 )
  , m_unified_space( 0 )
  , m_team_begin( 0 )
  , m_shmem_begin( 0 )
  , m_shmem_size( 0 )
  , m_scratch_ptr{NULL,NULL}
  , m_scratch_size{
    arg_policy.scratch_size(0,( 0 <= arg_policy.team_size() ? arg_policy.team_size() :
        Kokkos::Impl::cuda_get_opt_block_size< ParallelReduce, LaunchBounds >( arg_functor , arg_policy.vector_length(),
                                                                 arg_policy.team_scratch_size(0),arg_policy.thread_scratch_size(0) ) /
                                                                 arg_policy.vector_length() )
    ), arg_policy.scratch_size(1,( 0 <= arg_policy.team_size() ? arg_policy.team_size() :
        Kokkos::Impl::cuda_get_opt_block_size< ParallelReduce, LaunchBounds >( arg_functor , arg_policy.vector_length(),
                                                                 arg_policy.team_scratch_size(0),arg_policy.thread_scratch_size(0) ) /
                                                                 arg_policy.vector_length() )
        )}
  , m_league_size( arg_policy.league_size() )
  , m_team_size( 0 <= arg_policy.team_size() ? arg_policy.team_size() :
      Kokkos::Impl::cuda_get_opt_block_size< ParallelReduce, LaunchBounds >( arg_functor , arg_policy.vector_length(),
                                                               arg_policy.team_scratch_size(0),arg_policy.thread_scratch_size(0) ) /
                                                               arg_policy.vector_length() )
  , m_vector_size( arg_policy.vector_length() )
  {
    // Return Init value if the number of worksets is zero
    if( arg_policy.league_size() == 0) {
      ValueInit::init( ReducerConditional::select(m_functor , m_reducer) , arg_result.data() );
      return ;
    }

    m_team_begin = UseShflReduction?0:cuda_single_inter_block_reduce_scan_shmem<false,FunctorType,WorkTag>( arg_functor , m_team_size );
    m_shmem_begin = sizeof(double) * ( m_team_size + 2 );
    m_shmem_size = arg_policy.scratch_size(0,m_team_size) + FunctorTeamShmemSize< FunctorType >::value( arg_functor , m_team_size );
    m_scratch_ptr[1] = cuda_resize_scratch_space(static_cast<std::int64_t>(m_scratch_size[1])*(static_cast<std::int64_t>(Cuda::concurrency()/(m_team_size*m_vector_size))));
    m_scratch_size[0] = m_shmem_size;
    m_scratch_size[1] = arg_policy.scratch_size(1,m_team_size);

    // The global parallel_reduce does not support vector_length other than 1 at the moment
    if( (arg_policy.vector_length() > 1) && !UseShflReduction )
      Impl::throw_runtime_exception( "Kokkos::parallel_reduce with a TeamPolicy using a vector length of greater than 1 is not currently supported for CUDA for dynamic sized reduction types.");

    if( (m_team_size < 32) && !UseShflReduction )
      Impl::throw_runtime_exception( "Kokkos::parallel_reduce with a TeamPolicy using a team_size smaller than 32 is not currently supported with CUDA for dynamic sized reduction types.");

    // Functor's reduce memory, team scan memory, and team shared memory depend upon team size.

    const int shmem_size_total = m_team_begin + m_shmem_begin + m_shmem_size ;

    if (! Kokkos::Impl::is_integral_power_of_two( m_team_size )  && !UseShflReduction ) {
      Kokkos::Impl::throw_runtime_exception(std::string("Kokkos::Impl::ParallelReduce< Cuda > bad team size"));
    }

    if ( CudaTraits::SharedMemoryCapacity < shmem_size_total ) {
      Kokkos::Impl::throw_runtime_exception(std::string("Kokkos::Impl::ParallelReduce< Cuda > requested too much L0 scratch memory"));
    }

    if ( int(m_team_size) > arg_policy.team_size_max(m_functor,ParallelReduceTag()) ) {
      Kokkos::Impl::throw_runtime_exception(std::string("Kokkos::Impl::ParallelReduce< Cuda > requested too large team size."));
    }

  }

  ParallelReduce( const FunctorType  & arg_functor
                , const Policy       & arg_policy
                , const ReducerType & reducer)
  : m_functor( arg_functor )
  , m_policy( arg_policy )
  , m_reducer( reducer )
  , m_result_ptr( reducer.view().data() )
  , m_result_ptr_device_accessible(MemorySpaceAccess< Kokkos::CudaSpace , typename ReducerType::result_view_type::memory_space>::accessible )
  , m_scratch_space( 0 )
  , m_scratch_flags( 0 )
  , m_unified_space( 0 )
  , m_team_begin( 0 )
  , m_shmem_begin( 0 )
  , m_shmem_size( 0 )
  , m_scratch_ptr{NULL,NULL}
  , m_league_size( arg_policy.league_size() )
  , m_team_size( 0 <= arg_policy.team_size() ? arg_policy.team_size() :
      Kokkos::Impl::cuda_get_opt_block_size< ParallelReduce, LaunchBounds >( arg_functor , arg_policy.vector_length(),
                                                               arg_policy.team_scratch_size(0),arg_policy.thread_scratch_size(0) ) /
      arg_policy.vector_length() )
  , m_vector_size( arg_policy.vector_length() )
  {
    // Return Init value if the number of worksets is zero
    if( arg_policy.league_size() == 0) {
      ValueInit::init( ReducerConditional::select(m_functor , m_reducer) , m_result_ptr );
      return ;
    }

    m_team_begin = UseShflReduction?0:cuda_single_inter_block_reduce_scan_shmem<false,FunctorType,WorkTag>( arg_functor , m_team_size );
    m_shmem_begin = sizeof(double) * ( m_team_size + 2 );
    m_shmem_size = arg_policy.scratch_size(0,m_team_size) + FunctorTeamShmemSize< FunctorType >::value( arg_functor , m_team_size );
    m_scratch_ptr[1] = cuda_resize_scratch_space(static_cast<ptrdiff_t>(m_scratch_size[1])*static_cast<ptrdiff_t>(Cuda::concurrency()/(m_team_size*m_vector_size)));
    m_scratch_size[0] = m_shmem_size;
    m_scratch_size[1] = arg_policy.scratch_size(1,m_team_size);

    // The global parallel_reduce does not support vector_length other than 1 at the moment
    if( (arg_policy.vector_length() > 1) && !UseShflReduction )
      Impl::throw_runtime_exception( "Kokkos::parallel_reduce with a TeamPolicy using a vector length of greater than 1 is not currently supported for CUDA for dynamic sized reduction types.");

    if( (m_team_size < 32) && !UseShflReduction )
      Impl::throw_runtime_exception( "Kokkos::parallel_reduce with a TeamPolicy using a team_size smaller than 32 is not currently supported with CUDA for dynamic sized reduction types.");

    // Functor's reduce memory, team scan memory, and team shared memory depend upon team size.

    const int shmem_size_total = m_team_begin + m_shmem_begin + m_shmem_size ;

    if ( (! Kokkos::Impl::is_integral_power_of_two( m_team_size )  && !UseShflReduction ) ||
         CudaTraits::SharedMemoryCapacity < shmem_size_total ) {
      Kokkos::Impl::throw_runtime_exception(std::string("Kokkos::Impl::ParallelReduce< Cuda > bad team size"));
    }
    if ( int(m_team_size) > arg_policy.team_size_max(m_functor,ParallelReduceTag()) ) {
      Kokkos::Impl::throw_runtime_exception(std::string("Kokkos::Impl::ParallelReduce< Cuda > requested too large team size."));
    }

  }
};

} // namespace Impl
} // namespace Kokkos

#endif /* defined( __CUDACC__ ) */
#endif /* #ifndef KOKKOS_CUDA_PARALLEL_HPP */

