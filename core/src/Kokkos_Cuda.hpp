/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
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

#ifndef KOKKOS_CUDA_HPP
#define KOKKOS_CUDA_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_CUDA)

#include <Kokkos_Core_fwd.hpp>

#include <iosfwd>
#include <vector>

#include <impl/Kokkos_AnalyzePolicy.hpp>
#include <Kokkos_CudaSpace.hpp>

#include <Kokkos_Parallel.hpp>
#include <Kokkos_TaskScheduler.hpp>
#include <Kokkos_Layout.hpp>
#include <Kokkos_ScratchSpace.hpp>
#include <Kokkos_MemoryTraits.hpp>
#include <impl/Kokkos_Tags.hpp>

/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {
class CudaExec;
class CudaInternal;
}  // namespace Impl
}  // namespace Kokkos

/*--------------------------------------------------------------------------*/

namespace Kokkos {

namespace Impl {
namespace Experimental {
enum class CudaLaunchMechanism : unsigned {
  Default        = 0,
  ConstantMemory = 1,
  GlobalMemory   = 2,
  LocalMemory    = 4
};

constexpr inline CudaLaunchMechanism operator|(CudaLaunchMechanism p1,
                                               CudaLaunchMechanism p2) {
  return static_cast<CudaLaunchMechanism>(static_cast<unsigned>(p1) |
                                          static_cast<unsigned>(p2));
}
constexpr inline CudaLaunchMechanism operator&(CudaLaunchMechanism p1,
                                               CudaLaunchMechanism p2) {
  return static_cast<CudaLaunchMechanism>(static_cast<unsigned>(p1) &
                                          static_cast<unsigned>(p2));
}

template <CudaLaunchMechanism l>
struct CudaDispatchProperties {
  CudaLaunchMechanism launch_mechanism = l;
};
}  // namespace Experimental
}  // namespace Impl
/// \class Cuda
/// \brief Kokkos Execution Space that uses CUDA to run on GPUs.
///
/// An "execution space" represents a parallel execution model.  It tells Kokkos
/// how to parallelize the execution of kernels in a parallel_for or
/// parallel_reduce.  For example, the Threads execution space uses Pthreads or
/// C++11 threads on a CPU, the OpenMP execution space uses the OpenMP language
/// extensions, and the Serial execution space executes "parallel" kernels
/// sequentially.  The Cuda execution space uses NVIDIA's CUDA programming
/// model to execute kernels in parallel on GPUs.
class Cuda {
 public:
  //! \name Type declarations that all Kokkos execution spaces must provide.
  //@{

  //! Tag this class as a kokkos execution space
  typedef Cuda execution_space;

#if defined(KOKKOS_ENABLE_CUDA_UVM)
  //! This execution space's preferred memory space.
  typedef CudaUVMSpace memory_space;
#else
  //! This execution space's preferred memory space.
  typedef CudaSpace memory_space;
#endif

  //! This execution space preferred device_type
  typedef Kokkos::Device<execution_space, memory_space> device_type;

  //! The size_type best suited for this execution space.
  typedef memory_space::size_type size_type;

  //! This execution space's preferred array layout.
  typedef LayoutLeft array_layout;

  //!
  typedef ScratchMemorySpace<Cuda> scratch_memory_space;

  //@}
  //--------------------------------------------------
  //! \name Functions that all Kokkos devices must implement.
  //@{

  /// \brief True if and only if this method is being called in a
  ///   thread-parallel function.
  KOKKOS_INLINE_FUNCTION static int in_parallel() {
#if defined(__CUDA_ARCH__)
    return true;
#else
    return false;
#endif
  }

  /** \brief  Set the device in a "sleep" state.
   *
   * This function sets the device in a "sleep" state in which it is
   * not ready for work.  This may consume less resources than if the
   * device were in an "awake" state, but it may also take time to
   * bring the device from a sleep state to be ready for work.
   *
   * \return True if the device is in the "sleep" state, else false if
   *   the device is actively working and could not enter the "sleep"
   *   state.
   */
  static bool sleep();

  /// \brief Wake the device from the 'sleep' state so it is ready for work.
  ///
  /// \return True if the device is in the "ready" state, else "false"
  ///  if the device is actively working (which also means that it's
  ///  awake).
  static bool wake();

  /// \brief Wait until all dispatched functors complete.
  ///
  /// The parallel_for or parallel_reduce dispatch of a functor may
  /// return asynchronously, before the functor completes.  This
  /// method does not return until all dispatched functors on this
  /// device have completed.
  static void impl_static_fence();

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE
  static void fence();
#else
  void fence() const;
#endif

  /** \brief  Return the maximum amount of concurrency.  */
  static int concurrency();

  //! Print configuration information to the given output stream.
  static void print_configuration(std::ostream&, const bool detail = false);

  //@}
  //--------------------------------------------------
  //! \name  Cuda space instances

  KOKKOS_INLINE_FUNCTION
  ~Cuda() {}

  Cuda();

  Cuda(Cuda&&)      = default;
  Cuda(const Cuda&) = default;
  Cuda& operator=(Cuda&&) = default;
  Cuda& operator=(const Cuda&) = default;

  Cuda(cudaStream_t stream);

  //--------------------------------------------------------------------------
  //! \name Device-specific functions
  //@{

  struct SelectDevice {
    int cuda_device_id;
    SelectDevice() : cuda_device_id(0) {}
    explicit SelectDevice(int id) : cuda_device_id(id) {}
  };

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE
  //! Free any resources being consumed by the device.
  static void finalize();

  //! Has been initialized
  static int is_initialized();

  //! Initialize, telling the CUDA run-time library which device to use.
  static void initialize(const SelectDevice         = SelectDevice(),
                         const size_t num_instances = 1);
#else
  //! Free any resources being consumed by the device.
  static void impl_finalize();

  //! Has been initialized
  static int impl_is_initialized();

  //! Initialize, telling the CUDA run-time library which device to use.
  static void impl_initialize(const SelectDevice         = SelectDevice(),
                              const size_t num_instances = 1);
#endif

  /// \brief Cuda device architecture of the selected device.
  ///
  /// This matches the __CUDA_ARCH__ specification.
  static size_type device_arch();

  //! Query device count.
  static size_type detect_device_count();

  /** \brief  Detect the available devices and their architecture
   *          as defined by the __CUDA_ARCH__ specification.
   */
  static std::vector<unsigned> detect_device_arch();

  cudaStream_t cuda_stream() const;
  int cuda_device() const;

  //@}
  //--------------------------------------------------------------------------

  static const char* name();

  inline Impl::CudaInternal* impl_internal_space_instance() const {
    return m_space_instance;
  }

 private:
  Impl::CudaInternal* m_space_instance;
};

}  // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

template <>
struct MemorySpaceAccess<Kokkos::CudaSpace,
                         Kokkos::Cuda::scratch_memory_space> {
  enum { assignable = false };
  enum { accessible = true };
  enum { deepcopy = false };
};

#if defined(KOKKOS_ENABLE_CUDA_UVM)

// If forcing use of UVM everywhere
// then must assume that CudaUVMSpace
// can be a stand-in for CudaSpace.
// This will fail when a strange host-side execution space
// that defines CudaUVMSpace as its preferredmemory space.

template <>
struct MemorySpaceAccess<Kokkos::CudaUVMSpace,
                         Kokkos::Cuda::scratch_memory_space> {
  enum { assignable = false };
  enum { accessible = true };
  enum { deepcopy = false };
};

#endif

template <>
struct VerifyExecutionCanAccessMemorySpace<Kokkos::CudaSpace,
                                           Kokkos::Cuda::scratch_memory_space> {
  enum { value = true };
  KOKKOS_INLINE_FUNCTION static void verify(void) {}
  KOKKOS_INLINE_FUNCTION static void verify(const void*) {}
};

template <>
struct VerifyExecutionCanAccessMemorySpace<Kokkos::HostSpace,
                                           Kokkos::Cuda::scratch_memory_space> {
  enum { value = false };
  inline static void verify(void) { CudaSpace::access_error(); }
  inline static void verify(const void* p) { CudaSpace::access_error(p); }
};

}  // namespace Impl
}  // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

#include <Cuda/Kokkos_Cuda_KernelLaunch.hpp>
#include <Cuda/Kokkos_Cuda_Instance.hpp>
#include <Cuda/Kokkos_Cuda_View.hpp>
#include <Cuda/Kokkos_Cuda_Team.hpp>
#include <Cuda/Kokkos_Cuda_Parallel.hpp>
#include <Cuda/Kokkos_Cuda_Task.hpp>
#include <Cuda/Kokkos_Cuda_UniqueToken.hpp>

#include <KokkosExp_MDRangePolicy.hpp>
//----------------------------------------------------------------------------

#endif /* #if defined( KOKKOS_ENABLE_CUDA ) */
#endif /* #ifndef KOKKOS_CUDA_HPP */
