//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#include <Kokkos_Macros.hpp>
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif
#ifndef KOKKOS_OPENMP_HPP
#define KOKKOS_OPENMP_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_OPENMP)

#include <Kokkos_Core_fwd.hpp>

#include <cstddef>
#include <iosfwd>
#include <Kokkos_HostSpace.hpp>

#ifdef KOKKOS_ENABLE_HBWSPACE
#include <Kokkos_HBWSpace.hpp>
#endif

#include <Kokkos_ScratchSpace.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_TaskScheduler.hpp>
#include <Kokkos_Layout.hpp>
#include <impl/Kokkos_HostSharedPtr.hpp>
#include <impl/Kokkos_Profiling_Interface.hpp>
#include <impl/Kokkos_InitializationSettings.hpp>

#include <omp.h>

#include <vector>

/*--------------------------------------------------------------------------*/

namespace Kokkos {

namespace Impl {
class OpenMPInternal;
}

/// \class OpenMP
/// \brief Kokkos device for multicore processors in the host memory space.
class OpenMP {
 public:
  //! Tag this class as a kokkos execution space
  using execution_space = OpenMP;

  using memory_space =
#ifdef KOKKOS_ENABLE_HBWSPACE
      Experimental::HBWSpace;
#else
      HostSpace;
#endif

  //! This execution space preferred device_type
  using device_type          = Kokkos::Device<execution_space, memory_space>;
  using array_layout         = LayoutRight;
  using size_type            = memory_space::size_type;
  using scratch_memory_space = ScratchMemorySpace<OpenMP>;

  OpenMP();

  OpenMP(int pool_size);

  /// \brief Print configuration information to the given output stream.
  void print_configuration(std::ostream& os, bool verbose = false) const;

  /// \brief is the instance running a parallel algorithm
  static bool in_parallel(OpenMP const& = OpenMP()) noexcept;

  /// \brief Wait until all dispatched functors complete on the given instance
  ///
  ///  This is a no-op on OpenMP
  static void impl_static_fence(std::string const& name);

  void fence(std::string const& name =
                 "Kokkos::OpenMP::fence: Unnamed Instance Fence") const;

  /// \brief Does the given instance return immediately after launching
  /// a parallel algorithm
  ///
  /// This always returns false on OpenMP
  inline static bool is_asynchronous(OpenMP const& = OpenMP()) noexcept;

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_3
  /// \brief Partition the default instance and call 'f' on each new 'master'
  /// thread
  ///
  /// Func is a functor with the following signiture
  ///   void( int partition_id, int num_partitions )
  template <typename F>
  KOKKOS_DEPRECATED static void partition_master(
      F const& f, int requested_num_partitions = 0,
      int requested_partition_size = 0);
#endif

  static int concurrency(OpenMP const& = OpenMP());

  static void impl_initialize(InitializationSettings const&);

  /// \brief is the default execution space initialized for current 'master'
  /// thread
  static bool impl_is_initialized() noexcept;

  /// \brief Free any resources being consumed by the default execution space
  static void impl_finalize();

  static int impl_thread_pool_size(OpenMP const& = OpenMP()) noexcept;

  /** \brief  The rank of the executing thread in this thread pool */
  inline static int impl_thread_pool_rank() noexcept;

  inline static int impl_thread_pool_size(int depth, OpenMP const& = OpenMP());

  // use UniqueToken
  static int impl_max_hardware_threads() noexcept;

  // use UniqueToken
  KOKKOS_INLINE_FUNCTION
  static int impl_hardware_thread_id() noexcept;

  static int impl_get_current_max_threads() noexcept;

  Impl::OpenMPInternal* impl_internal_space_instance() const {
    return m_space_instance.get();
  }

  static constexpr const char* name() noexcept { return "OpenMP"; }
  uint32_t impl_instance_id() const noexcept { return 1; }

 private:
  friend bool operator==(OpenMP const& lhs, OpenMP const& rhs) {
    return lhs.impl_internal_space_instance() ==
           rhs.impl_internal_space_instance();
  }
  friend bool operator!=(OpenMP const& lhs, OpenMP const& rhs) {
    return !(lhs == rhs);
  }
  Kokkos::Impl::HostSharedPtr<Impl::OpenMPInternal> m_space_instance;
};

inline int OpenMP::impl_thread_pool_rank() noexcept {
  // FIXME_OPENMP Can we remove this when removing partition_master? It's only
  // used in one partition_master test
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_3
  KOKKOS_IF_ON_HOST(
      (return Impl::t_openmp_instance ? 0 : omp_get_thread_num();))
#else
  KOKKOS_IF_ON_HOST((return omp_get_thread_num();))
#endif

  KOKKOS_IF_ON_DEVICE((return -1;))
}

inline void OpenMP::impl_static_fence(std::string const& name) {
  Kokkos::Tools::Experimental::Impl::profile_fence_event<Kokkos::OpenMP>(
      name,
      Kokkos::Tools::Experimental::SpecialSynchronizationCases::
          GlobalDeviceSynchronization,
      []() {});
}

inline bool OpenMP::is_asynchronous(OpenMP const& /*instance*/) noexcept {
  return false;
}

inline int OpenMP::impl_thread_pool_size(int depth, OpenMP const& exec_space) {
  return depth < 2 ? impl_thread_pool_size(exec_space) : 1;
}

KOKKOS_INLINE_FUNCTION
int OpenMP::impl_hardware_thread_id() noexcept {
  KOKKOS_IF_ON_HOST((return omp_get_thread_num();))

  KOKKOS_IF_ON_DEVICE((return -1;))
}

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_3
template <typename F>
KOKKOS_DEPRECATED void OpenMP::partition_master(F const& f, int num_partitions,
                                                int partition_size) {
#if _OPENMP >= 201511
  if (omp_get_max_active_levels() > 1) {
#else
  if (omp_get_nested()) {
#endif
    using Exec = Impl::OpenMPInternal;

    Exec* prev_instance = &Impl::OpenMPInternal::singleton();

    Exec::validate_partition_impl(prev_instance->m_pool_size, num_partitions,
                                  partition_size);

    OpenMP::memory_space space;

#pragma omp parallel num_threads(num_partitions)
    {
      Exec thread_local_instance(partition_size);
      Impl::t_openmp_instance = &thread_local_instance;

      size_t pool_reduce_bytes  = 32 * partition_size;
      size_t team_reduce_bytes  = 32 * partition_size;
      size_t team_shared_bytes  = 1024 * partition_size;
      size_t thread_local_bytes = 1024;

      thread_local_instance.resize_thread_data(
          pool_reduce_bytes, team_reduce_bytes, team_shared_bytes,
          thread_local_bytes);

      omp_set_num_threads(partition_size);
      f(omp_get_thread_num(), omp_get_num_threads());
      Impl::t_openmp_instance = nullptr;
    }
  } else {
    // nested openmp not enabled
    f(0, 1);
  }
}
#endif

namespace Tools {
namespace Experimental {
template <>
struct DeviceTypeTraits<OpenMP> {
  static constexpr DeviceType id = DeviceType::OpenMP;
  static int device_id(const OpenMP&) { return 0; }
};
}  // namespace Experimental
}  // namespace Tools
}  // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

template <>
struct MemorySpaceAccess<Kokkos::OpenMP::memory_space,
                         Kokkos::OpenMP::scratch_memory_space> {
  enum : bool { assignable = false };
  enum : bool { accessible = true };
  enum : bool { deepcopy = false };
};

}  // namespace Impl
}  // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

#include <OpenMP/Kokkos_OpenMP_Instance.hpp>
#include <OpenMP/Kokkos_OpenMP_Team.hpp>
#include <OpenMP/Kokkos_OpenMP_Parallel.hpp>
#include <OpenMP/Kokkos_OpenMP_Task.hpp>

#include <KokkosExp_MDRangePolicy.hpp>
/*--------------------------------------------------------------------------*/

#endif /* #if defined( KOKKOS_ENABLE_OPENMP ) && defined( _OPENMP ) */
#endif /* #ifndef KOKKOS_OPENMP_HPP */
