// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#include <Kokkos_Macros.hpp>
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif
#ifndef KOKKOS_HOSTSPACE_HPP
#define KOKKOS_HOSTSPACE_HPP

#include <cstring>
#include <string>
#include <iosfwd>
#include <iostream>
#include <typeinfo>

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Concepts.hpp>
#include <Kokkos_MemoryTraits.hpp>

#ifdef KOKKOS_ENABLE_HWLOC
#include <Kokkos_hwloc.hpp>
#include <Kokkos_NUMANodes.hpp>
#endif // KOKKOS_ENABLE_HWLOC

#include <impl/Kokkos_Traits.hpp>
#include <impl/Kokkos_Error.hpp>
#include <impl/Kokkos_SharedAlloc.hpp>
#include <impl/Kokkos_Tools.hpp>

#include "impl/Kokkos_HostSpace_deepcopy.hpp"

/*--------------------------------------------------------------------------*/

namespace Kokkos {
/// \class HostSpace
/// \brief Memory management for host memory.
///
/// HostSpace is a memory space that governs host memory.  "Host"
/// memory means the usual CPU-accessible memory.
class HostSpace {
 public:
  //! Tag this class as a kokkos memory space
  using memory_space = HostSpace;
  using size_type    = size_t;
  using index_type   = std::make_signed_t<size_type>;

  /// \typedef execution_space
  /// \brief Default execution space for this memory space.
  ///
  /// Every memory space has a default execution space.  This is
  /// useful for things like initializing a View (which happens in
  /// parallel using the View's default execution space).
  using execution_space = DefaultHostExecutionSpace;

  //! This memory space preferred device_type
  using device_type = Kokkos::Device<execution_space, memory_space>;

  HostSpace() = default;

#ifdef KOKKOS_ENABLE_HWLOC
  HostSpace(const HostSpace& rhs) {
    if (nullptr != rhs.membind_set) {
      this->membind_set = hwloc_bitmap_dup(rhs.membind_set);
      this->membind_policy = rhs.membind_policy;
      this->membind_flags = rhs.membind_flags;
#ifdef KOKKOS_ENABLE_DEBUG_HWLOC
      char *set_str;
      hwloc_bitmap_list_asprintf(&set_str, this->membind_set);
      std::cout << "INFO: " << __PRETTY_FUNCTION__
                << " membind_set: " << set_str
                << " membind_policy: " << this->membind_policy
                << " membind_flags: " << this->membind_flags
                << std::endl;
      free(set_str);
#endif // KOKKOS_ENABLE_DEBUG_HWLOC
    }
  }

  template<typename T,
    std::enable_if_t<std::is_same_v<std::decay_t<T>, hwloc_bitmap_t>
    // hwloc_memattr_id_e was introduced in version 2.3.0 (0x00020300)
#if HWLOC_API_VERSION >= 0x00020300
                  || std::is_same_v<std::decay_t<T>, hwloc_memattr_id_e>
#endif // HWLOC_API_VERSION
                  || Kokkos::is_kokkos_numanodes_v<T>
                  , int> = 0>
  HostSpace(T&& arg,
    const hwloc_membind_policy_t policy = KOKKOS_HWLOC_DEFAULT_MEMBIND_POLICY,
    const int flags = KOKKOS_HWLOC_DEFAULT_MEMBIND_FLAGS)
  : membind_policy(policy), membind_flags(flags) {
    hwloc_bitmap_t const membind_set_cur = static_cast<hwloc_bitmap_t>(
        Kokkos::hwloc::get_membind_set());
    const hwloc_topology_t topology = Kokkos::hwloc::get_topology();

    hwloc_bitmap_t membind_set_asked = hwloc_bitmap_alloc();

    if (nullptr == this->membind_set) {
      this->membind_set = hwloc_bitmap_alloc();
    }

    // build the user-requested set
    using rawT = std::decay_t<T>;

    if constexpr (std::is_same_v<rawT, hwloc_bitmap_t>) {

      hwloc_bitmap_copy(membind_set_asked, arg);

    // hwloc_memattr_id_e was introduced in version 2.3.0 (0x00020300)
#if HWLOC_API_VERSION >= 0x00020300
    } else if constexpr (std::is_same_v<rawT, hwloc_memattr_id_e>) {

      hwloc_bitmap_t process_binding = Kokkos::hwloc::get_process_binding();
      hwloc_obj_t best_node;
      struct hwloc_location initiator;
      initiator.type = HWLOC_LOCATION_TYPE_CPUSET;
      initiator.location.cpuset = process_binding;

      int err = hwloc_memattr_get_best_target(topology, arg,
            &initiator, 0, &best_node, nullptr);
      if (0 == err) {
        hwloc_bitmap_copy(membind_set_asked, best_node->nodeset);
      }

      // If the cpuset is spread across multiple Sockets or Packages, hwloc
      // might fail to decide THE best target, since the best for a core
      // may not be the best for another core. In this case, we need to do
      // more dichotomy and gather the best target of each core in cpuset.
      if (hwloc_bitmap_iszero(membind_set_asked)) {
        unsigned id;
        hwloc_bitmap_foreach_begin(id, process_binding) {
          // get cpuset of this core
          hwloc_obj_t obj_core = hwloc_get_obj_by_type(topology, HWLOC_OBJ_CORE, id);

          // get best numa node of this core, then OR with membind_set_asked
          if (nullptr != obj_core) {
            initiator.location.cpuset = obj_core->cpuset;
            err = hwloc_memattr_get_best_target(topology, arg, &initiator, 0, &best_node, nullptr);
            if (0 == err) {
              hwloc_bitmap_or(membind_set_asked, membind_set_asked, best_node->nodeset);
            }
          }

        } hwloc_bitmap_foreach_end();
      }
      // Foolproof: mark bitmap as processed as nodeset
      membind_flags |= HWLOC_MEMBIND_BYNODESET;
#endif // HWLOC_API_VERSION

    } else if constexpr (Kokkos::is_kokkos_numanodes_v<T>) {

      const bool is_physical = arg.is_physical();
      const int nb_nodes = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_NUMANODE);
      for (size_t i = 0; i < arg.size(); ++i) {
        const unsigned id = arg[i];
        // sanity check that asked node id exists
        if (((int)id) >= nb_nodes) {
#ifdef KOKKOS_ENABLE_DEBUG_HWLOC
          std::cout << "WARNING: " << __PRETTY_FUNCTION__
                    << " asked node id " << id << " does not exist"
                    << " and will be ignored"
                    << " (available nb_nodes = " << nb_nodes << ")"
                    << std::endl;
#endif // KOKKOS_ENABLE_DEBUG_HWLOC
          continue;
        }
        hwloc_obj_t obj = is_physical
                   ? hwloc_get_numanode_obj_by_os_index(topology, id)
                   : hwloc_get_obj_by_type(topology, HWLOC_OBJ_NUMANODE, id);
        if (nullptr != obj->nodeset) {
          hwloc_bitmap_or(membind_set_asked, membind_set_asked, obj->nodeset);
        }
      }
      // Foolproof: mark bitmap as processed as nodeset
      membind_flags |= HWLOC_MEMBIND_BYNODESET;

    } else {
      __builtin_unreachable();
    }

    // fit in available nodes in membind_set_cur
    // membind_set = membind_set_asked & membind_set_cur
    hwloc_bitmap_and(this->membind_set, membind_set_asked, membind_set_cur);

    // little check
    if (hwloc_bitmap_iszero(this->membind_set)) {
      char *set_cur, *set_asked;
      hwloc_bitmap_list_asprintf(&set_cur, membind_set_cur);
      hwloc_bitmap_list_asprintf(&set_asked, membind_set_asked);

      std::cout << "ERROR: " << __PRETTY_FUNCTION__
                << " membind_set is empty. Further allocation might fail."
                << " Please review nodelist argument and/or numactl runtime."
                << " Current membind nodeset: " << set_cur
                << " Asked membind nodeset: " << set_asked
                << std::endl;
      free(set_cur);
      free(set_asked);
    }

    hwloc_bitmap_free(membind_set_asked);
  }

  ~HostSpace() {
    if (nullptr != membind_set) {
      hwloc_bitmap_free(membind_set);
      membind_set = nullptr;
    }
  }
#endif  // KOKKOS_ENABLE_HWLOC

  /**\brief  Allocate untracked memory in the space */
  template <typename ExecutionSpace>
  void* allocate(const ExecutionSpace&, const size_t arg_alloc_size) const {
    return allocate(arg_alloc_size);
  }
  template <typename ExecutionSpace>
  void* allocate(const ExecutionSpace&, const char* arg_label,
                 const size_t arg_alloc_size,
                 const size_t arg_logical_size = 0) const {
    return allocate(arg_label, arg_alloc_size, arg_logical_size);
  }
  void* allocate(const size_t arg_alloc_size) const;
  void* allocate(const char* arg_label, const size_t arg_alloc_size,
                 const size_t arg_logical_size = 0) const;

  /**\brief  Deallocate untracked memory in the space */
  void deallocate(void* const arg_alloc_ptr, const size_t arg_alloc_size) const;
  void deallocate(const char* arg_label, void* const arg_alloc_ptr,
                  const size_t arg_alloc_size,
                  const size_t arg_logical_size = 0) const;

  void* impl_allocate(const char* arg_label, const size_t arg_alloc_size,
                      const size_t arg_logical_size = 0,
                      const Kokkos::Tools::SpaceHandle =
                          Kokkos::Tools::make_space_handle(name())) const;
  void impl_deallocate(const char* arg_label, void* const arg_alloc_ptr,
                       const size_t arg_alloc_size,
                       const size_t arg_logical_size = 0,
                       const Kokkos::Tools::SpaceHandle =
                           Kokkos::Tools::make_space_handle(name())) const;

  /**\brief Return Name of the MemorySpace */
  static constexpr const char* name() { return m_name; }

#ifdef KOKKOS_ENABLE_HWLOC
  hwloc_bitmap_t get_membind_set(void) const {
    return membind_set;
  }
  hwloc_membind_policy_t get_membind_policy(void) const {
    return membind_policy;
  }
  int get_membind_flags(void) const {
    return membind_flags;
  }
#endif  // KOKKOS_ENABLE_HWLOC

 private:
  static constexpr const char* m_name = "Host";

#ifdef KOKKOS_ENABLE_HWLOC
  hwloc_bitmap_t membind_set{nullptr};
  hwloc_membind_policy_t membind_policy;
  int membind_flags;
#endif  // KOKKOS_ENABLE_HWLOC

};

}  // namespace Kokkos

//----------------------------------------------------------------------------

namespace Kokkos {

namespace Impl {

static_assert(Kokkos::Impl::MemorySpaceAccess<Kokkos::HostSpace,
                                              Kokkos::HostSpace>::assignable);

template <typename MemSpace>
struct HostMirror {
 private:
  static_assert(is_memory_space_v<MemSpace>);

  // If input execution space can access HostSpace then keep it.
  // Example: Kokkos::OpenMP can access, Kokkos::Cuda cannot
  enum {
    keep_exe = Kokkos::SpaceAccessibility<typename MemSpace::execution_space,
                                          Kokkos::HostSpace>::accessible
  };
  // If HostSpace can access memory space then keep it.
  // Example: Cannot access Kokkos::CudaSpace, can access Kokkos::CudaUVMSpace
  enum {
    keep_mem =
        Kokkos::Impl::MemorySpaceAccess<Kokkos::HostSpace, MemSpace>::accessible
  };

 public:
  // Construct a device mirror type
  // Decision logic: First check if HostSpace can access the memory space.
  // If yes, keep it and check execution space compatibility.
  // If no, fall back to HostSpace::device_type.

  // keep_exe | keep_mem | Result
  // ---------|----------|-------
  //    T     |    T     | MemSpace::device_type
  //    F     |    T     | Device<HostSpace::execution_space, MemSpace>
  //    T     |    F     | HostSpace::device_type
  //    F     |    F     | HostSpace::device_type

  using device_type = std::conditional_t<
      keep_mem,
      std::conditional_t<
          keep_exe, typename MemSpace::device_type,
          Kokkos::Device<Kokkos::HostSpace::execution_space, MemSpace>>,
      Kokkos::HostSpace::device_type>;

  using execution_space = typename device_type::execution_space;
  using memory_space    = typename device_type::memory_space;

  // FIXME: should be deprecated eventually
  using Space = memory_space;
};

}  // namespace Impl

}  // namespace Kokkos

//----------------------------------------------------------------------------

KOKKOS_IMPL_SHARED_ALLOCATION_SPECIALIZATION(Kokkos::HostSpace);

//----------------------------------------------------------------------------

namespace Kokkos {

namespace Impl {

template <class ExecutionSpace>
struct DeepCopy<HostSpace, HostSpace, ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
    hostspace_parallel_deepcopy(dst, src, n);
  }

  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    if constexpr (!Kokkos::SpaceAccessibility<ExecutionSpace,
                                              Kokkos::HostSpace>::accessible) {
      exec.fence(
          "Kokkos::Impl::DeepCopy<HostSpace, HostSpace, "
          "ExecutionSpace>::DeepCopy: fence before copy");
      hostspace_parallel_deepcopy_async(dst, src, n);
    } else {
      hostspace_parallel_deepcopy_async(exec, dst, src, n);
    }
  }
};

}  // namespace Impl

}  // namespace Kokkos

#endif  // #define KOKKOS_HOSTSPACE_HPP
