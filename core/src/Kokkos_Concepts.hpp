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
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
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

#ifndef KOKKOS_CORE_CONCEPTS_HPP
#define KOKKOS_CORE_CONCEPTS_HPP

#include <type_traits>

// Needed for 'is_space<S>::host_mirror_space
#include <Kokkos_Core_fwd.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

// Schedules for Execution Policies
struct Static {};
struct Dynamic {};

// Schedule Wrapper Type
template <class T>
struct Schedule {
  static_assert(std::is_same<T, Static>::value ||
                    std::is_same<T, Dynamic>::value,
                "Kokkos: Invalid Schedule<> type.");
  using schedule_type = Schedule;
  using type          = T;
};

// Specify Iteration Index Type
template <typename T>
struct IndexType {
  static_assert(std::is_integral<T>::value, "Kokkos: Invalid IndexType<>.");
  using index_type = IndexType;
  using type       = T;
};

namespace Experimental {
struct WorkItemProperty {
  template <unsigned long Property>
  struct ImplWorkItemProperty {
    static const unsigned value = Property;
    using work_item_property    = ImplWorkItemProperty<Property>;
  };

  constexpr static const ImplWorkItemProperty<0> None =
      ImplWorkItemProperty<0>();
  constexpr static const ImplWorkItemProperty<1> HintLightWeight =
      ImplWorkItemProperty<1>();
  constexpr static const ImplWorkItemProperty<2> HintHeavyWeight =
      ImplWorkItemProperty<2>();
  constexpr static const ImplWorkItemProperty<4> HintRegular =
      ImplWorkItemProperty<4>();
  constexpr static const ImplWorkItemProperty<8> HintIrregular =
      ImplWorkItemProperty<8>();
  using None_t            = ImplWorkItemProperty<0>;
  using HintLightWeight_t = ImplWorkItemProperty<1>;
  using HintHeavyWeight_t = ImplWorkItemProperty<2>;
  using HintRegular_t     = ImplWorkItemProperty<4>;
  using HintIrregular_t   = ImplWorkItemProperty<8>;
};

template <unsigned long pv1, unsigned long pv2>
inline constexpr WorkItemProperty::ImplWorkItemProperty<pv1 | pv2> operator|(
    WorkItemProperty::ImplWorkItemProperty<pv1>,
    WorkItemProperty::ImplWorkItemProperty<pv2>) {
  return WorkItemProperty::ImplWorkItemProperty<pv1 | pv2>();
}

template <unsigned long pv1, unsigned long pv2>
inline constexpr WorkItemProperty::ImplWorkItemProperty<pv1 & pv2> operator&(
    WorkItemProperty::ImplWorkItemProperty<pv1>,
    WorkItemProperty::ImplWorkItemProperty<pv2>) {
  return WorkItemProperty::ImplWorkItemProperty<pv1 & pv2>();
}

template <unsigned long pv1, unsigned long pv2>
inline constexpr bool operator==(WorkItemProperty::ImplWorkItemProperty<pv1>,
                                 WorkItemProperty::ImplWorkItemProperty<pv2>) {
  return pv1 == pv2;
}

}  // namespace Experimental

/**\brief Specify Launch Bounds for CUDA execution.
 *
 *  If no launch bounds specified then do not set launch bounds.
 */
template <unsigned int maxT = 0 /* Max threads per block */
          ,
          unsigned int minB = 0 /* Min blocks per SM */
          >
struct LaunchBounds {
  using launch_bounds = LaunchBounds;
  using type          = LaunchBounds<maxT, minB>;
  static unsigned int constexpr maxTperB{maxT};
  static unsigned int constexpr minBperSM{minB};
};

}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

#define KOKKOS_IMPL_IS_CONCEPT(CONCEPT)                                        \
  template <typename T>                                                        \
  struct is_##CONCEPT {                                                        \
   private:                                                                    \
    template <typename, typename = std::true_type>                             \
    struct have : std::false_type {};                                          \
    template <typename U>                                                      \
    struct have<U, typename std::is_base_of<typename U::CONCEPT, U>::type>     \
        : std::true_type {};                                                   \
    template <typename U>                                                      \
    struct have<U,                                                             \
                typename std::is_base_of<typename U::CONCEPT##_type, U>::type> \
        : std::true_type {};                                                   \
                                                                               \
   public:                                                                     \
    static constexpr bool value =                                              \
        is_##CONCEPT::template have<typename std::remove_cv<T>::type>::value;  \
    constexpr operator bool() const noexcept { return value; }                 \
  };

// Public concept:

KOKKOS_IMPL_IS_CONCEPT(memory_space)
KOKKOS_IMPL_IS_CONCEPT(memory_traits)
KOKKOS_IMPL_IS_CONCEPT(execution_space)
KOKKOS_IMPL_IS_CONCEPT(execution_policy)
KOKKOS_IMPL_IS_CONCEPT(array_layout)
KOKKOS_IMPL_IS_CONCEPT(reducer)
namespace Experimental {
KOKKOS_IMPL_IS_CONCEPT(work_item_property)
}

namespace Impl {

// For backward compatibility:

using Kokkos::is_array_layout;
using Kokkos::is_execution_policy;
using Kokkos::is_execution_space;
using Kokkos::is_memory_space;
using Kokkos::is_memory_traits;

// Implementation concept:

KOKKOS_IMPL_IS_CONCEPT(iteration_pattern)
KOKKOS_IMPL_IS_CONCEPT(schedule_type)
KOKKOS_IMPL_IS_CONCEPT(index_type)
KOKKOS_IMPL_IS_CONCEPT(launch_bounds)
KOKKOS_IMPL_IS_CONCEPT(thread_team_member)
KOKKOS_IMPL_IS_CONCEPT(host_thread_team_member)

}  // namespace Impl

#undef KOKKOS_IMPL_IS_CONCEPT

}  // namespace Kokkos

namespace Kokkos {
namespace Impl {

template <class Object>
class has_member_team_shmem_size {
  template <typename T>
  static int32_t test_for_member(decltype(&T::team_shmem_size)) {
    return int32_t(0);
  }
  template <typename T>
  static int64_t test_for_member(...) {
    return int64_t(0);
  }

 public:
  constexpr static bool value =
      sizeof(test_for_member<Object>(nullptr)) == sizeof(int32_t);
};

template <class Object>
class has_member_shmem_size {
  template <typename T>
  static int32_t test_for_member(decltype(&T::shmem_size_me)) {
    return int32_t(0);
  }
  template <typename T>
  static int64_t test_for_member(...) {
    return int64_t(0);
  }

 public:
  constexpr static bool value =
      sizeof(test_for_member<Object>(0)) == sizeof(int32_t);
};

}  // namespace Impl
}  // namespace Kokkos
//----------------------------------------------------------------------------

namespace Kokkos {

template <class ExecutionSpace, class MemorySpace>
struct Device {
  static_assert(Kokkos::is_execution_space<ExecutionSpace>::value,
                "Execution space is not valid");
  static_assert(Kokkos::is_memory_space<MemorySpace>::value,
                "Memory space is not valid");
  using execution_space = ExecutionSpace;
  using memory_space    = MemorySpace;
  using device_type     = Device<execution_space, memory_space>;
};

namespace Impl {

template <typename T>
struct is_device_helper : std::false_type {};

template <typename ExecutionSpace, typename MemorySpace>
struct is_device_helper<Device<ExecutionSpace, MemorySpace>> : std::true_type {
};

}  // namespace Impl

template <typename T>
using is_device =
    typename Impl::is_device_helper<typename std::remove_cv<T>::type>::type;

//----------------------------------------------------------------------------

template <typename T>
struct is_space {
 private:
  template <typename, typename = void>
  struct exe : std::false_type {
    using space = void;
  };

  template <typename, typename = void>
  struct mem : std::false_type {
    using space = void;
  };

  template <typename, typename = void>
  struct dev : std::false_type {
    using space = void;
  };

  template <typename U>
  struct exe<U, typename std::conditional<true, void,
                                          typename U::execution_space>::type>
      : std::is_same<U, typename U::execution_space>::type {
    using space = typename U::execution_space;
  };

  template <typename U>
  struct mem<
      U, typename std::conditional<true, void, typename U::memory_space>::type>
      : std::is_same<U, typename U::memory_space>::type {
    using space = typename U::memory_space;
  };

  template <typename U>
  struct dev<
      U, typename std::conditional<true, void, typename U::device_type>::type>
      : std::is_same<U, typename U::device_type>::type {
    using space = typename U::device_type;
  };

  using is_exe =
      typename is_space<T>::template exe<typename std::remove_cv<T>::type>;
  using is_mem =
      typename is_space<T>::template mem<typename std::remove_cv<T>::type>;
  using is_dev =
      typename is_space<T>::template dev<typename std::remove_cv<T>::type>;

 public:
  static constexpr bool value = is_exe::value || is_mem::value || is_dev::value;

  constexpr operator bool() const noexcept { return value; }

  using execution_space = typename is_exe::space;
  using memory_space    = typename is_mem::space;

  // For backward compatibility, deprecated in favor of
  // Kokkos::Impl::HostMirror<S>::host_mirror_space

  using host_memory_space = typename std::conditional<
      std::is_same<memory_space, Kokkos::HostSpace>::value
#if defined(KOKKOS_ENABLE_CUDA)
          || std::is_same<memory_space, Kokkos::CudaUVMSpace>::value ||
          std::is_same<memory_space, Kokkos::CudaHostPinnedSpace>::value
#endif /* #if defined( KOKKOS_ENABLE_CUDA ) */
      ,
      memory_space, Kokkos::HostSpace>::type;

#if defined(KOKKOS_ENABLE_CUDA)
  using host_execution_space = typename std::conditional<
      std::is_same<execution_space, Kokkos::Cuda>::value,
      Kokkos::DefaultHostExecutionSpace, execution_space>::type;
#else
#if defined(KOKKOS_ENABLE_OPENMPTARGET)
  using host_execution_space = typename std::conditional<
      std::is_same<execution_space, Kokkos::Experimental::OpenMPTarget>::value,
      Kokkos::DefaultHostExecutionSpace, execution_space>::type;
#else
  using host_execution_space = execution_space;
#endif
#endif

  using host_mirror_space = typename std::conditional<
      std::is_same<execution_space, host_execution_space>::value &&
          std::is_same<memory_space, host_memory_space>::value,
      T, Kokkos::Device<host_execution_space, host_memory_space>>::type;
};

// For backward compatibility

namespace Impl {

using Kokkos::is_space;

}

}  // namespace Kokkos

//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

/**\brief  Access relationship between DstMemorySpace and SrcMemorySpace
 *
 *  The default case can assume accessibility for the same space.
 *  Specializations must be defined for different memory spaces.
 */
template <typename DstMemorySpace, typename SrcMemorySpace>
struct MemorySpaceAccess {
  static_assert(Kokkos::is_memory_space<DstMemorySpace>::value &&
                    Kokkos::is_memory_space<SrcMemorySpace>::value,
                "template arguments must be memory spaces");

  /**\brief  Can a View (or pointer) to memory in SrcMemorySpace
   *         be assigned to a View (or pointer) to memory marked DstMemorySpace.
   *
   *  1. DstMemorySpace::execution_space == SrcMemorySpace::execution_space
   *  2. All execution spaces that can access DstMemorySpace can also access
   *     SrcMemorySpace.
   */
  enum { assignable = std::is_same<DstMemorySpace, SrcMemorySpace>::value };

  /**\brief  For all DstExecSpace::memory_space == DstMemorySpace
   *         DstExecSpace can access SrcMemorySpace.
   */
  enum { accessible = assignable };

  /**\brief  Does a DeepCopy capability exist
   *         to DstMemorySpace from SrcMemorySpace
   */
  enum { deepcopy = assignable };
};

}  // namespace Impl
}  // namespace Kokkos

namespace Kokkos {

/**\brief  Can AccessSpace access MemorySpace ?
 *
 *   Requires:
 *     Kokkos::is_space< AccessSpace >::value
 *     Kokkos::is_memory_space< MemorySpace >::value
 *
 *   Can AccessSpace::execution_space access MemorySpace ?
 *     enum : bool { accessible };
 *
 *   Is View<AccessSpace::memory_space> assignable from View<MemorySpace> ?
 *     enum : bool { assignable };
 *
 *   If ! accessible then through which intercessory memory space
 *   should a be used to deep copy memory for
 *     AccessSpace::execution_space
 *   to get access.
 *   When AccessSpace::memory_space == Kokkos::HostSpace
 *   then space is the View host mirror space.
 */
template <typename AccessSpace, typename MemorySpace>
struct SpaceAccessibility {
 private:
  static_assert(Kokkos::is_space<AccessSpace>::value,
                "template argument #1 must be a Kokkos space");

  static_assert(Kokkos::is_memory_space<MemorySpace>::value,
                "template argument #2 must be a Kokkos memory space");

  // The input AccessSpace may be a Device<ExecSpace,MemSpace>
  // verify that it is a valid combination of spaces.
  static_assert(Kokkos::Impl::MemorySpaceAccess<
                    typename AccessSpace::execution_space::memory_space,
                    typename AccessSpace::memory_space>::accessible,
                "template argument #1 is an invalid space");

  using exe_access = Kokkos::Impl::MemorySpaceAccess<
      typename AccessSpace::execution_space::memory_space, MemorySpace>;

  using mem_access =
      Kokkos::Impl::MemorySpaceAccess<typename AccessSpace::memory_space,
                                      MemorySpace>;

 public:
  /**\brief  Can AccessSpace::execution_space access MemorySpace ?
   *
   *  Default based upon memory space accessibility.
   *  Specialization required for other relationships.
   */
  enum { accessible = exe_access::accessible };

  /**\brief  Can assign to AccessSpace from MemorySpace ?
   *
   *  Default based upon memory space accessibility.
   *  Specialization required for other relationships.
   */
  enum {
    assignable = is_memory_space<AccessSpace>::value && mem_access::assignable
  };

  /**\brief  Can deep copy to AccessSpace::memory_Space from MemorySpace ?  */
  enum { deepcopy = mem_access::deepcopy };

  // What intercessory space for AccessSpace::execution_space
  // to be able to access MemorySpace?
  // If same memory space or not accessible use the AccessSpace
  // else construct a device with execution space and memory space.
  using space = typename std::conditional<
      std::is_same<typename AccessSpace::memory_space, MemorySpace>::value ||
          !exe_access::accessible,
      AccessSpace,
      Kokkos::Device<typename AccessSpace::execution_space, MemorySpace>>::type;
};

}  // namespace Kokkos

namespace Kokkos {
namespace Impl {

using Kokkos::SpaceAccessibility;  // For backward compatibility

}
}  // namespace Kokkos

//----------------------------------------------------------------------------

#endif  // KOKKOS_CORE_CONCEPTS_HPP
