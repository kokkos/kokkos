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

#ifndef KOKKOS_LOGICALMEMORYSPACE_HPP
#define KOKKOS_LOGICALMEMORYSPACE_HPP

#include <cstring>
#include <string>
#include <iosfwd>
#include <typeinfo>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Concepts.hpp>
#include <Kokkos_MemoryTraits.hpp>

#include <impl/Kokkos_Traits.hpp>
#include <impl/Kokkos_Error.hpp>
#include <impl/Kokkos_SharedAlloc.hpp>

#include "impl/Kokkos_HostSpace_deepcopy.hpp"
#include "impl/Kokkos_Profiling.hpp"

/*--------------------------------------------------------------------------*/

namespace Kokkos {

/// \class LogicalMemorySpace
/// \brief
///
/// LogicalMemorySpace is a space that is identical to another space, but
/// differentiable by name and template argument

template <class Namer, class BaseSpace, class DefaultExecutionSpace = void,
          bool SharesAccessWithBase = true>
class LogicalMemorySpace {
 public:
  //! Tag this class as a kokkos memory space
  typedef LogicalMemorySpace<Namer, BaseSpace, DefaultExecutionSpace,
                             SharesAccessWithBase>
      memory_space;
  typedef size_t size_type;

  /// \typedef execution_space
  /// \brief Default execution space for this memory space.
  ///
  /// Every memory space has a default execution space.  This is
  /// useful for things like initializing a View (which happens in
  /// parallel using the View's default execution space).

  using execution_space =
      typename std::conditional<std::is_void<DefaultExecutionSpace>::value,
                                typename BaseSpace::execution_space,
                                DefaultExecutionSpace>::type;

  typedef Kokkos::Device<execution_space, memory_space> device_type;

  /**\brief  Default memory space instance */
  LogicalMemorySpace() : underlying_allocator(){};
  LogicalMemorySpace(LogicalMemorySpace&& rhs)      = default;
  LogicalMemorySpace(const LogicalMemorySpace& rhs) = default;
  LogicalMemorySpace& operator=(LogicalMemorySpace&&) = default;
  LogicalMemorySpace& operator=(const LogicalMemorySpace&) = default;
  ~LogicalMemorySpace()                                    = default;

  BaseSpace underlying_allocator;

  template <typename... Args>
  LogicalMemorySpace(Args&&... args) : underlying_allocator(args...) {}

  /**\brief  Allocate untracked memory in the space */
  void* allocate(const size_t arg_alloc_size) const {
    return underlying_allocator.allocate(arg_alloc_size);
  }

  /**\brief  Deallocate untracked memory in the space */
  void deallocate(void* const arg_alloc_ptr,
                  const size_t arg_alloc_size) const {
    return underlying_allocator.deallocate(arg_alloc_ptr, arg_alloc_size);
  }

  /**\brief Return Name of the MemorySpace */
  constexpr static const char* name() { return Namer::name(); }

 private:
  friend class Kokkos::Impl::SharedAllocationRecord<memory_space, void>;
};

}  // namespace Kokkos

//----------------------------------------------------------------------------

namespace Kokkos {

namespace Impl {

template <class Namer, typename BaseSpace, typename DefaultExecutionSpace,
          typename OtherSpace>
struct MemorySpaceAccess<
    Kokkos::LogicalMemorySpace<Namer, BaseSpace, DefaultExecutionSpace, true>,
    OtherSpace> {
  enum { assignable = MemorySpaceAccess<BaseSpace, OtherSpace>::assignable };
  enum { accessible = MemorySpaceAccess<BaseSpace, OtherSpace>::accessible };
  enum { deepcopy = MemorySpaceAccess<BaseSpace, OtherSpace>::deepcopy };
};

template <class Namer, typename BaseSpace, typename DefaultExecutionSpace>
struct MemorySpaceAccess<
    HostSpace, Kokkos::LogicalMemorySpace<Namer, BaseSpace,
                                          DefaultExecutionSpace, true> > {
  enum { assignable = true };
  enum { accessible = true };
  enum { deepcopy = true };
};

}  // namespace Impl

}  // namespace Kokkos

//----------------------------------------------------------------------------

namespace Kokkos {

namespace Impl {

template <class Namer, class BaseSpace, class DefaultExecutionSpace,
          bool SharesAccessSemanticsWithBase>
class SharedAllocationRecord<
    Kokkos::LogicalMemorySpace<Namer, BaseSpace, DefaultExecutionSpace,
                               SharesAccessSemanticsWithBase>,
    void> : public SharedAllocationRecord<void, void> {
 private:
  typedef Kokkos::LogicalMemorySpace<Namer, BaseSpace, DefaultExecutionSpace,
                                     SharesAccessSemanticsWithBase>
      SpaceType;
  typedef SharedAllocationRecord<void, void> RecordBase;

  friend SpaceType;
  SharedAllocationRecord(const SharedAllocationRecord&) = delete;
  SharedAllocationRecord& operator=(const SharedAllocationRecord&) = delete;

  static void deallocate(RecordBase* arg_rec) {
    delete static_cast<SharedAllocationRecord*>(arg_rec);
  }

#ifdef KOKKOS_DEBUG
  /**\brief  Root record for tracked allocations from this LogicalMemorySpace
   * instance */
  static RecordBase s_root_record;
#endif

  const SpaceType m_space;

 protected:
  ~SharedAllocationRecord() {
#if defined(KOKKOS_ENABLE_PROFILING)
    if (Kokkos::Profiling::profileLibraryLoaded()) {
      Kokkos::Profiling::deallocateData(
          Kokkos::Profiling::make_space_handle(m_space.name()),
          RecordBase::m_alloc_ptr->m_label, data(), size());
    }
#endif

    m_space.deallocate(SharedAllocationRecord<void, void>::m_alloc_ptr,
                       SharedAllocationRecord<void, void>::m_alloc_size);
  }
  SharedAllocationRecord() = default;

  SharedAllocationRecord(
      const SpaceType& arg_space, const std::string& arg_label,
      const size_t arg_alloc_size,
      const RecordBase::function_type arg_dealloc = &deallocate)
      : SharedAllocationRecord<void, void>(
#ifdef KOKKOS_DEBUG
            &SharedAllocationRecord<SpaceType, void>::s_root_record,
#endif
            reinterpret_cast<SharedAllocationHeader*>(arg_space.allocate(
                sizeof(SharedAllocationHeader) + arg_alloc_size)),
            sizeof(SharedAllocationHeader) + arg_alloc_size, arg_dealloc),
        m_space(arg_space) {
#if defined(KOKKOS_ENABLE_PROFILING)
    if (Kokkos::Profiling::profileLibraryLoaded()) {
      Kokkos::Profiling::allocateData(
          Kokkos::Profiling::make_space_handle(arg_space.name()), arg_label,
          data(), arg_alloc_size);
    }
#endif
    // Fill in the Header information
    RecordBase::m_alloc_ptr->m_record =
        static_cast<SharedAllocationRecord<void, void>*>(this);

    strncpy(RecordBase::m_alloc_ptr->m_label, arg_label.c_str(),
            SharedAllocationHeader::maximum_label_length);
    // Set last element zero, in case c_str is too long
    RecordBase::m_alloc_ptr
        ->m_label[SharedAllocationHeader::maximum_label_length - 1] = (char)0;
  }

 public:
  inline std::string get_label() const {
    return std::string(RecordBase::head()->m_label);
  }
  KOKKOS_INLINE_FUNCTION static SharedAllocationRecord* allocate(
      const SpaceType& arg_space, const std::string& arg_label,
      const size_t arg_alloc_size) {
#if defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST)
    return new SharedAllocationRecord(arg_space, arg_label, arg_alloc_size);
#else
    (void)arg_space;
    (void)arg_label;
    (void)arg_alloc_size;
    return (SharedAllocationRecord*)0;
#endif
  }

  /**\brief  Allocate tracked memory in the space */
  static void* allocate_tracked(const SpaceType& arg_space,
                                const std::string& arg_label,
                                const size_t arg_alloc_size) {
    if (!arg_alloc_size) return (void*)0;

    SharedAllocationRecord* const r =
        allocate(arg_space, arg_label, arg_alloc_size);

    RecordBase::increment(r);

    return r->data();
  }

  /**\brief  Reallocate tracked memory in the space */
  static void* reallocate_tracked(void* const arg_alloc_ptr,
                                  const size_t arg_alloc_size) {
    SharedAllocationRecord* const r_old = get_record(arg_alloc_ptr);
    SharedAllocationRecord* const r_new =
        allocate(r_old->m_space, r_old->get_label(), arg_alloc_size);

    Kokkos::Impl::DeepCopy<SpaceType, SpaceType>(
        r_new->data(), r_old->data(), std::min(r_old->size(), r_new->size()));

    RecordBase::increment(r_new);
    RecordBase::decrement(r_old);

    return r_new->data();
  }
  /**\brief  Deallocate tracked memory in the space */
  static void deallocate_tracked(void* const arg_alloc_ptr) {
    if (arg_alloc_ptr != 0) {
      SharedAllocationRecord* const r = get_record(arg_alloc_ptr);

      RecordBase::decrement(r);
    }
  }

  static SharedAllocationRecord* get_record(void* alloc_ptr) {
    typedef SharedAllocationHeader Header;
    typedef SharedAllocationRecord<SpaceType, void> RecordHost;

    SharedAllocationHeader const* const head =
        alloc_ptr ? Header::get_header(alloc_ptr) : (SharedAllocationHeader*)0;
    RecordHost* const record =
        head ? static_cast<RecordHost*>(head->m_record) : (RecordHost*)0;

    if (!alloc_ptr || record->m_alloc_ptr != head) {
      Kokkos::Impl::throw_runtime_exception(
          std::string("Kokkos::Impl::SharedAllocationRecord< SpaceType , "
                      "void >::get_record ERROR"));
    }

    return record;
  }
#ifdef KOKKOS_DEBUG
  static void print_records(std::ostream& s, const SpaceType&,
                            bool detail = false) {
    SharedAllocationRecord<void, void>::print_host_accessible_records(
        s, "HostSpace", &s_root_record, detail);
  }
#else
  static void print_records(std::ostream&, const SpaceType&,
                            bool detail = false) {
    (void)detail;
    throw_runtime_exception(
        "SharedAllocationRecord<HostSpace>::print_records only works with "
        "KOKKOS_DEBUG enabled");
  }
#endif
};
#ifdef KOKKOS_DEBUG
/**\brief  Root record for tracked allocations from this HostSpace instance */
template <const char* Name, class BaseSpace, class DefaultExecutionSpace,
          bool SharesAccessSemanticsWithBase>
RecordBase<BaseSpace, DefaultExecutionSpace,
           SharesAccessSemanticsWithBase>::s_root_record;
#endif

}  // namespace Impl

}  // namespace Kokkos

//----------------------------------------------------------------------------

namespace Kokkos {

namespace Impl {

template <class Namer, class BaseSpace, class DefaultExecutionSpace,
          bool SharesAccessSemanticsWithBase, class ExecutionSpace>
struct DeepCopy<
    Kokkos::LogicalMemorySpace<Namer, BaseSpace, DefaultExecutionSpace,
                               SharesAccessSemanticsWithBase>,
    Kokkos::LogicalMemorySpace<Namer, BaseSpace, DefaultExecutionSpace,
                               SharesAccessSemanticsWithBase>,
    ExecutionSpace> {
  DeepCopy(void* dst, void* src, size_t n) {
    DeepCopy<BaseSpace, BaseSpace, ExecutionSpace>(dst, src, n);
  }
  DeepCopy(const ExecutionSpace& exec, void* dst, void* src, size_t n) {
    DeepCopy<BaseSpace, BaseSpace, ExecutionSpace>(exec, dst, src, n);
  }
};

template <class Namer, class BaseSpace, class DefaultExecutionSpace,
          bool SharesAccessSemanticsWithBase, class ExecutionSpace,
          class SourceSpace>
struct DeepCopy<
    SourceSpace,
    Kokkos::LogicalMemorySpace<Namer, BaseSpace, DefaultExecutionSpace,
                               SharesAccessSemanticsWithBase>,
    ExecutionSpace> {
  DeepCopy(void* dst, void* src, size_t n) {
    DeepCopy<SourceSpace, BaseSpace, ExecutionSpace>(dst, src, n);
  }
  DeepCopy(const ExecutionSpace& exec, void* dst, void* src, size_t n) {
    DeepCopy<SourceSpace, BaseSpace, ExecutionSpace>(exec, dst, src, n);
  }
};

template <class Namer, class BaseSpace, class DefaultExecutionSpace,
          bool SharesAccessSemanticsWithBase, class ExecutionSpace,
          class DestinationSpace>
struct DeepCopy<
    Kokkos::LogicalMemorySpace<Namer, BaseSpace, DefaultExecutionSpace,
                               SharesAccessSemanticsWithBase>,
    DestinationSpace, ExecutionSpace> {
  DeepCopy(void* dst, void* src, size_t n) {
    DeepCopy<BaseSpace, DestinationSpace, ExecutionSpace>(dst, src, n);
  }
  DeepCopy(const ExecutionSpace& exec, void* dst, void* src, size_t n) {
    DeepCopy<BaseSpace, DestinationSpace, ExecutionSpace>(exec, dst, src, n);
  }
};
}  // namespace Impl

}  // namespace Kokkos

#endif  // #define KOKKOS_LOGICALMEMORYSPACE_HPP
