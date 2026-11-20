// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_IMPL_SHAREDALLOC_TIMPL_HPP
#define KOKKOS_IMPL_SHAREDALLOC_TIMPL_HPP

#include <Kokkos_Macros.hpp>
#include <Kokkos_Core_fwd.hpp>

#include <impl/Kokkos_SharedAlloc.hpp>

#include <algorithm>
#include <cstdio>
#include <ostream>
#include <string>

namespace Kokkos {
namespace Impl {

namespace detail {

template <class MemorySpace, class ExecutionSpace>
void initialize_shared_allocation_header(
    SharedAllocationRecord<void, void>* record,
    SharedAllocationHeader* allocation_header,
    ExecutionSpace const& execution_space, std::string const& label,
    bool fence_after_copy) {
  if constexpr (MemorySpaceAccess<HostSpace, MemorySpace>::accessible) {
    fill_host_accessible_header_info(record, *allocation_header, label);
  } else {
    SharedAllocationHeader header;
    fill_host_accessible_header_info(record, header, label);

    Kokkos::Impl::DeepCopy<MemorySpace, HostSpace>(
        execution_space, allocation_header, &header,
        sizeof(SharedAllocationHeader));

    if (fence_after_copy) {
      execution_space.fence(
          std::string("SharedAllocationRecord<") + MemorySpace::name() +
          ", void>::SharedAllocationRecord(): fence after copying header "
          "from HostSpace");
    }
  }
}

}  // namespace detail

template <class MemorySpace>
template <class ExecutionSpace>
SharedAllocationRecord<MemorySpace, void>::SharedAllocationRecord(
    ExecutionSpace const& execution_space, MemorySpace const& space,
    std::string const& label, std::size_t alloc_size,
    SharedAllocationRecord<void, void>::function_type dealloc)
    : SharedAllocationRecord<void, void>(
#ifdef KOKKOS_ENABLE_DEBUG
          &s_root_record,
#endif
          checked_allocation_with_header(execution_space, space, label,
                                         alloc_size),
          sizeof(SharedAllocationHeader) + alloc_size, dealloc, label),
      m_space(space) {
  detail::initialize_shared_allocation_header<MemorySpace>(
      this, this->m_alloc_ptr, execution_space, label, false);
}

template <class MemorySpace>
SharedAllocationRecord<MemorySpace, void>::SharedAllocationRecord(
    MemorySpace const& space, std::string const& label, std::size_t alloc_size,
    SharedAllocationRecord<void, void>::function_type dealloc)
    : SharedAllocationRecord<void, void>(
#ifdef KOKKOS_ENABLE_DEBUG
          &s_root_record,
#endif
          checked_allocation_with_header(space, label, alloc_size),
          sizeof(SharedAllocationHeader) + alloc_size, dealloc, label),
      m_space(space) {
  if constexpr (MemorySpaceAccess<HostSpace, MemorySpace>::accessible) {
    fill_host_accessible_header_info(this, *this->m_alloc_ptr, label);
  } else {
    typename MemorySpace::execution_space execution_space;
    detail::initialize_shared_allocation_header<MemorySpace>(
        this, this->m_alloc_ptr, execution_space, label, true);
  }
}

// NOLINTBEGIN(bugprone-exception-escape)
template <class MemorySpace>
SharedAllocationRecord<MemorySpace, void>::~SharedAllocationRecord() {
  auto alloc_ptr  = this->m_alloc_ptr;
  auto alloc_size = this->m_alloc_size;
  auto label      = this->m_label;
  m_space.deallocate(label.c_str(), alloc_ptr, alloc_size,
                     alloc_size - sizeof(SharedAllocationHeader));
}
// NOLINTEND(bugprone-exception-escape)

template <class MemorySpace>
SharedAllocationRecord<MemorySpace, void>* allocate_shared_allocation_record(
    MemorySpace const& space, std::string const& label, size_t alloc_size) {
  return new SharedAllocationRecord<MemorySpace, void>(space, label,
                                                       alloc_size);
}

template <class MemorySpace>
typename SharedAllocationRecord<MemorySpace, void>::derived_t*
SharedAllocationRecord<MemorySpace, void>::allocate(MemorySpace const& space,
                                                    std::string const& label,
                                                    size_t alloc_size) {
  return allocate_shared_allocation_record<MemorySpace>(space, label,
                                                        alloc_size);
}

template <class MemorySpace>
void* allocate_tracked_shared_allocation(MemorySpace const& space,
                                         std::string const& label,
                                         size_t alloc_size) {
  if (!alloc_size) return nullptr;

  auto* record =
      allocate_shared_allocation_record<MemorySpace>(space, label, alloc_size);
  SharedAllocationRecord<void, void>::increment(record);
  return record->data();
}

template <class MemorySpace>
void* SharedAllocationRecord<MemorySpace, void>::allocate_tracked(
    MemorySpace const& space, std::string const& label, size_t alloc_size) {
  return allocate_tracked_shared_allocation<MemorySpace>(space, label,
                                                         alloc_size);
}

template <class MemorySpace>
void deallocate_tracked_shared_allocation(void* alloc_ptr) {
  if (alloc_ptr != nullptr) {
    auto* record = get_shared_allocation_record<MemorySpace>(alloc_ptr);
    SharedAllocationRecord<void, void>::decrement(record);
  }
}

template <class MemorySpace>
void SharedAllocationRecord<MemorySpace, void>::deallocate_tracked(
    void* alloc_ptr) {
  deallocate_tracked_shared_allocation<MemorySpace>(alloc_ptr);
}

template <class MemorySpace>
SharedAllocationRecord<MemorySpace, void>* get_shared_allocation_record(
    void* alloc_ptr) {
  using record_type = SharedAllocationRecord<MemorySpace, void>;
  using header_type = SharedAllocationHeader;

  header_type const* const allocation_header =
      alloc_ptr ? header_type::get_header(alloc_ptr) : nullptr;

  if constexpr (MemorySpaceAccess<HostSpace, MemorySpace>::accessible) {
    if (!alloc_ptr ||
        allocation_header->m_record->head() != allocation_header) {
      Kokkos::Impl::throw_runtime_exception(
          std::string("Kokkos::Impl::SharedAllocationRecord<") +
          std::string(MemorySpace::name()) +
          std::string(", void>::get_record ERROR"));
    }

    return static_cast<record_type*>(allocation_header->m_record);
  } else {
    header_type header;
    if (alloc_ptr) {
      typename MemorySpace::execution_space execution_space;
      Kokkos::Impl::DeepCopy<HostSpace, MemorySpace, decltype(execution_space)>(
          execution_space, &header, allocation_header,
          sizeof(SharedAllocationHeader));
      execution_space.fence(
          std::string("SharedAllocationRecord<") + MemorySpace::name() +
          ", void>::get_record(): fence after copying header to HostSpace");
    }

    record_type* const record =
        alloc_ptr ? static_cast<record_type*>(header.m_record) : nullptr;

    if (!alloc_ptr || !record || record->head() != allocation_header) {
      Kokkos::Impl::throw_runtime_exception(
          std::string("Kokkos::Impl::SharedAllocationRecord<") +
          std::string(MemorySpace::name()) +
          std::string(", void>::get_record ERROR"));
    }

    return record;
  }
}

template <class MemorySpace>
typename SharedAllocationRecord<MemorySpace, void>::derived_t*
SharedAllocationRecord<MemorySpace, void>::get_record(void* alloc_ptr) {
  return get_shared_allocation_record<MemorySpace>(alloc_ptr);
}

template <class MemorySpace, class ExecutionSpace>
void* reallocate_tracked_shared_allocation(void* alloc_ptr, size_t alloc_size) {
  using record_type = SharedAllocationRecord<MemorySpace, void>;

  record_type* const old_record =
      get_shared_allocation_record<MemorySpace>(alloc_ptr);
  record_type* const new_record =
      allocate_shared_allocation_record<MemorySpace>(
          old_record->m_space, old_record->get_label(), alloc_size);

  Kokkos::Impl::DeepCopy<MemorySpace, MemorySpace>(
      ExecutionSpace{}, new_record->data(), old_record->data(),
      std::min(old_record->size(), new_record->size()));
  Kokkos::fence(std::string("SharedAllocationRecord<") + MemorySpace::name() +
                ", void>::reallocate_tracked(): fence after copying data");

  SharedAllocationRecord<void, void>::increment(new_record);
  SharedAllocationRecord<void, void>::decrement(old_record);

  return new_record->data();
}

template <class MemorySpace>
template <class ExecutionSpace>
void* SharedAllocationRecord<MemorySpace, void>::reallocate_tracked(
    void* alloc_ptr, size_t alloc_size) {
  return reallocate_tracked_shared_allocation<MemorySpace, ExecutionSpace>(
      alloc_ptr, alloc_size);
}

template <class MemorySpace, class ExecutionSpace>
void print_shared_allocation_records([[maybe_unused]] std::ostream& stream,
                                     MemorySpace const&,
                                     [[maybe_unused]] bool detail) {
#ifdef KOKKOS_ENABLE_DEBUG
  if constexpr (MemorySpaceAccess<HostSpace, MemorySpace>::accessible) {
    SharedAllocationRecord<void, void>::print_host_accessible_records(
        stream, MemorySpace::name(),
        &SharedAllocationRecord<MemorySpace, void>::s_root_record, detail);
  } else {
    SharedAllocationRecord<void, void>* record =
        &SharedAllocationRecord<MemorySpace, void>::s_root_record;

    char buffer[256];
    SharedAllocationHeader header;

    if (detail) {
      do {
        if (record->m_alloc_ptr) {
          Kokkos::Impl::DeepCopy<HostSpace, MemorySpace, ExecutionSpace>(
              ExecutionSpace{}, &header, record->m_alloc_ptr,
              sizeof(SharedAllocationHeader));
          Kokkos::fence(
              "SharedAllocationRecord::print_records(): fence after copying "
              "header to HostSpace");
        } else {
          header.m_label[0] = 0;
        }

        // Formatting dependent on sizeof(uintptr_t)
        const char* format_string;

        if (sizeof(uintptr_t) == sizeof(unsigned long)) {
          format_string =
              "%s addr( 0x%.12lx ) list( 0x%.12lx 0x%.12lx ) extent[ "
              "0x%.12lx + %.8ld ] count(%d) dealloc(0x%.12lx) %s\n";
        } else if (sizeof(uintptr_t) == sizeof(unsigned long long)) {
          format_string =
              "%s addr( 0x%.12llx ) list( 0x%.12llx 0x%.12llx ) extent[ "
              "0x%.12llx + %.8ld ] count(%d) dealloc(0x%.12llx) %s\n";
        }

        std::snprintf(
            buffer, 256, format_string, MemorySpace::execution_space::name(),
            reinterpret_cast<uintptr_t>(record),
            reinterpret_cast<uintptr_t>(record->m_prev),
            reinterpret_cast<uintptr_t>(record->m_next),
            reinterpret_cast<uintptr_t>(record->m_alloc_ptr),
            record->m_alloc_size, record->m_count,
            reinterpret_cast<uintptr_t>(record->m_dealloc), header.m_label);
        stream << buffer;
        record = record->m_next;
      } while (record !=
               &SharedAllocationRecord<MemorySpace, void>::s_root_record);
    } else {
      do {
        if (record->m_alloc_ptr) {
          Kokkos::Impl::DeepCopy<HostSpace, MemorySpace, ExecutionSpace>(
              ExecutionSpace{}, &header, record->m_alloc_ptr,
              sizeof(SharedAllocationHeader));
          Kokkos::fence(
              "SharedAllocationRecord::print_records(): fence after copying "
              "header to HostSpace");

          // Formatting dependent on sizeof(uintptr_t)
          const char* format_string;

          if (sizeof(uintptr_t) == sizeof(unsigned long)) {
            format_string = "%s [ 0x%.12lx + %ld ] %s\n";
          } else if (sizeof(uintptr_t) == sizeof(unsigned long long)) {
            format_string = "%s [ 0x%.12llx + %ld ] %s\n";
          }

          std::snprintf(buffer, 256, format_string,
                        MemorySpace::execution_space::name(),
                        reinterpret_cast<uintptr_t>(record->data()),
                        record->size(), header.m_label);
        } else {
          std::snprintf(buffer, 256, "%s [ 0 + 0 ]\n",
                        MemorySpace::execution_space::name());
        }
        stream << buffer;
        record = record->m_next;
      } while (record !=
               &SharedAllocationRecord<MemorySpace, void>::s_root_record);
    }
  }
#else
  Kokkos::Impl::throw_runtime_exception(
      std::string("SharedAllocationHeader<") +
      std::string(MemorySpace::name()) +
      std::string(
          ">::print_records only works with KOKKOS_ENABLE_DEBUG enabled"));
#endif
}

template <class MemorySpace>
template <class ExecutionSpace>
void SharedAllocationRecord<MemorySpace, void>::print_records(
    std::ostream& stream, MemorySpace const& space, bool detail) {
  print_shared_allocation_records<MemorySpace, ExecutionSpace>(stream, space,
                                                               detail);
}

template <class MemorySpace>
std::string SharedAllocationRecord<MemorySpace, void>::get_label() const {
  return this->m_label;
}

}  // end namespace Impl
}  // end namespace Kokkos

#endif  // KOKKOS_IMPL_SHAREDALLOC_TIMPL_HPP
