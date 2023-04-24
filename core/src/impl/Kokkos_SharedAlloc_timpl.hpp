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

#ifndef KOKKOS_IMPL_SHAREDALLOC_TIMPL_HPP
#define KOKKOS_IMPL_SHAREDALLOC_TIMPL_HPP

#include <Kokkos_Macros.hpp>
#include <Kokkos_Core_fwd.hpp>

#include <impl/Kokkos_SharedAlloc.hpp>

#include <Kokkos_HostSpace.hpp>  // used with HostInaccessible specializations

#include <string>    // std::string
#include <cstring>   // strncpy
#include <iostream>  // ostream

namespace Kokkos {
namespace Impl {

template <class MemorySpace>
auto SharedAllocationRecordCommon<MemorySpace>::allocate(
    MemorySpace const& arg_space, std::string const& arg_label,
    size_t arg_alloc_size) -> derived_t* {
  return new derived_t(arg_space, arg_label, arg_alloc_size);
}

template <class MemorySpace>
void* SharedAllocationRecordCommon<MemorySpace>::allocate_tracked(
    const MemorySpace& arg_space, const std::string& arg_alloc_label,
    size_t arg_alloc_size) {
  if (!arg_alloc_size) return nullptr;

  SharedAllocationRecord* const r =
      allocate(arg_space, arg_alloc_label, arg_alloc_size);

  record_base_t::increment(r);

  return r->data();
}

template <class MemorySpace>
void SharedAllocationRecordCommon<MemorySpace>::deallocate(
    SharedAllocationRecordCommon::record_base_t* arg_rec) {
  delete static_cast<derived_t*>(arg_rec);
}

template <class MemorySpace>
void SharedAllocationRecordCommon<MemorySpace>::deallocate_tracked(
    void* arg_alloc_ptr) {
  if (arg_alloc_ptr != nullptr) {
    SharedAllocationRecord* const r = derived_t::get_record(arg_alloc_ptr);
    record_base_t::decrement(r);
  }
}

template <class MemorySpace>
void* SharedAllocationRecordCommon<MemorySpace>::reallocate_tracked(
    void* arg_alloc_ptr, size_t arg_alloc_size) {
  derived_t* const r_old = derived_t::get_record(arg_alloc_ptr);
  derived_t* const r_new =
      allocate(r_old->m_space, r_old->get_label(), arg_alloc_size);

  Kokkos::Impl::DeepCopy<MemorySpace, MemorySpace>(
      r_new->data(), r_old->data(), std::min(r_old->size(), r_new->size()));
  Kokkos::fence(
      "SharedAllocationRecord<Kokkos::Experimental::HBWSpace, "
      "void>::reallocate_tracked(): fence after copying data");

  record_base_t::increment(r_new);
  record_base_t::decrement(r_old);

  return r_new->data();
}

template <class MemorySpace>
auto SharedAllocationRecordCommon<MemorySpace>::get_record(void* alloc_ptr)
    -> derived_t* {
  using Header = SharedAllocationHeader;

  Header const* const h = alloc_ptr ? Header::get_header(alloc_ptr) : nullptr;

  if (!alloc_ptr || h->m_record->m_alloc_ptr != h) {
    Kokkos::Impl::throw_runtime_exception(
        std::string("Kokkos::Impl::SharedAllocationRecordCommon<") +
        std::string(MemorySpace::name()) +
        std::string(">::get_record() ERROR"));
  }

  return static_cast<derived_t*>(h->m_record);
}

template <class MemorySpace>
std::string SharedAllocationRecordCommon<MemorySpace>::get_label() const {
  return record_base_t::m_label;
}

template <class MemorySpace>
void SharedAllocationRecordCommon<MemorySpace>::
    _fill_host_accessible_header_info(SharedAllocationHeader& arg_header,
                                      std::string const& arg_label) {
  // Fill in the Header information, directly accessible on the host

  arg_header.m_record = &self();

  strncpy(arg_header.m_label, arg_label.c_str(),
          SharedAllocationHeader::maximum_label_length);
  // Set last element zero, in case c_str is too long
  arg_header.m_label[SharedAllocationHeader::maximum_label_length - 1] = '\0';
}

template <class MemorySpace>
void SharedAllocationRecordCommon<MemorySpace>::print_records(
    std::ostream& s, const MemorySpace&, bool detail) {
  (void)s;
  (void)detail;
#ifdef KOKKOS_ENABLE_DEBUG
  SharedAllocationRecord<void, void>::print_host_accessible_records(
      s, MemorySpace::name(), &derived_t::s_root_record, detail);
#else
  Kokkos::Impl::throw_runtime_exception(
      std::string("SharedAllocationHeader<") +
      std::string(MemorySpace::name()) +
      std::string(
          ">::print_records only works with KOKKOS_ENABLE_DEBUG enabled"));
#endif
}

template <class MemorySpace>
void HostInaccessibleSharedAllocationRecordCommon<MemorySpace>::print_records(
    std::ostream& s, const MemorySpace&, bool detail) {
  (void)s;
  (void)detail;
#ifdef KOKKOS_ENABLE_DEBUG
  SharedAllocationRecord<void, void>* r = &derived_t::s_root_record;

  char buffer[256];

  SharedAllocationHeader head;

  if (detail) {
    do {
      if (r->m_alloc_ptr) {
        Kokkos::Impl::DeepCopy<HostSpace, MemorySpace>(
            &head, r->m_alloc_ptr, sizeof(SharedAllocationHeader));
        Kokkos::fence(
            "HostInaccessibleSharedAllocationRecordCommon::print_records(): "
            "fence after copying header to HostSpace");
      } else {
        head.m_label[0] = 0;
      }

      // Formatting dependent on sizeof(uintptr_t)
      const char* format_string;

      if (sizeof(uintptr_t) == sizeof(unsigned long)) {
        format_string =
            "%s addr( 0x%.12lx ) list( 0x%.12lx 0x%.12lx ) extent[ 0x%.12lx "
            "+ %.8ld ] count(%d) dealloc(0x%.12lx) %s\n";
      } else if (sizeof(uintptr_t) == sizeof(unsigned long long)) {
        format_string =
            "%s addr( 0x%.12llx ) list( 0x%.12llx 0x%.12llx ) extent[ "
            "0x%.12llx + %.8ld ] count(%d) dealloc(0x%.12llx) %s\n";
      }

      snprintf(buffer, 256, format_string, MemorySpace::execution_space::name(),
               reinterpret_cast<uintptr_t>(r),
               reinterpret_cast<uintptr_t>(r->m_prev),
               reinterpret_cast<uintptr_t>(r->m_next),
               reinterpret_cast<uintptr_t>(r->m_alloc_ptr), r->m_alloc_size,
               r->m_count, reinterpret_cast<uintptr_t>(r->m_dealloc),
               head.m_label);
      s << buffer;
      r = r->m_next;
    } while (r != &derived_t::s_root_record);
  } else {
    do {
      if (r->m_alloc_ptr) {
        Kokkos::Impl::DeepCopy<HostSpace, MemorySpace>(
            &head, r->m_alloc_ptr, sizeof(SharedAllocationHeader));
        Kokkos::fence(
            "HostInaccessibleSharedAllocationRecordCommon::print_records(): "
            "fence after copying header to HostSpace");

        // Formatting dependent on sizeof(uintptr_t)
        const char* format_string;

        if (sizeof(uintptr_t) == sizeof(unsigned long)) {
          format_string = "%s [ 0x%.12lx + %ld ] %s\n";
        } else if (sizeof(uintptr_t) == sizeof(unsigned long long)) {
          format_string = "%s [ 0x%.12llx + %ld ] %s\n";
        }

        snprintf(
            buffer, 256, format_string, MemorySpace::execution_space::name(),
            reinterpret_cast<uintptr_t>(r->data()), r->size(), head.m_label);
      } else {
        snprintf(buffer, 256, "%s [ 0 + 0 ]\n",
                 MemorySpace::execution_space::name());
      }
      s << buffer;
      r = r->m_next;
    } while (r != &derived_t::s_root_record);
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
auto HostInaccessibleSharedAllocationRecordCommon<MemorySpace>::get_record(
    void* alloc_ptr) -> derived_t* {
  // Copy the header from the allocation
  SharedAllocationHeader head;

  SharedAllocationHeader const* const head_cuda =
      alloc_ptr ? SharedAllocationHeader::get_header(alloc_ptr) : nullptr;

  if (alloc_ptr) {
    typename MemorySpace::execution_space exec_space;
    Kokkos::Impl::DeepCopy<HostSpace, MemorySpace, decltype(exec_space)>(
        exec_space, &head, head_cuda, sizeof(SharedAllocationHeader));
    exec_space.fence(
        "HostInaccessibleSharedAllocationRecordCommon::get_record(): fence "
        "after copying header to HostSpace");
  }

  derived_t* const record =
      alloc_ptr ? static_cast<derived_t*>(head.m_record) : nullptr;

  if (!alloc_ptr || record->m_alloc_ptr != head_cuda) {
    Kokkos::Impl::throw_runtime_exception(
        std::string("Kokkos::Impl::SharedAllocationRecord<") +
        std::string(MemorySpace::name()) +
        std::string(", void>::get_record ERROR"));
  }

  return record;
}

template <class MemorySpace>
std::string
HostInaccessibleSharedAllocationRecordCommon<MemorySpace>::get_label() const {
  return record_base_t::m_label;
}

}  // end namespace Impl
}  // end namespace Kokkos

#endif  // KOKKOS_IMPL_SHAREDALLOC_TIMPL_HPP
