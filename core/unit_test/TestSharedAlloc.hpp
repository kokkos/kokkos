// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>

#include <sstream>
#include <iostream>
#include <cstddef>
#include <cstring>
#include <type_traits>

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
import kokkos.core_impl;
#else
#include <Kokkos_Core.hpp>
#endif

/*--------------------------------------------------------------------------*/

namespace Test {

// This memory space is deliberately defined in the test, outside Kokkos's
// backend implementation.  It exercises the ordinary template-instantiation
// path used by external memory-space implementations.
struct CustomHostMemorySpace {
  using memory_space    = CustomHostMemorySpace;
  using execution_space = Kokkos::DefaultHostExecutionSpace;
  using device_type     = Kokkos::Device<execution_space, memory_space>;
  using size_type       = std::size_t;
  using index_type      = std::make_signed_t<size_type>;

  static constexpr const char* name() { return "TestCustomHost"; }

  void* allocate(const size_t size) const {
    return Kokkos::HostSpace{}.allocate(size);
  }
  void* allocate(const char* label, const size_t size) const {
    return Kokkos::HostSpace{}.allocate(label, size);
  }
  void* allocate(const char* label, const size_t size,
                 const size_t logical_size) const {
    return Kokkos::HostSpace{}.allocate(label, size, logical_size);
  }

  template <class ExecutionSpace>
  void* allocate(ExecutionSpace const& exec, const size_t size) const {
    return Kokkos::HostSpace{}.allocate(exec, size);
  }
  template <class ExecutionSpace>
  void* allocate(ExecutionSpace const& exec, const char* label,
                 const size_t size) const {
    return Kokkos::HostSpace{}.allocate(exec, label, size);
  }
  template <class ExecutionSpace>
  void* allocate(ExecutionSpace const& exec, const char* label,
                 const size_t size, const size_t logical_size) const {
    return Kokkos::HostSpace{}.allocate(exec, label, size, logical_size);
  }

  void deallocate(void* ptr, const size_t size) const {
    Kokkos::HostSpace{}.deallocate(ptr, size);
  }
  void deallocate(const char* label, void* ptr, const size_t size) const {
    Kokkos::HostSpace{}.deallocate(label, ptr, size);
  }
  void deallocate(const char* label, void* ptr, const size_t size,
                  const size_t logical_size) const {
    Kokkos::HostSpace{}.deallocate(label, ptr, size, logical_size);
  }
};

// This variant deliberately reports that HostSpace cannot access it.  Its
// storage still delegates to HostSpace so the test can exercise the header
// copy path without requiring a device backend.
struct CustomHostInaccessibleMemorySpace : CustomHostMemorySpace {
  using memory_space    = CustomHostInaccessibleMemorySpace;
  using execution_space = Kokkos::DefaultHostExecutionSpace;
  using device_type     = Kokkos::Device<execution_space, memory_space>;

  static constexpr const char* name() { return "TestCustomInaccessible"; }
};

}  // namespace Test

namespace Kokkos::Impl {

template <>
struct MemorySpaceAccess<Kokkos::HostSpace, Test::CustomHostMemorySpace> {
  enum { assignable = false, accessible = true };
};

template <>
struct MemorySpaceAccess<Kokkos::HostSpace,
                         Test::CustomHostInaccessibleMemorySpace> {
  enum { assignable = false, accessible = false };
};

template <class ExecutionSpace>
struct DeepCopy<Test::CustomHostInaccessibleMemorySpace, Kokkos::HostSpace,
                ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t size) {
    std::memcpy(dst, src, size);
  }
  DeepCopy(const ExecutionSpace&, void* dst, const void* src, size_t size) {
    std::memcpy(dst, src, size);
  }
};

template <class ExecutionSpace>
struct DeepCopy<Kokkos::HostSpace, Test::CustomHostInaccessibleMemorySpace,
                ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t size) {
    std::memcpy(dst, src, size);
  }
  DeepCopy(const ExecutionSpace&, void* dst, const void* src, size_t size) {
    std::memcpy(dst, src, size);
  }
};

}  // namespace Kokkos::Impl

namespace Test {

struct SharedAllocDestroy {
  volatile int* count;

  SharedAllocDestroy() = default;
  SharedAllocDestroy(int* arg) : count(arg) {}

  void destroy_shared_allocation() { Kokkos::atomic_inc(count); }
};

template <class MemorySpace, class ExecutionSpace>
void test_shared_alloc() {
  using Header     = const Kokkos::Impl::SharedAllocationHeader;
  using Tracker    = Kokkos::Impl::SharedAllocationTracker;
  using RecordBase = Kokkos::Impl::SharedAllocationRecord<void, void>;
  using RecordMemS = Kokkos::Impl::SharedAllocationRecord<MemorySpace, void>;
  using RecordFull =
      Kokkos::Impl::SharedAllocationRecord<MemorySpace, SharedAllocDestroy>;

  static_assert(sizeof(Tracker) == sizeof(int*),
                "SharedAllocationTracker has wrong size!");

  MemorySpace s;

  const size_t N    = 1200;
  const size_t size = 8;

  RecordMemS* rarray[N];
  Header* harray[N];

  RecordMemS** const r = rarray;
  Header** const h     = harray;

  Kokkos::RangePolicy<ExecutionSpace> range(0, N);

  {
    // Since always executed on host space, leave [=]
    Kokkos::parallel_for(range, [=](int i) {
      char name[64];
      snprintf(name, 64, "test_%.2d", i);

      r[i] = RecordMemS::allocate(s, name, size * (i + 1));
      h[i] = Header::get_header(r[i]->data());

      ASSERT_EQ(r[i]->use_count(), 0);

      for (int j = 0; j < (i / 10) + 1; ++j) RecordBase::increment(r[i]);

      ASSERT_EQ(r[i]->use_count(), (i / 10) + 1);
      ASSERT_EQ(r[i], RecordMemS::get_record(r[i]->data()));
    });

    Kokkos::fence();

#ifdef KOKKOS_ENABLE_DEBUG
    // Sanity check for the whole set of allocation records to which this record
    // belongs.
    RecordBase::is_sane(r[0]);
    // RecordMemS::print_records( std::cout, s, true );
#endif

    // This must be a plain for-loop since deallocation (which can be triggered
    // by RecordBase::decrement) fences all execution space instances. If this
    // is a parallel_for, the test can hang with the parallel_for blocking
    // waiting for itself to complete.
    for (auto i = range.begin(); i < range.end(); ++i) {
      while (nullptr !=
             (r[i] = static_cast<RecordMemS*>(RecordBase::decrement(r[i])))) {
#ifdef KOKKOS_ENABLE_DEBUG
        if (r[i]->use_count() == 1) RecordBase::is_sane(r[i]);
#endif
      }
    }

    Kokkos::fence();
  }

  {
    int destroy_count = 0;
    SharedAllocDestroy counter(&destroy_count);

    Kokkos::parallel_for(range, [=](size_t i) {
      char name[64];
      snprintf(name, 64, "test_%.2d", int(i));

      RecordFull* rec = RecordFull::allocate(s, name, size * (i + 1));

      rec->m_destroy = counter;

      r[i] = rec;
      h[i] = Header::get_header(r[i]->data());

      ASSERT_EQ(r[i]->use_count(), 0);

      for (size_t j = 0; j < (i / 10) + 1; ++j) RecordBase::increment(r[i]);

      ASSERT_EQ(r[i]->use_count(), int((i / 10) + 1));
      ASSERT_EQ(r[i], RecordMemS::get_record(r[i]->data()));
    });

    Kokkos::fence();

#ifdef KOKKOS_ENABLE_DEBUG
    RecordBase::is_sane(r[0]);
#endif

    // This must be a plain for-loop since deallocation (which can be triggered
    // by RecordBase::decrement) fences all execution space instances. If this
    // is a parallel_for, the test can hang with the parallel_for blocking
    // waiting for itself to complete.
    for (auto i = range.begin(); i < range.end(); ++i) {
      while (nullptr !=
             (r[i] = static_cast<RecordMemS*>(RecordBase::decrement(r[i])))) {
#ifdef KOKKOS_ENABLE_DEBUG
        if (r[i]->use_count() == 1) RecordBase::is_sane(r[i]);
#endif
      }
    }

    Kokkos::fence();

    ASSERT_EQ(destroy_count, int(N));
  }

  {
    int destroy_count = 0;

    {
      RecordFull* rec = RecordFull::allocate(s, "test", size);

      // ... Construction of the allocated { rec->data(), rec->size() }

      // Copy destruction function object into the allocation record.
      rec->m_destroy = SharedAllocDestroy(&destroy_count);

      ASSERT_EQ(rec->use_count(), 0);

      // Start tracking, increments the use count from 0 to 1.
      Tracker track;

      track.assign_allocated_record_to_uninitialized(rec);

      ASSERT_EQ(rec->use_count(), 1);
      ASSERT_EQ(track.use_count(), 1);

      // Verify construction / destruction increment.
      for (size_t i = 0; i < N; ++i) {
        ASSERT_EQ(rec->use_count(), 1);

        {
          Tracker local_tracker;
          local_tracker.assign_allocated_record_to_uninitialized(rec);
          ASSERT_EQ(rec->use_count(), 2);
          ASSERT_EQ(local_tracker.use_count(), 2);
        }

        ASSERT_EQ(rec->use_count(), 1);
        ASSERT_EQ(track.use_count(), 1);
      }

      Kokkos::parallel_for(range, [=](size_t) {
        Tracker local_tracker;
        local_tracker.assign_allocated_record_to_uninitialized(rec);
        ASSERT_GT(rec->use_count(), 1);
      });

      Kokkos::fence();

      ASSERT_EQ(rec->use_count(), 1);
      ASSERT_EQ(track.use_count(), 1);

      // Destruction of 'track' object deallocates the 'rec' and invokes the
      // destroy function object.
    }

    ASSERT_EQ(destroy_count, 1);
  }
}

TEST(TEST_CATEGORY, impl_shared_alloc) {
#ifdef TEST_CATEGORY_NUMBER
#if (TEST_CATEGORY_NUMBER < 4)  // serial threads openmp hpx
  test_shared_alloc<Kokkos::HostSpace, TEST_EXECSPACE>();
#elif (TEST_CATEGORY_NUMBER == 5)  // cuda
  test_shared_alloc<Kokkos::CudaSpace, Kokkos::DefaultHostExecutionSpace>();
#elif (TEST_CATEGORY_NUMBER == 6)  // hip
  test_shared_alloc<Kokkos::HIPSpace, Kokkos::DefaultHostExecutionSpace>();
#elif (TEST_CATEGORY_NUMBER == 7)  // sycl
  test_shared_alloc<Kokkos::SYCLDeviceUSMSpace,
                    Kokkos::DefaultHostExecutionSpace>();
#elif (TEST_CATEGORY_NUMBER == 8)  // openacc
  test_shared_alloc<Kokkos::Experimental::OpenACCSpace,
                    Kokkos::DefaultHostExecutionSpace>();
#endif
#else
  test_shared_alloc<TEST_EXECSPACE, Kokkos::DefaultHostExecutionSpace>();
#endif
}

TEST(TEST_CATEGORY, impl_shared_alloc_custom_memory_space) {
  using RecordBase = Kokkos::Impl::SharedAllocationRecord<void, void>;
  using Record =
      Kokkos::Impl::SharedAllocationRecord<CustomHostMemorySpace, void>;

  CustomHostMemorySpace space;
  constexpr size_t allocation_size = 64;

  auto* record = Record::allocate(space, "custom allocation", allocation_size);
  ASSERT_NE(record, nullptr);
  EXPECT_EQ(record, Record::get_record(record->data()));
  EXPECT_EQ(record->size(), allocation_size);
  EXPECT_EQ(record->get_label(), "custom allocation");
  EXPECT_STREQ(
      Kokkos::Impl::SharedAllocationHeader::get_header(record->data())->label(),
      "custom allocation");

  RecordBase::increment(record);
  EXPECT_EQ(record->use_count(), 1);
  EXPECT_EQ(RecordBase::decrement(record), nullptr);

  void* tracked = Record::allocate_tracked(space, "custom tracked", 32);
  ASSERT_NE(tracked, nullptr);
  auto* tracked_record = Record::get_record(tracked);
  EXPECT_EQ(tracked_record->get_label(), "custom tracked");
  EXPECT_EQ(tracked_record, Record::get_record(tracked));
#ifdef KOKKOS_ENABLE_DEBUG
  std::ostringstream records;
  Record::print_records(records, space);
  EXPECT_NE(records.str().find("custom tracked"), std::string::npos);
#endif
  Record::deallocate_tracked(tracked);
}

TEST(TEST_CATEGORY, impl_shared_alloc_custom_inaccessible_memory_space) {
  using RecordBase = Kokkos::Impl::SharedAllocationRecord<void, void>;
  using Record =
      Kokkos::Impl::SharedAllocationRecord<CustomHostInaccessibleMemorySpace,
                                           void>;

  CustomHostInaccessibleMemorySpace space;
  auto* record = Record::allocate(space, "custom inaccessible", 64);
  ASSERT_NE(record, nullptr);
  EXPECT_EQ(record, Record::get_record(record->data()));
  EXPECT_EQ(record->get_label(), "custom inaccessible");
  EXPECT_STREQ(
      Kokkos::Impl::SharedAllocationHeader::get_header(record->data())->label(),
      "custom inaccessible");

  RecordBase::increment(record);
  EXPECT_EQ(RecordBase::decrement(record), nullptr);

  void* tracked =
      Record::allocate_tracked(space, "custom inaccessible tracked", 32);
  ASSERT_NE(tracked, nullptr);
  EXPECT_EQ(Record::get_record(tracked)->get_label(),
            "custom inaccessible tracked");
#ifdef KOKKOS_ENABLE_DEBUG
  std::ostringstream records;
  Record::print_records(records, space);
  EXPECT_NE(records.str().find("custom inaccessible tracked"),
            std::string::npos);
#endif
  Record::deallocate_tracked(tracked);
}

}  // namespace Test
