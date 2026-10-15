// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_NEXTSILICON_PAGE_ALIGNED_DATA_HPP
#define KOKKOS_NEXTSILICON_PAGE_ALIGNED_DATA_HPP

#include <type_traits>
#include <utility>

namespace Kokkos::Impl {

inline constexpr int PAGE_SIZE = 4096;

enum class PageLocation {
  Host,
  Device,
  Any,
};

template <PageLocation>
void migrate_after_initialize(void* obj, size_t size);

// This struct is aligned to 4096 bytes (page size) to work around
// issues with NextSilicon page migration. It can be pinned to the host
// to prevent migration to device memory, which would
// cause problems when we try to access it from the host.
template <typename T, PageLocation Location = PageLocation::Any>
struct alignas(PAGE_SIZE) PageAlignedData {
  static constexpr PageLocation location = Location;

  T data;

  operator T&() { return data; }
  operator const T&() const { return data; }

  template <typename... Args>
    requires(!(sizeof...(Args) == 1 &&
               (std::is_same_v<std::decay_t<Args>, PageAlignedData> && ...)))
  PageAlignedData(Args&&... args) : data{std::forward<Args>(args)...} {
    migrate_after_initialize<location>(this, sizeof(*this));
  }

  PageAlignedData& operator=(const T& data_) {
    data = data_;
    return *this;
  }

  PageAlignedData(const PageAlignedData&)            = delete;
  PageAlignedData& operator=(const PageAlignedData&) = delete;
  PageAlignedData(PageAlignedData&&)                 = delete;
  PageAlignedData& operator=(PageAlignedData&&)      = delete;
};

static_assert(sizeof(PageAlignedData<int>) == PAGE_SIZE);
static_assert(sizeof(PageAlignedData<bool>) == PAGE_SIZE);

}  // namespace Kokkos::Impl

#endif
