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

#ifndef KOKKOS_EXPERIMENTAL_CUDA_VIEW_HPP
#define KOKKOS_EXPERIMENTAL_CUDA_VIEW_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_CUDA)

#include <cuda/std/version>

// FIXME_CUDA <cuda/annotated_ptr> couldn't be included in multiple compilation
// units (https://github.com/NVIDIA/libcudacxx/issues/270)
#if defined(_LIBCUDACXX_CUDA_API_VERSION) && \
    _LIBCUDACXX_CUDA_API_VERSION >= 1008001
#include <cuda/annotated_ptr>
#endif

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

#if defined(_LIBCUDACXX_CUDA_API_VERSION) && \
    _LIBCUDACXX_CUDA_API_VERSION >= 1008001

template <typename ValueType, typename MemorySpace>
struct CudaAnnotatedHandleHandle {
  mutable std::conditional_t<
      std::is_same_v<MemorySpace, Kokkos::Cuda::scratch_memory_space_l0>,
      cuda::annotated_ptr<ValueType, cuda::access_property::shared>,
      cuda::annotated_ptr<ValueType, cuda::access_property::persisting>>
      m_ptr;

  template <typename iType>
  KOKKOS_FORCEINLINE_FUNCTION ValueType& operator()(const iType& i) const {
    return m_ptr[i];
  }

  template <typename iType>
  KOKKOS_FORCEINLINE_FUNCTION ValueType& operator[](const iType& i) const {
    return m_ptr[i];
  }

  KOKKOS_FUNCTION
  operator ValueType*() const { return static_cast<ValueType*>(m_ptr.get()); }

  KOKKOS_DEFAULTED_FUNCTION
  CudaAnnotatedHandleHandle() = default;

  KOKKOS_FUNCTION
  explicit CudaAnnotatedHandleHandle(ValueType* const arg_ptr)
      : m_ptr(arg_ptr) {}

  KOKKOS_FUNCTION
  CudaAnnotatedHandleHandle(const CudaAnnotatedHandleHandle& arg_handle,
                            size_t offset)
      : m_ptr(arg_handle.m_ptr.get() + offset) {}

  CudaAnnotatedHandleHandle& operator=(ValueType* const arg_ptr) {
    return *this = CudaAnnotatedHandleHandle(arg_ptr);
  }
};

template <class Traits>
struct ViewDataHandle<
    Traits,
    std::enable_if_t<std::is_void<typename Traits::specialize>::value &&
                     !Traits::memory_traits::is_aligned &&
                     !Traits::memory_traits::is_restrict &&
                     !Traits::memory_traits::is_atomic &&
                     (std::is_same_v<typename Traits::memory_space,
                                     Kokkos::Cuda::scratch_memory_space_l0> ||
                      std::is_same_v<typename Traits::memory_space,
                                     Kokkos::Cuda::scratch_memory_space_l1>)>> {
  // private:
  using value_type   = typename Traits::value_type;
  using memory_space = typename Traits::memory_space;
  using handle_type  = CudaAnnotatedHandleHandle<value_type, memory_space>;
  using return_type  = typename Traits::value_type&;
  using track_type   = Kokkos::Impl::SharedAllocationTracker;

 public:
  KOKKOS_INLINE_FUNCTION
  static handle_type assign(value_type* const arg_data_ptr,
                            track_type const& /*arg_tracker*/) {
    return handle_type(arg_data_ptr);
  }

  KOKKOS_INLINE_FUNCTION
  static handle_type assign(value_type* const arg_data_ptr, size_t offset) {
    return handle_type(arg_data_ptr + offset);
  }

  KOKKOS_INLINE_FUNCTION
  static handle_type assign(const handle_type arg_data_ptr, size_t offset) {
    return handle_type(arg_data_ptr, offset);
  }
};

#endif

template <typename ValueType, typename AliasType>
struct CudaLDGFetch {
  const ValueType* m_ptr;

  template <typename iType>
  KOKKOS_FUNCTION ValueType operator[](const iType& i) const {
#if defined(KOKKOS_ARCH_KEPLER30) || defined(KOKKOS_ARCH_KEPLER32)
    return m_ptr[i];
#else
    KOKKOS_IF_ON_DEVICE(
        (AliasType v = __ldg(reinterpret_cast<const AliasType*>(&m_ptr[i]));
         return *(reinterpret_cast<ValueType*>(&v));))
    KOKKOS_IF_ON_HOST((return m_ptr[i];))
#endif
  }

  KOKKOS_FUNCTION
  operator const ValueType*() const { return m_ptr; }

  KOKKOS_DEFAULTED_FUNCTION
  CudaLDGFetch() = default;

  KOKKOS_FUNCTION
  explicit CudaLDGFetch(const ValueType* const arg_ptr) : m_ptr(arg_ptr) {}

  KOKKOS_FUNCTION
  CudaLDGFetch(CudaLDGFetch const rhs, size_t offset)
      : m_ptr(rhs.m_ptr + offset) {}
};

}  // namespace Impl
}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

/** \brief  Replace Default ViewDataHandle with CudaLDGFetch
 * specialization if 'const' value type, CudaSpace and random access.
 */
template <class Traits>
class ViewDataHandle<
    Traits, std::enable_if_t<(
                // Is Cuda memory space
                (std::is_same<typename Traits::memory_space,
                              Kokkos::CudaSpace>::value ||
                 std::is_same<typename Traits::memory_space,
                              Kokkos::CudaUVMSpace>::value) &&
                // Is a trivial const value of 4, 8, or 16 bytes
                std::is_trivial<typename Traits::const_value_type>::value &&
                std::is_same<typename Traits::const_value_type,
                             typename Traits::value_type>::value &&
                (sizeof(typename Traits::const_value_type) == 4 ||
                 sizeof(typename Traits::const_value_type) == 8 ||
                 sizeof(typename Traits::const_value_type) == 16) &&
                // Random access trait
                (Traits::memory_traits::is_random_access != 0))>> {
 public:
  using track_type = Kokkos::Impl::SharedAllocationTracker;

  using value_type  = typename Traits::const_value_type;
  using return_type = typename Traits::const_value_type;  // NOT a reference

  using alias_type = std::conditional_t<
      (sizeof(value_type) == 4), int,
      std::conditional_t<
          (sizeof(value_type) == 8), ::int2,
          std::conditional_t<(sizeof(value_type) == 16), ::int4, void>>>;

  using handle_type = Kokkos::Impl::CudaLDGFetch<value_type, alias_type>;

  KOKKOS_INLINE_FUNCTION
  static handle_type const& assign(handle_type const& arg_handle,
                                   track_type const& /* arg_tracker */) {
    return arg_handle;
  }

  KOKKOS_INLINE_FUNCTION
  static handle_type const assign(handle_type const& arg_handle,
                                  size_t offset) {
    return handle_type(arg_handle, offset);
  }

  KOKKOS_INLINE_FUNCTION
  static handle_type assign(value_type* arg_data_ptr,
                            track_type const& /*arg_tracker*/) {
    if (arg_data_ptr == nullptr) return handle_type();
    return handle_type(arg_data_ptr);
  }
};

}  // namespace Impl
}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #if defined( KOKKOS_ENABLE_CUDA ) */
#endif /* #ifndef KOKKOS_CUDA_VIEW_HPP */
