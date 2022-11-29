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

#ifndef KOKKOS_EXPERIMENTAL_CUDA_VIEW_HPP
#define KOKKOS_EXPERIMENTAL_CUDA_VIEW_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_CUDA)

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template <typename ValueType, typename AliasType>
struct CudaLDGFetch {
  const ValueType* m_ptr;

  template <typename iType>
  KOKKOS_INLINE_FUNCTION ValueType operator[](const iType& i) const {
#if defined(KOKKOS_ARCH_KEPLER30) || defined(KOKKOS_ARCH_KEPLER32)
    return m_ptr[i];
#else
    KOKKOS_IF_ON_DEVICE(
        (AliasType v = __ldg(reinterpret_cast<const AliasType*>(&m_ptr[i]));
         return *(reinterpret_cast<ValueType*>(&v));))
    KOKKOS_IF_ON_HOST((return m_ptr[i];))
#endif
  }

  KOKKOS_INLINE_FUNCTION
  operator const ValueType*() const { return m_ptr; }

  KOKKOS_INLINE_FUNCTION
  CudaLDGFetch() : m_ptr() {}

  KOKKOS_DEFAULTED_FUNCTION
  ~CudaLDGFetch() = default;

  KOKKOS_INLINE_FUNCTION
  CudaLDGFetch(const CudaLDGFetch& rhs) : m_ptr(rhs.m_ptr) {}

  KOKKOS_INLINE_FUNCTION
  CudaLDGFetch(CudaLDGFetch&& rhs) : m_ptr(rhs.m_ptr) {}

  KOKKOS_INLINE_FUNCTION
  CudaLDGFetch& operator=(const CudaLDGFetch& rhs) {
    m_ptr = rhs.m_ptr;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  CudaLDGFetch& operator=(CudaLDGFetch&& rhs) {
    m_ptr = rhs.m_ptr;
    return *this;
  }

  template <class CudaMemorySpace>
  inline explicit CudaLDGFetch(
      const ValueType* const arg_ptr,
      Kokkos::Impl::SharedAllocationRecord<CudaMemorySpace, void>*)
      : m_ptr(arg_ptr) {}

  KOKKOS_INLINE_FUNCTION
  CudaLDGFetch(CudaLDGFetch const rhs, size_t offset)
      : m_ptr(rhs.m_ptr + offset) {}
};

}  // namespace Impl
}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

/** \brief  Replace Default ViewDataHandle with Cuda texture fetch
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
                            track_type const& arg_tracker) {
    if (arg_data_ptr == nullptr) return handle_type();

    KOKKOS_IF_ON_HOST((
        // Assignment of texture = non-texture requires creation of a texture
        // object which can only occur on the host.  In addition, 'get_record'
        // is only valid if called in a host execution space

        using memory_space = typename Traits::memory_space;
        using record =
            typename Impl::SharedAllocationRecord<memory_space, void>;

        record* const r = arg_tracker.template get_record<memory_space>();

        return handle_type(arg_data_ptr, r);))

    KOKKOS_IF_ON_DEVICE(
        ((void)arg_tracker; Kokkos::Impl::cuda_abort(
             "Cannot create Cuda texture object from within a Cuda kernel");
         return handle_type();))
  }
};

}  // namespace Impl
}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #if defined( KOKKOS_ENABLE_CUDA ) */
#endif /* #ifndef KOKKOS_CUDA_VIEW_HPP */
