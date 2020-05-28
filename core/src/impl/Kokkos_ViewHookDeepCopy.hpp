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

#ifndef KOKKOS_VIEWHOOK_DEEPCOPY_HPP
#define KOKKOS_VIEWHOOK_DEEPCOPY_HPP

#include <Kokkos_Core_fwd.hpp>
#include <functional>
#include <memory>
#include <type_traits>

namespace Kokkos {
namespace Experimental {
/**
 * Here we define copy_buffer utility functions
 * to copy data to and from Views. We need
 * specialized variants for ScratchMemorySpace
 * and AnonymousSpace as they don't participate
 * in the ViewHooks system, and we need copies
 * to and from Views
 */
template <class ViewType>
typename std::enable_if<(std::is_same<Kokkos::AnonymousSpace,
                                      typename ViewType::memory_space>::value ||
                         std::is_same<Kokkos::ScratchMemorySpace<
                                          typename ViewType::execution_space>,
                                      typename ViewType::memory_space>::value),
                        void>::type
copy_buffer(unsigned char *, const ViewType &) {}

template <class ViewType>
typename std::enable_if<
    !(std::is_same<Kokkos::AnonymousSpace,
                   typename ViewType::memory_space>::value ||
      std::is_same<
          Kokkos::ScratchMemorySpace<typename ViewType::execution_space>,
          typename ViewType::memory_space>::value),
    void>::type
copy_buffer(unsigned char *buff, const ViewType &v) {
  Kokkos::Impl::DeepCopy<Kokkos::HostSpace, typename ViewType::memory_space,
                         typename ViewType::execution_space>(
      (void *)buff, v.data(), v.span() * sizeof(typename ViewType::value_type));
}

template <class ViewType>
typename std::enable_if<(std::is_same<Kokkos::AnonymousSpace,
                                      typename ViewType::memory_space>::value ||
                         std::is_same<Kokkos::ScratchMemorySpace<
                                          typename ViewType::execution_space>,
                                      typename ViewType::memory_space>::value ||
                         std::is_const<typename ViewType::value_type>::value),
                        void>::type
copy_buffer(const ViewType &, unsigned char *) {}

template <class ViewType>
typename std::enable_if<
    !(std::is_same<Kokkos::AnonymousSpace,
                   typename ViewType::memory_space>::value ||
      std::is_same<
          Kokkos::ScratchMemorySpace<typename ViewType::execution_space>,
          typename ViewType::memory_space>::value ||
      std::is_const<typename ViewType::value_type>::value),
    void>::type
copy_buffer(const ViewType &v, unsigned char *buff) {
  Kokkos::Impl::DeepCopy<typename ViewType::memory_space, Kokkos::HostSpace,
                         typename ViewType::execution_space>(
      v.data(), (void *)buff, v.span() * sizeof(typename ViewType::value_type));
}

template <class ViewType>
class ViewHookDeepCopy<ViewType, void> {
 public:
  using view_type = ViewType;

  static inline void update_view(view_type &, const void *) {}

  // default buffer is assumed to be host space...
  static void deep_copy(unsigned char *buff, const view_type &v) {
    copy_buffer(buff, v);
  }

  static void deep_copy(const view_type &v, unsigned char *buff) {
    copy_buffer(v, buff);
  }

  static constexpr const char *m_name = "Default";
};
}  // namespace Experimental
}  // namespace Kokkos

#endif  // KOKKOS_VIEWHOOK_DEEPCOPY_HPP
