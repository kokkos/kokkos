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

template <class DataType, class... Properties>
Kokkos::View<
    typename Kokkos::ViewTraits<DataType, Properties...>::non_const_data_type,
    typename Kokkos::ViewTraits<DataType, Properties...>::array_layout,
    Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
make_unmanaged_host_view(const Kokkos::View<DataType, Properties...> &view,
                         unsigned char *buff) {
  using traits_type   = Kokkos::ViewTraits<DataType, Properties...>;
  using new_data_type = typename traits_type::non_const_data_type;
  using layout_type   = typename traits_type::array_layout;
  using new_view_type =
      Kokkos::View<new_data_type, layout_type, Kokkos::HostSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

  return new_view_type(
      reinterpret_cast<typename new_view_type::pointer_type>(buff),
      view.rank_dynamic > 0 ? view.extent(0) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      view.rank_dynamic > 1 ? view.extent(1) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      view.rank_dynamic > 2 ? view.extent(2) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      view.rank_dynamic > 3 ? view.extent(3) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      view.rank_dynamic > 4 ? view.extent(4) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      view.rank_dynamic > 5 ? view.extent(5) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      view.rank_dynamic > 6 ? view.extent(6) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      view.rank_dynamic > 7 ? view.extent(7) : KOKKOS_IMPL_CTOR_DEFAULT_ARG);
}

template <class ViewType>
class ViewHookCopyView<
    ViewType,
    typename std::enable_if<
        (!std::is_const<ViewType>::value &&
         !std::is_same<Kokkos::AnonymousSpace,
                       typename ViewType::memory_space>::value &&
         std::is_same<typename ViewType::memory_space::resilient_space,
                      typename ViewType::memory_space>::value),
        void>::type> {
 public:
  static inline void copy_view(ViewType &view, const void *ptr) {
    auto src = make_unmanaged_host_view(view, ptr);
    Kokkos::deep_copy(view, src);
  }

  static inline void copy_view(const void *ptr, ViewType &view) {
    auto src = make_unmanaged_host_view(view, ptr);
    Kokkos::deep_copy(src, view);
  }

  static constexpr const char *m_name = "Non-ConstImpl";
};
}  // namespace Experimental
}  // namespace Kokkos

#endif  // KOKKOS_VIEWHOOK_DEEPCOPY_HPP
