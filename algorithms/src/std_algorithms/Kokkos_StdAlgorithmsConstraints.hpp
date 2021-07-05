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

#ifndef KOKKOS_STD_ALGORITHMS_CONSTRAINTS_HPP_
#define KOKKOS_STD_ALGORITHMS_CONSTRAINTS_HPP_

#include <Kokkos_Core.hpp>

namespace Kokkos {
namespace Experimental {

template <typename T, typename enable = void>
struct is_admissible_view_to_kokkos_std_non_modifying_sequence_op
    : std::false_type {};

template <typename T>
struct is_admissible_view_to_kokkos_std_non_modifying_sequence_op<
  T,
  std::enable_if_t<
    ::Kokkos::is_view<T>::value and T::rank == 1 and
    (std::is_same<typename T::traits::array_layout, Kokkos::LayoutLeft>::value
     or std::is_same<typename T::traits::array_layout, Kokkos::LayoutRight>::value
     or std::is_same<typename T::traits::array_layout, Kokkos::LayoutStride>::value)
    >
  >
  : std::true_type {};


template <typename ... Args>
using is_admissible_view_to_kokkos_std_modifying_sequence_op =
  is_admissible_view_to_kokkos_std_non_modifying_sequence_op<Args...>;

}  // namespace Experimental
}  // namespace Kokkos

#endif
