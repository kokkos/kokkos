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

#ifndef KOKKOS_BEGIN_END_HPP
#define KOKKOS_BEGIN_END_HPP

/// \file Kokkos_NonModifyingSequenceOperations.hpp
/// \brief Kokkos non-modifying sequence operations

namespace Kokkos {
namespace Experimental {

template <class DataType, class... Properties>
auto begin(const Kokkos::View<DataType, Properties...>& v)
    -> decltype(v.data()) {
  using ViewInType = Kokkos::View<DataType, Properties...>;
  static_assert(is_admissible_view_to_kokkos_std_non_modifying_sequence_op<
                    ViewInType>::value,
                "Currently, Kokkos::Experimental::begin only accepts 1D "
                "contiguous Views.");

  KOKKOS_EXPECTS(v.span_is_contiguous());
  return v.data();
}

template <class DataType, class... Properties>
auto end(const Kokkos::View<DataType, Properties...>& v) -> decltype(v.data()) {
  using ViewInType = Kokkos::View<DataType, Properties...>;
  static_assert(
      is_admissible_view_to_kokkos_std_non_modifying_sequence_op<
          ViewInType>::value,
      "Currently, Kokkos::Experimental::end only accepts 1D contiguous Views.");

  KOKKOS_EXPECTS(v.span_is_contiguous());
  return v.data() + v.size();
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
