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

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#include <Kokkos_Macros.hpp>
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif
#ifndef KOKKOS_LOCALDEEPCOPY_HPP_
#define KOKKOS_LOCALDEEPCOPY_HPP_

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

namespace Experimental {
/** \brief  A local deep copy between views of the default specialization,
 * compatible type, same non-zero rank.
 */
template <class TeamType, class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION
local_deep_copy_contiguous(const TeamType& team, const View<DT, DP...>& dst,
                           const View<ST, SP...>& src) {
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, src.span()),
                       [&](const int& i) { dst.data()[i] = src.data()[i]; });
}
//----------------------------------------------------------------------------
template <class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy_contiguous(
    const View<DT, DP...>& dst, const View<ST, SP...>& src) {
  for (size_t i = 0; i < src.span(); ++i) {
    dst.data()[i] = src.data()[i];
  }
}
//----------------------------------------------------------------------------
template <class TeamType, class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType& team, const View<DT, DP...>& dst,
    const View<ST, SP...>& src,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 1 &&
                      unsigned(ViewTraits<ST, SP...>::rank) == 1)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0);

  team.team_barrier();
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, N),
                       [&](const int& i) { dst(i) = src(i); });
  team.team_barrier();
}
//----------------------------------------------------------------------------
template <class TeamType, class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType& team, const View<DT, DP...>& dst,
    const View<ST, SP...>& src,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 2 &&
                      unsigned(ViewTraits<ST, SP...>::rank) == 2)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0) * dst.extent(1);

  if (dst.span_is_contiguous() && src.span_is_contiguous()) {
    team.team_barrier();
    local_deep_copy_contiguous(team, dst, src);
    team.team_barrier();
  } else {
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, N), [&](const int& i) {
      int i0      = i % dst.extent(0);
      int i1      = i / dst.extent(0);
      dst(i0, i1) = src(i0, i1);
    });
    team.team_barrier();
  }
}
//----------------------------------------------------------------------------
template <class TeamType, class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType& team, const View<DT, DP...>& dst,
    const View<ST, SP...>& src,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 3 &&
                      unsigned(ViewTraits<ST, SP...>::rank) == 3)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0) * dst.extent(1) * dst.extent(2);

  if (dst.span_is_contiguous() && src.span_is_contiguous()) {
    team.team_barrier();
    local_deep_copy_contiguous(team, dst, src);
    team.team_barrier();
  } else {
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, N), [&](const int& i) {
      int i0          = i % dst.extent(0);
      int itmp        = i / dst.extent(0);
      int i1          = itmp % dst.extent(1);
      int i2          = itmp / dst.extent(1);
      dst(i0, i1, i2) = src(i0, i1, i2);
    });
    team.team_barrier();
  }
}
//----------------------------------------------------------------------------
template <class TeamType, class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType& team, const View<DT, DP...>& dst,
    const View<ST, SP...>& src,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 4 &&
                      unsigned(ViewTraits<ST, SP...>::rank) == 4)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N =
      dst.extent(0) * dst.extent(1) * dst.extent(2) * dst.extent(3);

  if (dst.span_is_contiguous() && src.span_is_contiguous()) {
    team.team_barrier();
    local_deep_copy_contiguous(team, dst, src);
    team.team_barrier();
  } else {
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, N), [&](const int& i) {
      int i0              = i % dst.extent(0);
      int itmp            = i / dst.extent(0);
      int i1              = itmp % dst.extent(1);
      itmp                = itmp / dst.extent(1);
      int i2              = itmp % dst.extent(2);
      int i3              = itmp / dst.extent(2);
      dst(i0, i1, i2, i3) = src(i0, i1, i2, i3);
    });
    team.team_barrier();
  }
}
//----------------------------------------------------------------------------
template <class TeamType, class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType& team, const View<DT, DP...>& dst,
    const View<ST, SP...>& src,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 5 &&
                      unsigned(ViewTraits<ST, SP...>::rank) == 5)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0) * dst.extent(1) * dst.extent(2) *
                   dst.extent(3) * dst.extent(4);

  if (dst.span_is_contiguous() && src.span_is_contiguous()) {
    team.team_barrier();
    local_deep_copy_contiguous(team, dst, src);
    team.team_barrier();
  } else {
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, N), [&](const int& i) {
      int i0                  = i % dst.extent(0);
      int itmp                = i / dst.extent(0);
      int i1                  = itmp % dst.extent(1);
      itmp                    = itmp / dst.extent(1);
      int i2                  = itmp % dst.extent(2);
      itmp                    = itmp / dst.extent(2);
      int i3                  = itmp % dst.extent(3);
      int i4                  = itmp / dst.extent(3);
      dst(i0, i1, i2, i3, i4) = src(i0, i1, i2, i3, i4);
    });
    team.team_barrier();
  }
}
//----------------------------------------------------------------------------
template <class TeamType, class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType& team, const View<DT, DP...>& dst,
    const View<ST, SP...>& src,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 6 &&
                      unsigned(ViewTraits<ST, SP...>::rank) == 6)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0) * dst.extent(1) * dst.extent(2) *
                   dst.extent(3) * dst.extent(4) * dst.extent(5);

  if (dst.span_is_contiguous() && src.span_is_contiguous()) {
    team.team_barrier();
    local_deep_copy_contiguous(team, dst, src);
    team.team_barrier();
  } else {
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, N), [&](const int& i) {
      int i0                      = i % dst.extent(0);
      int itmp                    = i / dst.extent(0);
      int i1                      = itmp % dst.extent(1);
      itmp                        = itmp / dst.extent(1);
      int i2                      = itmp % dst.extent(2);
      itmp                        = itmp / dst.extent(2);
      int i3                      = itmp % dst.extent(3);
      itmp                        = itmp / dst.extent(3);
      int i4                      = itmp % dst.extent(4);
      int i5                      = itmp / dst.extent(4);
      dst(i0, i1, i2, i3, i4, i5) = src(i0, i1, i2, i3, i4, i5);
    });
    team.team_barrier();
  }
}
//----------------------------------------------------------------------------
template <class TeamType, class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType& team, const View<DT, DP...>& dst,
    const View<ST, SP...>& src,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 7 &&
                      unsigned(ViewTraits<ST, SP...>::rank) == 7)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0) * dst.extent(1) * dst.extent(2) *
                   dst.extent(3) * dst.extent(4) * dst.extent(5) *
                   dst.extent(6);

  if (dst.span_is_contiguous() && src.span_is_contiguous()) {
    team.team_barrier();
    local_deep_copy_contiguous(team, dst, src);
    team.team_barrier();
  } else {
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, N), [&](const int& i) {
      int i0                          = i % dst.extent(0);
      int itmp                        = i / dst.extent(0);
      int i1                          = itmp % dst.extent(1);
      itmp                            = itmp / dst.extent(1);
      int i2                          = itmp % dst.extent(2);
      itmp                            = itmp / dst.extent(2);
      int i3                          = itmp % dst.extent(3);
      itmp                            = itmp / dst.extent(3);
      int i4                          = itmp % dst.extent(4);
      itmp                            = itmp / dst.extent(4);
      int i5                          = itmp % dst.extent(5);
      int i6                          = itmp / dst.extent(5);
      dst(i0, i1, i2, i3, i4, i5, i6) = src(i0, i1, i2, i3, i4, i5, i6);
    });
    team.team_barrier();
  }
}
//----------------------------------------------------------------------------
template <class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...>& dst, const View<ST, SP...>& src,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 1 &&
                      unsigned(ViewTraits<ST, SP...>::rank) == 1)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0);

  for (size_t i = 0; i < N; ++i) {
    dst(i) = src(i);
  }
}
//----------------------------------------------------------------------------
template <class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...>& dst, const View<ST, SP...>& src,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 2 &&
                      unsigned(ViewTraits<ST, SP...>::rank) == 2)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  if (dst.span_is_contiguous() && src.span_is_contiguous()) {
    local_deep_copy_contiguous(dst, src);
  } else {
    for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
      for (size_t i1 = 0; i1 < dst.extent(1); ++i1) dst(i0, i1) = src(i0, i1);
  }
}
//----------------------------------------------------------------------------
template <class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...>& dst, const View<ST, SP...>& src,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 3 &&
                      unsigned(ViewTraits<ST, SP...>::rank) == 3)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  if (dst.span_is_contiguous() && src.span_is_contiguous()) {
    local_deep_copy_contiguous(dst, src);
  } else {
    for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
      for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
        for (size_t i2 = 0; i2 < dst.extent(2); ++i2)
          dst(i0, i1, i2) = src(i0, i1, i2);
  }
}
//----------------------------------------------------------------------------
template <class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...>& dst, const View<ST, SP...>& src,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 4 &&
                      unsigned(ViewTraits<ST, SP...>::rank) == 4)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  if (dst.span_is_contiguous() && src.span_is_contiguous()) {
    local_deep_copy_contiguous(dst, src);
  } else {
    for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
      for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
        for (size_t i2 = 0; i2 < dst.extent(2); ++i2)
          for (size_t i3 = 0; i3 < dst.extent(3); ++i3)
            dst(i0, i1, i2, i3) = src(i0, i1, i2, i3);
  }
}
//----------------------------------------------------------------------------
template <class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...>& dst, const View<ST, SP...>& src,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 5 &&
                      unsigned(ViewTraits<ST, SP...>::rank) == 5)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  if (dst.span_is_contiguous() && src.span_is_contiguous()) {
    local_deep_copy_contiguous(dst, src);
  } else {
    for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
      for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
        for (size_t i2 = 0; i2 < dst.extent(2); ++i2)
          for (size_t i3 = 0; i3 < dst.extent(3); ++i3)
            for (size_t i4 = 0; i4 < dst.extent(4); ++i4)
              dst(i0, i1, i2, i3, i4) = src(i0, i1, i2, i3, i4);
  }
}
//----------------------------------------------------------------------------
template <class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...>& dst, const View<ST, SP...>& src,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 6 &&
                      unsigned(ViewTraits<ST, SP...>::rank) == 6)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  if (dst.span_is_contiguous() && src.span_is_contiguous()) {
    local_deep_copy_contiguous(dst, src);
  } else {
    for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
      for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
        for (size_t i2 = 0; i2 < dst.extent(2); ++i2)
          for (size_t i3 = 0; i3 < dst.extent(3); ++i3)
            for (size_t i4 = 0; i4 < dst.extent(4); ++i4)
              for (size_t i5 = 0; i5 < dst.extent(5); ++i5)
                dst(i0, i1, i2, i3, i4, i5) = src(i0, i1, i2, i3, i4, i5);
  }
}
//----------------------------------------------------------------------------
template <class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...>& dst, const View<ST, SP...>& src,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 7 &&
                      unsigned(ViewTraits<ST, SP...>::rank) == 7)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  if (dst.span_is_contiguous() && src.span_is_contiguous()) {
    local_deep_copy_contiguous(dst, src);
  } else {
    for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
      for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
        for (size_t i2 = 0; i2 < dst.extent(2); ++i2)
          for (size_t i3 = 0; i3 < dst.extent(3); ++i3)
            for (size_t i4 = 0; i4 < dst.extent(4); ++i4)
              for (size_t i5 = 0; i5 < dst.extent(5); ++i5)
                for (size_t i6 = 0; i6 < dst.extent(6); ++i6)
                  dst(i0, i1, i2, i3, i4, i5, i6) =
                      src(i0, i1, i2, i3, i4, i5, i6);
  }
}
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
/** \brief  Deep copy a value into a view.  */
template <class TeamType, class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy_contiguous(
    const TeamType& team, const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    std::enable_if_t<std::is_same<typename ViewTraits<DT, DP...>::specialize,
                                  void>::value>* = nullptr) {
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, dst.span()),
                       [&](const int& i) { dst.data()[i] = value; });
}
//----------------------------------------------------------------------------
template <class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy_contiguous(
    const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    std::enable_if_t<std::is_same<typename ViewTraits<DT, DP...>::specialize,
                                  void>::value>* = nullptr) {
  for (size_t i = 0; i < dst.span(); ++i) {
    dst.data()[i] = value;
  }
}
//----------------------------------------------------------------------------
template <class TeamType, class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType& team, const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 1)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0);

  team.team_barrier();
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, N),
                       [&](const int& i) { dst(i) = value; });
  team.team_barrier();
}
//----------------------------------------------------------------------------
template <class TeamType, class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType& team, const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 2)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0) * dst.extent(1);

  if (dst.span_is_contiguous()) {
    team.team_barrier();
    local_deep_copy_contiguous(team, dst, value);
    team.team_barrier();
  } else {
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, N), [&](const int& i) {
      int i0      = i % dst.extent(0);
      int i1      = i / dst.extent(0);
      dst(i0, i1) = value;
    });
    team.team_barrier();
  }
}
//----------------------------------------------------------------------------
template <class TeamType, class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType& team, const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 3)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0) * dst.extent(1) * dst.extent(2);

  if (dst.span_is_contiguous()) {
    team.team_barrier();
    local_deep_copy_contiguous(team, dst, value);
    team.team_barrier();
  } else {
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, N), [&](const int& i) {
      int i0          = i % dst.extent(0);
      int itmp        = i / dst.extent(0);
      int i1          = itmp % dst.extent(1);
      int i2          = itmp / dst.extent(1);
      dst(i0, i1, i2) = value;
    });
    team.team_barrier();
  }
}
//----------------------------------------------------------------------------
template <class TeamType, class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType& team, const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 4)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N =
      dst.extent(0) * dst.extent(1) * dst.extent(2) * dst.extent(3);

  if (dst.span_is_contiguous()) {
    team.team_barrier();
    local_deep_copy_contiguous(team, dst, value);
    team.team_barrier();
  } else {
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, N), [&](const int& i) {
      int i0              = i % dst.extent(0);
      int itmp            = i / dst.extent(0);
      int i1              = itmp % dst.extent(1);
      itmp                = itmp / dst.extent(1);
      int i2              = itmp % dst.extent(2);
      int i3              = itmp / dst.extent(2);
      dst(i0, i1, i2, i3) = value;
    });
    team.team_barrier();
  }
}
//----------------------------------------------------------------------------
template <class TeamType, class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType& team, const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 5)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0) * dst.extent(1) * dst.extent(2) *
                   dst.extent(3) * dst.extent(4);

  if (dst.span_is_contiguous()) {
    team.team_barrier();
    local_deep_copy_contiguous(team, dst, value);
    team.team_barrier();
  } else {
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, N), [&](const int& i) {
      int i0                  = i % dst.extent(0);
      int itmp                = i / dst.extent(0);
      int i1                  = itmp % dst.extent(1);
      itmp                    = itmp / dst.extent(1);
      int i2                  = itmp % dst.extent(2);
      itmp                    = itmp / dst.extent(2);
      int i3                  = itmp % dst.extent(3);
      int i4                  = itmp / dst.extent(3);
      dst(i0, i1, i2, i3, i4) = value;
    });
    team.team_barrier();
  }
}
//----------------------------------------------------------------------------
template <class TeamType, class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType& team, const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 6)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0) * dst.extent(1) * dst.extent(2) *
                   dst.extent(3) * dst.extent(4) * dst.extent(5);

  if (dst.span_is_contiguous()) {
    team.team_barrier();
    local_deep_copy_contiguous(team, dst, value);
    team.team_barrier();
  } else {
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, N), [&](const int& i) {
      int i0                      = i % dst.extent(0);
      int itmp                    = i / dst.extent(0);
      int i1                      = itmp % dst.extent(1);
      itmp                        = itmp / dst.extent(1);
      int i2                      = itmp % dst.extent(2);
      itmp                        = itmp / dst.extent(2);
      int i3                      = itmp % dst.extent(3);
      itmp                        = itmp / dst.extent(3);
      int i4                      = itmp % dst.extent(4);
      int i5                      = itmp / dst.extent(4);
      dst(i0, i1, i2, i3, i4, i5) = value;
    });
    team.team_barrier();
  }
}
//----------------------------------------------------------------------------
template <class TeamType, class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType& team, const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 7)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0) * dst.extent(1) * dst.extent(2) *
                   dst.extent(3) * dst.extent(4) * dst.extent(5) *
                   dst.extent(6);

  if (dst.span_is_contiguous()) {
    team.team_barrier();
    local_deep_copy_contiguous(team, dst, value);
    team.team_barrier();
  } else {
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, N), [&](const int& i) {
      int i0                          = i % dst.extent(0);
      int itmp                        = i / dst.extent(0);
      int i1                          = itmp % dst.extent(1);
      itmp                            = itmp / dst.extent(1);
      int i2                          = itmp % dst.extent(2);
      itmp                            = itmp / dst.extent(2);
      int i3                          = itmp % dst.extent(3);
      itmp                            = itmp / dst.extent(3);
      int i4                          = itmp % dst.extent(4);
      itmp                            = itmp / dst.extent(4);
      int i5                          = itmp % dst.extent(5);
      int i6                          = itmp / dst.extent(5);
      dst(i0, i1, i2, i3, i4, i5, i6) = value;
    });
    team.team_barrier();
  }
}
//----------------------------------------------------------------------------
template <class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 1)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0);

  for (size_t i = 0; i < N; ++i) {
    dst(i) = value;
  }
}
//----------------------------------------------------------------------------
template <class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 2)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  if (dst.span_is_contiguous()) {
    local_deep_copy_contiguous(dst, value);
  } else {
    for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
      for (size_t i1 = 0; i1 < dst.extent(1); ++i1) dst(i0, i1) = value;
  }
}
//----------------------------------------------------------------------------
template <class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 3)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  if (dst.span_is_contiguous()) {
    local_deep_copy_contiguous(dst, value);
  } else {
    for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
      for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
        for (size_t i2 = 0; i2 < dst.extent(2); ++i2) dst(i0, i1, i2) = value;
  }
}
//----------------------------------------------------------------------------
template <class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 4)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  if (dst.span_is_contiguous()) {
    local_deep_copy_contiguous(dst, value);
  } else {
    for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
      for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
        for (size_t i2 = 0; i2 < dst.extent(2); ++i2)
          for (size_t i3 = 0; i3 < dst.extent(3); ++i3)
            dst(i0, i1, i2, i3) = value;
  }
}
//----------------------------------------------------------------------------
template <class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 5)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  if (dst.span_is_contiguous()) {
    local_deep_copy_contiguous(dst, value);
  } else {
    for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
      for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
        for (size_t i2 = 0; i2 < dst.extent(2); ++i2)
          for (size_t i3 = 0; i3 < dst.extent(3); ++i3)
            for (size_t i4 = 0; i4 < dst.extent(4); ++i4)
              dst(i0, i1, i2, i3, i4) = value;
  }
}
//----------------------------------------------------------------------------
template <class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 6)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  if (dst.span_is_contiguous()) {
    local_deep_copy_contiguous(dst, value);
  } else {
    for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
      for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
        for (size_t i2 = 0; i2 < dst.extent(2); ++i2)
          for (size_t i3 = 0; i3 < dst.extent(3); ++i3)
            for (size_t i4 = 0; i4 < dst.extent(4); ++i4)
              for (size_t i5 = 0; i5 < dst.extent(5); ++i5)
                dst(i0, i1, i2, i3, i4, i5) = value;
  }
}
//----------------------------------------------------------------------------
template <class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    std::enable_if_t<(unsigned(ViewTraits<DT, DP...>::rank) == 7)>* = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  if (dst.span_is_contiguous()) {
    local_deep_copy_contiguous(dst, value);
  } else {
    for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
      for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
        for (size_t i2 = 0; i2 < dst.extent(2); ++i2)
          for (size_t i3 = 0; i3 < dst.extent(3); ++i3)
            for (size_t i4 = 0; i4 < dst.extent(4); ++i4)
              for (size_t i5 = 0; i5 < dst.extent(5); ++i5)
                for (size_t i6 = 0; i6 < dst.extent(6); ++i6)
                  dst(i0, i1, i2, i3, i4, i5, i6) = value;
  }
}
} /* namespace Experimental */
} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

/** \brief  Deep copy a value from Host memory into a view. ExecSpace can access
 * dst */
template <class ExecSpace, class DT, class... DP>
inline void deep_copy(
    const ExecSpace& space, const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    std::enable_if_t<
        Kokkos::is_execution_space<ExecSpace>::value &&
        std::is_void<typename ViewTraits<DT, DP...>::specialize>::value &&
        Kokkos::SpaceAccessibility<ExecSpace, typename ViewTraits<DT, DP...>::
                                                  memory_space>::accessible>* =
        nullptr) {
  using dst_traits = ViewTraits<DT, DP...>;
  static_assert(std::is_same<typename dst_traits::non_const_value_type,
                             typename dst_traits::value_type>::value,
                "deep_copy requires non-const type");
  using dst_memory_space = typename dst_traits::memory_space;
  if (Kokkos::Tools::Experimental::get_callbacks().begin_deep_copy != nullptr) {
    Kokkos::Profiling::beginDeepCopy(
        Kokkos::Profiling::make_space_handle(dst_memory_space::name()),
        dst.label(), dst.data(),
        Kokkos::Profiling::make_space_handle(Kokkos::HostSpace::name()),
        "(none)", &value, dst.span() * sizeof(typename dst_traits::value_type));
  }
  if (dst.data() == nullptr) {
    space.fence("Kokkos::deep_copy: scalar copy on space, dst data is null");
  } else if (dst.span_is_contiguous()) {
    Impl::contiguous_fill_or_memset(space, dst, value);
  } else {
    using ViewType = View<DT, DP...>;
    // Figure out iteration order to do the ViewFill
    int64_t strides[ViewType::Rank + 1];
    dst.stride(strides);
    Kokkos::Iterate iterate;
    if (std::is_same<typename ViewType::array_layout,
                     Kokkos::LayoutRight>::value) {
      iterate = Kokkos::Iterate::Right;
    } else if (std::is_same<typename ViewType::array_layout,
                            Kokkos::LayoutLeft>::value) {
      iterate = Kokkos::Iterate::Left;
    } else if (std::is_same<typename ViewType::array_layout,
                            Kokkos::LayoutStride>::value) {
      if (strides[0] > strides[ViewType::Rank > 0 ? ViewType::Rank - 1 : 0])
        iterate = Kokkos::Iterate::Right;
      else
        iterate = Kokkos::Iterate::Left;
    } else {
      if (std::is_same<typename ViewType::execution_space::array_layout,
                       Kokkos::LayoutRight>::value)
        iterate = Kokkos::Iterate::Right;
      else
        iterate = Kokkos::Iterate::Left;
    }

    // Lets call the right ViewFill functor based on integer space needed and
    // iteration type
    using ViewTypeUniform =
        std::conditional_t<ViewType::Rank == 0,
                           typename ViewType::uniform_runtime_type,
                           typename ViewType::uniform_runtime_nomemspace_type>;
    if (dst.span() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
      if (iterate == Kokkos::Iterate::Right)
        Kokkos::Impl::ViewFill<ViewTypeUniform, Kokkos::LayoutRight, ExecSpace,
                               ViewType::Rank, int64_t>(dst, value, space);
      else
        Kokkos::Impl::ViewFill<ViewTypeUniform, Kokkos::LayoutLeft, ExecSpace,
                               ViewType::Rank, int64_t>(dst, value, space);
    } else {
      if (iterate == Kokkos::Iterate::Right)
        Kokkos::Impl::ViewFill<ViewTypeUniform, Kokkos::LayoutRight, ExecSpace,
                               ViewType::Rank, int32_t>(dst, value, space);
      else
        Kokkos::Impl::ViewFill<ViewTypeUniform, Kokkos::LayoutLeft, ExecSpace,
                               ViewType::Rank, int32_t>(dst, value, space);
    }
  }
  if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
    Kokkos::Profiling::endDeepCopy();
  }
}

/** \brief  Deep copy a value from Host memory into a view. ExecSpace can not
 * access dst */
template <class ExecSpace, class DT, class... DP>
inline void deep_copy(
    const ExecSpace& space, const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    std::enable_if_t<
        Kokkos::is_execution_space<ExecSpace>::value &&
        std::is_void<typename ViewTraits<DT, DP...>::specialize>::value &&
        !Kokkos::SpaceAccessibility<ExecSpace, typename ViewTraits<DT, DP...>::
                                                   memory_space>::accessible>* =
        nullptr) {
  using dst_traits = ViewTraits<DT, DP...>;
  static_assert(std::is_same<typename dst_traits::non_const_value_type,
                             typename dst_traits::value_type>::value,
                "deep_copy requires non-const type");
  using dst_memory_space = typename dst_traits::memory_space;
  if (Kokkos::Tools::Experimental::get_callbacks().begin_deep_copy != nullptr) {
    Kokkos::Profiling::beginDeepCopy(
        Kokkos::Profiling::make_space_handle(dst_memory_space::name()),
        dst.label(), dst.data(),
        Kokkos::Profiling::make_space_handle(Kokkos::HostSpace::name()),
        "(none)", &value, dst.span() * sizeof(typename dst_traits::value_type));
  }
  if (dst.data() == nullptr) {
    space.fence(
        "Kokkos::deep_copy: scalar-to-view copy on space, dst data is null");
  } else {
    space.fence("Kokkos::deep_copy: scalar-to-view copy on space, pre copy");
    using fill_exec_space = typename dst_traits::memory_space::execution_space;
    if (dst.span_is_contiguous()) {
      Impl::contiguous_fill_or_memset(fill_exec_space(), dst, value);
    } else {
      using ViewTypeUniform = std::conditional_t<
          View<DT, DP...>::Rank == 0,
          typename View<DT, DP...>::uniform_runtime_type,
          typename View<DT, DP...>::uniform_runtime_nomemspace_type>;
      Kokkos::Impl::ViewFill<ViewTypeUniform, typename dst_traits::array_layout,
                             fill_exec_space>(dst, value, fill_exec_space());
    }
    fill_exec_space().fence(
        "Kokkos::deep_copy: scalar-to-view copy on space, fence after fill");
  }
  if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
    Kokkos::Profiling::endDeepCopy();
  }
}

/** \brief  Deep copy into a value in Host memory from a view.  */
template <class ExecSpace, class ST, class... SP>
inline void deep_copy(
    const ExecSpace& exec_space,
    typename ViewTraits<ST, SP...>::non_const_value_type& dst,
    const View<ST, SP...>& src,
    std::enable_if_t<Kokkos::is_execution_space<ExecSpace>::value &&
                     std::is_same<typename ViewTraits<ST, SP...>::specialize,
                                  void>::value>* = nullptr) {
  using src_traits       = ViewTraits<ST, SP...>;
  using src_memory_space = typename src_traits::memory_space;
  static_assert(src_traits::rank == 0,
                "ERROR: Non-rank-zero view in deep_copy( value , View )");
  if (Kokkos::Tools::Experimental::get_callbacks().begin_deep_copy != nullptr) {
    Kokkos::Profiling::beginDeepCopy(
        Kokkos::Profiling::make_space_handle(Kokkos::HostSpace::name()),
        "(none)", &dst,
        Kokkos::Profiling::make_space_handle(src_memory_space::name()),
        src.label(), src.data(), sizeof(ST));
  }

  if (src.data() == nullptr) {
    exec_space.fence(
        "Kokkos::deep_copy: view-to-scalar copy on space, src data is null");
    if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
      Kokkos::Profiling::endDeepCopy();
    }
    return;
  }

  Kokkos::Impl::DeepCopy<HostSpace, src_memory_space, ExecSpace>(
      exec_space, &dst, src.data(), sizeof(ST));
  if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
    Kokkos::Profiling::endDeepCopy();
  }
}

//----------------------------------------------------------------------------
/** \brief  A deep copy between views of compatible type, and rank zero.  */
template <class ExecSpace, class DT, class... DP, class ST, class... SP>
inline void deep_copy(
    const ExecSpace& exec_space, const View<DT, DP...>& dst,
    const View<ST, SP...>& src,
    std::enable_if_t<
        (Kokkos::is_execution_space<ExecSpace>::value &&
         std::is_void<typename ViewTraits<DT, DP...>::specialize>::value &&
         std::is_void<typename ViewTraits<ST, SP...>::specialize>::value &&
         (unsigned(ViewTraits<DT, DP...>::rank) == unsigned(0) &&
          unsigned(ViewTraits<ST, SP...>::rank) == unsigned(0)))>* = nullptr) {
  using src_traits = ViewTraits<ST, SP...>;
  using dst_traits = ViewTraits<DT, DP...>;

  using src_memory_space = typename src_traits::memory_space;
  using dst_memory_space = typename dst_traits::memory_space;
  static_assert(std::is_same<typename dst_traits::value_type,
                             typename src_traits::non_const_value_type>::value,
                "deep_copy requires matching non-const destination type");

  if (Kokkos::Tools::Experimental::get_callbacks().begin_deep_copy != nullptr) {
    Kokkos::Profiling::beginDeepCopy(
        Kokkos::Profiling::make_space_handle(dst_memory_space::name()),
        dst.label(), dst.data(),
        Kokkos::Profiling::make_space_handle(src_memory_space::name()),
        src.label(), src.data(), sizeof(DT));
  }

  if (dst.data() == nullptr && src.data() == nullptr) {
    exec_space.fence(
        "Kokkos::deep_copy: view-to-view copy on space, data is null");
    if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
      Kokkos::Profiling::endDeepCopy();
    }
    return;
  }

  if (dst.data() != src.data()) {
    Kokkos::Impl::DeepCopy<dst_memory_space, src_memory_space, ExecSpace>(
        exec_space, dst.data(), src.data(),
        sizeof(typename dst_traits::value_type));
  }
  if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
    Kokkos::Profiling::endDeepCopy();
  }
}

//----------------------------------------------------------------------------
/** \brief  A deep copy between views of the default specialization, compatible
 * type, same non-zero rank
 */
template <class ExecSpace, class DT, class... DP, class ST, class... SP>
inline void deep_copy(
    const ExecSpace& exec_space, const View<DT, DP...>& dst,
    const View<ST, SP...>& src,
    std::enable_if_t<
        (Kokkos::is_execution_space<ExecSpace>::value &&
         std::is_void<typename ViewTraits<DT, DP...>::specialize>::value &&
         std::is_void<typename ViewTraits<ST, SP...>::specialize>::value &&
         (unsigned(ViewTraits<DT, DP...>::rank) != 0 ||
          unsigned(ViewTraits<ST, SP...>::rank) != 0))>* = nullptr) {
  using dst_type = View<DT, DP...>;
  using src_type = View<ST, SP...>;

  static_assert(std::is_same<typename dst_type::value_type,
                             typename dst_type::non_const_value_type>::value,
                "deep_copy requires non-const destination type");

  static_assert((unsigned(dst_type::rank) == unsigned(src_type::rank)),
                "deep_copy requires Views of equal rank");

  using dst_execution_space = typename dst_type::execution_space;
  using src_execution_space = typename src_type::execution_space;
  using dst_memory_space    = typename dst_type::memory_space;
  using src_memory_space    = typename src_type::memory_space;
  using dst_value_type      = typename dst_type::value_type;
  using src_value_type      = typename src_type::value_type;

  if (Kokkos::Tools::Experimental::get_callbacks().begin_deep_copy != nullptr) {
    Kokkos::Profiling::beginDeepCopy(
        Kokkos::Profiling::make_space_handle(dst_memory_space::name()),
        dst.label(), dst.data(),
        Kokkos::Profiling::make_space_handle(src_memory_space::name()),
        src.label(), src.data(), dst.span() * sizeof(dst_value_type));
  }

  dst_value_type* dst_start = dst.data();
  dst_value_type* dst_end   = dst.data() + dst.span();
  src_value_type* src_start = src.data();
  src_value_type* src_end   = src.data() + src.span();

  // Early dropout if identical range
  if ((dst_start == nullptr || src_start == nullptr) ||
      ((std::ptrdiff_t(dst_start) == std::ptrdiff_t(src_start)) &&
       (std::ptrdiff_t(dst_end) == std::ptrdiff_t(src_end)))) {
    // throw if dimension mismatch
    if ((src.extent(0) != dst.extent(0)) || (src.extent(1) != dst.extent(1)) ||
        (src.extent(2) != dst.extent(2)) || (src.extent(3) != dst.extent(3)) ||
        (src.extent(4) != dst.extent(4)) || (src.extent(5) != dst.extent(5)) ||
        (src.extent(6) != dst.extent(6)) || (src.extent(7) != dst.extent(7))) {
      std::string message(
          "Deprecation Error: Kokkos::deep_copy extents of views don't "
          "match: ");
      message += dst.label();
      message += "(";
      for (int r = 0; r < dst_type::Rank - 1; r++) {
        message += std::to_string(dst.extent(r));
        message += ",";
      }
      message += std::to_string(dst.extent(dst_type::Rank - 1));
      message += ") ";
      message += src.label();
      message += "(";
      for (int r = 0; r < src_type::Rank - 1; r++) {
        message += std::to_string(src.extent(r));
        message += ",";
      }
      message += std::to_string(src.extent(src_type::Rank - 1));
      message += ") ";

      Kokkos::Impl::throw_runtime_exception(message);
    }
    if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
      Kokkos::Profiling::endDeepCopy();
    }
    return;
  }

  enum {
    ExecCanAccessSrcDst =
        Kokkos::SpaceAccessibility<ExecSpace, dst_memory_space>::accessible &&
        Kokkos::SpaceAccessibility<ExecSpace, src_memory_space>::accessible
  };
  enum {
    DstExecCanAccessSrc =
        Kokkos::SpaceAccessibility<dst_execution_space,
                                   src_memory_space>::accessible
  };

  enum {
    SrcExecCanAccessDst =
        Kokkos::SpaceAccessibility<src_execution_space,
                                   dst_memory_space>::accessible
  };

  // Error out for non-identical overlapping views.
  if ((((std::ptrdiff_t)dst_start < (std::ptrdiff_t)src_end) &&
       ((std::ptrdiff_t)dst_end > (std::ptrdiff_t)src_start)) &&
      ((dst.span_is_contiguous() && src.span_is_contiguous()))) {
    std::string message("Error: Kokkos::deep_copy of overlapping views: ");
    message += dst.label();
    message += "(";
    message += std::to_string((std::ptrdiff_t)dst_start);
    message += ",";
    message += std::to_string((std::ptrdiff_t)dst_end);
    message += ") ";
    message += src.label();
    message += "(";
    message += std::to_string((std::ptrdiff_t)src_start);
    message += ",";
    message += std::to_string((std::ptrdiff_t)src_end);
    message += ") ";
    Kokkos::Impl::throw_runtime_exception(message);
  }

  // Check for same extents
  if ((src.extent(0) != dst.extent(0)) || (src.extent(1) != dst.extent(1)) ||
      (src.extent(2) != dst.extent(2)) || (src.extent(3) != dst.extent(3)) ||
      (src.extent(4) != dst.extent(4)) || (src.extent(5) != dst.extent(5)) ||
      (src.extent(6) != dst.extent(6)) || (src.extent(7) != dst.extent(7))) {
    std::string message(
        "Deprecation Error: Kokkos::deep_copy extents of views don't match: ");
    message += dst.label();
    message += "(";
    for (int r = 0; r < dst_type::Rank - 1; r++) {
      message += std::to_string(dst.extent(r));
      message += ",";
    }
    message += std::to_string(dst.extent(dst_type::Rank - 1));
    message += ") ";
    message += src.label();
    message += "(";
    for (int r = 0; r < src_type::Rank - 1; r++) {
      message += std::to_string(src.extent(r));
      message += ",";
    }
    message += std::to_string(src.extent(src_type::Rank - 1));
    message += ") ";

    Kokkos::Impl::throw_runtime_exception(message);
  }

  // If same type, equal layout, equal dimensions, equal span, and contiguous
  // memory then can byte-wise copy

  if (std::is_same<typename dst_type::value_type,
                   typename src_type::non_const_value_type>::value &&
      (std::is_same<typename dst_type::array_layout,
                    typename src_type::array_layout>::value ||
       (dst_type::rank == 1 && src_type::rank == 1)) &&
      dst.span_is_contiguous() && src.span_is_contiguous() &&
      ((dst_type::rank < 1) || (dst.stride_0() == src.stride_0())) &&
      ((dst_type::rank < 2) || (dst.stride_1() == src.stride_1())) &&
      ((dst_type::rank < 3) || (dst.stride_2() == src.stride_2())) &&
      ((dst_type::rank < 4) || (dst.stride_3() == src.stride_3())) &&
      ((dst_type::rank < 5) || (dst.stride_4() == src.stride_4())) &&
      ((dst_type::rank < 6) || (dst.stride_5() == src.stride_5())) &&
      ((dst_type::rank < 7) || (dst.stride_6() == src.stride_6())) &&
      ((dst_type::rank < 8) || (dst.stride_7() == src.stride_7()))) {
    const size_t nbytes = sizeof(typename dst_type::value_type) * dst.span();
    if ((void*)dst.data() != (void*)src.data()) {
      Kokkos::Impl::DeepCopy<dst_memory_space, src_memory_space, ExecSpace>(
          exec_space, dst.data(), src.data(), nbytes);
    }
  } else {
    // Copying data between views in accessible memory spaces and either
    // non-contiguous or incompatible shape.
    if (ExecCanAccessSrcDst) {
      Impl::view_copy(exec_space, dst, src);
    } else if (DstExecCanAccessSrc || SrcExecCanAccessDst) {
      using cpy_exec_space =
          std::conditional_t<DstExecCanAccessSrc, dst_execution_space,
                             src_execution_space>;
      exec_space.fence(
          "Kokkos::deep_copy: view-to-view noncontiguous copy on space, pre "
          "copy");
      Impl::view_copy(cpy_exec_space(), dst, src);
      cpy_exec_space().fence(
          "Kokkos::deep_copy: view-to-view noncontiguous copy on space, post "
          "copy");
    } else {
      Kokkos::Impl::throw_runtime_exception(
          "deep_copy given views that would require a temporary allocation");
    }
  }
  if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
    Kokkos::Profiling::endDeepCopy();
  }
}

}  // namespace Kokkos

#endif  // KOKKOS_LOCALDEEPCOPY_HPP_
