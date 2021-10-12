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

#ifndef KOKKOS_KOKKOS_VIEWVERIFY_HPP
#define KOKKOS_KOKKOS_VIEWVERIFY_HPP

#include <Kokkos_Macros.hpp>

#include <impl/Kokkos_ViewTracker.hpp>

namespace Kokkos {
namespace Impl {

template <unsigned, class MapType>
KOKKOS_INLINE_FUNCTION bool view_verify_operator_bounds(const MapType&) {
  return true;
}

template <unsigned R, class MapType, class iType, class... Args>
KOKKOS_INLINE_FUNCTION bool view_verify_operator_bounds(const MapType& map,
                                                        const iType& i,
                                                        Args... args) {
  return (size_t(i) < map.extent(R)) &&
         view_verify_operator_bounds<R + 1>(map, args...);
}

template <unsigned, class MapType>
inline void view_error_operator_bounds(char*, int, const MapType&) {}

template <unsigned R, class MapType, class iType, class... Args>
inline void view_error_operator_bounds(char* buf, int len, const MapType& map,
                                       const iType& i, Args... args) {
  const int n = snprintf(
      buf, len, " %ld < %ld %c", static_cast<unsigned long>(i),
      static_cast<unsigned long>(map.extent(R)), (sizeof...(Args) ? ',' : ')'));
  view_error_operator_bounds<R + 1>(buf + n, len - n, map, args...);
}

#if !defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST)

/* Check #3: is the View managed as determined by the MemoryTraits? */
template <class MapType, bool is_managed = (MapType::is_managed != 0)>
struct OperatorBoundsErrorOnDevice;

template <class MapType>
struct OperatorBoundsErrorOnDevice<MapType, false> {
  KOKKOS_INLINE_FUNCTION
  static void run(MapType const&) { Kokkos::abort("View bounds error"); }
};

template <class MapType>
struct OperatorBoundsErrorOnDevice<MapType, true> {
  KOKKOS_INLINE_FUNCTION
  static void run(MapType const& map) {
    SharedAllocationHeader const* const header =
        SharedAllocationHeader::get_header((void*)(map.data()));
    char const* const label = header->label();
    enum { LEN = 128 };
    char msg[LEN];
    char const* const first_part = "View bounds error of view ";
    char* p                      = msg;
    char* const end              = msg + LEN - 1;
    for (char const* p2 = first_part; (*p2 != '\0') && (p < end); ++p, ++p2) {
      *p = *p2;
    }
    for (char const* p2 = label; (*p2 != '\0') && (p < end); ++p, ++p2) {
      *p = *p2;
    }
    *p = '\0';
    Kokkos::abort(msg);
  }
};

/* Check #2: does the ViewMapping have the printable_label_typedef defined?
   See above that only the non-specialized standard-layout ViewMapping has
   this defined by default.
   The existence of this alias indicates the existence of MapType::is_managed
 */
template <class T, class Enable = void>
struct has_printable_label_typedef : public std::false_type {};

template <class T>
struct has_printable_label_typedef<
    T, typename enable_if_type<typename T::printable_label_typedef>::type>
    : public std::true_type {};

template <class MapType>
KOKKOS_INLINE_FUNCTION void operator_bounds_error_on_device(MapType const&,
                                                            std::false_type) {
  Kokkos::abort("View bounds error");
}

template <class MapType>
KOKKOS_INLINE_FUNCTION void operator_bounds_error_on_device(MapType const& map,
                                                            std::true_type) {
  OperatorBoundsErrorOnDevice<MapType>::run(map);
}

#endif  // ! defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )

template <class MemorySpace, class ViewType, class MapType, class... Args>
KOKKOS_INLINE_FUNCTION void view_verify_operator_bounds(
    Kokkos::Impl::ViewTracker<ViewType> const& tracker, const MapType& map,
    Args... args) {
  if (!view_verify_operator_bounds<0>(map, args...)) {
#if defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST)
    enum { LEN = 1024 };
    char buffer[LEN];
    const std::string label =
        tracker.m_tracker.template get_label<MemorySpace>();
    int n =
        snprintf(buffer, LEN, "View bounds error of view %s (", label.c_str());
    view_error_operator_bounds<0>(buffer + n, LEN - n, map, args...);
    Kokkos::Impl::throw_runtime_exception(std::string(buffer));
#else
    /* Check #1: is there a SharedAllocationRecord?
       (we won't use it, but if its not there then there isn't
        a corresponding SharedAllocationHeader containing a label).
       This check should cover the case of Views that don't
       have the Unmanaged trait but were initialized by pointer. */
    if (tracker.m_tracker.has_record()) {
      operator_bounds_error_on_device<MapType>(
          map, has_printable_label_typedef<MapType>());
    } else {
      Kokkos::abort("View bounds error");
    }
#endif
  }
}

//==============================================================================
// <editor-fold desc="ViewVerifySpace"> {{{1

template <class Space,
          bool =
              MemorySpaceAccess<ActiveExecutionMemorySpace, Space>::accessible>
struct ViewVerifySpace {
  KOKKOS_FORCEINLINE_FUNCTION static void check() {}
};

template <class Space>
struct ViewVerifySpace<Space, false> {
  KOKKOS_FORCEINLINE_FUNCTION static void check() {
    Kokkos::abort(
        "Kokkos::View ERROR: attempt to access inaccessible memory space");
  }
};

// </editor-fold> end ViewVerifySpace }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="runtime_check_rank_*"> {{{1

template <typename IntType>
KOKKOS_INLINE_FUNCTION std::size_t count_valid_integers(
    const IntType i0, const IntType i1, const IntType i2, const IntType i3,
    const IntType i4, const IntType i5, const IntType i6, const IntType i7) {
  static_assert(std::is_integral<IntType>::value,
                "count_valid_integers() must have integer arguments.");

  return (i0 != KOKKOS_INVALID_INDEX) + (i1 != KOKKOS_INVALID_INDEX) +
         (i2 != KOKKOS_INVALID_INDEX) + (i3 != KOKKOS_INVALID_INDEX) +
         (i4 != KOKKOS_INVALID_INDEX) + (i5 != KOKKOS_INVALID_INDEX) +
         (i6 != KOKKOS_INVALID_INDEX) + (i7 != KOKKOS_INVALID_INDEX);
}

KOKKOS_INLINE_FUNCTION
void runtime_check_rank_device(const size_t dyn_rank, const bool is_void_spec,
                               const size_t i0, const size_t i1,
                               const size_t i2, const size_t i3,
                               const size_t i4, const size_t i5,
                               const size_t i6, const size_t i7) {
  if (is_void_spec) {
    const size_t num_passed_args =
        count_valid_integers(i0, i1, i2, i3, i4, i5, i6, i7);

    if (num_passed_args != dyn_rank && is_void_spec) {
      Kokkos::abort(
          "Number of arguments passed to Kokkos::View() constructor must match "
          "the dynamic rank of the view.");
    }
  }
}

#ifdef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
KOKKOS_INLINE_FUNCTION
void runtime_check_rank_host(const size_t dyn_rank, const bool is_void_spec,
                             const size_t i0, const size_t i1, const size_t i2,
                             const size_t i3, const size_t i4, const size_t i5,
                             const size_t i6, const size_t i7,
                             const std::string& label) {
  if (is_void_spec) {
    const size_t num_passed_args =
        count_valid_integers(i0, i1, i2, i3, i4, i5, i6, i7);

    if (num_passed_args != dyn_rank) {
      const std::string message =
          "Constructor for Kokkos View '" + label +
          "' has mismatched number of arguments. Number of arguments = " +
          std::to_string(num_passed_args) +
          " but dynamic rank = " + std::to_string(dyn_rank) + " \n";
      Kokkos::abort(message.c_str());
    }
  }
}
#endif

// </editor-fold> end runtime_check_rank_* }}}1
//==============================================================================

}  // end namespace Impl
}  // end namespace Kokkos

#endif  // KOKKOS_KOKKOS_VIEWVERIFY_HPP
