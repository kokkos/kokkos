/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
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

#ifndef KOKKOS_PROPERTIES_EXECPOLICYCHUNKSIZE_HPP
#define KOKKOS_PROPERTIES_EXECPOLICYCHUNKSIZE_HPP

#include <Kokkos_Core_fwd.hpp>

#include <Kokkos_Concepts.hpp>
#include <Properties/Kokkos_IsApplicableProperty.hpp>

// TODO these should include forward declarations, not implementations
#include <Kokkos_ExecPolicy.hpp>

namespace Kokkos {
namespace Experimental {
namespace ExecutionPolicyProperties {

struct chunk_size_t {
public:

  static constexpr auto is_requirable = true;
  static constexpr auto is_preferable = true;

  using polymorphic_query_result_type = int;

  KOKKOS_INLINE_FUNCTION
  constexpr chunk_size_t() = default;

  KOKKOS_INLINE_FUNCTION
  constexpr explicit
  chunk_size_t(int size) noexcept
    : m_value(size)
  { }

  // can't be statically queried, so we don't need a static_query_v

  KOKKOS_INLINE_FUNCTION
  constexpr
  int value() const noexcept { return m_value; }


  // For chunk_size_t, we'll give a call operator, so that
  // the syntaxes chunk_size_t(10) and chunk_size(10) both work
  KOKKOS_INLINE_FUNCTION
  constexpr chunk_size_t
  operator()(int value) const {
    return chunk_size_t{value};
  }

private:
  // Chunk size of 0 should be interpreted as "auto"
  int m_value = 0;
};

template <class T>
struct is_applicable_property<
  T, chunk_size_t,
  typename std::enable_if<
    Kokkos::is_execution_policy<T>::value
  >::type
> : std::true_type { };

namespace {

constexpr const auto chunk_size = chunk_size_t{};
constexpr const auto auto_chunk_size = chunk_size_t{};

} // end anonymous namespace

// Non-intrusive overloads for known ExecutionPolicies

template <
  class... RangeProperties
>
KOKKOS_INLINE_FUNCTION
Kokkos::RangePolicy<RangeProperties...>
require_property(
  Kokkos::RangePolicy<RangeProperties...> const& policy,
  chunk_size_t const& property
) {
  return policy.set_chunk_size(property.value());
}

template <
  class... RangeProperties
>
KOKKOS_INLINE_FUNCTION
Kokkos::RangePolicy<RangeProperties...>
require_property(
  Kokkos::RangePolicy<RangeProperties...> const& policy,
  chunk_size_t const& property
) {
  return policy.set_chunk_size(property.value());
}

template <
  class... RangeProperties
>
KOKKOS_INLINE_FUNCTION
int
query_property(
  Kokkos::RangePolicy<RangeProperties...> const& policy,
  chunk_size_t const& property
) {
  return policy.chunk_size(property.value());
}

} // end namespace ExecutionPolicyProperties

} // end namespace Experimental
} // end namespace Kokkos

#endif //KOKKOS_PROPERTIES_EXECPOLICYCHUNKSIZE_HPP
