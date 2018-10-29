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

#ifndef KOKKOS_IMPL_RANGE_POLICY_HPP
#define KOKKOS_IMPL_RANGE_POLICY_HPP

#include <Kokkos_Core_fwd.hpp>
#include <impl/Kokkos_Traits.hpp>
#include <impl/Kokkos_Error.hpp>
#include <impl/Kokkos_AnalyzePolicy.hpp>
#include <Kokkos_Concepts.hpp>

//----------------------------------------------------------------------------
//
namespace Kokkos { namespace Impl {

template <typename IndexType>
class WorkRange
{
public:
  using index_type = IndexType;

  constexpr KOKKOS_INLINE_FUNCTION
  index_type begin() const noexcept { return m_begin; }

  constexpr KOKKOS_INLINE_FUNCTION
  index_type end() const noexcept { return m_end; }

  template <typename Policy>
  constexpr KOKKOS_INLINE_FUNCTION
  WorkRange( const Policy & range
           , const int part_rank
           , const int part_size
           ) noexcept
    : m_begin{ offset( range.begin()
                     , range.size()
                     , range.num_chunks()
                     , range.chunk_size()
                     , part_rank
                     , part_size
                     )
             }
    , m_end{ offset( range.begin()
                   , range.size()
                   , range.num_chunks()
                   , range.chunk_size()
                   , part_rank+1
                   , part_size
                   )
           }
  {}

private:


  static constexpr KOKKOS_INLINE_FUNCTION
  index_type chunks_per_part( const index_type num_chunks_
                            , const index_type part_size_
                            ) noexcept
  { return (num_chunks_ + (part_size_-1)) / part_size_; }

  static constexpr KOKKOS_INLINE_FUNCTION
  index_type work_offset( const index_type chunk_size_
                        , const index_type chunks_per_part_
                        , const index_type part_rank_
                        ) noexcept
  { return chunk_size_ * chunks_per_part_ * part_rank_; }

  static constexpr KOKKOS_INLINE_FUNCTION
  index_type clamp_offset( const index_type range_offset_
                         , const index_type work_offset_
                         , const index_type size_
                         ) noexcept
  {
    return range_offset_ + (work_offset_ < size_ ? work_offset_ : size_);
  }

  static constexpr KOKKOS_INLINE_FUNCTION
  index_type offset( const index_type range_offset_
                   , const index_type size_
                   , const index_type num_chunks_
                   , const index_type chunk_size_
                   , const int part_rank_
                   , const int part_size_
                   ) noexcept
  {
    return clamp_offset( range_offset_
                       , work_offset( chunk_size_
                                    , chunks_per_part( num_chunks_
                                                     , part_size_
                                                     )
                                    , part_rank_
                                    )
                       , size_
                       );
  }

  index_type m_begin {0};
  index_type m_end   {0};
};

template <typename PolicyTraits>
class RangePolicy
  : public PolicyTraits
{
  static_assert( is_policy_traits_base<PolicyTraits>::value
               , "Error: PolicyTraits parameter not a PolicyTraitsBase" );

private:
  using traits = PolicyTraits;

public:
  using execution_policy = RangePolicy< traits >;
  using execution_space  = typename traits::execution_space;

  using index_type  = typename traits::index_type;
  using member_type = typename traits::index_type;

  using WorkRange = Impl::WorkRange<index_type>;

  static_assert( std::is_integral<index_type>::value
               , "Error: index_type is not an integral type" );

  constexpr RangePolicy()                    noexcept = default;
  constexpr RangePolicy(const RangePolicy &) noexcept = default;
  constexpr RangePolicy(RangePolicy &&)      noexcept = default;

  RangePolicy( const execution_space& exec_
             , const index_type begin_
             , const index_type end_
             ) noexcept
    : m_exec{exec_}
    , m_offset{begin_}
    , m_size{end_ - begin_}
  {
    auto_chunk_size();
  }

  constexpr RangePolicy( const execution_space& exec_
                       , const index_type begin_
                       , const index_type end_
                       , const ChunkSize chunk_size_
                       ) noexcept
    : m_exec{exec_}
    , m_offset{begin_}
    , m_size{end_ - begin_}
    , m_chunk_size{ static_cast<index_type>(chunk_size_.value) }
  {}

  RangePolicy( const index_type begin_
             , const index_type end_
             ) noexcept
    : m_offset{begin_}
    , m_size{end_ - begin_}
  {
    auto_chunk_size();
  }

  constexpr RangePolicy( const index_type begin_
                       , const index_type end_
                       , const ChunkSize chunk_size_
                       ) noexcept
    : m_offset{begin_}
    , m_size{end_ - begin_}
    , m_chunk_size{ static_cast<index_type>(chunk_size_.value) }
  {}

  RangePolicy& operator=(const RangePolicy &) noexcept = default;
  RangePolicy& operator=(RangePolicy &&)      noexcept = default;

  KOKKOS_INLINE_FUNCTION
  constexpr execution_space& space() const noexcept
  { return m_exec; }

  KOKKOS_INLINE_FUNCTION
  constexpr index_type begin() const noexcept
  { return m_offset; }

  KOKKOS_INLINE_FUNCTION
  constexpr index_type end() const noexcept
  { return m_offset + m_size; }

  KOKKOS_INLINE_FUNCTION
  constexpr index_type size() const noexcept
  { return m_size; }

  KOKKOS_INLINE_FUNCTION
  constexpr index_type num_chunks() const noexcept
  { return (m_size + (m_chunk_size-static_cast<index_type>(1))) / m_chunk_size; }

  KOKKOS_INLINE_FUNCTION
  constexpr index_type chunk_size() const noexcept
  {
    return m_chunk_size;
  }

  RangePolicy & set(const ChunkSize& chunk_size_) noexcept
  {
    m_chunk_size = chunk_size_.value;
    return *this;
  }

  RangePolicy & set_chunk_size(const int chunk_size_) noexcept
  {
    m_chunk_size = chunk_size_;
    return *this;
  }

  RangePolicy & auto_chunk_size() noexcept
  {
    const index_type concurrency =
      static_cast<index_type>(execution_space::concurrency());

    constexpr index_type zero =
      static_cast<index_type>(0);

    constexpr index_type j_scale =
      static_cast<index_type>(40);

    const index_type J = concurrency > zero
                       ? concurrency * j_scale
                       : j_scale
                       ;

    constexpr index_type k_scale =
      static_cast<index_type>(100);

    const index_type K = concurrency > zero
                       ? concurrency * k_scale
                       : k_scale
                       ;

    int shift = 0;

    while (J < (m_size >> shift) && (shift < 7)) { ++shift; }
    while (K < (m_size >> shift)) { ++shift; }

    m_chunk_size = static_cast<index_type>(1) << shift;

    return *this;
  }


private:
  execution_space m_exec        {};
  index_type      m_offset     {0};
  index_type      m_size       {0};
  index_type      m_chunk_size {0};

};

}} // namespace Kokkos::Impl

#endif //KOKKOS_IMPL_RANGE_POLICY_HPP
