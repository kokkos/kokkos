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
// Questions? Contact  H. Carter Edwards (hcedwar@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_HOST_EXP_ITERATE_TILE_HPP
#define KOKKOS_HOST_EXP_ITERATE_TILE_HPP

#include <iostream>
#include <algorithm>
#include <stdio.h>

#include <Kokkos_Macros.hpp>

namespace Kokkos { namespace Experimental { namespace Impl {

template < typename RP
         , typename Functor
         , typename Dim 
         , typename Tag
         , typename ValueType
         >
struct HostIterateTile 
  : protected HostIterateTile<RP, Functor, std::integral_constant<int,Dim::value-1>, Tag, ValueType>
{
  static_assert( Dim::value <= RP::rank, "Error: greater than rank");
  static_assert( Dim::value > 0, "Error: less than 0");

  using base_type = HostIterateTile<RP, Functor, std::integral_constant<int,Dim::value-1>, Tag, ValueType>;
  using final_type = HostIterateTile<RP, Functor, std::integral_constant<int,0>, Tag, ValueType>;

  using index_type = typename RP::index_type;

  template< typename... Args >
  //KOKKOS_INLINE_FUNCTION //compiler warning
  HostIterateTile( Args &&... args )
    : base_type( std::forward<Args>(args)... )
  {}

  void apply()
  {
    if (RP::inner_direction == RP::Left) {
      apply_left();
    } else {
      apply_right();
    }
  }

protected:

  template< typename... Args>
  void apply_left( Args &&... args )
  {
    for (index_type i = final_type::m_begin[Dim::value]; i < final_type::m_end[Dim::value]; ++i) {
      base_type::apply_left(i, std::forward<Args>(args)...);
    }
  }

  template< typename... Args>
  void apply_right( Args &&... args )
  {
    for (index_type i = final_type::m_begin[RP::rank-Dim::value-1]; i < final_type::m_end[RP::rank-Dim::value-1]; ++i) {
      base_type::apply_right(std::forward<Args>(args)...,i);
    }
  }

};

template < typename RP
         , typename Functor
         >
struct HostIterateTile<RP, Functor, std::integral_constant<int,0>, void, void>
{
  using index_type = typename RP::index_type;
  using point_type = typename RP::point_type;

  //KOKKOS_INLINE_FUNCTION
  HostIterateTile( RP const& rp, Functor const& func, index_type tile_idx )
    : m_rp{rp}
    , m_func{func}
  {
    if (RP::outer_direction == RP::Left) {
      for (int i=0; i<RP::rank; ++i) {
        m_begin[i] = (tile_idx % rp.m_tile_end[i]) * rp.m_tile[i] + rp.m_lower[i] ;
        m_end[i] = ((m_begin[i] + rp.m_tile[i]) <= rp.m_upper[i]) ? (m_begin[i] + rp.m_tile[i]) : rp.m_upper[i] ;
        tile_idx /= rp.m_tile_end[i];
      }
    }
    else {
      for (int i=RP::rank-1; i>=0; --i) {
        m_begin[i] = (tile_idx % rp.m_tile_end[i]) * rp.m_tile[i] + rp.m_lower[i] ;
        m_end[i] = ((m_begin[i] + rp.m_tile[i]) <= rp.m_upper[i]) ? (m_begin[i] + rp.m_tile[i]) : rp.m_upper[i] ;
        tile_idx /= rp.m_tile_end[i];
      }
    }
  }

  void apply()
  {
    if (RP::inner_direction == RP::Left) {
      apply_left();
    } else {
      apply_right();
    }
  }

protected:

  template< typename... Args>
  void apply_left( Args &&... args )
  {
    for (index_type i = m_begin[0]; i < m_end[0]; ++i) {
      m_func(i, std::forward<Args>(args)...);
    }
  }

  template< typename... Args>
  void apply_right( Args &&... args )
  {
    for (index_type i = m_begin[RP::rank-1]; i < m_end[RP::rank-1]; ++i) {
      m_func(std::forward<Args>(args)...,i);
    }
  }

  RP        const& m_rp;
  Functor   const& m_func;
  point_type m_begin;
  point_type m_end;
};


template < typename RP
         , typename Functor
         , typename Tag
         >
struct HostIterateTile<RP, Functor, std::integral_constant<int,0>, Tag, void>
{
  using index_type = typename RP::index_type;
  using point_type = typename RP::point_type;

  KOKKOS_INLINE_FUNCTION
  HostIterateTile( RP const& rp, Functor const& func, index_type tile_idx )
    : m_rp{rp}
    , m_func{func}
  {
    if (RP::outer_direction == RP::Left) {
      for (int i=0; i<RP::rank; ++i) {
        m_begin[i] = (tile_idx % rp.m_tile_end[i]) * rp.m_tile[i] + rp.m_lower[i] ;
        m_end[i] = ((m_begin[i] + rp.m_tile[i]) <= rp.m_upper[i]) ? (m_begin[i] + rp.m_tile[i]) : rp.m_upper[i] ;
        tile_idx /= rp.m_tile_end[i];
      }
    }
    else {
      for (int i=RP::rank-1; i>=0; --i) {
        m_begin[i] = (tile_idx % rp.m_tile_end[i]) * rp.m_tile[i] + rp.m_lower[i] ;
        m_end[i] = ((m_begin[i] + rp.m_tile[i]) <= rp.m_upper[i]) ? (m_begin[i] + rp.m_tile[i]) : rp.m_upper[i] ;
        tile_idx /= rp.m_tile_end[i];
      }
    }
  }

  void apply()
  {
    if (RP::inner_direction == RP::Left) {
      apply_left();
    } else {
      apply_right();
    }
  }

protected:

  template< typename... Args>
  void apply_left( Args &&... args )
  {
    for (index_type i = m_begin[0]; i < m_end[0]; ++i) {
      m_func(m_tag, i, std::forward<Args>(args)...);
    }
  }

  template< typename... Args>
  void apply_right( Args &&... args )
  {
    for (index_type i = m_begin[RP::rank-1]; i < m_end[RP::rank-1]; ++i) {
      m_func(m_tag, std::forward<Args>(args)...,i);
    }
  }

  RP         const& m_rp;
  Functor    const& m_func;
  point_type m_begin;
  point_type m_end;
  Tag        m_tag;
};

template < typename RP
         , typename Functor
         , typename ValueType
         >
struct HostIterateTile<RP, Functor, std::integral_constant<int,0>, void, ValueType>
{
  using index_type = typename RP::index_type;
  using point_type = typename RP::point_type;

  using value_type = ValueType;

  KOKKOS_INLINE_FUNCTION
  HostIterateTile( RP const& rp, Functor const& func, index_type tile_idx, value_type & v )
    : m_rp{rp}
    , m_func{func}
    , m_v{v}
  {
    if (RP::outer_direction == RP::Left) {
      for (int i=0; i<RP::rank; ++i) {
        m_begin[i] =  (tile_idx % rp.m_tile_end[i]) * rp.m_tile[i] + rp.m_lower[i] ;
        m_end[i] = ((m_begin[i] + rp.m_tile[i]) <= rp.m_upper[i]) ? (m_begin[i] + rp.m_tile[i]) : rp.m_upper[i] ;
        tile_idx /= rp.m_tile_end[i];
      }
    }
    else {
      for (int i=RP::rank-1; i>=0; --i) {
        m_begin[i] = (tile_idx % rp.m_tile_end[i]) * rp.m_tile[i] + rp.m_lower[i] ;
        m_end[i] = ((m_begin[i] + rp.m_tile[i]) <= rp.m_upper[i]) ? (m_begin[i] + rp.m_tile[i]) : rp.m_upper[i] ;
        tile_idx /= rp.m_tile_end[i];
      }
    }
  }

  void apply()
  {
    if (RP::inner_direction == RP::Left) {
      apply_left();
    } else {
      apply_right();
    }
  }

protected:

  template< typename... Args>
  void apply_left( Args &&... args )
  {
    for (index_type i = m_begin[0]; i < m_end[0]; ++i) {
      m_func(i, std::forward<Args>(args)..., m_v);
    }
  }

  template< typename... Args>
  void apply_right( Args &&... args )
  {
    for (index_type i = m_begin[RP::rank-1]; i < m_end[RP::rank-1]; ++i) {
      m_func(std::forward<Args>(args)...,i, m_v);
    }
  }

  RP        const& m_rp;
  Functor   const& m_func;
  value_type     & m_v;
  point_type m_begin;
  point_type m_end;
};

template < typename RP
         , typename Functor
         , typename Tag
         , typename ValueType
         >
struct HostIterateTile<RP, Functor, std::integral_constant<int,0>, Tag, ValueType>
{
  using index_type = typename RP::index_type;
  using point_type = typename RP::point_type;

  using value_type = ValueType;

  KOKKOS_INLINE_FUNCTION
  HostIterateTile( RP const& rp, Functor const& func, index_type tile_idx, value_type & v )
    : m_rp{rp}
    , m_func{func}
    , m_v{v}
  {
    if (RP::outer_direction == RP::Left) {
      for (int i=0; i<RP::rank; ++i) {
        m_begin[i] = (tile_idx % rp.m_tile_end[i]) * rp.m_tile[i] + rp.m_lower[i] ;
        m_end[i] = ((m_begin[i] + rp.m_tile[i]) <= rp.m_upper[i]) ? (m_begin[i] + rp.m_tile[i]) : rp.m_upper[i] ;
        tile_idx /= rp.m_tile_end[i];
      }
    }
    else {
      for (int i=RP::rank-1; i>=0; --i) {
        m_begin[i] = (tile_idx % rp.m_tile_end[i]) * rp.m_tile[i] + rp.m_lower[i] ;
        m_end[i] = ((m_begin[i] + rp.m_tile[i]) <= rp.m_upper[i]) ? (m_begin[i] + rp.m_tile[i]) : rp.m_upper[i] ;
        tile_idx /= rp.m_tile_end[i];
      }
    }
  }

  void apply()
  {
    if (RP::inner_direction == RP::Left) {
      apply_left();
    } else {
      apply_right();
    }
  }

protected:

  template< typename... Args>
  void apply_left( Args &&... args )
  {
    for (index_type i = m_begin[0]; i < m_end[0]; ++i) {
      m_func(m_tag, i, std::forward<Args>(args)..., m_v);
    }
  }

  template< typename... Args>
  void apply_right( Args &&... args )
  {
    for (index_type i = m_begin[RP::rank-1]; i < m_end[RP::rank-1]; ++i) {
      m_func(m_tag, std::forward<Args>(args)...,i, m_v);
    }
  }

  RP         const& m_rp;
  Functor    const& m_func;
  value_type     & m_v;
  point_type m_begin;
  point_type m_end;
  Tag        m_tag;
};


// ------------------------------------------------------------------ //

// MDFunctor - wraps the range_policy and functor to pass to IterateTile
// Serial, Threads, OpenMP
// Cuda uses DeviceIterateTile directly within md_parallel_for
template < typename MDRange, typename Functor, typename ValueType = void >
struct MDFunctor
{
  using range_policy = MDRange;
  using functor_type = Functor;
  using value_type   = ValueType;
  using work_tag     = typename range_policy::work_tag;
  using index_type   = typename range_policy::index_type;
  using iterate_type = typename Kokkos::Experimental::Impl::HostIterateTile< MDRange
                                                                           , Functor
                                                                           , std::integral_constant<int,MDRange::rank - 1>
                                                                           , work_tag
                                                                           , value_type  
                                                                           >;

  KOKKOS_INLINE_FUNCTION
  MDFunctor( MDRange const& range, Functor const& f, ValueType & v )
    : m_range( range )
    , m_func( f )
    , m_v( v )
  {}


  KOKKOS_INLINE_FUNCTION
  MDFunctor( MDFunctor const& ) = default;

  KOKKOS_INLINE_FUNCTION
  MDFunctor& operator=( MDFunctor const& ) = default;

  KOKKOS_INLINE_FUNCTION
  MDFunctor( MDFunctor && ) = default;

  KOKKOS_INLINE_FUNCTION
  MDFunctor& operator=( MDFunctor && ) = default;

  KOKKOS_FORCEINLINE_FUNCTION
  void operator()(index_type t, value_type & v) const
  {
    iterate_type(m_range, m_func, t, v).apply();
  }
  
  MDRange   m_range;
  Functor   m_func;
  ValueType m_v;
};

template < typename MDRange, typename Functor >
struct MDFunctor< MDRange, Functor, void >
{
  using range_policy = MDRange;
  using functor_type = Functor;
  using work_tag     = typename range_policy::work_tag;
  using index_type   = typename range_policy::index_type;
  using iterate_type = typename Kokkos::Experimental::Impl::HostIterateTile< MDRange
                                                                           , Functor
                                                                           , std::integral_constant<int,MDRange::rank - 1>
                                                                           , work_tag
                                                                           , void  
                                                                           >;

  KOKKOS_INLINE_FUNCTION
  MDFunctor( MDRange const& range, Functor const& f )
    : m_range( range )
    , m_func( f )
  {}


  KOKKOS_INLINE_FUNCTION
  MDFunctor( MDFunctor const& ) = default;

  KOKKOS_INLINE_FUNCTION
  MDFunctor& operator=( MDFunctor const& ) = default;

  KOKKOS_INLINE_FUNCTION
  MDFunctor( MDFunctor && ) = default;

  KOKKOS_INLINE_FUNCTION
  MDFunctor& operator=( MDFunctor && ) = default;

  KOKKOS_FORCEINLINE_FUNCTION
  void operator()(index_type t) const
  {
    iterate_type(m_range, m_func, t).apply();
  }
  
  MDRange m_range;
  Functor m_func;
};

} } } //end namespace Kokkos::Experimental::Impl


#endif
