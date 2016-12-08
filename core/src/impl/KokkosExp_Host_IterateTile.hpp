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

#if defined(KOKKOS_OPT_RANGE_AGGRESSIVE_VECTORIZATION) && defined(KOKKOS_HAVE_PRAGMA_IVDEP) && !defined(__CUDA_ARCH__)
#define KOKKOS_MDRANGE_IVDEP
#endif


namespace Kokkos { namespace Experimental { namespace Impl {

#define LOOP_1L(tile) \
  for( int i0=0; i0<static_cast<int>(tile[0]); ++i0)

#define LOOP_2L(tile) \
  for( int i1=0; i1<static_cast<int>(tile[1]); ++i1) \
  LOOP_1L(tile)

#define LOOP_3L(tile) \
  for( int i2=0; i2<static_cast<int>(tile[2]); ++i2) \
  LOOP_2L(tile)

#define LOOP_4L(tile) \
  for( int i3=0; i3<static_cast<int>(tile[3]); ++i3) \
  LOOP_3L(tile)

#define LOOP_5L(tile) \
  for( int i4=0; i4<static_cast<int>(tile[4]); ++i4) \
  LOOP_4L(tile)

#define LOOP_6L(tile) \
  for( int i5=0; i5<static_cast<int>(tile[5]); ++i5) \
  LOOP_5L(tile)

#define LOOP_7L(tile) \
  for( int i6=0; i6<static_cast<int>(tile[6]); ++i6) \
  LOOP_6L(tile)

#define LOOP_8L(tile) \
  for( int i7=0; i7<static_cast<int>(tile[7]); ++i7) \
  LOOP_7L(tile)


#define LOOP_1R(tile) \
  for ( int i0=0; i0<static_cast<int>(tile[0]); ++i0 )

#define LOOP_2R(tile) \
  LOOP_1R(tile) \
  for ( int i1=0; i1<static_cast<int>(tile[1]); ++i1 )

#define LOOP_3R(tile) \
  LOOP_2R(tile) \
  for ( int i2=0; i2<static_cast<int>(tile[2]); ++i2 )

#define LOOP_4R(tile) \
  LOOP_3R(tile) \
  for ( int i3=0; i3<static_cast<int>(tile[3]); ++i3 )

#define LOOP_5R(tile) \
  LOOP_4R(tile) \
  for ( int i4=0; i4<static_cast<int>(tile[4]); ++i4 )

#define LOOP_6R(tile) \
  LOOP_5R(tile) \
  for ( int i5=0; i5<static_cast<int>(tile[5]); ++i5 )

#define LOOP_7R(tile) \
  LOOP_6R(tile) \
  for ( int i6=0; i6<static_cast<int>(tile[6]); ++i6 )

#define LOOP_8R(tile) \
  LOOP_7R(tile) \
  for ( int i7=0; i7<static_cast<int>(tile[7]); ++i7 )


#define LOOP_ARGS_1 i0 + m_offset[0]
#define LOOP_ARGS_2 LOOP_ARGS_1, i1 + m_offset[1]
#define LOOP_ARGS_3 LOOP_ARGS_2, i2 + m_offset[2]
#define LOOP_ARGS_4 LOOP_ARGS_3, i3 + m_offset[3]
#define LOOP_ARGS_5 LOOP_ARGS_4, i4 + m_offset[4]
#define LOOP_ARGS_6 LOOP_ARGS_5, i5 + m_offset[5]
#define LOOP_ARGS_7 LOOP_ARGS_6, i6 + m_offset[6]
#define LOOP_ARGS_8 LOOP_ARGS_7, i7 + m_offset[7]


#define REMOVEOPERATOR 0

template <typename T>
using is_void = std::is_same< T , void >;

template < typename RP
         , typename Functor
         , typename Tag = void
         , typename ValueType = void
         , typename Enable = void
         >
struct HostIterateTile;

template < typename RP
         , typename Functor
         , typename Tag
         , typename ValueType
         >
struct HostIterateTile < RP , Functor , Tag , ValueType , typename std::enable_if< is_void<ValueType >::value >::type >
{
  using index_type = typename RP::index_type;
  using point_type = typename RP::point_type;

  using value_type = ValueType;

  inline
  HostIterateTile( RP const& rp, Functor const& func )
    : m_rp{rp}
    , m_func{func}
  {
  }

  inline
  bool check_iteration_bounds( point_type& partial_tile , point_type& offset ) const {
    bool is_full_tile = true;

      for ( int i = 0; i < RP::rank; ++i ) {
        if ((offset[i] + m_rp.m_tile[i]) <= m_rp.m_upper[i]) {
            partial_tile[i] = m_rp.m_tile[i] ;
        }
        else {
          is_full_tile = false ;
            partial_tile[i] = (m_rp.m_upper[i] - 1 - offset[i]) == 0 ? 1 
                            : (m_rp.m_upper[i] - m_rp.m_tile[i]) > 0 ? (m_rp.m_upper[i] - offset[i]) 
                            : (m_rp.m_upper[i] - m_rp.m_lower[i]) ; // when single tile encloses range
        }
      }

    return is_full_tile ;
  } // end check bounds


  template <int Rank>
  struct RankTag 
  {
    typedef RankTag type;
    enum { value = (int)Rank };
  };

#if REMOVEOPERATOR

  template <typename IType>
  inline
  typename std::enable_if< ( std::is_integral<IType>::value && (RP::rank == 2) ) >::type 
  apply_impl( IType tile_idx ) const
  {
    point_type m_offset;
    point_type m_tiledims;

    if (RP::outer_direction == RP::Left) {
      for (int i=0; i<RP::rank; ++i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ; 
        tile_idx /= m_rp.m_tile_end[i];
      }
    }
    else {
      for (int i=RP::rank-1; i>=0; --i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ;
        tile_idx /= m_rp.m_tile_end[i];
      }
    }

    //Check if offset+tiledim in bounds - if not, replace tile dims with the partial tile dims
    const bool full_tile = check_iteration_bounds(m_tiledims , m_offset) ; 

    if (RP::inner_direction == RP::Left) {
     if ( full_tile ) {
//      #pragma simd
        LOOP_2L(m_tiledims) {
          apply( LOOP_ARGS_2 );
        }
      } else {
//      #pragma simd
        LOOP_2L(m_tiledims) {
          apply( LOOP_ARGS_2 );
        }
      }
    } // end RP::Left
    else {
     if ( full_tile ) {
//      #pragma simd
        LOOP_2R(m_tiledims) {
          apply( LOOP_ARGS_2 );
        }
      } else {
//      #pragma simd
        LOOP_2R(m_tiledims) {
          apply( LOOP_ARGS_2 );
        }
      }
    } // end RP::Right

  } //end op() rank == 2


  template <typename IType>
  inline
  typename std::enable_if< ( std::is_integral<IType>::value && (RP::rank == 3) ) >::type 
  apply_impl( IType tile_idx ) const
  {
    point_type m_offset;
    point_type m_tiledims;

    if (RP::outer_direction == RP::Left) {
      for (int i=0; i<RP::rank; ++i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ; 
        tile_idx /= m_rp.m_tile_end[i];
      }
    }
    else {
      for (int i=RP::rank-1; i>=0; --i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ;
        tile_idx /= m_rp.m_tile_end[i];
      }
    }

    //Check if offset+tiledim in bounds - if not, replace tile dims with the partial tile dims
    const bool full_tile = check_iteration_bounds(m_tiledims , m_offset) ;

    if (RP::inner_direction == RP::Left) {
     if ( full_tile ) {
//      #pragma simd
        LOOP_3L(m_tiledims) {
          apply( LOOP_ARGS_3 );
        }
      } else {
//      #pragma simd
        LOOP_3L(m_tiledims) {
          apply( LOOP_ARGS_3 );
        }
      }
    } // end RP::Left
    else {
     if ( full_tile ) {
//      #pragma simd
        LOOP_3R(m_tiledims) {
          apply( LOOP_ARGS_3 );
        }
      } else {
//      #pragma simd
        LOOP_3R(m_tiledims) {
          apply( LOOP_ARGS_3 );
        }
      }
    } // end RP::Right

  } //end op() rank == 3


  template <typename IType>
  inline
  typename std::enable_if< ( std::is_integral<IType>::value && (RP::rank == 4) ) >::type 
  apply_impl( IType tile_idx ) const
  {
    point_type m_offset;
    point_type m_tiledims;

    if (RP::outer_direction == RP::Left) {
      for (int i=0; i<RP::rank; ++i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ; 
        tile_idx /= m_rp.m_tile_end[i];
      }
    }
    else {
      for (int i=RP::rank-1; i>=0; --i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ;
        tile_idx /= m_rp.m_tile_end[i];
      }
    }

    //Check if offset+tiledim in bounds - if not, replace tile dims with the partial tile dims
    const bool full_tile = check_iteration_bounds(m_tiledims , m_offset) ;

    if (RP::inner_direction == RP::Left) {
     if ( full_tile ) {
//      #pragma simd
        LOOP_4L(m_tiledims) {
          apply( LOOP_ARGS_4 );
        }
      } else {
//      #pragma simd
        LOOP_4L(m_tiledims) {
          apply( LOOP_ARGS_4 );
        }
      }
    } // end RP::Left
    else {
     if ( full_tile ) {
//      #pragma simd
        LOOP_4R(m_tiledims) {
          apply( LOOP_ARGS_4 );
        }
      } else {
//      #pragma simd
        LOOP_4R(m_tiledims) {
          apply( LOOP_ARGS_4 );
        }
      }
    } // end RP::Right

  } //end op() rank == 4


  template <typename IType>
  inline
  typename std::enable_if< ( std::is_integral<IType>::value && (RP::rank == 5) ) >::type 
  apply_impl( IType tile_idx ) const
  {
    point_type m_offset;
    point_type m_tiledims;

    if (RP::outer_direction == RP::Left) {
      for (int i=0; i<RP::rank; ++i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ; 
        tile_idx /= m_rp.m_tile_end[i];
      }
    }
    else {
      for (int i=RP::rank-1; i>=0; --i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ;
        tile_idx /= m_rp.m_tile_end[i];
      }
    }

    //Check if offset+tiledim in bounds - if not, replace tile dims with the partial tile dims
    const bool full_tile = check_iteration_bounds(m_tiledims , m_offset) ;

    if (RP::inner_direction == RP::Left) {
     if ( full_tile ) {
//      #pragma simd
        LOOP_5L(m_tiledims) {
          apply( LOOP_ARGS_5 );
        }
      } else {
//      #pragma simd
        LOOP_5L(m_tiledims) {
          apply( LOOP_ARGS_5 );
        }
      }
    } // end RP::Left
    else {
     if ( full_tile ) {
//      #pragma simd
        LOOP_5R(m_tiledims) {
          apply( LOOP_ARGS_5 );
        }
      } else {
//      #pragma simd
        LOOP_5R(m_tiledims) {
          apply( LOOP_ARGS_5 );
        }
      }
    } // end RP::Right

  } //end op() rank == 5


  template <typename IType>
  inline
  typename std::enable_if< ( std::is_integral<IType>::value && (RP::rank == 6) ) >::type 
  apply_impl( IType tile_idx ) const
  {
    point_type m_offset;
    point_type m_tiledims;

    if (RP::outer_direction == RP::Left) {
      for (int i=0; i<RP::rank; ++i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ; 
        tile_idx /= m_rp.m_tile_end[i];
      }
    }
    else {
      for (int i=RP::rank-1; i>=0; --i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ;
        tile_idx /= m_rp.m_tile_end[i];
      }
    }

    //Check if offset+tiledim in bounds - if not, replace tile dims with the partial tile dims
    const bool full_tile = check_iteration_bounds(m_tiledims , m_offset) ;

    if (RP::inner_direction == RP::Left) {
     if ( full_tile ) {
//      #pragma simd
        LOOP_6L(m_tiledims) {
          apply( LOOP_ARGS_6 );
        }
      } else {
//      #pragma simd
        LOOP_6L(m_tiledims) {
          apply( LOOP_ARGS_6 );
        }
      }
    } // end RP::Left
    else {
     if ( full_tile ) {
//      #pragma simd
        LOOP_6R(m_tiledims) {
          apply( LOOP_ARGS_6 );
        }
      } else {
//      #pragma simd
        LOOP_6R(m_tiledims) {
          apply( LOOP_ARGS_6 );
        }
      }
    } // end RP::Right

  } //end op() rank == 6


  template <typename IType>
  inline
  typename std::enable_if< ( std::is_integral<IType>::value && (RP::rank == 7) ) >::type 
  apply_impl( IType tile_idx ) const
  {
    point_type m_offset;
    point_type m_tiledims;

    if (RP::outer_direction == RP::Left) {
      for (int i=0; i<RP::rank; ++i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ; 
        tile_idx /= m_rp.m_tile_end[i];
      }
    }
    else {
      for (int i=RP::rank-1; i>=0; --i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ;
        tile_idx /= m_rp.m_tile_end[i];
      }
    }

    //Check if offset+tiledim in bounds - if not, replace tile dims with the partial tile dims
    const bool full_tile = check_iteration_bounds(m_tiledims , m_offset) ;

    if (RP::inner_direction == RP::Left) {
     if ( full_tile ) {
//      #pragma simd
        LOOP_7L(m_tiledims) {
          apply( LOOP_ARGS_7 );
        }
      } else {
//      #pragma simd
        LOOP_7L(m_tiledims) {
          apply( LOOP_ARGS_7 );
        }
      }
    } // end RP::Left
    else {
     if ( full_tile ) {
//      #pragma simd
        LOOP_7R(m_tiledims) {
          apply( LOOP_ARGS_7 );
        }
      } else {
//      #pragma simd
        LOOP_7R(m_tiledims) {
          apply( LOOP_ARGS_7 );
        }
      }
    } // end RP::Right

  } //end op() rank == 7


  template <typename IType>
  inline
  typename std::enable_if< ( std::is_integral<IType>::value && (RP::rank == 8) ) >::type 
  apply_impl( IType tile_idx ) const
  {
    point_type m_offset;
    point_type m_tiledims;

    if (RP::outer_direction == RP::Left) {
      for (int i=0; i<RP::rank; ++i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ; 
        tile_idx /= m_rp.m_tile_end[i];
      }
    }
    else {
      for (int i=RP::rank-1; i>=0; --i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ;
        tile_idx /= m_rp.m_tile_end[i];
      }
    }

    //Check if offset+tiledim in bounds - if not, replace tile dims with the partial tile dims
    const bool full_tile = check_iteration_bounds(m_tiledims , m_offset) ;

    if (RP::inner_direction == RP::Left) {
     if ( full_tile ) {
//      #pragma simd
        LOOP_8L(m_tiledims) {
          apply( LOOP_ARGS_8 );
        }
      } else {
//      #pragma simd
        LOOP_8L(m_tiledims) {
          apply( LOOP_ARGS_8 );
        }
      }
    } // end RP::Left
    else {
     if ( full_tile ) {
//      #pragma simd
        LOOP_8R(m_tiledims) {
          apply( LOOP_ARGS_8 );
        }
      } else {
//      #pragma simd
        LOOP_8R(m_tiledims) {
          apply( LOOP_ARGS_8 );
        }
      }
    } // end RP::Right

  } //end op() rank == 8


#else
  template <typename IType>
  inline
  void
  operator()(IType tile_idx) const
  { operator_impl( tile_idx , RankTag<RP::rank>() ); }
  // added due to compiler error when using sfinae to choose operator based on rank w/ cuda+serial

  template <typename IType>
  inline
  void operator_impl( IType tile_idx , const RankTag<2> ) const
  {
    point_type m_offset;
    point_type m_tiledims;

    if (RP::outer_direction == RP::Left) {
      for (int i=0; i<RP::rank; ++i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ; 
        tile_idx /= m_rp.m_tile_end[i];
      }
    }
    else {
      for (int i=RP::rank-1; i>=0; --i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ;
        tile_idx /= m_rp.m_tile_end[i];
      }
    }

    //Check if offset+tiledim in bounds - if not, replace tile dims with the partial tile dims
    const bool full_tile = check_iteration_bounds(m_tiledims , m_offset) ; 

    if (RP::inner_direction == RP::Left) {
     if ( full_tile ) {
//      #pragma simd
        LOOP_2L(m_tiledims) {
          apply( LOOP_ARGS_2 );
        }
      } else {
//      #pragma simd
        LOOP_2L(m_tiledims) {
          apply( LOOP_ARGS_2 );
        }
      }
    } // end RP::Left
    else {
     if ( full_tile ) {
//      #pragma simd
        LOOP_2R(m_tiledims) {
          apply( LOOP_ARGS_2 );
        }
      } else {
//      #pragma simd
        LOOP_2R(m_tiledims) {
          apply( LOOP_ARGS_2 );
        }
      }
    } // end RP::Right

  } //end op() rank == 2


  template <typename IType>
  inline
  void operator_impl( IType tile_idx , const RankTag<3> ) const
  {
    point_type m_offset;
    point_type m_tiledims;

    if (RP::outer_direction == RP::Left) {
      for (int i=0; i<RP::rank; ++i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ; 
        tile_idx /= m_rp.m_tile_end[i];
      }
    }
    else {
      for (int i=RP::rank-1; i>=0; --i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ;
        tile_idx /= m_rp.m_tile_end[i];
      }
    }

    //Check if offset+tiledim in bounds - if not, replace tile dims with the partial tile dims
    const bool full_tile = check_iteration_bounds(m_tiledims , m_offset) ;

    if (RP::inner_direction == RP::Left) {
     if ( full_tile ) {
//      #pragma simd
        LOOP_3L(m_tiledims) {
          apply( LOOP_ARGS_3 );
        }
      } else {
//      #pragma simd
        LOOP_3L(m_tiledims) {
          apply( LOOP_ARGS_3 );
        }
      }
    } // end RP::Left
    else {
     if ( full_tile ) {
//      #pragma simd
        LOOP_3R(m_tiledims) {
          apply( LOOP_ARGS_3 );
        }
      } else {
//      #pragma simd
        LOOP_3R(m_tiledims) {
          apply( LOOP_ARGS_3 );
        }
      }
    } // end RP::Right

  } //end op() rank == 3


  template <typename IType>
  inline
  void operator_impl( IType tile_idx , const RankTag<4> ) const
  {
    point_type m_offset;
    point_type m_tiledims;

    if (RP::outer_direction == RP::Left) {
      for (int i=0; i<RP::rank; ++i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ; 
        tile_idx /= m_rp.m_tile_end[i];
      }
    }
    else {
      for (int i=RP::rank-1; i>=0; --i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ;
        tile_idx /= m_rp.m_tile_end[i];
      }
    }

    //Check if offset+tiledim in bounds - if not, replace tile dims with the partial tile dims
    const bool full_tile = check_iteration_bounds(m_tiledims , m_offset) ;

    if (RP::inner_direction == RP::Left) {
     if ( full_tile ) {
//      #pragma simd
        LOOP_4L(m_tiledims) {
          apply( LOOP_ARGS_4 );
        }
      } else {
//      #pragma simd
        LOOP_4L(m_tiledims) {
          apply( LOOP_ARGS_4 );
        }
      }
    } // end RP::Left
    else {
     if ( full_tile ) {
//      #pragma simd
        LOOP_4R(m_tiledims) {
          apply( LOOP_ARGS_4 );
        }
      } else {
//      #pragma simd
        LOOP_4R(m_tiledims) {
          apply( LOOP_ARGS_4 );
        }
      }
    } // end RP::Right

  } //end op() rank == 4


  template <typename IType>
  inline
  void operator_impl( IType tile_idx , const RankTag<5> ) const
  {
    point_type m_offset;
    point_type m_tiledims;

    if (RP::outer_direction == RP::Left) {
      for (int i=0; i<RP::rank; ++i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ; 
        tile_idx /= m_rp.m_tile_end[i];
      }
    }
    else {
      for (int i=RP::rank-1; i>=0; --i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ;
        tile_idx /= m_rp.m_tile_end[i];
      }
    }

    //Check if offset+tiledim in bounds - if not, replace tile dims with the partial tile dims
    const bool full_tile = check_iteration_bounds(m_tiledims , m_offset) ;

    if (RP::inner_direction == RP::Left) {
     if ( full_tile ) {
//      #pragma simd
        LOOP_5L(m_tiledims) {
          apply( LOOP_ARGS_5 );
        }
      } else {
//      #pragma simd
        LOOP_5L(m_tiledims) {
          apply( LOOP_ARGS_5 );
        }
      }
    } // end RP::Left
    else {
     if ( full_tile ) {
//      #pragma simd
        LOOP_5R(m_tiledims) {
          apply( LOOP_ARGS_5 );
        }
      } else {
//      #pragma simd
        LOOP_5R(m_tiledims) {
          apply( LOOP_ARGS_5 );
        }
      }
    } // end RP::Right

  } //end op() rank == 5


  template <typename IType>
  inline
  void operator_impl( IType tile_idx , const RankTag<6> ) const
  {
    point_type m_offset;
    point_type m_tiledims;

    if (RP::outer_direction == RP::Left) {
      for (int i=0; i<RP::rank; ++i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ; 
        tile_idx /= m_rp.m_tile_end[i];
      }
    }
    else {
      for (int i=RP::rank-1; i>=0; --i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ;
        tile_idx /= m_rp.m_tile_end[i];
      }
    }

    //Check if offset+tiledim in bounds - if not, replace tile dims with the partial tile dims
    const bool full_tile = check_iteration_bounds(m_tiledims , m_offset) ;

    if (RP::inner_direction == RP::Left) {
     if ( full_tile ) {
//      #pragma simd
        LOOP_6L(m_tiledims) {
          apply( LOOP_ARGS_6 );
        }
      } else {
//      #pragma simd
        LOOP_6L(m_tiledims) {
          apply( LOOP_ARGS_6 );
        }
      }
    } // end RP::Left
    else {
     if ( full_tile ) {
//      #pragma simd
        LOOP_6R(m_tiledims) {
          apply( LOOP_ARGS_6 );
        }
      } else {
//      #pragma simd
        LOOP_6R(m_tiledims) {
          apply( LOOP_ARGS_6 );
        }
      }
    } // end RP::Right

  } //end op() rank == 6


  template <typename IType>
  inline
  void operator_impl( IType tile_idx , const RankTag<7> ) const
  {
    point_type m_offset;
    point_type m_tiledims;

    if (RP::outer_direction == RP::Left) {
      for (int i=0; i<RP::rank; ++i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ; 
        tile_idx /= m_rp.m_tile_end[i];
      }
    }
    else {
      for (int i=RP::rank-1; i>=0; --i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ;
        tile_idx /= m_rp.m_tile_end[i];
      }
    }

    //Check if offset+tiledim in bounds - if not, replace tile dims with the partial tile dims
    const bool full_tile = check_iteration_bounds(m_tiledims , m_offset) ;

    if (RP::inner_direction == RP::Left) {
     if ( full_tile ) {
//      #pragma simd
        LOOP_7L(m_tiledims) {
          apply( LOOP_ARGS_7 );
        }
      } else {
//      #pragma simd
        LOOP_7L(m_tiledims) {
          apply( LOOP_ARGS_7 );
        }
      }
    } // end RP::Left
    else {
     if ( full_tile ) {
//      #pragma simd
        LOOP_7R(m_tiledims) {
          apply( LOOP_ARGS_7 );
        }
      } else {
//      #pragma simd
        LOOP_7R(m_tiledims) {
          apply( LOOP_ARGS_7 );
        }
      }
    } // end RP::Right

  } //end op() rank == 7


  template <typename IType>
  inline
  void operator_impl( IType tile_idx , const RankTag<8> ) const
  {
    point_type m_offset;
    point_type m_tiledims;

    if (RP::outer_direction == RP::Left) {
      for (int i=0; i<RP::rank; ++i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ; 
        tile_idx /= m_rp.m_tile_end[i];
      }
    }
    else {
      for (int i=RP::rank-1; i>=0; --i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ;
        tile_idx /= m_rp.m_tile_end[i];
      }
    }

    //Check if offset+tiledim in bounds - if not, replace tile dims with the partial tile dims
    const bool full_tile = check_iteration_bounds(m_tiledims , m_offset) ;

    if (RP::inner_direction == RP::Left) {
     if ( full_tile ) {
//      #pragma simd
        LOOP_8L(m_tiledims) {
          apply( LOOP_ARGS_8 );
        }
      } else {
//      #pragma simd
        LOOP_8L(m_tiledims) {
          apply( LOOP_ARGS_8 );
        }
      }
    } // end RP::Left
    else {
     if ( full_tile ) {
//      #pragma simd
        LOOP_8R(m_tiledims) {
          apply( LOOP_ARGS_8 );
        }
      } else {
//      #pragma simd
        LOOP_8R(m_tiledims) {
          apply( LOOP_ARGS_8 );
        }
      }
    } // end RP::Right

  } //end op() rank == 8
#endif

    template <typename... Args>
    typename std::enable_if<( sizeof...(Args) == RP::rank && std::is_same<Tag,void>::value), void>::type
    apply(Args &&... args) const
    {
      m_func(args...);
    }

    template <typename... Args>
    typename std::enable_if<( sizeof...(Args) == RP::rank && !std::is_same<Tag,void>::value), void>::type
    apply(Args &&... args) const
    {
      m_func( m_tag, args...);
    }


  RP         const& m_rp;
  Functor    const& m_func;
  typename std::conditional< std::is_same<Tag,void>::value,int,Tag>::type m_tag{};
//  value_type  & m_v;

};


// ValueType: For reductions
template < typename RP
         , typename Functor
         , typename Tag
         , typename ValueType
         >
struct HostIterateTile < RP , Functor , Tag , ValueType , typename std::enable_if< !is_void<ValueType >::value >::type >
{
  using index_type = typename RP::index_type;
  using point_type = typename RP::point_type;

  using value_type = ValueType;

  inline
  HostIterateTile( RP const& rp, Functor const& func, value_type & v )
    : m_rp{rp}
    , m_func{func}
    , m_v{v} // use with non-void ValueType struct
  {
  }

  inline
  bool check_iteration_bounds( point_type& partial_tile , point_type& offset ) const {
    bool is_full_tile = true;

      for ( int i = 0; i < RP::rank; ++i ) {
        if ((offset[i] + m_rp.m_tile[i]) <= m_rp.m_upper[i]) {
            partial_tile[i] = m_rp.m_tile[i] ;
        }
        else {
          is_full_tile = false ;
            partial_tile[i] = (m_rp.m_upper[i] - 1 - offset[i]) == 0 ? 1 
                            : (m_rp.m_upper[i] - m_rp.m_tile[i]) > 0 ? (m_rp.m_upper[i] - offset[i]) 
                            : (m_rp.m_upper[i] - m_rp.m_lower[i]) ; // when single tile encloses range
        }
      }

    return is_full_tile ;
  } // end check bounds


  template <int Rank>
  struct RankTag 
  {
    typedef RankTag type;
    enum { value = (int)Rank };
  };


#if REMOVEOPERATOR
  template <typename IType>
  inline
  typename std::enable_if< ( std::is_integral<IType>::value && (RP::rank == 2) ) >::type 
  apply_impl( IType tile_idx ) const
  {
    point_type m_offset;
    point_type m_tiledims;

    if (RP::outer_direction == RP::Left) {
      for (int i=0; i<RP::rank; ++i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ; 
        tile_idx /= m_rp.m_tile_end[i];
      }
    }
    else {
      for (int i=RP::rank-1; i>=0; --i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ;
        tile_idx /= m_rp.m_tile_end[i];
      }
    }

    //Check if offset+tiledim in bounds - if not, replace tile dims with the partial tile dims
    const bool full_tile = check_iteration_bounds(m_tiledims , m_offset) ; 

    if (RP::inner_direction == RP::Left) {
     if ( full_tile ) {
//      #pragma simd
        LOOP_2L(m_tiledims) {
          apply( LOOP_ARGS_2 );
        }
      } else {
//      #pragma simd
        LOOP_2L(m_tiledims) {
          apply( LOOP_ARGS_2 );
        }
      }
    } // end RP::Left
    else {
     if ( full_tile ) {
//      #pragma simd
        LOOP_2R(m_tiledims) {
          apply( LOOP_ARGS_2 );
        }
      } else {
//      #pragma simd
        LOOP_2R(m_tiledims) {
          apply( LOOP_ARGS_2 );
        }
      }
    } // end RP::Right

  } //end op() rank == 2


  template <typename IType>
  inline
  typename std::enable_if< ( std::is_integral<IType>::value && (RP::rank == 3) ) >::type 
  apply_impl( IType tile_idx ) const
  {
    point_type m_offset;
    point_type m_tiledims;

    if (RP::outer_direction == RP::Left) {
      for (int i=0; i<RP::rank; ++i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ; 
        tile_idx /= m_rp.m_tile_end[i];
      }
    }
    else {
      for (int i=RP::rank-1; i>=0; --i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ;
        tile_idx /= m_rp.m_tile_end[i];
      }
    }

    //Check if offset+tiledim in bounds - if not, replace tile dims with the partial tile dims
    const bool full_tile = check_iteration_bounds(m_tiledims , m_offset) ;

    if (RP::inner_direction == RP::Left) {
     if ( full_tile ) {
//      #pragma simd
        LOOP_3L(m_tiledims) {
          apply( LOOP_ARGS_3 );
        }
      } else {
//      #pragma simd
        LOOP_3L(m_tiledims) {
          apply( LOOP_ARGS_3 );
        }
      }
    } // end RP::Left
    else {
     if ( full_tile ) {
//      #pragma simd
        LOOP_3R(m_tiledims) {
          apply( LOOP_ARGS_3 );
        }
      } else {
//      #pragma simd
        LOOP_3R(m_tiledims) {
          apply( LOOP_ARGS_3 );
        }
      }
    } // end RP::Right

  } //end op() rank == 3


  template <typename IType>
  inline
  typename std::enable_if< ( std::is_integral<IType>::value && (RP::rank == 4) ) >::type 
  apply_impl( IType tile_idx ) const
  {
    point_type m_offset;
    point_type m_tiledims;

    if (RP::outer_direction == RP::Left) {
      for (int i=0; i<RP::rank; ++i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ; 
        tile_idx /= m_rp.m_tile_end[i];
      }
    }
    else {
      for (int i=RP::rank-1; i>=0; --i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ;
        tile_idx /= m_rp.m_tile_end[i];
      }
    }

    //Check if offset+tiledim in bounds - if not, replace tile dims with the partial tile dims
    const bool full_tile = check_iteration_bounds(m_tiledims , m_offset) ;

    if (RP::inner_direction == RP::Left) {
     if ( full_tile ) {
//      #pragma simd
        LOOP_4L(m_tiledims) {
          apply( LOOP_ARGS_4 );
        }
      } else {
//      #pragma simd
        LOOP_4L(m_tiledims) {
          apply( LOOP_ARGS_4 );
        }
      }
    } // end RP::Left
    else {
     if ( full_tile ) {
//      #pragma simd
        LOOP_4R(m_tiledims) {
          apply( LOOP_ARGS_4 );
        }
      } else {
//      #pragma simd
        LOOP_4R(m_tiledims) {
          apply( LOOP_ARGS_4 );
        }
      }
    } // end RP::Right

  } //end op() rank == 4


  template <typename IType>
  inline
  typename std::enable_if< ( std::is_integral<IType>::value && (RP::rank == 5) ) >::type 
  apply_impl( IType tile_idx ) const
  {
    point_type m_offset;
    point_type m_tiledims;

    if (RP::outer_direction == RP::Left) {
      for (int i=0; i<RP::rank; ++i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ; 
        tile_idx /= m_rp.m_tile_end[i];
      }
    }
    else {
      for (int i=RP::rank-1; i>=0; --i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ;
        tile_idx /= m_rp.m_tile_end[i];
      }
    }

    //Check if offset+tiledim in bounds - if not, replace tile dims with the partial tile dims
    const bool full_tile = check_iteration_bounds(m_tiledims , m_offset) ;

    if (RP::inner_direction == RP::Left) {
     if ( full_tile ) {
//      #pragma simd
        LOOP_5L(m_tiledims) {
          apply( LOOP_ARGS_5 );
        }
      } else {
//      #pragma simd
        LOOP_5L(m_tiledims) {
          apply( LOOP_ARGS_5 );
        }
      }
    } // end RP::Left
    else {
     if ( full_tile ) {
//      #pragma simd
        LOOP_5R(m_tiledims) {
          apply( LOOP_ARGS_5 );
        }
      } else {
//      #pragma simd
        LOOP_5R(m_tiledims) {
          apply( LOOP_ARGS_5 );
        }
      }
    } // end RP::Right

  } //end op() rank == 5


  template <typename IType>
  inline
  typename std::enable_if< ( std::is_integral<IType>::value && (RP::rank == 6) ) >::type 
  apply_impl( IType tile_idx ) const
  {
    point_type m_offset;
    point_type m_tiledims;

    if (RP::outer_direction == RP::Left) {
      for (int i=0; i<RP::rank; ++i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ; 
        tile_idx /= m_rp.m_tile_end[i];
      }
    }
    else {
      for (int i=RP::rank-1; i>=0; --i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ;
        tile_idx /= m_rp.m_tile_end[i];
      }
    }

    //Check if offset+tiledim in bounds - if not, replace tile dims with the partial tile dims
    const bool full_tile = check_iteration_bounds(m_tiledims , m_offset) ;

    if (RP::inner_direction == RP::Left) {
     if ( full_tile ) {
//      #pragma simd
        LOOP_6L(m_tiledims) {
          apply( LOOP_ARGS_6 );
        }
      } else {
//      #pragma simd
        LOOP_6L(m_tiledims) {
          apply( LOOP_ARGS_6 );
        }
      }
    } // end RP::Left
    else {
     if ( full_tile ) {
//      #pragma simd
        LOOP_6R(m_tiledims) {
          apply( LOOP_ARGS_6 );
        }
      } else {
//      #pragma simd
        LOOP_6R(m_tiledims) {
          apply( LOOP_ARGS_6 );
        }
      }
    } // end RP::Right

  } //end op() rank == 6


  template <typename IType>
  inline
  typename std::enable_if< ( std::is_integral<IType>::value && (RP::rank == 7) ) >::type 
  apply_impl( IType tile_idx ) const
  {
    point_type m_offset;
    point_type m_tiledims;

    if (RP::outer_direction == RP::Left) {
      for (int i=0; i<RP::rank; ++i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ; 
        tile_idx /= m_rp.m_tile_end[i];
      }
    }
    else {
      for (int i=RP::rank-1; i>=0; --i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ;
        tile_idx /= m_rp.m_tile_end[i];
      }
    }

    //Check if offset+tiledim in bounds - if not, replace tile dims with the partial tile dims
    const bool full_tile = check_iteration_bounds(m_tiledims , m_offset) ;

    if (RP::inner_direction == RP::Left) {
     if ( full_tile ) {
//      #pragma simd
        LOOP_7L(m_tiledims) {
          apply( LOOP_ARGS_7 );
        }
      } else {
//      #pragma simd
        LOOP_7L(m_tiledims) {
          apply( LOOP_ARGS_7 );
        }
      }
    } // end RP::Left
    else {
     if ( full_tile ) {
//      #pragma simd
        LOOP_7R(m_tiledims) {
          apply( LOOP_ARGS_7 );
        }
      } else {
//      #pragma simd
        LOOP_7R(m_tiledims) {
          apply( LOOP_ARGS_7 );
        }
      }
    } // end RP::Right

  } //end op() rank == 7


  template <typename IType>
  inline
  typename std::enable_if< ( std::is_integral<IType>::value && (RP::rank == 8) ) >::type 
  apply_impl( IType tile_idx ) const
  {
    point_type m_offset;
    point_type m_tiledims;

    if (RP::outer_direction == RP::Left) {
      for (int i=0; i<RP::rank; ++i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ; 
        tile_idx /= m_rp.m_tile_end[i];
      }
    }
    else {
      for (int i=RP::rank-1; i>=0; --i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ;
        tile_idx /= m_rp.m_tile_end[i];
      }
    }

    //Check if offset+tiledim in bounds - if not, replace tile dims with the partial tile dims
    const bool full_tile = check_iteration_bounds(m_tiledims , m_offset) ;

    if (RP::inner_direction == RP::Left) {
     if ( full_tile ) {
//      #pragma simd
        LOOP_8L(m_tiledims) {
          apply( LOOP_ARGS_8 );
        }
      } else {
//      #pragma simd
        LOOP_8L(m_tiledims) {
          apply( LOOP_ARGS_8 );
        }
      }
    } // end RP::Left
    else {
     if ( full_tile ) {
//      #pragma simd
        LOOP_8R(m_tiledims) {
          apply( LOOP_ARGS_8 );
        }
      } else {
//      #pragma simd
        LOOP_8R(m_tiledims) {
          apply( LOOP_ARGS_8 );
        }
      }
    } // end RP::Right

  } //end op() rank == 8


#else
  template <typename IType>
  inline
  void
  operator()(IType tile_idx) const
  { operator_impl( tile_idx , RankTag<RP::rank>() ); }
  // added due to compiler error when using sfinae to choose operator based on rank


  template <typename IType>
  inline
  void operator_impl( IType tile_idx , const RankTag<2> ) const
  {
    point_type m_offset;
    point_type m_tiledims;

    if (RP::outer_direction == RP::Left) {
      for (int i=0; i<RP::rank; ++i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ; 
        tile_idx /= m_rp.m_tile_end[i];
      }
    }
    else {
      for (int i=RP::rank-1; i>=0; --i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ;
        tile_idx /= m_rp.m_tile_end[i];
      }
    }

    //Check if offset+tiledim in bounds - if not, replace tile dims with the partial tile dims
    const bool full_tile = check_iteration_bounds(m_tiledims , m_offset) ; 

    if (RP::inner_direction == RP::Left) {
     if ( full_tile ) {
//      #pragma simd
        LOOP_2L(m_tiledims) {
          apply( LOOP_ARGS_2 );
        }
      } else {
//      #pragma simd
        LOOP_2L(m_tiledims) {
          apply( LOOP_ARGS_2 );
        }
      }
    } // end RP::Left
    else {
     if ( full_tile ) {
//      #pragma simd
        LOOP_2R(m_tiledims) {
          apply( LOOP_ARGS_2 );
        }
      } else {
//      #pragma simd
        LOOP_2R(m_tiledims) {
          apply( LOOP_ARGS_2 );
        }
      }
    } // end RP::Right

  } //end op() rank == 2


  template <typename IType>
  inline
  void operator_impl( IType tile_idx , const RankTag<3> ) const
  {
    point_type m_offset;
    point_type m_tiledims;

    if (RP::outer_direction == RP::Left) {
      for (int i=0; i<RP::rank; ++i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ; 
        tile_idx /= m_rp.m_tile_end[i];
      }
    }
    else {
      for (int i=RP::rank-1; i>=0; --i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ;
        tile_idx /= m_rp.m_tile_end[i];
      }
    }

    //Check if offset+tiledim in bounds - if not, replace tile dims with the partial tile dims
    const bool full_tile = check_iteration_bounds(m_tiledims , m_offset) ;

    if (RP::inner_direction == RP::Left) {
     if ( full_tile ) {
//      #pragma simd
        LOOP_3L(m_tiledims) {
          apply( LOOP_ARGS_3 );
        }
      } else {
//      #pragma simd
        LOOP_3L(m_tiledims) {
          apply( LOOP_ARGS_3 );
        }
      }
    } // end RP::Left
    else {
     if ( full_tile ) {
//      #pragma simd
        LOOP_3R(m_tiledims) {
          apply( LOOP_ARGS_3 );
        }
      } else {
//      #pragma simd
        LOOP_3R(m_tiledims) {
          apply( LOOP_ARGS_3 );
        }
      }
    } // end RP::Right

  } //end op() rank == 3


  template <typename IType>
  inline
  void operator_impl( IType tile_idx , const RankTag<4> ) const
  {
    point_type m_offset;
    point_type m_tiledims;

    if (RP::outer_direction == RP::Left) {
      for (int i=0; i<RP::rank; ++i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ; 
        tile_idx /= m_rp.m_tile_end[i];
      }
    }
    else {
      for (int i=RP::rank-1; i>=0; --i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ;
        tile_idx /= m_rp.m_tile_end[i];
      }
    }

    //Check if offset+tiledim in bounds - if not, replace tile dims with the partial tile dims
    const bool full_tile = check_iteration_bounds(m_tiledims , m_offset) ;

    if (RP::inner_direction == RP::Left) {
     if ( full_tile ) {
//      #pragma simd
        LOOP_4L(m_tiledims) {
          apply( LOOP_ARGS_4 );
        }
      } else {
//      #pragma simd
        LOOP_4L(m_tiledims) {
          apply( LOOP_ARGS_4 );
        }
      }
    } // end RP::Left
    else {
     if ( full_tile ) {
//      #pragma simd
        LOOP_4R(m_tiledims) {
          apply( LOOP_ARGS_4 );
        }
      } else {
//      #pragma simd
        LOOP_4R(m_tiledims) {
          apply( LOOP_ARGS_4 );
        }
      }
    } // end RP::Right

  } //end op() rank == 4


  template <typename IType>
  inline
  void operator_impl( IType tile_idx , const RankTag<5> ) const
  {
    point_type m_offset;
    point_type m_tiledims;

    if (RP::outer_direction == RP::Left) {
      for (int i=0; i<RP::rank; ++i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ; 
        tile_idx /= m_rp.m_tile_end[i];
      }
    }
    else {
      for (int i=RP::rank-1; i>=0; --i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ;
        tile_idx /= m_rp.m_tile_end[i];
      }
    }

    //Check if offset+tiledim in bounds - if not, replace tile dims with the partial tile dims
    const bool full_tile = check_iteration_bounds(m_tiledims , m_offset) ;

    if (RP::inner_direction == RP::Left) {
     if ( full_tile ) {
//      #pragma simd
        LOOP_5L(m_tiledims) {
          apply( LOOP_ARGS_5 );
        }
      } else {
//      #pragma simd
        LOOP_5L(m_tiledims) {
          apply( LOOP_ARGS_5 );
        }
      }
    } // end RP::Left
    else {
     if ( full_tile ) {
//      #pragma simd
        LOOP_5R(m_tiledims) {
          apply( LOOP_ARGS_5 );
        }
      } else {
//      #pragma simd
        LOOP_5R(m_tiledims) {
          apply( LOOP_ARGS_5 );
        }
      }
    } // end RP::Right

  } //end op() rank == 5


  template <typename IType>
  inline
  void operator_impl( IType tile_idx , const RankTag<6> ) const
  {
    point_type m_offset;
    point_type m_tiledims;

    if (RP::outer_direction == RP::Left) {
      for (int i=0; i<RP::rank; ++i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ; 
        tile_idx /= m_rp.m_tile_end[i];
      }
    }
    else {
      for (int i=RP::rank-1; i>=0; --i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ;
        tile_idx /= m_rp.m_tile_end[i];
      }
    }

    //Check if offset+tiledim in bounds - if not, replace tile dims with the partial tile dims
    const bool full_tile = check_iteration_bounds(m_tiledims , m_offset) ;

    if (RP::inner_direction == RP::Left) {
     if ( full_tile ) {
//      #pragma simd
        LOOP_6L(m_tiledims) {
          apply( LOOP_ARGS_6 );
        }
      } else {
//      #pragma simd
        LOOP_6L(m_tiledims) {
          apply( LOOP_ARGS_6 );
        }
      }
    } // end RP::Left
    else {
     if ( full_tile ) {
//      #pragma simd
        LOOP_6R(m_tiledims) {
          apply( LOOP_ARGS_6 );
        }
      } else {
//      #pragma simd
        LOOP_6R(m_tiledims) {
          apply( LOOP_ARGS_6 );
        }
      }
    } // end RP::Right

  } //end op() rank == 6


  template <typename IType>
  inline
  void operator_impl( IType tile_idx , const RankTag<7> ) const
  {
    point_type m_offset;
    point_type m_tiledims;

    if (RP::outer_direction == RP::Left) {
      for (int i=0; i<RP::rank; ++i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ; 
        tile_idx /= m_rp.m_tile_end[i];
      }
    }
    else {
      for (int i=RP::rank-1; i>=0; --i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ;
        tile_idx /= m_rp.m_tile_end[i];
      }
    }

    //Check if offset+tiledim in bounds - if not, replace tile dims with the partial tile dims
    const bool full_tile = check_iteration_bounds(m_tiledims , m_offset) ;

    if (RP::inner_direction == RP::Left) {
     if ( full_tile ) {
//      #pragma simd
        LOOP_7L(m_tiledims) {
          apply( LOOP_ARGS_7 );
        }
      } else {
//      #pragma simd
        LOOP_7L(m_tiledims) {
          apply( LOOP_ARGS_7 );
        }
      }
    } // end RP::Left
    else {
     if ( full_tile ) {
//      #pragma simd
        LOOP_7R(m_tiledims) {
          apply( LOOP_ARGS_7 );
        }
      } else {
//      #pragma simd
        LOOP_7R(m_tiledims) {
          apply( LOOP_ARGS_7 );
        }
      }
    } // end RP::Right

  } //end op() rank == 7


  template <typename IType>
  inline
  void operator_impl( IType tile_idx , const RankTag<8> ) const
  {
    point_type m_offset;
    point_type m_tiledims;

    if (RP::outer_direction == RP::Left) {
      for (int i=0; i<RP::rank; ++i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ; 
        tile_idx /= m_rp.m_tile_end[i];
      }
    }
    else {
      for (int i=RP::rank-1; i>=0; --i) {
        m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i] ;
        tile_idx /= m_rp.m_tile_end[i];
      }
    }

    //Check if offset+tiledim in bounds - if not, replace tile dims with the partial tile dims
    const bool full_tile = check_iteration_bounds(m_tiledims , m_offset) ;

    if (RP::inner_direction == RP::Left) {
     if ( full_tile ) {
//      #pragma simd
        LOOP_8L(m_tiledims) {
          apply( LOOP_ARGS_8 );
        }
      } else {
//      #pragma simd
        LOOP_8L(m_tiledims) {
          apply( LOOP_ARGS_8 );
        }
      }
    } // end RP::Left
    else {
     if ( full_tile ) {
//      #pragma simd
        LOOP_8R(m_tiledims) {
          apply( LOOP_ARGS_8 );
        }
      } else {
//      #pragma simd
        LOOP_8R(m_tiledims) {
          apply( LOOP_ARGS_8 );
        }
      }
    } // end RP::Right

  } //end op() rank == 8
#endif


    template <typename... Args>
    typename std::enable_if<( sizeof...(Args) == RP::rank && std::is_same<Tag,void>::value), void>::type
    apply(Args &&... args) const
    {
      m_func(args... , m_v);
    }

    template <typename... Args>
    typename std::enable_if<( sizeof...(Args) == RP::rank && !std::is_same<Tag,void>::value), void>::type
    apply(Args &&... args) const
    {
      m_func( m_tag, args... , m_v);
    }


  RP         const& m_rp;
  Functor    const& m_func;
  value_type  & m_v;
  typename std::conditional< std::is_same<Tag,void>::value,int,Tag>::type m_tag{};

};


// ------------------------------------------------------------------ //

// MDFunctor - wraps the range_policy and functor to pass to IterateTile
// Serial, Threads, OpenMP
// Cuda uses DeviceIterateTile directly within md_parallel_for
// ParallelReduce
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
                                                                           , work_tag
                                                                           , value_type
                                                                           >;


  inline
  MDFunctor( MDRange const& range, Functor const& f, ValueType & v )
    : m_range( range )
    , m_func( f )
  {}

  inline
  MDFunctor( MDFunctor const& ) = default;

  inline
  MDFunctor& operator=( MDFunctor const& ) = default;

  inline
  MDFunctor( MDFunctor && ) = default;

  inline
  MDFunctor& operator=( MDFunctor && ) = default;

//  KOKKOS_FORCEINLINE_FUNCTION //Caused cuda warning - __host__ warning
  inline
  void operator()(index_type t, value_type & v) const
  {
  #if REMOVEOPERATOR
    iterate_type(m_range, m_func, v).apply_impl(t);
  #else
    iterate_type(m_range, m_func, v)(t);
  #endif 
  }

  MDRange   m_range;
  Functor   m_func;
};

// ParallelFor
template < typename MDRange, typename Functor >
struct MDFunctor< MDRange, Functor, void >
{
  using range_policy = MDRange;
  using functor_type = Functor;
  using work_tag     = typename range_policy::work_tag;
  using index_type   = typename range_policy::index_type;
  using iterate_type = typename Kokkos::Experimental::Impl::HostIterateTile< MDRange
                                                                           , Functor
                                                                           , work_tag
                                                                           , void
                                                                           >;


  inline
  MDFunctor( MDRange const& range, Functor const& f )
    : m_range( range )
    , m_func( f )
  {}

  inline
  MDFunctor( MDFunctor const& ) = default;

  inline
  MDFunctor& operator=( MDFunctor const& ) = default;

  inline
  MDFunctor( MDFunctor && ) = default;

  inline
  MDFunctor& operator=( MDFunctor && ) = default;

  inline
  void operator()(index_type t) const
  {
  #if REMOVEOPERATOR
    iterate_type(m_range, m_func).apply_impl(t);
  #else
    iterate_type(m_range, m_func)(t);
  #endif
  }

  MDRange m_range;
  Functor m_func;
};

#undef REMOVEOPERATOR

} } } //end namespace Kokkos::Experimental::Impl


#endif
