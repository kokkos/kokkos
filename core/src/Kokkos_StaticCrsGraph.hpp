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

#ifndef KOKKOS_STATICCRSGRAPH_HPP
#define KOKKOS_STATICCRSGRAPH_HPP

#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

namespace Kokkos {

namespace Impl {
  template<class RowOffsetsType, class RowBlockOffsetsType>
  struct StaticCrsGraphBalancerFunctor {
    typedef typename RowOffsetsType::non_const_value_type int_type;
    RowOffsetsType row_offsets;
    RowBlockOffsetsType row_block_offsets;

    int_type cost_per_row, num_blocks;

    StaticCrsGraphBalancerFunctor(RowOffsetsType row_offsets_,
                                  RowBlockOffsetsType row_block_offsets_,
                                  int_type cost_per_row_, int_type num_blocks_):
                                    row_offsets(row_offsets_),
                                    row_block_offsets(row_block_offsets_),
                                    cost_per_row(cost_per_row_),
                                    num_blocks(num_blocks_){}

    KOKKOS_INLINE_FUNCTION
    void operator() (const int_type& iRow) const {
      const int_type num_rows = row_offsets.dimension_0()-1;
      const int_type num_entries = row_offsets(num_rows);
      const int_type total_cost = num_entries + num_rows*cost_per_row;

      const double cost_per_workset = 1.0*total_cost/num_blocks;

      const int_type row_cost = row_offsets(iRow+1)-row_offsets(iRow) + cost_per_row;

      int_type count = row_offsets(iRow+1) + cost_per_row*iRow;

      if(iRow == num_rows-1) row_block_offsets(num_blocks) = num_rows;

      if(true) {
        int_type current_block = (count-row_cost-cost_per_row)/cost_per_workset;
        int_type end_block = count/cost_per_workset;

        // Handle some corner cases for the last two blocks.
        if(current_block >= num_blocks-2) {
          if((current_block == num_blocks-2) && (count >= (current_block + 1) * cost_per_workset)) {
            int_type row = iRow;
            int_type cc = count-row_cost-cost_per_row;
            int_type block = cc/cost_per_workset;
            while((block>0) && (block==current_block)) {
              cc = row_offsets(row)+row*cost_per_row;
              block = cc/cost_per_workset;
              row--;
            }
            if((count-cc-row_cost-cost_per_row) < num_entries-row_offsets(iRow+1)) {
              row_block_offsets(current_block+1) = iRow+1;
            } else {
              row_block_offsets(current_block+1) = iRow;
            }
          }
        } else {
          if((count >= (current_block + 1) * cost_per_workset) ||
             (iRow+2 == row_offsets.dimension_0())) {
            if(end_block>current_block+1) {
              int_type num_block = end_block-current_block;
              row_block_offsets(current_block+1) = iRow;
              for(int_type block = current_block+2; block <= end_block; block++)
                if((block<current_block+2+(num_block-1)/2))
                  row_block_offsets(block) = iRow;
                else
                  row_block_offsets(block) = iRow+1;
            } else {
              row_block_offsets(current_block+1) = iRow+1;
            }
          }
        }

      }
    }
  };
}

/// \class StaticCrsGraph
/// \brief Compressed row storage array.
///
/// \tparam DataType The type of stored entries.  If a StaticCrsGraph is
///   used as the graph of a sparse matrix, then this is usually an
///   integer type, the type of the column indices in the sparse
///   matrix.
///
/// \tparam Arg1Type The second template parameter, corresponding
///   either to the Device type (if there are no more template
///   parameters) or to the Layout type (if there is at least one more
///   template parameter).
///
/// \tparam Arg2Type The third template parameter, which if provided
///   corresponds to the Device type.
///
/// \tparam SizeType The type of row offsets.  Usually the default
///   parameter suffices.  However, setting a nondefault value is
///   necessary in some cases, for example, if you want to have a
///   sparse matrices with dimensions (and therefore column indices)
///   that fit in \c int, but want to store more than <tt>INT_MAX</tt>
///   entries in the sparse matrix.
///
/// A row has a range of entries:
/// <ul>
/// <li> <tt> row_map[i0] <= entry < row_map[i0+1] </tt> </li>
/// <li> <tt> 0 <= i1 < row_map[i0+1] - row_map[i0] </tt> </li>
/// <li> <tt> entries( entry ,            i2 , i3 , ... ); </tt> </li>
/// <li> <tt> entries( row_map[i0] + i1 , i2 , i3 , ... ); </tt> </li>
/// </ul>
template< class DataType,
          class Arg1Type,
          class Arg2Type = void,
          typename SizeType = typename ViewTraits<DataType*, Arg1Type, Arg2Type, void >::size_type>
class StaticCrsGraph {
private:
  typedef ViewTraits<DataType*, Arg1Type, Arg2Type, void> traits;

public:
  typedef DataType                                            data_type;
  typedef typename traits::array_layout                       array_layout;
  typedef typename traits::execution_space                    execution_space;
  typedef typename traits::device_type                        device_type;
  typedef SizeType                                            size_type;

  typedef StaticCrsGraph< DataType , Arg1Type , Arg2Type , SizeType > staticcrsgraph_type;
  typedef StaticCrsGraph< DataType , array_layout , typename traits::host_mirror_space , SizeType > HostMirror;
  typedef View< const size_type* , array_layout, device_type >  row_map_type;
  typedef View<       DataType*  , array_layout, device_type >  entries_type;
  typedef View< const size_type* , array_layout, device_type >  row_block_type;

  entries_type entries;
  row_map_type row_map;
  row_block_type row_block_offsets;

  //! Construct an empty view.
  StaticCrsGraph () : entries(), row_map(), row_block_offsets() {}

  //! Copy constructor (shallow copy).
  StaticCrsGraph (const StaticCrsGraph& rhs) : entries (rhs.entries), row_map (rhs.row_map),
                                               row_block_offsets(rhs.row_block_offsets)
  {}

  template<class EntriesType, class RowMapType>
  StaticCrsGraph (const EntriesType& entries_,const RowMapType& row_map_) : entries (entries_), row_map (row_map_),
  row_block_offsets()
  {}

  /** \brief  Assign to a view of the rhs array.
   *          If the old view is the last view
   *          then allocated memory is deallocated.
   */
  StaticCrsGraph& operator= (const StaticCrsGraph& rhs) {
    entries = rhs.entries;
    row_map = rhs.row_map;
    row_block_offsets = rhs.row_block_offsets;
    return *this;
  }

  /**  \brief  Destroy this view of the array.
   *           If the last view then allocated memory is deallocated.
   */
  ~StaticCrsGraph() {}

  /**  \brief  Return number of rows in the graph
   */
  KOKKOS_INLINE_FUNCTION
  size_type numRows() const {
    return (row_map.dimension_0 () != 0) ?
      row_map.dimension_0 () - static_cast<size_type> (1) :
      static_cast<size_type> (0);
  }

  /**  \brief  Create a row partitioning into a given number of blocks
   *           balancing non-zeros + a fixed cost per row.
   */
  void create_block_partitioning(size_type num_blocks, size_type fix_cost_per_row = 4) {
    View< size_type* , array_layout, device_type >
      block_offsets("StatisCrsGraph::load_balance_offsets",num_blocks+1);

    Impl::StaticCrsGraphBalancerFunctor<row_map_type,View< size_type* , array_layout, device_type > >
      partitioner(row_map,block_offsets,fix_cost_per_row,num_blocks);

    Kokkos::parallel_for(Kokkos::RangePolicy<execution_space>(0,numRows()),partitioner);
    Kokkos::fence();

    row_block_offsets = block_offsets;
  }
};

//----------------------------------------------------------------------------

template< class StaticCrsGraphType , class InputSizeType >
typename StaticCrsGraphType::staticcrsgraph_type
create_staticcrsgraph( const std::string & label ,
                 const std::vector< InputSizeType > & input );

template< class StaticCrsGraphType , class InputSizeType >
typename StaticCrsGraphType::staticcrsgraph_type
create_staticcrsgraph( const std::string & label ,
                 const std::vector< std::vector< InputSizeType > > & input );

//----------------------------------------------------------------------------

template< class DataType ,
          class Arg1Type ,
          class Arg2Type ,
          typename SizeType >
typename StaticCrsGraph< DataType , Arg1Type , Arg2Type , SizeType >::HostMirror
create_mirror_view( const StaticCrsGraph<DataType,Arg1Type,Arg2Type,SizeType > & input );

template< class DataType ,
          class Arg1Type ,
          class Arg2Type ,
          typename SizeType >
typename StaticCrsGraph< DataType , Arg1Type , Arg2Type , SizeType >::HostMirror
create_mirror( const StaticCrsGraph<DataType,Arg1Type,Arg2Type,SizeType > & input );

} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#include <impl/Kokkos_StaticCrsGraph_factory.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template< class GraphType >
struct StaticCrsGraphMaximumEntry {

  typedef typename GraphType::execution_space execution_space ;
  typedef typename GraphType::data_type value_type ;

  const typename GraphType::entries_type entries ;

  StaticCrsGraphMaximumEntry( const GraphType & graph ) : entries( graph.entries ) {}

  KOKKOS_INLINE_FUNCTION
  void operator()( const unsigned i , value_type & update ) const
    { if ( update < entries(i) ) update = entries(i); }

  KOKKOS_INLINE_FUNCTION
  void init( value_type & update ) const
    { update = 0 ; }

  KOKKOS_INLINE_FUNCTION
  void join( volatile value_type & update ,
             volatile const value_type & input ) const
    { if ( update < input ) update = input ; }
};

}

template< class DataType, class Arg1Type, class Arg2Type, typename SizeType >
DataType maximum_entry( const StaticCrsGraph< DataType , Arg1Type , Arg2Type , SizeType > & graph )
{
  typedef StaticCrsGraph<DataType,Arg1Type,Arg2Type,SizeType> GraphType ;
  typedef Impl::StaticCrsGraphMaximumEntry< GraphType > FunctorType ;

  DataType result = 0 ;
  Kokkos::parallel_reduce( graph.entries.dimension_0(),
                           FunctorType(graph), result );
  return result ;
}

} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #ifndef KOKKOS_CRSARRAY_HPP */

