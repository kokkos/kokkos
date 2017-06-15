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

#ifndef KOKKOS_CRS_HPP
#define KOKKOS_CRS_HPP

namespace Kokkos {
namespace Experimental {

/// \class Crs
/// \brief Compressed row storage array.
///
/// \tparam DataType The type of stored entries.  If a Crs is
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
class Crs {
protected:
  typedef ViewTraits<DataType*, Arg1Type, Arg2Type, void> traits;

public:
  typedef DataType                                            data_type;
  typedef typename traits::array_layout                       array_layout;
  typedef typename traits::execution_space                    execution_space;
  typedef typename traits::device_type                        device_type;
  typedef SizeType                                            size_type;

  typedef Crs< DataType , Arg1Type , Arg2Type , SizeType > staticcrsgraph_type;
  typedef Crs< DataType , array_layout , typename traits::host_mirror_space , SizeType > HostMirror;
  typedef View< const size_type* , array_layout, device_type >  row_map_type;
  typedef View<       DataType*  , array_layout, device_type >  entries_type;

  entries_type entries;
  row_map_type row_map;

  //! Construct an empty view.
  Crs () : entries(), row_map() {}

  //! Copy constructor (shallow copy).
  Crs (const Crs& rhs) : entries (rhs.entries), row_map (rhs.row_map)
  {}

  template<class EntriesType, class RowMapType>
  Crs (const EntriesType& entries_,const RowMapType& row_map_) : entries (entries_), row_map (row_map_)
  {}

  /** \brief  Assign to a view of the rhs array.
   *          If the old view is the last view
   *          then allocated memory is deallocated.
   */
  Crs& operator= (const Crs& rhs) {
    entries = rhs.entries;
    row_map = rhs.row_map;
    return *this;
  }

  /**  \brief  Destroy this view of the array.
   *           If the last view then allocated memory is deallocated.
   */
  ~Crs() {}

  /**  \brief  Return number of rows in the graph
   */
  KOKKOS_INLINE_FUNCTION
  size_type numRows() const {
    return (row_map.dimension_0 () != 0) ?
      row_map.dimension_0 () - static_cast<size_type> (1) :
      static_cast<size_type> (0);
  }
};

} // namespace Experimental
} // namespace Kokkos

#endif /* #define KOKKOS_CRS_HPP */
