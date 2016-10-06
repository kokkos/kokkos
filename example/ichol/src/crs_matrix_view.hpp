#pragma once
#ifndef __CRS_MATRIX_VIEW_HPP__
#define __CRS_MATRIX_VIEW_HPP__

/// \file crs_matrix_view.hpp
/// \brief CRS matrix view object creates 2D view to setup a computing region.
/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "util.hpp"

namespace Tacho { 

  using namespace std;

  template<typename CrsMatBaseType>  
  class CrsRowView;

  template<typename CrsMatBaseType>
  class CrsMatrixView {
  public:
    typedef typename CrsMatBaseType::space_type    space_type;
    
    typedef typename CrsMatBaseType::value_type    value_type;
    typedef typename CrsMatBaseType::ordinal_type  ordinal_type;
    typedef typename CrsMatBaseType::size_type     size_type;

    typedef CrsMatBaseType             mat_base_type;
    typedef CrsRowView<mat_base_type>  row_view_type;

    // be careful this use rcp and atomic operation
    // - use setView to create a view if _rows is not necessary
    // - copy constructor and assignment operator will do soft copy of the object
    typedef Kokkos::View<row_view_type*,space_type,Kokkos::MemoryUnmanaged> row_view_type_array;
    
  private:
    CrsMatBaseType _base;    // shallow copy of the base object
    ordinal_type  _offm;     // offset in rows
    ordinal_type  _offn;     // offset in cols
    ordinal_type  _m;        // # of rows
    ordinal_type  _n;        // # of cols

    row_view_type_array _rows;
    
  public:

    KOKKOS_INLINE_FUNCTION
    void setRowViewArray( const row_view_type_array & arg_rows )
      {
        _rows = arg_rows ;

        for (ordinal_type i=0;i<_m;++i) {
          _rows[i].setView(*this, i);
        }
      }

    KOKKOS_INLINE_FUNCTION
    row_view_type& RowView(const ordinal_type i) const { return _rows[i]; }

    KOKKOS_INLINE_FUNCTION
    void setView(const CrsMatBaseType &base,
                 const ordinal_type offm, const ordinal_type m,
                 const ordinal_type offn, const ordinal_type n) {
      _base = base;

      _offm = offm; _m = m;
      _offn = offn; _n = n;
    }

    KOKKOS_INLINE_FUNCTION
    const CrsMatBaseType & BaseObject() const { return _base; }

    KOKKOS_INLINE_FUNCTION
    ordinal_type  OffsetRows() const { return _offm; }

    KOKKOS_INLINE_FUNCTION
    ordinal_type  OffsetCols() const { return _offn; }

    KOKKOS_INLINE_FUNCTION    
    ordinal_type  NumRows() const { return _m; }

    KOKKOS_INLINE_FUNCTION
    ordinal_type  NumCols() const { return _n; }

    KOKKOS_INLINE_FUNCTION
    bool hasNumNonZeros() const { 
      const ordinal_type m = NumRows();
      for (ordinal_type i=0;i<m;++i) {
        row_view_type row;
        row.setView(*this, i);
        if (row.NumNonZeros()) return true;
      }
      return false;
    }

    inline
    size_type countNumNonZeros() const { 
      size_type nnz = 0;
      const ordinal_type m = NumRows();
      for (ordinal_type i=0;i<m;++i) {
        row_view_type row;
        row.setView(*this, i);
        nnz += row.NumNonZeros();
      }
      return nnz; 
    }

    KOKKOS_INLINE_FUNCTION
    CrsMatrixView()
      : _base(),
        _offm(0),
        _offn(0),
        _m(0),
        _n(0),
        _rows()
    { } 

    KOKKOS_INLINE_FUNCTION
    CrsMatrixView(const CrsMatrixView &b)
      : _base(b._base),
        _offm(b._offm),
        _offn(b._offn),
        _m(b._m),
        _n(b._n),
        _rows(b._rows)
    { } 

    KOKKOS_INLINE_FUNCTION
    CrsMatrixView(const CrsMatBaseType & b)
      : _base(b),
        _offm(0),
        _offn(0),
        _m(b.NumRows()),
        _n(b.NumCols()),
        _rows()
    { } 

    CrsMatrixView(const CrsMatBaseType & b,
                  const ordinal_type offm, const ordinal_type m,
                  const ordinal_type offn, const ordinal_type n) 
      : _base(b),
        _offm(offm),
        _offn(offn),
        _m(m),
        _n(n),
        _rows()
    { } 

    ostream& showMe(ostream &os) const {
      const int w = 4;
      os << "CrsMatrixView, "
         << " Offs ( " << setw(w) << _offm << ", " << setw(w) << _offn << " ); "
         << " Dims ( " << setw(w) << _m    << ", " << setw(w) << _n    << " ); "
         << " NumNonZeros = " << countNumNonZeros() << ";";

      return os;
    }

  };
}


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif
