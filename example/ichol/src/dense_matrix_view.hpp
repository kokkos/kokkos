#pragma once
#ifndef __DENSE_MATRIX_VIEW_HPP__
#define __DENSE_MATRIX_VIEW_HPP__

/// \file dense_matrix_view.hpp
/// \brief dense matrix view object creates 2D view to setup a computing region.
/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "util.hpp"

namespace Tacho { 

  using namespace std;

  template<typename DenseMatBaseType>
  class DenseMatrixView : public Disp {
  public:
    typedef typename DenseMatBaseType::space_type    space_type;
    typedef typename DenseMatBaseType::memory_traits memory_traits;
    
    typedef typename DenseMatBaseType::value_type    value_type;
    typedef typename DenseMatBaseType::ordinal_type  ordinal_type;
    typedef typename DenseMatBaseType::size_type     size_type;

    typedef DenseMatBaseType mat_base_type;

  private:
    DenseMatBaseType *_base;   // pointer to the base object

    ordinal_type  _offm;     // offset in rows
    ordinal_type  _offn;     // offset in cols

    ordinal_type  _m;        // # of rows
    ordinal_type  _n;        // # of cols

  public:

    KOKKOS_INLINE_FUNCTION
    void setView(DenseMatBaseType *base,
                 const ordinal_type offm, const ordinal_type m,
                 const ordinal_type offn, const ordinal_type n) {
      _base = base;

      _offm = offm; _m = m;
      _offn = offn; _n = n;
    }

    KOKKOS_INLINE_FUNCTION
    DenseMatBaseType* BaseObject() const { return _base; }

    KOKKOS_INLINE_FUNCTION
    ordinal_type  OffsetRows() const { return _offm; }

    KOKKOS_INLINE_FUNCTION
    ordinal_type  OffsetCols() const { return _offn; }

    KOKKOS_INLINE_FUNCTION    
    ordinal_type  NumRows() const { return _m; }

    KOKKOS_INLINE_FUNCTION
    ordinal_type  NumCols() const { return _n; }

    KOKKOS_INLINE_FUNCTION
    value_type& Value(const ordinal_type i,
                      const ordinal_type j) { return _base->Value(_offm+i, _offn+j); }

    KOKKOS_INLINE_FUNCTION
    value_type Value(const ordinal_type i,
                     const ordinal_type j) const { return _base->Value(_offm+i, _offn+j); }

    KOKKOS_INLINE_FUNCTION
    value_type* ValuePtr() const { return &_base->Value(_offm, _offn); }

    DenseMatrixView()
      : _base(NULL),
        _offm(0),
        _offn(0),
        _m(0),
        _n(0)
    { } 

    DenseMatrixView(const DenseMatrixView &b)
      : _base(b._base),
        _offm(b._offm),
        _offn(b._offn),
        _m(b._m),
        _n(b._n)
    { } 

    DenseMatrixView(DenseMatBaseType *b)
      : _base(b),
        _offm(0),
        _offn(0),
        _m(b->NumRows()),
        _n(b->NumCols())
    { } 

    DenseMatrixView(DenseMatBaseType *b,
                    const ordinal_type offm, const ordinal_type m,
                    const ordinal_type offn, const ordinal_type n) 
      : _base(b),
        _offm(offm),
        _offn(offn),
        _m(m),
        _n(n) 
    { } 

    ostream& showMe(ostream &os) const {
      const int w = 4;
      if (_base != NULL) 
        os << _base->Label() << "::View, "
           << " Offs ( " << setw(w) << _offm << ", " << setw(w) << _offn << " ); "
           << " Dims ( " << setw(w) << _m    << ", " << setw(w) << _n    << " ); ";
      else 
        os << "-- Base object is null --";

      return os;
    }
    
    ostream& showMeDetail(ostream &os) const {
      showMe(os) << endl;
      
      streamsize prec = os.precision();
      os.precision(8);
      os << scientific;
      
      const int w = 10;
      if (_base != NULL) {
        for (ordinal_type i=0;i<NumRows();++i) {
          for (ordinal_type j=0;j<NumCols();++j) {
            const value_type val = this->Value(i,j);
            os << setw(w) << val << "  ";
          }
          os << endl;
        }
      }

      os.unsetf(ios::scientific);
      os.precision(prec);
      
      return os;
    }

  };
}

//----------------------------------------------------------------------------

#endif
