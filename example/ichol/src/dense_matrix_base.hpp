#pragma once
#ifndef __DENSE_MATRIX_BASE_HPP__
#define __DENSE_MATRIX_BASE_HPP__

/// \file dense_matrix_base.hpp
/// \brief dense matrix base object interfaces 
/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "util.hpp"

namespace Tacho { 

  using namespace std;
  
  /// \class DenseMatrixBase
  /// \breif Dense matrix base object using Kokkos view and subview
  template<typename ValueType,
           typename OrdinalType, 
           typename SizeType = OrdinalType,
           typename SpaceType = void,
           typename MemoryTraits = void>
  class DenseMatrixBase : public Disp {
  public:
    typedef ValueType    value_type;
    typedef OrdinalType  ordinal_type;
    typedef SpaceType    space_type;
    typedef SizeType     size_type;
    typedef MemoryTraits memory_traits;

    // 1D view, layout does not matter; no template parameters for that
    typedef Kokkos::View<size_type*,   space_type,memory_traits> size_type_array;
    typedef Kokkos::View<ordinal_type*,space_type,memory_traits> ordinal_type_array;
    typedef Kokkos::View<value_type*,  space_type,memory_traits> value_type_array;

    typedef typename size_type_array::value_type*    size_type_array_ptr;
    typedef typename ordinal_type_array::value_type* ordinal_type_array_ptr;
    typedef typename value_type_array::value_type*   value_type_array_ptr;

    friend class DenseMatrixHelper;

  private:
    string             _label;   //!< object label
    
    ordinal_type       _m;       //!< # of rows
    ordinal_type       _n;       //!< # of cols

    ordinal_type       _cs;      //!< column stride
    ordinal_type       _rs;      //!< row stride
    
    value_type_array   _a;       //!< values
    
  protected:
    void createInternalArrays(const ordinal_type m, 
                              const ordinal_type n) {
      // matrix diemsntion
      _m = m;
      _n = n;
      
      // the default layout is column major
      _cs = m;
      _rs = 1;

      // grow buffer dimension
      const size_type size = _m*_n;
      if (static_cast<size_type>(_a.dimension_0()) < size)
        _a = value_type_array(_label+"::ValuesArray", size);
    }

  public:

    KOKKOS_INLINE_FUNCTION
    void setLabel(const string label) { _label = label; }

    KOKKOS_INLINE_FUNCTION
    string Label() const { return _label; }

    KOKKOS_INLINE_FUNCTION
    ordinal_type NumRows() const { return _m; }

    KOKKOS_INLINE_FUNCTION
    ordinal_type NumCols() const { return _n; }

    KOKKOS_INLINE_FUNCTION
    ordinal_type ColStride() const { return _cs; }

    KOKKOS_INLINE_FUNCTION
    ordinal_type RowStride() const { return _rs; }

    KOKKOS_INLINE_FUNCTION
    value_type& Value(const ordinal_type i, 
                      const ordinal_type j) { return _a[i*_rs + j*_cs]; }

    KOKKOS_INLINE_FUNCTION
    value_type Value(const ordinal_type i, 
                     const ordinal_type j) const { return _a[i*_rs + j*_cs]; }

    KOKKOS_INLINE_FUNCTION
    value_type* ValuePtr() const { return &_a[0]; }
    
    /// \brief Default constructor.
    DenseMatrixBase() 
      : _label("DenseMatrixBase"),
        _m(0),
        _n(0),
        _cs(0),
        _rs(0),
        _a()
    { }

    /// \brief Constructor with label
    DenseMatrixBase(const string label) 
      : _label(label),
        _m(0),
        _n(0),
        _cs(0),
        _rs(0),
        _a()
    { }

    /// \brief Copy constructor (shallow copy), for deep-copy use a method copy
    template<typename VT,
             typename OT,
             typename ST,
             typename SpT,
             typename MT>
    DenseMatrixBase(const DenseMatrixBase<VT,OT,ST,SpT,MT> &b) 
      : _label(b._label),
        _m(b._m),
        _n(b._n),
        _cs(b._cs),
        _rs(b._rs),
        _a(b._a) 
    { }
    
    /// \brief Constructor to allocate internal data structures.
    DenseMatrixBase(const string label,
                    const ordinal_type m, 
                    const ordinal_type n)
      : _label(label), 
        _m(m),
        _n(n),
        _cs(m),
        _rs(1),
        _a(_label+"::ValuesArray", m*n)  
    { }
      
    /// \brief Constructor to attach external arrays to the matrix.
    DenseMatrixBase(const string label,
                    const ordinal_type m, 
                    const ordinal_type n,
                    const ordinal_type cs,
                    const ordinal_type rs,
                    const value_type_array &a) 
      : _label(label), 
        _m(m),
        _n(n),
        _cs(cs == -1 ? m : cs),
        _rs(rs == -1 ? 1 : rs),
        _a(a) 
    { }
    
    /// \brief deep copy of matrix b
    template<typename VT,
             typename OT,
             typename ST,
             typename SpT,
             typename MT>
    int 
    copy(const DenseMatrixBase<VT,OT,ST,SpT,MT> &b) {
      //createInternalArrays(b._m, b._n);

      for (ordinal_type j=0;j<b._n;++j)
        for (ordinal_type i=0;i<b._m;++i)
          this->Value(i,j) = b.Value(i,j);

      return 0;
    }

    /// \brief deep copy of lower/upper triangular of matrix b
    template<typename VT,
             typename OT,
             typename ST,
             typename SpT,
             typename MT>
    int 
    copy(const int uplo, 
         const DenseMatrixBase<VT,OT,ST,SpT,MT> &b) { 
      //createInternalArrays(b._m, b._n);

      // assume that matrix b is sorted.
      switch (uplo) {
      case Uplo::Lower: {
        for (ordinal_type j=0;j<b._n;++j) 
          for (ordinal_type i=j;i<b._m;++i) 
            this->Value(i, j) = b.Value(i, j); 
        break;
      }
      case Uplo::Upper: {
        for (ordinal_type j=0;j<b._n;++j) 
          for (ordinal_type i=0;i<(j+1);++i) 
            this->Value(i, j) = b.Value(i, j); 
        break;
      }
      }
      
      return 0;
    }

    /// \brief deep copy of matrix b with given permutation vectors
    template<typename VT,
             typename OT,
             typename ST,
             typename SpT,
             typename MT>
    int
    copy(const typename DenseMatrixBase<VT,OT,ST,SpT,MT>::ordinal_type_array &ip,
         const DenseMatrixBase<VT,OT,ST,SpT,MT> &b) {
      //createInternalArrays(b._m, b._n);

      for (ordinal_type i=0;i<b._m;++i) {
        const ordinal_type ii = ip[i];
        for (ordinal_type j=0;j<b._n;++j) 
          this->Value(i,j) = b.Value(ii, j);
      }

      return 0;
    }

    ostream& showMe(ostream &os) const {
      streamsize prec = os.precision();
      os.precision(8);
      os << scientific;

      os << " -- " << _label << " -- " << endl
         << "    # of Rows              = " << _m << endl
         << "    # of Cols              = " << _n << endl
         << "    Col Stride             = " << _cs << endl
         << "    Row Stride             = " << _rs << endl
         << endl
         << "    ValueArray dimensions  = " << _a.dimension_0() << endl
         << endl;
      
      const int w = 10;
      if (_a.size()) {
        for (ordinal_type i=0;i<_m;++i) {
          for (ordinal_type j=0;j<_n;++j) {
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

#endif
