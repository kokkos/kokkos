#pragma once
#ifndef __CRS_MATRIX_VIEW_EXT_HPP__
#define __CRS_MATRIX_VIEW_EXT_HPP__

/// \file crs_matrix_view_ext.hpp
/// \brief Extended matrix view that has nested dense block.
/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "dense_matrix_helper.hpp"
namespace Tacho {

  template<typename CrsMatViewType,
           typename DenseFlatViewType,
           typename DenseHierViewType>
  class CrsMatrixViewExt : public CrsMatViewType {
  public:
    typedef typename CrsMatViewType::space_type    space_type;
    typedef typename CrsMatViewType::memory_traits memory_traits;

    typedef typename CrsMatViewType::value_type    value_type;
    typedef typename CrsMatViewType::ordinal_type  ordinal_type;
    typedef typename CrsMatViewType::size_type     size_type;

    typedef          DenseFlatViewType                dense_flat_view_type;
    typedef typename DenseFlatViewType::mat_base_type dense_flat_base_type;

    typedef          DenseHierViewType                dense_hier_view_type;
    typedef typename DenseHierViewType::mat_base_type dense_hier_base_type;

  private:
    dense_flat_base_type _A;
    dense_hier_base_type _H;

  public:
    bool hasDenseFlatBase() const {
      return (_A.NumRows() && _A.NumCols());
    }
    bool hasDenseHierBase() const {
      return (_H.NumRows() && _H.NumCols());
    }

    bool isDenseFlatBaseValid() const {
      return (_A.NumRows() >= this->NumRows() && _A.NumCols() >= this->NumCols());
    }

    void createDenseFlatBase() {
      _A = dense_flat_base_type("NestedDenseFlatBase", this->NumRows(), this->NumCols());
    }
    void createDenseHierBase(const ordinal_type mb, const ordinal_type nb) {
      if (hasDenseFlatBase() && isDenseFlatBaseValid()) {
        _H.setLabel("NestedDenseHierBase");
        DenseMatrixHelper::flat2hier(_A, _H, mb, nb);
      }
    }

    dense_flat_base_type* DenseFlatBaseObject() { return &_A; }
    dense_hier_base_type* DenseHierBaseObject() { return &_H; }

    int copyToDenseFlatBase() {
      int r_val = 0;
      if (hasDenseFlatBase() && isDenseFlatBaseValid())  {
        const ordinal_type nrows = this->NumRows();
        for (ordinal_type i=0;i<nrows;++i) {
          auto row = this->RowView(i);
          const ordinal_type nnz = row.NumNonZeros();
          for (ordinal_type j=0;j<nnz;++j) 
            _A.Value(i, row.Col(j)) = row.Value(j);
        }
      } else {
        r_val = -1;
      }
      return r_val;
    }

    int copyToCrsMatrixView() {
      int r_val = 0;
      if (hasDenseFlatBase() && isDenseFlatBaseValid())  {
        const ordinal_type nrows = this->NumRows();
        for (ordinal_type i=0;i<nrows;++i) {
          auto row = this->RowView(i);
          const ordinal_type nnz = row.NumNonZeros();
          for (ordinal_type j=0;j<nnz;++j)
            row.Value(j) = _A.Value(i, row.Col(j));
        }
      } else {
        r_val = -1;
      }
      return r_val;
    }

    CrsMatrixViewExt()
      : CrsMatViewType(), _A(), _H()
    { }

    CrsMatrixViewExt(const CrsMatrixViewExt &b)
      : CrsMatViewType(b), _A(b._A), _H(b._H)
    { }

    CrsMatrixViewExt(typename CrsMatViewType::mat_base_type *b)
      : CrsMatViewType(b), _A(), _H()
    { }

    CrsMatrixViewExt(typename CrsMatViewType::mat_base_type *b,
                 const ordinal_type offm, const ordinal_type m,
                 const ordinal_type offn, const ordinal_type n)
      : CrsMatViewType(b, offm, m, offn, n), _A(), _H()
    { }

  };
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
  namespace Impl {

    //  The Kokkos::View allocation will by default assign each allocated datum to zero.
    //  This is not the required initialization behavior when
    //  non-trivial objects are used within a Kokkos::View.
    //  Create a partial specialization of the Kokkos::Impl::AViewDefaultConstruct
    //  to replace the assignment initialization with placement new initialization.
    //
    //  This work-around is necessary until a TBD design refactorization of Kokkos::View.

    template< class ExecSpace , typename T1, typename T2, typename T3 >
    struct ViewDefaultConstruct< ExecSpace , Tacho::CrsMatrixViewExt<T1,T2,T3> , true >
    {
      typedef Tacho::CrsMatrixViewExt<T1,T2,T3> type ;
      type * const m_ptr ;

      KOKKOS_FORCEINLINE_FUNCTION
      void operator()( const typename ExecSpace::size_type& i ) const
      { new(m_ptr+i) type(); }

      ViewDefaultConstruct( type * pointer , size_t capacity )
        : m_ptr( pointer )
      {
        Kokkos::RangePolicy< ExecSpace > range( 0 , capacity );
        parallel_for( range , *this );
        ExecSpace::fence();
      }
    };

  } // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif
