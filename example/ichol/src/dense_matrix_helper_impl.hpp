#pragma once
#ifndef __DENSE_MATRIX_HELPER_IMPL_HPP__
#define __DENSE_MATRIX_HELPER_IMPL_HPP__

/// \file dense_matrix_helper_impl.hpp
/// \brief This file includes utility functions to convert between flat and hierarchical matrices.
/// \author Kyungjoo Kim (kyukim@sandia.gov)  

#include "util.hpp"

namespace Tacho { 

  using namespace std;

  template<typename DenseFlatBase,
           typename DenseHierBase>
  KOKKOS_INLINE_FUNCTION
  int
  DenseMatrixHelper::flat2hier(DenseFlatBase &flat, 
                               DenseHierBase &hier) {
    typedef typename DenseHierBase::ordinal_type ordinal_type;

    hier.createInternalArrays(flat.NumRows(), flat.NumCols());
    
    for (ordinal_type j=0;j<flat.NumCols();++j) 
      for (ordinal_type i=0;i<flat.NumRows();++i) 
        hier.Value(i,j).setView(&flat, i, 1,
                                /**/   j, 1);
    
    return 0;
  } 

  template<typename DenseFlatBase,
           typename DenseHierBase>
  KOKKOS_INLINE_FUNCTION
  int
  DenseMatrixHelper::flat2hier(DenseFlatBase &flat, 
                               DenseHierBase &hier,
                               const typename DenseHierBase::ordinal_type mb,
                               const typename DenseHierBase::ordinal_type nb) {
    typedef typename DenseHierBase::ordinal_type       ordinal_type;
    typedef typename DenseHierBase::ordinal_type_array ordinal_type_array;

    const ordinal_type fm = flat.NumRows();
    const ordinal_type fn = flat.NumCols();

    const ordinal_type mbb = (mb == 0 ? fm : mb);
    const ordinal_type nbb = (nb == 0 ? fn : nb);

    const ordinal_type hm = fm/mbb + (fm%mbb > 0);
    const ordinal_type hn = fn/nbb + (fn%nbb > 0);

    ordinal_type_array offm("DenseMatrixHelper::flat2hier::offm", hm+1);
    {
      ordinal_type offs = 0;
      for (ordinal_type i=0;i<hm;++i) {
        offm[i] = offs;
        offs += mbb;
      }
      offm[hm] = fm;
    }

    ordinal_type_array offn("DenseMatrixHelper::flat2hier::offn", hn+1);
    {
      ordinal_type offs = 0;
      for (ordinal_type i=0;i<hn;++i) {
        offn[i] = offs;
        offs += nbb;
      }
      offn[hn] = fn;
    }

    flat2hier(flat, hier,
              hm, offm, hn, offn);
    
    return 0;
  }

  template<typename DenseFlatBase,
           typename DenseHierBase>
  KOKKOS_INLINE_FUNCTION
  int
  DenseMatrixHelper::flat2hier(DenseFlatBase &flat, 
                               DenseHierBase &hier,
                               const typename DenseHierBase::ordinal_type m,
                               const typename DenseHierBase::ordinal_type_array offm,
                               const typename DenseHierBase::ordinal_type nb) {
    typedef typename DenseHierBase::ordinal_type       ordinal_type;
    typedef typename DenseHierBase::ordinal_type_array ordinal_type_array;

    const ordinal_type fn = flat.NumCols();
    const ordinal_type nbb = (nb == 0 ? fn : nb);
    const ordinal_type hn = fn/nbb + (fn%nbb > 0);

    ordinal_type_array offn("DenseMatrixHelper::flat2hier::offn", hn+1);
    {
      ordinal_type offs = 0;
      for (ordinal_type i=0;i<hn;++i) {
        offn[i] = offs;
        offs += nbb;
      }
      offn[hn] = fn;
    }

    flat2hier(flat, hier,
              m, offm, hn, offn);
    
    return 0;
  }
  

  template<typename DenseFlatBase,
           typename DenseHierBase>
  KOKKOS_INLINE_FUNCTION
  int
  DenseMatrixHelper::flat2hier(DenseFlatBase &flat, 
                               DenseHierBase &hier,
                               const typename DenseHierBase::ordinal_type m,
                               const typename DenseHierBase::ordinal_type_array offm,
                               const typename DenseHierBase::ordinal_type n,
                               const typename DenseHierBase::ordinal_type_array offn) {
    typedef typename DenseHierBase::ordinal_type       ordinal_type;

    const ordinal_type mm = (m == 0 ? 1 : m);
    const ordinal_type nn = (n == 0 ? 1 : n);
    hier.createInternalArrays(mm, nn);

    if (mm == 1) {
      if (nn == 1) 
        hier.Value(0,0).setView(&flat, 0, flat.NumRows(),
                                /**/   0, flat.NumCols());
      else 
        for (ordinal_type j=0;j<nn;++j)
          hier.Value(0, j).setView(&flat, 0,       flat.NumRows(),
                                   /**/   offn[j], (offn[j+1] - offn[j]));
    } else {
      if (nn == 1)
        for (ordinal_type i=0;i<mm;++i)
          hier.Value(i, 0).setView(&flat, offm[i], (offm[i+1] - offm[i]),
                                   /**/   0,       flat.NumCols());
      else 
        for (ordinal_type j=0;j<n;++j)
          for (ordinal_type i=0;i<m;++i) 
            hier.Value(i, j).setView(&flat, offm[i], (offm[i+1] - offm[i]),
                                     /**/   offn[j], (offn[j+1] - offn[j]));
    }
    return 0;
  }

}


#endif

