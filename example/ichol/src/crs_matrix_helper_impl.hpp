#pragma once
#ifndef __CRS_MATRIX_HELPER_IMPL_HPP__
#define __CRS_MATRIX_HELPER_IMPL_HPP__

/// \file crs_matrix_helper_impl.hpp
/// \brief This file includes utility functions to convert between flat and hierarchical matrices.
/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "util.hpp"

namespace Tacho {

  using namespace std;
  
  template<typename CrsFlatBase>
  KOKKOS_INLINE_FUNCTION
  int
  CrsMatrixHelper::filterZeros(CrsFlatBase &flat) {
    typedef typename CrsFlatBase::ordinal_type           ordinal_type;
    typedef typename CrsFlatBase::size_type              size_type;
    typedef typename CrsFlatBase::value_type             value_type;
    
    typedef typename CrsFlatBase::ordinal_type_array_ptr ordinal_type_array_ptr;
    typedef typename CrsFlatBase::value_type_array_ptr   value_type_array_ptr;
    
    size_type nz = 0;
    const value_type zero(0);
    
    for (ordinal_type k=0;k<flat.NumNonZeros();++k) 
      nz += (flat.Value(k) == zero) ;
    
    if (nz) {
      CrsFlatBase resized(flat.Label() + "::ZeroFiltered", 
                          flat.NumRows(),
                          flat.NumCols(),
                          flat.NumNonZeros() - nz);
      
      ordinal_type_array_ptr rows = resized.RowPtr(); rows[0] = 0;
      ordinal_type_array_ptr cols = resized.ColPtr();
      value_type_array_ptr vals = resized.ValuePtr();    
      
      size_type nnz = 0;
      for (ordinal_type i=0;i<flat.NumRows();++i) {
        const ordinal_type nnz_in_row = flat.NumNonZerosInRow(i);
        const ordinal_type_array_ptr cols_in_row = flat.ColsInRow(i);
        const value_type_array_ptr vals_in_row = flat.ValuesInRow(i);
        
        for (ordinal_type j=0;j<nnz_in_row;++j) {
          if (vals_in_row[j] != zero) {
            cols[nnz] = cols_in_row[j];
            vals[nnz] = vals_in_row[j];
            ++nnz;
          }
        }
        rows[i+1] = nnz;
      }
      flat = resized;
      resized.setNumNonZeros();
    }

    return 0;
  }


  template<typename CrsFlatBase,
           typename CrsHierBase>
  KOKKOS_INLINE_FUNCTION
  int
  CrsMatrixHelper::flat2hier(CrsFlatBase &flat,
                             CrsHierBase &hier) {
    typedef typename CrsHierBase::ordinal_type           ordinal_type;
    typedef typename CrsHierBase::size_type              size_type;
    typedef typename CrsHierBase::ordinal_type_array_ptr ordinal_type_array_ptr;

    size_type nnz = 0;

    hier.createInternalArrays(flat.NumRows(), flat.NumCols(), flat.NumNonZeros());

    for (ordinal_type i=0;i<flat.NumRows();++i) {
      ordinal_type jsize = flat.NumNonZerosInRow(i);

      hier._ap[i] = nnz;
      ordinal_type_array_ptr ci = flat.ColsInRow(i);
      for (ordinal_type j=0;j<jsize;++j,++nnz) {
        hier._aj[nnz] = ci[j];
        hier._ax[nnz].setView( flat,     i, 1,
                              /**/   ci[j], 1);
      }
    }

    hier._ap[flat.NumRows()] = nnz;
    hier._nnz = nnz;

    return 0;
  }

  template<typename CrsFlatBase,
           typename CrsHierBase>
  KOKKOS_INLINE_FUNCTION
  int
  CrsMatrixHelper::flat2hier(int uplo,
                             CrsFlatBase &flat,
                             CrsHierBase &hier,
                             const typename CrsHierBase::ordinal_type       nblks,
                             const typename CrsHierBase::ordinal_type_array range,
                             const typename CrsHierBase::ordinal_type_array tree) {
    switch(uplo) {
    case Uplo::Upper: return flat2hier_upper(flat, hier, nblks, range, tree);
    case Uplo::Lower: return flat2hier_lower(flat, hier, nblks, range, tree);
    }
    return -1;
  }

  template<typename CrsFlatBase,
           typename CrsHierBase>
  KOKKOS_INLINE_FUNCTION
  int
  CrsMatrixHelper::flat2hier_upper(CrsFlatBase &flat, 
                                   CrsHierBase &hier,
                                   const typename CrsHierBase::ordinal_type       nblks,
                                   const typename CrsHierBase::ordinal_type_array range,
                                   const typename CrsHierBase::ordinal_type_array tree) {
    typedef typename CrsHierBase::ordinal_type            ordinal_type;
    typedef typename CrsHierBase::size_type               size_type;
    
    //typedef typename CrsHierBase::ordinal_type_array     ordinal_type_array;
    //typedef typename CrsHierBase::ordinal_type_array_ptr ordinal_type_array_ptr;
    //typedef typename CrsHierBase::value_type_array_ptr   value_type_array_ptr;
    
    size_type nnz = 0;
    
    // count nnz and nnz in rows for the upper triangular hier matrix
    for (ordinal_type i=0;i<nblks;++i) 
      for (ordinal_type j=i;j != -1;++nnz,j=tree[j]) ;
    
    // create upper triangular block matrix
    hier.createInternalArrays(nblks, nblks, nnz);    

    nnz = 0;
    for (ordinal_type i=0;i<nblks;++i) {
      hier._ap[i] = nnz;
      for (ordinal_type j=i;j != -1;++nnz,j=tree[j]) {
        hier._aj[nnz] = j;
        hier._ax[nnz].setView( flat, range[i], (range[i+1] - range[i]),
                              /**/   range[j], (range[j+1] - range[j]));
        // this checking might more expensive 
        if (!hier._ax[nnz].countNumNonZeros())
          --nnz;
      }
    }
    
    hier._ap[nblks] = nnz;
    hier._nnz = nnz;

    return 0;
  }

  // template<typename CrsFlatBase,
  //          typename CrsHierBase>
  // KOKKOS_INLINE_FUNCTION
  // int
  // CrsMatrixHelper::flat2hier_upper(CrsFlatBase &flat,
  //                                  CrsHierBase &hier,
  //                                  const typename CrsHierBase::ordinal_type       nblks,
  //                                  const typename CrsHierBase::ordinal_type_array range,
  //                                  const typename CrsHierBase::ordinal_type_array tree) {
  //   typedef typename CrsHierBase::ordinal_type            ordinal_type;
  //   typedef typename CrsHierBase::size_type               size_type;

  //   typedef typename CrsHierBase::ordinal_type_array     ordinal_type_array;
  //   //typedef typename CrsHierBase::ordinal_type_array_ptr ordinal_type_array_ptr;
  //   //typedef typename CrsHierBase::value_type_array_ptr   value_type_array_ptr;

  //   ordinal_type_array sibling("CrsMatrixHelper::flat2hier_upper::sibling", nblks);

  //   // check the end of adjacent siblings (if not adjacent, they are separators)
  //   ordinal_type p = tree[0];
  //   for (ordinal_type i=1;i<nblks;++i) {
  //     const ordinal_type j = tree[i];
  //     if (p != j) {
  //       p = j;
  //       sibling[i-1] = -1;
  //     }
  //   }
  //   sibling[nblks-1] = -1;

  //   size_type nnz = 0;

  //   // count nnz and nnz in rows for the upper triangular hier matrix
  //   for (ordinal_type i=0;i<nblks;++i) {                  // search for all rows
  //     for (ordinal_type j=i;j != -1;j=tree[j]) {          // move up
  //       ordinal_type k=j;
  //       do {
  //         ++nnz;
  //       } while (sibling[k++] != -1);
  //     }
  //   }

  //   // create upper triangular block matrix
  //   hier.createInternalArrays(nblks, nblks, nnz);

  //   nnz = 0;
  //   for (ordinal_type i=0;i<nblks;++i) {
  //     hier._ap[i] = nnz;
  //     for (ordinal_type j=i;j != -1;j=tree[j]) {
  //       ordinal_type k=j;
  //       do {
  //         hier._aj[nnz] = k;
  //         hier._ax[nnz].setView( flat, range[i], (range[i+1] - range[i]),
  //                               /**/   range[k], (range[k+1] - range[k]));

  //         // this checking might more expensive
  //         if (hier._ax[nnz].hasNumNonZeros())
  //           ++nnz;
  //       } while (sibling[k++] != -1);
  //     }
  //   }
  //   hier._ap[nblks] = nnz;
  //   hier._nnz = nnz;

  //   return 0;
  // }

  template<typename CrsFlatBase,
           typename CrsHierBase>
  KOKKOS_INLINE_FUNCTION
  int
  CrsMatrixHelper::flat2hier_lower(CrsFlatBase &flat,
                                   CrsHierBase &hier,
                                   const typename CrsHierBase::ordinal_type       nblks,
                                   const typename CrsHierBase::ordinal_type_array range,
                                   const typename CrsHierBase::ordinal_type_array tree) {
    ERROR(MSG_NOT_YET_IMPLEMENTED);

    // typedef typename CrsHierBase::ordinal_type           ordinal_type;
    // typedef typename CrsHierBase::size_type              size_type;

    // typedef typename CrsHierBase::ordinal_type_array     ordinal_type_array;
    // //typedef typename CrsHierBase::ordinal_type_array_ptr ordinal_type_array_ptr;
    // //typedef typename CrsHierBase::value_type_array_ptr   value_type_array_ptr;

    // ordinal_type_array tmp = ordinal_type_array("flat2hier:tmp", nblks+1);
    // size_type nnz = 0;

    // // count nnz and nnz in rows for lower triangular matrix
    // for (ordinal_type i=0;i<nblks;++i)
    //   for (ordinal_type j=i;j != -1;++nnz) {
    //     ++tmp[j];
    //     j = tree[j];
    //   }

    // // count nnz and nnz in rows for lower triangular matrix
    // hier.createInternalArrays(nblks, nblks, nnz);
    // for (ordinal_type i=1;i<(nblks+1);++i)
    //   hier._ap[i] = hier._ap[i-1] + tmp[i-1];

    // for (ordinal_type i=0;i<(nblks+1);++i)
    //   tmp[i] = hier._ap[i];

    // for (ordinal_type i=0;i<nblks;++i)
    //   for (ordinal_type j=i;j != -1;j=tree[j]) {
    //     hier._aj[tmp[j]] = i;
    //     hier._ax[tmp[j]].setView( flat, range[j], (range[j+1] - range[j]),
    //                              /**/   range[i], (range[i+1] - range[i]));
    //     ++tmp[j];
    //   }

    return 0;
  }

}


#endif

