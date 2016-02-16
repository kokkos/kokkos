#pragma once
#ifndef __DENSE_MATRIX_HELPER_HPP__
#define __DENSE_MATRIX_HELPER_HPP__

/// \file dense_matrix_helper.hpp
/// \brief This file includes utility functions to convert between flat and hierarchical matrices.
/// \author Kyungjoo Kim (kyukim@sandia.gov)  

#include "util.hpp"

namespace Tacho { 

  using namespace std;

  class DenseMatrixHelper {
  public:

    /// \brief Transform a scalar flat matrix to hierarchical matrix of matrices 1x1; testing only.
    template<typename DenseFlatBase,
             typename DenseHierBase>
    KOKKOS_INLINE_FUNCTION
    static int
    flat2hier(DenseFlatBase &flat, 
              DenseHierBase &hier);

    /// \brief Transform a scalar flat matrix to hierarchical matrix with given blocksizes.
    template<typename DenseFlatBase,
             typename DenseHierBase>
    KOKKOS_INLINE_FUNCTION
    static int
    flat2hier(DenseFlatBase &flat, 
              DenseHierBase &hier,
              const typename DenseHierBase::ordinal_type mb,
              const typename DenseHierBase::ordinal_type nb);


    /// \brief Transform a scalar flat matrix to hierarchical matrix with given range info.
    template<typename DenseFlatBase,
             typename DenseHierBase>
    KOKKOS_INLINE_FUNCTION
    static int
    flat2hier(DenseFlatBase &flat, 
              DenseHierBase &hier,
              const typename DenseHierBase::ordinal_type       m,
              const typename DenseHierBase::ordinal_type_array offm,
              const typename DenseHierBase::ordinal_type       nb);

    /// \brief Transform a scalar flat matrix to hierarchical matrix with given range info.
    template<typename DenseFlatBase,
             typename DenseHierBase>
    KOKKOS_INLINE_FUNCTION
    static int
    flat2hier(DenseFlatBase &flat, 
              DenseHierBase &hier,
              const typename DenseHierBase::ordinal_type       m,
              const typename DenseHierBase::ordinal_type_array offm,
              const typename DenseHierBase::ordinal_type       n,
              const typename DenseHierBase::ordinal_type_array offn);
  };

}

#include "dense_matrix_helper_impl.hpp"

#endif
