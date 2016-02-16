#pragma once
#ifndef __GEMM_NT_NT_EXTERNAL_BLAS_HPP__
#define __GEMM_NT_NT_EXTERNAL_BLAS_HPP__

/// \file gemm_nt_nt_external_blas.hpp
/// \brief BLAS matrix-matrix multiplication 
/// \author Kyungjoo Kim (kyukim@sandia.gov)
#ifdef HAVE_SHYLUTACHO_TEUCHOS
#include "Teuchos_BLAS.hpp"

namespace Tacho {

  using namespace std;

  // BLAS Gemm
  // =========
  template<>
  template<typename ScalarType,
           typename DenseExecViewTypeA,
           typename DenseExecViewTypeB,
           typename DenseExecViewTypeC>
  KOKKOS_INLINE_FUNCTION
  int
  Gemm<Trans::NoTranspose,Trans::NoTranspose,
       AlgoGemm::ExternalBlas>
  ::invoke(typename DenseExecViewTypeA::policy_type &policy,
           const typename DenseExecViewTypeA::policy_type::member_type &member,
           const ScalarType alpha,
           DenseExecViewTypeA &A,
           DenseExecViewTypeB &B,
           const ScalarType beta,
           DenseExecViewTypeC &C) {
    typedef typename DenseExecViewTypeA::ordinal_type ordinal_type;
    typedef typename DenseExecViewTypeA::value_type   value_type;

    if (member.team_rank() == 0) {
      Teuchos::BLAS<ordinal_type,value_type> blas;

      const ordinal_type m = C.NumRows();
      const ordinal_type n = C.NumCols();
      const ordinal_type k = B.NumRows();

      blas.GEMM(Teuchos::NO_TRANS, Teuchos::NO_TRANS,
                m, n, k,
                value_type(alpha),
                A.ValuePtr(), A.BaseObject()->ColStride(),
                B.ValuePtr(), B.BaseObject()->ColStride(),
                value_type(beta),
                C.ValuePtr(), C.BaseObject()->ColStride());
    }
    return 0;
  }

}

#endif
#endif
