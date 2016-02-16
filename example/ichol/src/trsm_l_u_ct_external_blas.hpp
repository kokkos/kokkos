#pragma once
#ifndef __TRSM_L_U_CT_EXTERNAL_BLAS_HPP__
#define __TRSM_L_U_CT_EXTERNAL_BLAS_HPP__

/// \file trsm_l_u_ct_external_blas.hpp
/// \brief Sparse triangular solve on given sparse patterns and multiple rhs.
/// \author Kyungjoo Kim (kyukim@sandia.gov)
///
#ifdef HAVE_SHYLUTACHO_TEUCHOS
#include "Teuchos_BLAS.hpp"

namespace Tacho {

  using namespace std;

  // Trsm used in the factorization phase: data parallel on b1t
  // ==========================================================
  template<>
  template<typename ScalarType,
           typename DenseExecViewTypeA,
           typename DenseExecViewTypeB>
  KOKKOS_INLINE_FUNCTION
  int
  Trsm<Side::Left,Uplo::Upper,Trans::ConjTranspose,
       AlgoTrsm::ExternalBlas>
  ::invoke(typename DenseExecViewTypeA::policy_type &policy,
           const typename DenseExecViewTypeA::policy_type::member_type &member,
           const int diagA,
           const ScalarType alpha,
           DenseExecViewTypeA &A,
           DenseExecViewTypeB &B) {
    typedef typename DenseExecViewTypeA::ordinal_type      ordinal_type;
    typedef typename DenseExecViewTypeA::value_type        value_type;

    if (member.team_rank() == 0) {
      Teuchos::BLAS<ordinal_type,value_type> blas;

      const ordinal_type m = A.NumRows();
      const ordinal_type n = B.NumCols();

      blas.TRSM(Teuchos::LEFT_SIDE, Teuchos::UPPER_TRI, Teuchos::CONJ_TRANS,
                (diagA == Diag::Unit ? Teuchos::UNIT_DIAG : Teuchos::NON_UNIT_DIAG),
                m, n, 
                alpha,
                A.ValuePtr(), A.BaseObject()->ColStride(),
                B.ValuePtr(), B.BaseObject()->ColStride());
    }
    return 0;
  }

}

#endif
#endif
