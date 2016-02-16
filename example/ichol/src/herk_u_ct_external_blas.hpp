#pragma once
#ifndef __HERK_U_CT_EXTERNAL_BLAS_HPP__
#define __HERK_U_CT_EXTERNAL_BLAS_HPP__

/// \file herk_u_ct_external_blas.hpp
/// \brief BLAS hermitian rank one update
/// \author Kyungjoo Kim (kyukim@sandia.gov)
#ifdef HAVE_SHYLUTACHO_TEUCHOS
#include "Teuchos_BLAS.hpp"

namespace Tacho {

  using namespace std;

  // BLAS Herk
  // =========
  template<>
  template<typename ScalarType,
           typename DenseExecViewTypeA,
           typename DenseExecViewTypeC>
  KOKKOS_INLINE_FUNCTION
  int
  Herk<Uplo::Upper,Trans::ConjTranspose,
       AlgoHerk::ExternalBlas>
  ::invoke(typename DenseExecViewTypeA::policy_type &policy,
           const typename DenseExecViewTypeA::policy_type::member_type &member,
           const ScalarType alpha,
           DenseExecViewTypeA &A,
           const ScalarType beta,
           DenseExecViewTypeC &C) {
    typedef typename DenseExecViewTypeA::ordinal_type      ordinal_type;
    typedef typename DenseExecViewTypeA::value_type        value_type;

    if (member.team_rank() == 0) {
      Teuchos::BLAS<ordinal_type,value_type> blas;

      // should be square
      const ordinal_type n = C.NumRows();
      const ordinal_type k = A.NumRows();

      blas.HERK(Teuchos::UPPER_TRI, Teuchos::CONJ_TRANS,
                n, k,
                alpha,
                A.ValuePtr(), A.BaseObject()->ColStride(),
                beta,
                C.ValuePtr(), C.BaseObject()->ColStride());
    }
    return 0;
  }

}
#endif
#endif
