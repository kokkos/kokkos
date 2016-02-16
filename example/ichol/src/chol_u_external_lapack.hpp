#pragma once
#ifndef __CHOL_U_EXTERNAL_LAPACK_HPP__
#define __CHOL_U_EXTERNAL_LAPACK_HPP__

/// \file chol_u_external_lapack.hpp
/// \brief BLAS Chloesky factorization.
/// \author Kyungjoo Kim (kyukim@sandia.gov)
#ifdef HAVE_SHYLUTACHO_TEUCHOS
#include "Teuchos_LAPACK.hpp"

namespace Tacho {

  using namespace std;

  template<>
  template<typename DenseExecViewType>
  KOKKOS_INLINE_FUNCTION
  int
  Chol<Uplo::Upper,AlgoChol::ExternalLapack>
  ::invoke(typename DenseExecViewType::policy_type &policy,
           const typename DenseExecViewType::policy_type::member_type &member,
           DenseExecViewType &A) {
    typedef typename DenseExecViewType::ordinal_type ordinal_type;
    typedef typename DenseExecViewType::value_type   value_type;

    int r_val = 0;
    if (member.team_rank() == 0) {
      Teuchos::LAPACK<ordinal_type,value_type> lapack;
      lapack.POTRF('U', 
                   A.NumRows(), 
                   A.ValuePtr(), A.BaseObject()->ColStride(),
                   &r_val);
    }

    return r_val;
  }

}

#endif
#endif
