#pragma once
#ifndef __CHOL_U_UNBLOCKED_DUMMY_HPP__
#define __CHOL_U_UNBLOCKED_DUMMY_HPP__

/// \file chol_u_unblocked_opt1.hpp
/// \brief Test code for data parallel interface
/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "util.hpp"
#include "partition.hpp"

namespace Tacho {

  using namespace std;

  template<>
  template<typename CrsExecViewType>
  KOKKOS_INLINE_FUNCTION
  int
  Chol<Uplo::Upper,AlgoChol::Dummy,Variant::One>
  ::invoke(typename CrsExecViewType::policy_type &policy,
           const typename CrsExecViewType::policy_type::member_type &member,
           CrsExecViewType &A) {

    typedef typename CrsExecViewType::value_type        value_type;
    typedef typename CrsExecViewType::ordinal_type      ordinal_type;
    typedef typename CrsExecViewType::row_view_type     row_view_type;

    // row_view_type r1t, r2t;

    for (ordinal_type k=0;k<A.NumRows();++k) {
      //r1t.setView(A, k);
      row_view_type &r1t = A.RowView(k);

      // extract diagonal from alpha11
      value_type &alpha = r1t.Value(0);
      const ordinal_type nnz_r1t = r1t.NumNonZeros();

      if (nnz_r1t) {
        // inverse scale
        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 1, nnz_r1t),
                             [&](const ordinal_type j) {
                               r1t.Value(j) /= alpha;
                             });
      }
    }
    return 0;
  }

}

#endif
