#pragma once
#ifndef __HERK_U_CT_FOR_FACTOR_BLOCKED_HPP__
#define __HERK_U_CT_FOR_FACTOR_BLOCKED_HPP__

/// \file herk_u_ct_for_factor_blocked.hpp
/// \brief Sparse hermitian rank one update on given sparse patterns.
/// \author Kyungjoo Kim (kyukim@sandia.gov)

namespace Tacho {

  using namespace std;


  // Herk used in the factorization phase
  // ====================================
  template<>
  template<typename ScalarType,
           typename CrsExecViewTypeA,
           typename CrsExecViewTypeC>
  KOKKOS_INLINE_FUNCTION
  int
  Herk<Uplo::Upper,Trans::ConjTranspose,
       AlgoHerk::ForFactorBlocked>
  ::invoke(typename CrsExecViewTypeA::policy_type &policy,
           const typename CrsExecViewTypeA::policy_type::member_type &member,
           const ScalarType alpha,
           CrsExecViewTypeA &A,
           const ScalarType beta,
           CrsExecViewTypeC &C) {
    typedef typename CrsExecViewTypeA::ordinal_type      ordinal_type;
    typedef typename CrsExecViewTypeA::value_type        value_type;
    typedef typename CrsExecViewTypeA::row_view_type     row_view_type;

    // scale the matrix C with beta
    scaleCrsMatrix(member, beta, C);

    // C(i,j) += alpha*A'(i,k)*A(k,j)
    for (ordinal_type k=0;k<A.NumRows();++k) {
      row_view_type &a = A.RowView(k);
      const ordinal_type nnz = a.NumNonZeros();

      if (nnz > 0) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, nnz),
                             [&](const ordinal_type i) {
                               const ordinal_type row_at_i  = a.Col(i);
                               const value_type   val_at_ik = conj(a.Value(i));

                               row_view_type &c = C.RowView(row_at_i);

                               ordinal_type idx = 0;
                               for (ordinal_type j=i;j<nnz && (idx > -2);++j) {
                                 const ordinal_type col_at_j  = a.Col(j);
                                 const value_type   val_at_kj = a.Value(j);

                                 idx = c.Index(col_at_j, idx);
                                 if (idx >= 0)
                                   c.Value(idx) += alpha*val_at_ik*val_at_kj;
                               }
                             });
        member.team_barrier();
      }
    }

    return 0;
  }

}

#endif
