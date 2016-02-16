#pragma once
#ifndef __GEMM_CT_NT_TRI_SOLVE_BLOCKED_HPP__
#define __GEMM_CT_NT_TRI_SOLVE_BLOCKED_HPP__

/// \file gemm_ct_nt_tri_solve_blocked.hpp
/// \brief Sparse matrix-matrix multiplication on given sparse patterns.
/// \author Kyungjoo Kim (kyukim@sandia.gov)

namespace Tacho {

  using namespace std;

  // Gemm used in the tri-solve phase
  // ================================
  template<>
  template<typename ScalarType,
           typename CrsExecViewTypeA,
           typename DenseExecViewTypeB,
           typename DenseExecViewTypeC>
  KOKKOS_INLINE_FUNCTION
  int
  Gemm<Trans::ConjTranspose,Trans::NoTranspose,
       AlgoGemm::ForTriSolveBlocked>
  ::invoke(typename CrsExecViewTypeA::policy_type &policy,
           const typename CrsExecViewTypeA::policy_type::member_type &member,
           const ScalarType alpha,
           CrsExecViewTypeA &A,
           DenseExecViewTypeB &B,
           const ScalarType beta,
           DenseExecViewTypeC &C) {
    typedef typename CrsExecViewTypeA::ordinal_type      ordinal_type;
    typedef typename CrsExecViewTypeA::value_type        value_type;
    typedef typename CrsExecViewTypeA::row_view_type     row_view_type;

    // scale the matrix C with beta
    scaleDenseMatrix(member, beta, C);

    // C(i,j) += alpha*A'(i,k)*B(k,j)
    const ordinal_type mA = A.NumRows();
    for (ordinal_type k=0;k<mA;++k) {
      row_view_type &a = A.RowView(k);
      const ordinal_type nnz_a = a.NumNonZeros();
      const ordinal_type nB = B.NumCols();

      if (nnz_a > 0 && nB > 0) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, nnz_a),
                             [&](const ordinal_type i) {
                               const ordinal_type row_at_i = a.Col(i);
                               const value_type   val_at_ik = conj(a.Value(i));

                               for (ordinal_type j=0;j<nB;++j) {
                                 const value_type val_at_kj = B.Value(k, j);
                                 C.Value(row_at_i, j) += alpha*val_at_ik*val_at_kj;
                               }
                             });
        member.team_barrier();
      }
    }

    return 0;
  }

}

#endif
