#pragma once
#ifndef __GEMM_NT_NT_FOR_TRI_SOLVE_BLOCKED_HPP__
#define __GEMM_NT_NT_FOR_TRI_SOLVE_BLOCKED_HPP__

/// \file gemm_nt_nt_tri_solve_blocked.hpp
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
  Gemm<Trans::NoTranspose,Trans::NoTranspose,
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

    // C(i,j) += alpha*A(i,k)*B(k,j)
    const ordinal_type mA = A.NumRows();
    const ordinal_type nB = B.NumCols();
    if (mA > 0 && nB > 0) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, mA),
                           [&](const ordinal_type i) {
                             row_view_type &a = A.RowView(i);
                             const ordinal_type nnz_a = a.NumNonZeros();

                             for (ordinal_type k=0;k<nnz_a;++k) {
                               for (ordinal_type j=0;j<nB;++j) {
                                 const ordinal_type col_at_ik = a.Col(k);
                                 const value_type   val_at_ik = a.Value(k);
                                 const value_type   val_at_kj = B.Value(col_at_ik, j);

                                 C.Value(i, j) += alpha*val_at_ik*val_at_kj;
                               }
                             }
                           });
      member.team_barrier();
    }

    return 0;
  }

}

#endif
