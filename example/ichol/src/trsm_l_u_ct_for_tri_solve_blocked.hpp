#pragma once
#ifndef __TRSM_L_U_CT_TRI_SOLVE_BLOCKED_HPP__
#define __TRSM_L_U_CT_TRI_SOLVE_BLOCKED_HPP__

/// \file trsm_l_u_ct_tri_solve_blocked.hpp
/// \brief Sparse triangular solve on given sparse patterns and multiple rhs.
/// \author Kyungjoo Kim (kyukim@sandia.gov)
///

namespace Tacho {

  using namespace std;

  // Trsm used in the tri-solve phase: Multiple RHS
  // ==============================================
  template<>
  template<typename ScalarType,
           typename CrsExecViewTypeA,
           typename DenseExecViewTypeB>
  KOKKOS_INLINE_FUNCTION
  int
  Trsm<Side::Left,Uplo::Upper,Trans::ConjTranspose,
       AlgoTrsm::ForTriSolveBlocked,Variant::One>
  ::invoke(typename CrsExecViewTypeA::policy_type &policy,
           const typename CrsExecViewTypeA::policy_type::member_type &member,
           const int diagA,
           const ScalarType alpha,
           CrsExecViewTypeA &A,
           DenseExecViewTypeB &B) {
    typedef typename CrsExecViewTypeA::ordinal_type      ordinal_type;
    typedef typename CrsExecViewTypeA::value_type        value_type;
    typedef typename CrsExecViewTypeA::row_view_type     row_view_type;

    // scale the matrix B with alpha
    scaleDenseMatrix(member, alpha, B);

    // Solve a system: AX = B -> B := inv(A) B
    const ordinal_type mA = A.NumRows();
    const ordinal_type nB = B.NumCols();

    if (nB > 0) {
      for (ordinal_type k=0;k<mA;++k) {
        row_view_type &a = A.RowView(k);
        const value_type cdiag = conj(a.Value(0));

        // invert
        if (diagA != Diag::Unit) {
          // b1t = b1t / conj(diag);
          Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, nB),
                               [&](const ordinal_type j) {
                                 B.Value(k, j) /= cdiag;
                               });
        }

        // update
        const ordinal_type nnz_a = a.NumNonZeros();
        if (nnz_a > 0) {
          // B2 = B2 - trans(conj(a12t)) b1t
          Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, nB),
                               [&](const ordinal_type j) {
                                 // grab b1t
                                 const value_type val_at_j = B.Value(k, j);
                            
                                 for (ordinal_type i=1;i<nnz_a;++i) {
                                   // grab a12t
                                   const ordinal_type row_at_i = a.Col(i);
                                   const value_type   val_at_i = conj(a.Value(i));
                              
                                   // update B2
                                   B.Value(row_at_i, j) -= val_at_i*val_at_j;
                                 }
                               });
        }
        member.team_barrier();
      }
    }

    return 0;
  }


  // Trsm used in the tri-solve phase: Single RHS
  // ============================================
  template<>
  template<typename ScalarType,
           typename CrsExecViewTypeA,
           typename DenseExecViewTypeB>
  KOKKOS_INLINE_FUNCTION
  int
  Trsm<Side::Left,Uplo::Upper,Trans::ConjTranspose,
       AlgoTrsm::ForTriSolveBlocked,Variant::Two>
  ::invoke(typename CrsExecViewTypeA::policy_type &policy,
           const typename CrsExecViewTypeA::policy_type::member_type &member,
           const int diagA,
           const ScalarType alpha,
           CrsExecViewTypeA &A,
           DenseExecViewTypeB &B) {
    typedef typename CrsExecViewTypeA::ordinal_type      ordinal_type;
    typedef typename CrsExecViewTypeA::value_type        value_type;
    typedef typename CrsExecViewTypeA::row_view_type     row_view_type;

    // scale the matrix B with alpha
    scaleDenseMatrix(member, alpha, B);

    // Solve a system: AX = B -> B := inv(A) B
    const ordinal_type mA = A.NumRows();
    const ordinal_type nB = B.NumCols();

    if (nB > 0) {
      for (ordinal_type k=0;k<mA;++k) {
        row_view_type &a = A.RowView(k);
        const value_type cdiag = conj(a.Value(0));

        // invert
        if (diagA != Diag::Unit) {
          // b1t = b1t / conj(diag);
          Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, nB),
                               [&](const ordinal_type j) {
                                 B.Value(k, j) /= cdiag;
                               });
          member.team_barrier();
        }

        // update
        const ordinal_type nnz_a = a.NumNonZeros();
        if (nnz_a > 0) {
          // B2 = B2 - trans(conj(a12t)) b1t
          Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 1, nnz_a),
                               [&](const ordinal_type i) {
                                 // grab a12t
                                 const ordinal_type row_at_i = a.Col(i);
                                 const value_type   val_at_i = conj(a.Value(i));

                                 for (ordinal_type j=0;j<nB;++j) {
                                   // grab b1t
                                   const ordinal_type col_at_j = j;
                                   const value_type   val_at_j = B.Value(k, j);

                                   // update B2
                                   B.Value(row_at_i, col_at_j) -= val_at_i*val_at_j;
                                 }
                               });
          member.team_barrier();
        }
      }
    }
    return 0;
  }
  
}

#endif
