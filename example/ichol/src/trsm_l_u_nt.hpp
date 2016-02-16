#pragma once
#ifndef __TRSM_L_U_NT_HPP__
#define __TRSM_L_U_NT_HPP__

/// \file trsm_l_u_nt.hpp
/// \brief Sparse triangular solve on given sparse patterns and multiple rhs.
/// \author Kyungjoo Kim (kyukim@sandia.gov)
///

namespace Tacho {

  using namespace std;

  // Trsm used in the tri-solve phase
  // ================================
  template<>
  template<typename ScalarType,
           typename CrsExecViewTypeA,
           typename DenseExecViewTypeB>
  KOKKOS_INLINE_FUNCTION
  int
  Trsm<Side::Left,Uplo::Upper,Trans::NoTranspose,
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
      for (ordinal_type k=mA-1;k>=0;--k) {
        row_view_type &a = A.RowView(k);
        const value_type diag = a.Value(0);
        
        // update
        const ordinal_type nnz_a = a.NumNonZeros();
        if (nnz_a > 0) {
          // b1t = b1t - a12t B2 
          Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, nB),
                               [&](const ordinal_type j) {
                                 for (ordinal_type i=1;i<nnz_a;++i) {
                                   const ordinal_type row_at_i = a.Col(i);   // grab B2 row
                                   const value_type   val_at_i = a.Value(i); // grab a12t value
                              
                                   // update b1t
                                   B.Value(k, j) -= val_at_i*B.Value(row_at_i, j);
                                 }
                               });
          
          // invert
          if (diagA != Diag::Unit) {
            // b1t = b1t / diag
            Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, nB),
                                 [&](const ordinal_type j) {
                                   B.Value(k, j) /= diag;
                                 });

          }
          member.team_barrier();
        }
      }
    }

    return 0;
  }

}

#endif
