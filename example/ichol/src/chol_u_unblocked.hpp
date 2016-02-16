#pragma once
#ifndef __CHOL_U_UNBLOCKED_HPP__
#define __CHOL_U_UNBLOCKED_HPP__

/// \file chol_u_unblocked.hpp
/// \brief Unblocked incomplete Chloesky factorization.
/// \author Kyungjoo Kim (kyukim@sandia.gov)

namespace Tacho {

  using namespace std;

  template<typename ScalarType,
           typename CrsExecViewType>
  KOKKOS_INLINE_FUNCTION
  int
  her_r(const typename CrsExecViewType::policy_type::member_type &member,
        const ScalarType alpha,
        CrsExecViewType &x,
        CrsExecViewType &A);

  template<>
  template<typename CrsExecViewType>
  KOKKOS_INLINE_FUNCTION
  int
  Chol<Uplo::Upper,AlgoChol::Unblocked>
  ::invoke(typename CrsExecViewType::policy_type &policy,
           const typename CrsExecViewType::policy_type::member_type &member,
           CrsExecViewType &A) {
    typedef typename CrsExecViewType::value_type        value_type;
    typedef typename CrsExecViewType::ordinal_type      ordinal_type;
    typedef typename CrsExecViewType::row_view_type     row_view_type;

    CrsExecViewType ATL, ATR,      A00,  a01,     A02,
      /**/          ABL, ABR,      a10t, alpha11, a12t,
      /**/                         A20,  a21,     A22;

    Part_2x2(A,   ATL, ATR,
             /**/ ABL, ABR,
             0, 0, Partition::TopLeft);

    value_type zero = 0.0;
    row_view_type alpha, r12t;

    while (ATL.NumRows() < A.NumRows()) {
      Part_2x2_to_3x3(ATL, ATR, /**/  A00,  a01,     A02,
                      /*******/ /**/  a10t, alpha11, a12t,
                      ABL, ABR, /**/  A20,  a21,     A22,
                      1, 1, Partition::BottomRight);
      // -----------------------------------------------------

      // extract diagonal from alpha11
      alpha.setView(alpha11, 0);
      value_type &alpha_val = (alpha.Col(0) ? zero : alpha.Value(0));

      if (member.team_rank() == 0) {
        // if encounter null diag, return -(row + 1)
        if (abs(alpha_val) == 0.0)
          return -(ATL.NumRows() + 1);

        // sqrt on diag
        alpha_val = sqrt(real(alpha_val));
      }

      // sparse inverse scale
      scaleCrsMatrix(member, 1.0/real(alpha_val), a12t);

      // hermitian rank update
      her_r(member, -1.0, a12t, A22);

      // -----------------------------------------------------
      Merge_3x3_to_2x2(A00,  a01,     A02,  /**/ ATL, ATR,
                       a10t, alpha11, a12t, /**/ /******/
                       A20,  a21,     A22,  /**/ ABL, ABR,
                       Partition::TopLeft);
    }

    return 0;
  }

  template<typename ScalarType,
           typename CrsExecViewType>
  KOKKOS_INLINE_FUNCTION
  int
  her_r(const typename CrsExecViewType::policy_type::member_type &member,
        const ScalarType alpha,
        CrsExecViewType &x,
        CrsExecViewType &A) {
    typedef typename CrsExecViewType::ordinal_type      ordinal_type;
    typedef typename CrsExecViewType::value_type        value_type;
    typedef typename CrsExecViewType::row_view_type     row_view_type;

    // input x is row vector

    row_view_type rx, ra;

    rx.setView(x, 0);
    ordinal_type nnz = rx.NumNonZeros();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, nnz),
                         [&](const ordinal_type i) {
                           const ordinal_type row_at_i = rx.Col(i);
                           const value_type   val_at_i = rx.Value(i);

                           ra.setView(A, row_at_i);
                           ordinal_type prev = 0;

                           for (ordinal_type j=0;j<nnz;++j) {
                             ordinal_type col_at_j = rx.Col(j);
                             value_type   val_at_j = rx.Value(j);

                             ordinal_type idx = ra.Index(col_at_j, prev);
                             if (idx >= 0) {
                               ra.Value(idx) += alpha*val_at_i*conj(val_at_j);
                               prev = idx;
                             }
                           }
                         });

    return 0;
  }

}

#endif
