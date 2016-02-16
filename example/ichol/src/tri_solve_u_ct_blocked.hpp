#pragma once
#ifndef __TRI_SOLVE_U_CT_BLOCKED_HPP__
#define __TRI_SOLVE_U_CT_BLOCKED_HPP__

/// \file tri_solve_u_ct_blocked.hpp
/// \brief Sparse triangular solve on given sparse patterns and multiple rhs.
/// \author Kyungjoo Kim (kyukim@sandia.gov)
///

namespace Tacho {

  using namespace std;

  template<>
  template<typename CrsExecViewTypeA,
           typename DenseExecViewTypeB>
  KOKKOS_INLINE_FUNCTION
  int
  TriSolve<Uplo::Upper,Trans::ConjTranspose,AlgoTriSolve::Blocked>
  ::invoke(typename CrsExecViewTypeA::policy_type &policy,
           const typename CrsExecViewTypeA::policy_type::member_type &member,
           const int diagA,
           CrsExecViewTypeA &A,
           DenseExecViewTypeB &B) {
    typedef typename CrsExecViewTypeA::ordinal_type ordinal_type;
    const ordinal_type mb = blocksize;

    CrsExecViewTypeA ATL, ATR,      A00, A01, A02,
      /**/           ABL, ABR,      A10, A11, A12,
      /**/                          A20, A21, A22;

    DenseExecViewTypeB BT,      B0,
      /**/             BB,      B1,
      /**/                      B2;

    Part_2x2(A,  ATL, ATR,
             /**/ABL, ABR,
             0, 0, Partition::TopLeft);

    Part_2x1(B,  BT,
             /**/BB,
             0, Partition::Top);

    while (ATL.NumRows() < A.NumRows()) {
      Part_2x2_to_3x3(ATL, ATR, /**/  A00, A01, A02,
                      /*******/ /**/  A10, A11, A12,
                      ABL, ABR, /**/  A20, A21, A22,
                      mb, mb, Partition::BottomRight);

      Part_2x1_to_3x1(BT,  /**/  B0,
                      /**/ /**/  B1,
                      BB,  /**/  B2,
                      mb, Partition::Bottom);

      // -----------------------------------------------------
      A11.fillRowViewArray();
      A12.fillRowViewArray();

      // B1 = inv(triu(A11))*B1
      Trsm<Side::Left,Uplo::Upper,Trans::ConjTranspose,AlgoTrsm::ForTriSolveBlocked>
        ::invoke(policy, member, diagA, 1.0, A11, B1);

      // B2 = B2 - A12'*B1
      Gemm<Trans::ConjTranspose,Trans::NoTranspose,AlgoGemm::ForTriSolveBlocked>
        ::invoke(policy, member, -1.0, A12, B1, 1.0, B2);

      // -----------------------------------------------------
      Merge_3x3_to_2x2(A00, A01, A02, /**/ ATL, ATR,
                       A10, A11, A12, /**/ /******/
                       A20, A21, A22, /**/ ABL, ABR,
                       Partition::TopLeft);

      Merge_3x1_to_2x1(B0, /**/   BT,
                       B1, /**/  /**/
                       B2, /**/   BB,
                       Partition::Top);
    }

    return 0;
  }

}

#endif
