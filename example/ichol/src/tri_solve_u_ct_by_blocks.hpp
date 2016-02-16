#pragma once
#ifndef __TRI_SOLVE_U_CT_BY_BLOCKS_HPP__
#define __TRI_SOLVE_U_CT_BY_BLOCKS_HPP__

/// \file tri_solve_u_ct_by_blocks.hpp
/// \brief Sparse triangular solve on given sparse patterns and multiple rhs.
/// \author Kyungjoo Kim (kyukim@sandia.gov)
///

namespace Tacho {

  using namespace std;

  template<typename CrsTaskViewTypeA,
           typename DenseTaskViewTypeB>
  KOKKOS_INLINE_FUNCTION
  static int genTrsmTasks_TriSolveUpperConjTransposeByBlocks(typename CrsTaskViewTypeA::policy_type &policy,
                                                             const int diagA,
                                                             CrsTaskViewTypeA &A, 
                                                             DenseTaskViewTypeB &B);

  template<typename CrsTaskViewTypeA,
           typename DenseTaskViewTypeB,
           typename DenseTaskViewTypeC>
  KOKKOS_INLINE_FUNCTION
  static int genGemmTasks_TriSolveUpperConjTransposeByBlocks(typename CrsTaskViewTypeA::policy_type &policy,
                                                             CrsTaskViewTypeA &A, 
                                                             DenseTaskViewTypeB &B, 
                                                             DenseTaskViewTypeC &C);
  
  template<>
  template<typename CrsTaskViewTypeA,
           typename DenseTaskViewTypeB>
  KOKKOS_INLINE_FUNCTION
  int
  TriSolve<Uplo::Upper,Trans::ConjTranspose,AlgoTriSolve::ByBlocks>
  ::invoke(typename CrsTaskViewTypeA::policy_type &policy,
           const typename CrsTaskViewTypeA::policy_type::member_type &member,
           const int diagA,
           CrsTaskViewTypeA &A,
           DenseTaskViewTypeB &B) {
    // this task generation should be done by a root
    // ---------------------------------------------
    if (member.team_rank() == 0) {
      CrsTaskViewTypeA ATL, ATR,      A00, A01, A02,
        /**/           ABL, ABR,      A10, A11, A12,
        /**/                          A20, A21, A22;

      DenseTaskViewTypeB BT,      B0,
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
                        1, 1, Partition::BottomRight);

        Part_2x1_to_3x1(BT,  /**/  B0,
                        /**/ /**/  B1,
                        BB,  /**/  B2,
                        1, Partition::Bottom);

        // -----------------------------------------------------

        // B1 = inv(triu(A11))*B1
        genTrsmTasks_TriSolveUpperConjTransposeByBlocks(policy, diagA, A11, B1);
        
        // B2 = B2 - A12'*B1
        genGemmTasks_TriSolveUpperConjTransposeByBlocks(policy, A12, B1, B2);

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
    }
    return 0;
  }

}

#include "tri_solve_u_ct_by_blocks_var1.hpp"

#endif
