#pragma once
#ifndef __CHOL_U_BLOCKED_HPP__
#define __CHOL_U_BLOCKED_HPP__

/// \file chol_u_blocked.hpp
/// \brief Blocked incomplete Chloesky factorization.
/// \author Kyungjoo Kim (kyukim@sandia.gov)
///
/// Unlike the dense matrix algebra, this blocked version of sparse
/// factorization does not lead to an efficient computation.
/// This algorithm is only for the testing and debugging purpose.

namespace Tacho {

  using namespace std;

  template<template<int> class ControlType,
           typename CrsExecViewType>
  class FunctionChol<Uplo::Upper,AlgoChol::Blocked,
                     ControlType,
                     CrsExecViewType> {
  private:
    int _r_val;
    
  public:
    KOKKOS_INLINE_FUNCTION
    int getReturnValue() const {
      return _r_val;
    }
    
    KOKKOS_INLINE_FUNCTION
    FunctionChol(typename CrsExecViewType::policy_type &policy,
                 const typename CrsExecViewType::policy_type::member_type &member,
                 CrsExecViewType &A) {
      typedef typename CrsExecViewType::ordinal_type ordinal_type;
      const ordinal_type mb = 32; // getBlocksize<AlgoChol::Blocked>();
      
      CrsExecViewType ATL, ATR,      A00, A01, A02,
        /**/          ABL, ABR,      A10, A11, A12,
        /**/                         A20, A21, A22;
      
      Part_2x2(A,  ATL, ATR,
               /**/ABL, ABR,
               0, 0, Partition::TopLeft);

      while (ATL.NumRows() < A.NumRows()) {
        Part_2x2_to_3x3(ATL, ATR, /**/  A00, A01, A02,
                        /*******/ /**/  A10, A11, A12,
                        ABL, ABR, /**/  A20, A21, A22,
                        mb, mb, Partition::BottomRight);
        // -----------------------------------------------------
        A11.fillRowViewArray();
        A12.fillRowViewArray();      
        A22.fillRowViewArray();      

        _r_val = Chol<Uplo::Upper,AlgoChol::UnblockedOpt1>
          ::invoke(policy, member, A11);

        if (_r_val) {
          _r_val += A00.NumRows();
          return;
        }

        Trsm<Side::Left,Uplo::Upper,Trans::ConjTranspose,AlgoTrsm::ForFactorBlocked>
          ::invoke(policy, member, Diag::NonUnit, 1.0, A11, A12);

        Herk<Uplo::Upper,Trans::ConjTranspose,AlgoHerk::ForFactorBlocked>
          ::invoke(policy, member, -1.0, A12, 1.0, A22);

        // -----------------------------------------------------
        Merge_3x3_to_2x2(A00, A01, A02, /**/ ATL, ATR,
                         A10, A11, A12, /**/ /******/
                         A20, A21, A22, /**/ ABL, ABR,
                         Partition::TopLeft);
      }

      _r_val = 0;
    }
  };
}

#endif
