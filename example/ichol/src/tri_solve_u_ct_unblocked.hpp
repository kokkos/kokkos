#pragma once
#ifndef __TRI_SOLVE_U_CT_UNBLOCKED_HPP__
#define __TRI_SOLVE_U_CT_UNBLOCKED_HPP__

/// \file tri_solve_u_ct_unblocked.hpp
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
  TriSolve<Uplo::Upper,Trans::ConjTranspose,AlgoTriSolve::Unblocked>
  ::invoke(typename CrsExecViewTypeA::policy_type &policy,
           const typename CrsExecViewTypeA::policy_type::member_type &member,
           const int diagA,
           CrsExecViewTypeA &A,
           DenseExecViewTypeB &B) {
    return Trsm<Side::Left,Uplo::Upper,Trans::ConjTranspose,AlgoTrsm::ForTriSolveBlocked>
      ::invoke(policy, member, diagA, 1.0, A, B);
  }

}

#endif
