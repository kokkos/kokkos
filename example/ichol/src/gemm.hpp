#pragma once
#ifndef __GEMM_HPP__
#define __GEMM_HPP__

/// \file gemm.hpp
/// \brief Sparse matrix-matrix multiplication on given sparse patterns.
/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "util.hpp"
#include "control.hpp"
#include "partition.hpp"

namespace Tacho {

  using namespace std;

  template<int ArgTransA, int ArgTransB, int ArgAlgo,
           int ArgVariant = Variant::One,
           template<int,int> class ControlType = Control>
  struct Gemm {

    // data-parallel interface
    // =======================
    template<typename ScalarType,
             typename ExecViewTypeA,
             typename ExecViewTypeB,
             typename ExecViewTypeC>
    KOKKOS_INLINE_FUNCTION
    static int invoke(typename ExecViewTypeA::policy_type &policy,
                      const typename ExecViewTypeA::policy_type::member_type &member,
                      const ScalarType alpha,
                      ExecViewTypeA &A,
                      ExecViewTypeB &B,
                      const ScalarType beta,
                      ExecViewTypeC &C);

    // task-data parallel interface
    // ============================
    template<typename ScalarType,
             typename ExecViewTypeA,
             typename ExecViewTypeB,
             typename ExecViewTypeC>
    class TaskFunctor {
    public:
      typedef typename ExecViewTypeA::policy_type policy_type;
      typedef typename policy_type::member_type member_type;
      typedef int value_type;

    private:
      ScalarType _alpha, _beta;
      ExecViewTypeA _A;
      ExecViewTypeB _B;
      ExecViewTypeC _C;

      policy_type &_policy;

    public:
      TaskFunctor(const ScalarType alpha,
                  const ExecViewTypeA A,
                  const ExecViewTypeB B,
                  const ScalarType beta,
                  const ExecViewTypeC C)
        : _alpha(alpha),
          _beta(beta),
          _A(A),
          _B(B),
          _C(C),
          _policy(ExecViewTypeA::task_factory_type::Policy())
      { }

      string Label() const { return "Gemm"; }

      // task execution
      void apply(value_type &r_val) {
        r_val = Gemm::invoke(_policy, _policy.member_single(),
                             _alpha, _A, _B, _beta, _C);
      }

      // task-data execution
      void apply(const member_type &member, value_type &r_val) {
        r_val = Gemm::invoke(_policy, member,
                             _alpha, _A, _B, _beta, _C);
      }

    };

  };

}


#include "gemm_nt_nt.hpp"
#include "gemm_ct_nt.hpp"

#endif
