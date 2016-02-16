#pragma once
#ifndef __HERK_HPP__
#define __HERK_HPP__

/// \file herk.hpp
/// \brief Sparse hermitian rank one update on given sparse patterns.
/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "util.hpp"
#include "control.hpp"
#include "partition.hpp"

namespace Tacho {

  using namespace std;

  template<int ArgUplo, int ArgTrans, int ArgAlgo,
           int ArgVariant = Variant::One,
           template<int,int> class ControlType = Control>
  struct Herk {

    // data-parallel interface
    // =======================
    template<typename ScalarType,
             typename ExecViewTypeA,
             typename ExecViewTypeC>
    KOKKOS_INLINE_FUNCTION
    static int invoke(typename ExecViewTypeA::policy_type &policy,
                      const typename ExecViewTypeA::policy_type::member_type &member,
                      const ScalarType alpha,
                      ExecViewTypeA &A,
                      const ScalarType beta,
                      ExecViewTypeC &C);

    // task-data parallel interface
    // ============================
    template<typename ScalarType,
             typename ExecViewTypeA,
             typename ExecViewTypeC>
    class TaskFunctor {
    public:
      typedef typename ExecViewTypeA::policy_type policy_type;
      typedef typename policy_type::member_type member_type;
      typedef int value_type;

    private:
      ScalarType _alpha, _beta;
      ExecViewTypeA _A;
      ExecViewTypeC _C;

      policy_type &_policy;

    public:
      TaskFunctor(const ScalarType alpha,
                  const ExecViewTypeA A,
                  const ScalarType beta,
                  const ExecViewTypeC C)
        : _alpha(alpha),
          _beta(beta),
          _A(A),
          _C(C),
          _policy(ExecViewTypeA::task_factory_type::Policy())
      { }

      string Label() const { return "Herk"; }

      // task execution
      void apply(value_type &r_val) {
        r_val = Herk::invoke(_policy, _policy.member_single(), 
                             _alpha, _A, _beta, _C);
      }

      // task-data execution
      void apply(const member_type &member, value_type &r_val) {
        r_val = Herk::invoke(_policy, member, 
                             _alpha, _A, _beta, _C);
      }

    };

  };

}

#include "herk_u_ct.hpp"

#endif
