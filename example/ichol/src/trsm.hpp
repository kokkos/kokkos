#pragma once
#ifndef __TRSM_HPP__
#define __TRSM_HPP__

/// \file trsm.hpp
/// \brief Sparse triangular solve on given sparse patterns and multiple rhs.
/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "util.hpp"
#include "control.hpp"
#include "partition.hpp"

namespace Tacho {

  using namespace std;

  template<int ArgSide,int ArgUplo, int ArgTrans, int ArgAlgo,
           int ArgVariant = Variant::One,
           template<int,int> class ControlType = Control>
  struct Trsm {

    // data-parallel interface
    // =======================
    template<typename ScalarType,
             typename ExecViewTypeA,
             typename ExecViewTypeB>
    KOKKOS_INLINE_FUNCTION
    static int invoke(typename ExecViewTypeA::policy_type &policy,
                      const typename ExecViewTypeA::policy_type::member_type &member,
                      const int diagA,
                      const ScalarType alpha,
                      ExecViewTypeA &A,
                      ExecViewTypeB &B);

    // task-data parallel interface
    // ============================
    template<typename ScalarType,
             typename ExecViewTypeA,
             typename ExecViewTypeB>
    class TaskFunctor {
    public:
      typedef typename ExecViewTypeA::policy_type policy_type;
      typedef typename policy_type::member_type member_type;
      typedef int value_type;

    private:
      int _diagA;
      ScalarType _alpha;
      ExecViewTypeA _A;
      ExecViewTypeB _B;

      policy_type &_policy;

    public:
      TaskFunctor(const int diagA,
                  const ScalarType alpha,
                  const ExecViewTypeA A,
                  const ExecViewTypeB B)
        : _diagA(diagA),
          _alpha(alpha),
          _A(A),
          _B(B),
          _policy(ExecViewTypeA::task_factory_type::Policy())
      { }

      string Label() const { return "Trsm"; }

      // task execution
      void apply(value_type &r_val) {
        r_val = Trsm::invoke(_policy, _policy.member_single(),
                             _diagA, _alpha, _A, _B);
      }

      // task-data execution
      void apply(const member_type &member, value_type &r_val) {
        r_val = Trsm::invoke(_policy, member, 
                             _diagA, _alpha, _A, _B);
      }

    };
  };

}

#include "trsm_l_u_nt.hpp"
#include "trsm_l_u_ct.hpp"

#endif
