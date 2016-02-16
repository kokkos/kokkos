#pragma once
#ifndef __CHOL_U_NESTED_DENSE_BLOCK_HPP__
#define __CHOL_U_NESTED_DENSE_BLOCK_HPP__

/// \file chol_u_nested_dense_block
/// \brief use a nested dense block (not hierarchical).
/// \author Kyungjoo Kim (kyukim@sandia.gov)

// basic utils
#include "util.hpp"
#include "control.hpp"
#include "partition.hpp"

namespace Tacho {

  using namespace std;

  // specialization for different task generation in right looking by-blocks algorithm
  // =================================================================================
  template<int ArgVariant, template<int,int> class ControlType>
  class Chol<Uplo::Upper,AlgoChol::NestedDenseBlock,ArgVariant,ControlType> {
  public:

    // function interface
    // ==================
    template<typename ExecViewType>
    KOKKOS_INLINE_FUNCTION
    static int invoke(typename ExecViewType::policy_type &policy,
                      const typename ExecViewType::policy_type::member_type &member,
                      ExecViewType &A) {
      typedef typename ExecViewType::dense_flat_view_type dense_flat_view_type;

      int r_val = 0;
      if (member.team_rank() == 0) {
        // need size threshold here or when the dense block is created
        if (A.copyToDenseFlatBase() == 0) {
          auto D = dense_flat_view_type(A.DenseFlatBaseObject());
          r_val = Chol<Uplo::Upper,CtrlDetail(ControlType,AlgoChol::NestedDenseBlock,ArgVariant,CholDense)>
            ::invoke(policy, member, D);
          A.copyToCrsMatrixView();
        } else {
          r_val = Chol<Uplo::Upper,CtrlDetail(ControlType,AlgoChol::NestedDenseBlock,ArgVariant,CholSparse)>
            ::invoke(policy, member, A);
        }
      }

      return r_val;
    }

    // task-data parallel interface
    // ============================
    template<typename ExecViewType>
    class TaskFunctor {
    public:
      typedef typename ExecViewType::policy_type policy_type;
      typedef typename policy_type::member_type member_type;

      typedef typename ExecViewType::task_factory_type task_factory_type;

      typedef int value_type;

    private:
      ExecViewType _A;

      policy_type &_policy;

    public:
      TaskFunctor(const ExecViewType A)
        : _A(A),
          _policy(task_factory_type::Policy())
      { }

      string Label() const { return "Chol"; }

      // task execution
      void apply(value_type &r_val) {
        r_val = Chol::invoke(_policy, _policy.member_single(), _A);
      }

      // task-data execution
      void apply(const member_type &member, value_type &r_val) {
        r_val = Chol::invoke(_policy, member, _A);
      }

    };

  };
}

#endif
