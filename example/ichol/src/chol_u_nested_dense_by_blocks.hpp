#pragma once
#ifndef __CHOL_U_NESTED_DENSE_BY_BLOCKS_HPP__
#define __CHOL_U_NESTED_DENSE_BY_BLOCKS_HPP__

/// \file chol_u_nested_dense_by_blocks.hpp
/// \brief use a nested dense by-block (hierarchical).
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
  class Chol<Uplo::Upper,AlgoChol::NestedDenseByBlocks,ArgVariant,ControlType> {
  public:

    // function interface
    // ==================
    template<typename ExecViewType>
    KOKKOS_INLINE_FUNCTION
    static int invoke(typename ExecViewType::policy_type &policy,
                      const typename ExecViewType::policy_type::member_type &member,
                      ExecViewType &A, 
                      bool &respawn,
                      typename ExecViewType::future_type &dependence) {
      typedef typename ExecViewType::dense_hier_view_type dense_hier_view_type;

      int r_val = 0;
      if (member.team_rank() == 0) {
        if (!respawn) {
          if (A.copyToDenseFlatBase() == 0) {
            auto H = dense_hier_view_type(A.DenseHierBaseObject());
            r_val = Chol<Uplo::Upper,CtrlDetail(ControlType,AlgoChol::NestedDenseByBlocks,ArgVariant,CholDenseByBlocks)>
              ::invoke(policy, member, H);
            respawn = true;
            dependence = H.Value(H.NumRows()-1, H.NumCols()-1).Future();
          } else {
            r_val = Chol<Uplo::Upper,CtrlDetail(ControlType,AlgoChol::NestedDenseBlock,ArgVariant,CholSparse)>
              ::invoke(policy, member, A);
          }
        } else {
          A.copyToCrsMatrixView();
          respawn = false;
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
      bool _respawn; 

    public:
      TaskFunctor(const ExecViewType A)
        : _A(A),
          _policy(task_factory_type::Policy()),
          _respawn(false)
      { }

      string Label() const { return "Chol"; }

      // task execution
      void apply(value_type &r_val) {
        typename ExecViewType::future_type dependence;
        r_val = Chol::invoke(_policy, _policy.member_single(), 
                             _A, 
                             _respawn, dependence);
        if (_respawn) {
          task_factory_type::clearDependence(_policy, this);
          task_factory_type::addDependence(_policy, this, dependence);
          task_factory_type::respawn(_policy, this);
        } 
      }

      // task-data execution
      void apply(const member_type &member, value_type &r_val) {
        typename ExecViewType::future_type dependence;
        r_val = Chol::invoke(_policy, member,
                             _A, 
                             _respawn, dependence);
        if (_respawn) {
          task_factory_type::clearDependence(_policy, this);
          task_factory_type::addDependence(_policy, this, dependence);
          task_factory_type::respawn(_policy, this);
        } 
      }

    };

  };
}

#endif
