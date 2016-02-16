#pragma once
#ifndef __CHOL_U_DENSE_BY_BLOCKS_HPP__
#define __CHOL_U_DENSE_BY_BLOCKS_HPP__

/// \file chol_u_dense_by_blocks.hpp
/// \brief Dense Cholesky factorization by-blocks
/// \author Kyungjoo Kim (kyukim@sandia.gov)

// basic utils
#include "util.hpp"
#include "control.hpp"
#include "partition.hpp"

namespace Tacho {

  using namespace std;

  // detailed workflow of by-blocks algorithm
  // ========================================
  template<int ArgVariant,
           template<int,int> class ControlType,
           typename CrsTaskViewType>
  class CholUpperDenseByBlocks {
  public:
    KOKKOS_INLINE_FUNCTION
    static int genScalarTask(typename CrsTaskViewType::policy_type &policy,
                             CrsTaskViewType &A) {
      typedef typename CrsTaskViewType::value_type        value_type;

      typedef typename CrsTaskViewType::future_type       future_type;
      typedef typename CrsTaskViewType::task_factory_type task_factory_type;

      value_type &aa = A.Value(0, 0);

      // construct a task
      future_type f = task_factory_type::create(policy,
                                                typename Chol<Uplo::Upper,
                                                CtrlDetail(ControlType,AlgoChol::DenseByBlocks,ArgVariant,Chol)>
                                                ::template TaskFunctor<value_type>(aa));

      // manage dependence
      task_factory_type::addDependence(policy, f, aa.Future());
      aa.setFuture(f);

      // spawn a task
      task_factory_type::spawn(policy, f);

      return 0;
    }

    KOKKOS_INLINE_FUNCTION
    static int genTrsmTasks(typename CrsTaskViewType::policy_type &policy,
                            CrsTaskViewType &A,
                            CrsTaskViewType &B) {
      typedef typename CrsTaskViewType::ordinal_type      ordinal_type;
      typedef typename CrsTaskViewType::value_type        value_type;

      typedef typename CrsTaskViewType::future_type       future_type;
      typedef typename CrsTaskViewType::task_factory_type task_factory_type;

      value_type &aa = A.Value(0, 0);

      const ordinal_type ncols = B.NumCols();
      for (ordinal_type j=0;j<ncols;++j) {
        value_type &bb = B.Value(0, j);

        future_type f = task_factory_type
          ::create(policy,
                   typename Trsm<Side::Left,Uplo::Upper,Trans::ConjTranspose,
                   CtrlDetail(ControlType,AlgoChol::DenseByBlocks,ArgVariant,Trsm)>
                   ::template TaskFunctor<double,value_type,value_type>(Diag::NonUnit, 1.0, aa, bb));

        // trsm dependence
        task_factory_type::addDependence(policy, f, aa.Future());

        // self
        task_factory_type::addDependence(policy, f, bb.Future());

        // place task signature on b
        bb.setFuture(f);

        // spawn a task
        task_factory_type::spawn(policy, f);
      }

      return 0;
    }

    KOKKOS_INLINE_FUNCTION
    static int genHerkTasks(typename CrsTaskViewType::policy_type &policy,
                            CrsTaskViewType &A,
                            CrsTaskViewType &C) {
      typedef typename CrsTaskViewType::ordinal_type      ordinal_type;
      typedef typename CrsTaskViewType::value_type        value_type;

      typedef typename CrsTaskViewType::future_type       future_type;
      typedef typename CrsTaskViewType::task_factory_type task_factory_type;

      // update herk
      const ordinal_type ncols = C.NumCols();

      for (ordinal_type j=0;j<ncols;++j) {
        {
          value_type &aa = A.Value(0, j);
          value_type &cc = C.Value(j, j);
          future_type f = task_factory_type
            ::create(policy,
                     typename Herk<Uplo::Upper,Trans::ConjTranspose,
                     CtrlDetail(ControlType,AlgoChol::DenseByBlocks,ArgVariant,Herk)>
                     ::template TaskFunctor<double,value_type,value_type>(-1.0, aa, 1.0, cc));

          // dependence
          task_factory_type::addDependence(policy, f, aa.Future());

          // self
          task_factory_type::addDependence(policy, f, cc.Future());

          // place task signature on y
          cc.setFuture(f);

          // spawn a task
          task_factory_type::spawn(policy, f);
        }
        for (ordinal_type i=0;i<j;++i) {
          value_type &aa = A.Value(0, i);
          value_type &bb = A.Value(0, j);
          value_type &cc = C.Value(i, j);
          future_type f = task_factory_type
            ::create(policy,
                     typename Gemm<Trans::ConjTranspose,Trans::NoTranspose,
                     CtrlDetail(ControlType,AlgoChol::DenseByBlocks,ArgVariant,Gemm)>
                     ::template TaskFunctor<double,value_type,value_type,value_type>(-1.0, aa, bb, 1.0, cc));

          // dependence
          task_factory_type::addDependence(policy, f, aa.Future());
          task_factory_type::addDependence(policy, f, bb.Future());

          // self
          task_factory_type::addDependence(policy, f, cc.Future());

          // place task signature on y
          cc.setFuture(f);

          // spawn a task
          task_factory_type::spawn(policy, f);
        }
      }

      return 0;
    }

  };

  // specialization for different task generation in right looking by-blocks algorithm
  // =================================================================================
  template<int ArgVariant, template<int,int> class ControlType>
  class Chol<Uplo::Upper,AlgoChol::DenseByBlocks,ArgVariant,ControlType> {
  public:

    // function interface
    // ==================
    template<typename ExecViewType>
    KOKKOS_INLINE_FUNCTION
    static int invoke(typename ExecViewType::policy_type &policy,
                      const typename ExecViewType::policy_type::member_type &member,
                      ExecViewType &A) {
      if (member.team_rank() == 0) {
        ExecViewType ATL, ATR,      A00, A01, A02,
          /**/       ABL, ABR,      A10, A11, A12,
          /**/                      A20, A21, A22;

        Part_2x2(A,  ATL, ATR,
                 /**/ABL, ABR,
                 0, 0, Partition::TopLeft);

        while (ATL.NumRows() < A.NumRows()) {
          Part_2x2_to_3x3(ATL, ATR, /**/  A00, A01, A02,
                          /*******/ /**/  A10, A11, A12,
                          ABL, ABR, /**/  A20, A21, A22,
                          1, 1, Partition::BottomRight);
          // -----------------------------------------------------

          // A11 = chol(A11)
          CholUpperDenseByBlocks<ArgVariant,ControlType,ExecViewType>
            ::genScalarTask(policy, A11);

          // A12 = inv(triu(A11)') * A12
          CholUpperDenseByBlocks<ArgVariant,ControlType,ExecViewType>
            ::genTrsmTasks(policy, A11, A12);

          // A22 = A22 - A12' * A12
          CholUpperDenseByBlocks<ArgVariant,ControlType,ExecViewType>
            ::genHerkTasks(policy, A12, A22);

          // -----------------------------------------------------
          Merge_3x3_to_2x2(A00, A01, A02, /**/ ATL, ATR,
                           A10, A11, A12, /**/ /******/
                           A20, A21, A22, /**/ ABL, ABR,
                           Partition::TopLeft);
        }
      }

      return 0;
    }

    // task-data parallel interface
    // ============================
    template<typename ExecViewType>
    class TaskFunctor {
    public:
      typedef typename ExecViewType::policy_type policy_type;
      typedef typename policy_type::member_type member_type;
      typedef int value_type;

    private:
      ExecViewType _A;

      policy_type &_policy;

    public:
      TaskFunctor(const ExecViewType A)
        : _A(A),
          _policy(ExecViewType::task_factory_type::Policy())
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
