#pragma once
#ifndef __GEMM_CT_NT_DENSE_BY_BLOCKS_HPP__
#define __GEMM_CT_NT_DENSE_BY_BLOCKS_HPP__

/// \file gemm_ct_nt_dense_by_blocks.hpp
/// \brief Dense matrix-matrix multiplication
/// \author Kyungjoo Kim (kyukim@sandia.gov)

namespace Tacho {

  using namespace std;

  // Gemm-By-Blocks
  // ==============
  template<int ArgVariant, template<int,int> class ControlType>
  class Gemm<Trans::ConjTranspose,Trans::NoTranspose,
             AlgoGemm::DenseByBlocks,ArgVariant,ControlType> {
  public:
    template<typename ScalarType,
             typename DenseTaskViewTypeA,
             typename DenseTaskViewTypeB,
             typename DenseTaskViewTypeC>
    KOKKOS_INLINE_FUNCTION
    static int invoke(typename DenseTaskViewTypeA::policy_type &policy,
                      const typename DenseTaskViewTypeA::policy_type::member_type &member,
                      const ScalarType alpha,
                      DenseTaskViewTypeA &A,
                      DenseTaskViewTypeB &B,
                      const ScalarType beta,
                      DenseTaskViewTypeC &C) {
      typedef ScalarType scalar_type;
      typedef typename DenseTaskViewTypeA::ordinal_type ordinal_type;
      typedef typename DenseTaskViewTypeA::value_type   value_type;

      typedef typename DenseTaskViewTypeA::future_type       future_type;
      typedef typename DenseTaskViewTypeA::task_factory_type task_factory_type;

      if (member.team_rank() == 0) {
        for (ordinal_type p=0;p<A.NumRows();++p) {
          const ScalarType beta_select = (p > 0 ? 1.0 : beta);
          for (ordinal_type k2=0;k2<C.NumCols();++k2) {
            value_type &bb = B.Value(p, k2);
            for (ordinal_type k1=0;k1<C.NumRows();++k1) {
              value_type &aa = A.Value(p , k1);
              value_type &cc = C.Value(k1, k2);

              future_type f = task_factory_type::create(policy,
                                                        typename Gemm<Trans::ConjTranspose,Trans::NoTranspose,
                                                        CtrlDetail(ControlType,AlgoGemm::DenseByBlocks,ArgVariant,Gemm)>
                                                        ::template TaskFunctor<scalar_type,value_type,value_type,value_type>
                                                        (alpha, aa, bb, beta_select, cc));

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
        }
      }

      return 0;
    }


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

#endif
