#pragma once
#ifndef __TRSM_L_U_CT_DENSE_BY_BLOCKS_HPP__
#define __TRSM_L_U_CT_DENSE_BY_BLOCKS_HPP__

/// \file trsm_l_u_ct_dense_by_blocks.hpp
/// \brief Dense triangular solver
/// \author Kyungjoo Kim (kyukim@sandia.gov)

namespace Tacho {

  using namespace std;

  // Trsm-By-Blocks
  // ==============
  template<int ArgVariant, template<int,int> class ControlType>
  class Trsm<Side::Left,Uplo::Upper,Trans::ConjTranspose,
             AlgoTrsm::DenseByBlocks,ArgVariant,ControlType> {
  public:
    template<typename ScalarType,
             typename DenseTaskViewTypeA,
             typename DenseTaskViewTypeB>
    KOKKOS_INLINE_FUNCTION
    static int invoke(typename DenseTaskViewTypeA::policy_type &policy,
                      const typename DenseTaskViewTypeA::policy_type::member_type &member,
                      const int diagA,
                      const ScalarType alpha,
                      DenseTaskViewTypeA &A,
                      DenseTaskViewTypeB &B) {
      typedef ScalarType scalar_type;
      typedef typename DenseTaskViewTypeA::ordinal_type ordinal_type;
      typedef typename DenseTaskViewTypeA::value_type   value_type;

      typedef typename DenseTaskViewTypeA::future_type       future_type;
      typedef typename DenseTaskViewTypeA::task_factory_type task_factory_type;

      if (member.team_rank() == 0) {
        for (ordinal_type k2=0;k2<B.NumCols();++k2) {
          for (ordinal_type k1=0;k1<B.NumRows();++k1) {
            for (ordinal_type p=0;p<k1;++p) {
              // update B first
              const ScalarType beta_select = (p > 0 ? 1.0 : alpha);
              
              value_type &aa = A.Value(p , k1);
              value_type &bb = B.Value(p,  k2);
              value_type &cc = B.Value(k1, k2);
              
              future_type f = task_factory_type::create(policy,
                                                        typename Gemm<Trans::ConjTranspose,Trans::NoTranspose,
                                                        CtrlDetail(ControlType,AlgoTrsm::DenseByBlocks,ArgVariant,Gemm)>
                                                        ::template TaskFunctor<scalar_type,value_type,value_type,value_type>
                                                        (-1.0, aa, bb, beta_select, cc));

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

            // Solve A
            {
              value_type &aa = A.Value(k1, k1);
              value_type &bb = B.Value(k1, k2);

              future_type f = task_factory_type::create(policy,
                                                        typename Trsm<Side::Left,Uplo::Upper,Trans::ConjTranspose,
                                                        CtrlDetail(ControlType,AlgoTrsm::DenseByBlocks,ArgVariant,Trsm)>
                                                        ::template TaskFunctor<double,value_type,value_type>(diagA, 1.0, aa, bb));

              // trsm dependence
              task_factory_type::addDependence(policy, f, aa.Future());

              // self
              task_factory_type::addDependence(policy, f, bb.Future());

              // place task signature on b
              bb.setFuture(f);

              // spawn a task
              task_factory_type::spawn(policy, f);
            }
          }
        }
      }

      return 0;
    }

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

#endif
