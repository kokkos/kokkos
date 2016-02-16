#pragma once
#ifndef __CHOL_U_RIGHT_LOOK_BY_BLOCKS_HPP__
#define __CHOL_U_RIGHT_LOOK_BY_BLOCKS_HPP__

/// \file chol_u_right_look_by_blocks.hpp
/// \brief Cholesky factorization by-blocks
/// \author Kyungjoo Kim (kyukim@sandia.gov)

/// The Partitioned-Block Matrix (PBM) is sparse and a block itself is a view of a sparse matrix. 
/// The algorithm generates tasks with a given sparse block matrix structure.

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
  class CholUpperRightLookByBlocks {
  public:
    KOKKOS_INLINE_FUNCTION
    static int genScalarTask(typename CrsTaskViewType::policy_type &policy,
                             CrsTaskViewType &A) {
      typedef typename CrsTaskViewType::value_type        value_type;
      typedef typename CrsTaskViewType::row_view_type     row_view_type;
      
      typedef typename CrsTaskViewType::future_type       future_type;
      typedef typename CrsTaskViewType::task_factory_type task_factory_type;
      
      row_view_type a(A, 0); 
      value_type &aa = a.Value(0);
      
      // construct a task
      future_type f = task_factory_type::create(policy,
                                                typename Chol<Uplo::Upper,
                                                CtrlDetail(ControlType,AlgoChol::ByBlocks,ArgVariant,Chol)>
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
      typedef typename CrsTaskViewType::row_view_type     row_view_type;
      
      typedef typename CrsTaskViewType::future_type       future_type;
      typedef typename CrsTaskViewType::task_factory_type task_factory_type;
      
      row_view_type a(A,0), b(B,0); 
      value_type &aa = a.Value(0);
      
      const ordinal_type nnz = b.NumNonZeros();
      for (ordinal_type j=0;j<nnz;++j) {
        value_type &bb = b.Value(j);
        
        future_type f = task_factory_type
          ::create(policy, 
                   typename Trsm<Side::Left,Uplo::Upper,Trans::ConjTranspose,
                   CtrlDetail(ControlType,AlgoChol::ByBlocks,ArgVariant,Trsm)>
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
      typedef typename CrsTaskViewType::row_view_type     row_view_type;
      
      typedef typename CrsTaskViewType::future_type       future_type;
      typedef typename CrsTaskViewType::task_factory_type task_factory_type;
      
      // case that X.transpose, A.no_transpose, Y.no_transpose
      
      row_view_type a(A,0), c; 
      
      const ordinal_type nnz = a.NumNonZeros();
      
      // update herk
      for (ordinal_type i=0;i<nnz;++i) {
        const ordinal_type row_at_i = a.Col(i);
        value_type &aa = a.Value(i);
        
        c.setView(C, row_at_i);
        
        ordinal_type idx = 0;
        for (ordinal_type j=i;j<nnz && (idx > -2);++j) {
          const ordinal_type col_at_j = a.Col(j);
          value_type &bb = a.Value(j);
          
          if (row_at_i == col_at_j) {
            idx = c.Index(row_at_i, idx);
            if (idx >= 0) {
              value_type &cc = c.Value(idx);
              future_type f = task_factory_type
                ::create(policy, 
                         typename Herk<Uplo::Upper,Trans::ConjTranspose,
                         CtrlDetail(ControlType,AlgoChol::ByBlocks,ArgVariant,Herk)>
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
          } else {
            idx = c.Index(col_at_j, idx);
            if (idx >= 0) {
              value_type &cc = c.Value(idx);
              future_type f = task_factory_type
                ::create(policy, 
                         typename Gemm<Trans::ConjTranspose,Trans::NoTranspose,
                         CtrlDetail(ControlType,AlgoChol::ByBlocks,ArgVariant,Gemm)>
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
        }
      }
    
      return 0;
    }
    
  };
  
  // specialization for different task generation in right looking by-blocks algorithm
  // =================================================================================
  template<int ArgVariant, template<int,int> class ControlType>
  class Chol<Uplo::Upper,AlgoChol::RightLookByBlocks,ArgVariant,ControlType> {
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
          CholUpperRightLookByBlocks<ArgVariant,ControlType,ExecViewType>
            ::genScalarTask(policy, A11);
          
          // A12 = inv(triu(A11)') * A12
          CholUpperRightLookByBlocks<ArgVariant,ControlType,ExecViewType>
            ::genTrsmTasks(policy, A11, A12);

          // A22 = A22 - A12' * A12
          CholUpperRightLookByBlocks<ArgVariant,ControlType,ExecViewType>
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
