#pragma once
#ifndef __TRI_SOLVE_U_CT_BY_BLOCKS_VAR1_HPP__
#define __TRI_SOLVE_U_CT_BY_BLOCKS_VAR1_HPP__

/// \file tri_solve_u_ct_by_blocks_var1.hpp
/// \brief  Sparse triangular solve on given sparse patterns and multiple rhs.
/// \author Kyungjoo Kim (kyukim@sandia.gov)
///
/// This naively generates tasks without any merging of task blocks.

namespace Tacho {
  
  using namespace std;
  
  template<typename CrsTaskViewTypeA,
           typename DenseTaskViewTypeB>
  KOKKOS_INLINE_FUNCTION
  int genTrsmTasks_TriSolveUpperConjTransposeByBlocks(typename CrsTaskViewTypeA::policy_type &policy,
                                                      const int diagA,
                                                      CrsTaskViewTypeA &A,
                                                      DenseTaskViewTypeB &B) {
    typedef typename CrsTaskViewTypeA::ordinal_type      ordinal_type;
    typedef typename CrsTaskViewTypeA::value_type        crs_value_type;
    typedef typename CrsTaskViewTypeA::row_view_type     row_view_type;
    
    typedef typename CrsTaskViewTypeA::future_type       future_type;
    typedef typename CrsTaskViewTypeA::task_factory_type task_factory_type;
    
    typedef typename DenseTaskViewTypeB::value_type      dense_value_type;

    row_view_type a(A,0);
    crs_value_type &aa = a.Value(0);
    
    for (ordinal_type j=0;j<B.NumCols();++j) {
      dense_value_type &bb = B.Value(0, j);
      
      future_type f = task_factory_type
        ::create(policy,
                 Trsm<Side::Left,Uplo::Upper,Trans::ConjTranspose,AlgoTrsm::ForTriSolveBlocked>
                 ::TaskFunctor<double,crs_value_type,dense_value_type>(diagA, 1.0, aa, bb));
      
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

  template<typename CrsTaskViewTypeA,
           typename DenseTaskViewTypeB,
           typename DenseTaskViewTypeC>
  KOKKOS_INLINE_FUNCTION
  int genGemmTasks_TriSolveUpperConjTransposeByBlocks(typename CrsTaskViewTypeA::policy_type &policy,
                                                      CrsTaskViewTypeA &A,
                                                      DenseTaskViewTypeB &B,
                                                      DenseTaskViewTypeC &C) {
    typedef typename CrsTaskViewTypeA::ordinal_type      ordinal_type;
    typedef typename CrsTaskViewTypeA::value_type        crs_value_type;
    typedef typename CrsTaskViewTypeA::row_view_type     row_view_type;
    
    typedef typename CrsTaskViewTypeA::future_type       future_type;
    typedef typename CrsTaskViewTypeA::task_factory_type task_factory_type;
    
    typedef typename DenseTaskViewTypeB::value_type      dense_value_type;

    row_view_type a(A,0);
    const ordinal_type nnz = a.NumNonZeros();
    
    for (ordinal_type i=0;i<nnz;++i) {
      const ordinal_type row_at_i = a.Col(i);
      crs_value_type &aa = a.Value(i);

      for (ordinal_type j=0;j<C.NumCols();++j) {
        dense_value_type &bb = B.Value(0, j);
        dense_value_type &cc = C.Value(row_at_i, j);
        
        future_type f = task_factory_type
          ::create(policy,
                   Gemm<Trans::ConjTranspose,Trans::NoTranspose,AlgoGemm::ForTriSolveBlocked>
                   ::TaskFunctor<double,crs_value_type,dense_value_type,dense_value_type>(-1.0, aa, bb, 1.0, cc));
        
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

}

#endif
