#pragma once
#ifndef __CHOL_HPP__
#define __CHOL_HPP__

/// \file chol.hpp
/// \brief Incomplete Cholesky factorization front interface.
/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "util.hpp"
#include "control.hpp"
#include "partition.hpp"

namespace Tacho { 

  using namespace std;

  // tasking interface
  // * default behavior is for non-by-blocks tasks
  // * control is only used for by-blocks algorithms
  // ===============================================
  template<int ArgUplo, int ArgAlgo, 
           int ArgVariant = Variant::One,                  
           template<int,int> class ControlType = Control>  
  class Chol {
  public:
    
    // function interface
    // ==================
    template<typename ExecViewType>
    KOKKOS_INLINE_FUNCTION
    static int invoke(typename ExecViewType::policy_type &policy, 
                      const typename ExecViewType::policy_type::member_type &member, 
                      ExecViewType &A) {
      // each algorithm and its variants should be specialized
      ERROR(MSG_INVALID_TEMPLATE_ARGS);
      return -1;
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


// unblocked version blas operations
#include "scale.hpp"

// blocked version blas operations
#include "gemm.hpp"
#include "trsm.hpp"
#include "herk.hpp"

// cholesky
#include "chol_u.hpp"

#endif
