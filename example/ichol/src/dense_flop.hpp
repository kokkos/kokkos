#pragma once
#ifndef __DENSE_FLOP_HPP__
#define __DENSE_FLOP_HPP__

#include "util.hpp"
#include "Teuchos_ScalarTraits.hpp"  

/// \file dense_flop.hpp
/// \author Kyungjoo Kim (kyukim@sandia.gov)

//  FLOP counting - From LAPACK working note #41  

using namespace std;

#define FLOP_MUL(value_type) ((Teuchos::ScalarTraits<value_type>::isComplex) ?  (6.0) : (1.0))
#define FLOP_ADD(value_type) ((Teuchos::ScalarTraits<value_type>::isComplex) ?  (2.0) : (1.0))

namespace Tacho {

  template<typename ValueType>
  KOKKOS_INLINE_FUNCTION
  double get_flop_gemm(int mm, int nn, int kk) {
    double m = (double)mm;    double n = (double)nn;    double k = (double)kk;
    return (FLOP_MUL(ValueType)*(m*n*k) +
            FLOP_ADD(ValueType)*(m*n*k));
  }

  template<typename ValueType>
  KOKKOS_INLINE_FUNCTION
  double get_flop_syrk(int kk, int nn) {
    double k = (double)kk;    double n = (double)nn;
    return (FLOP_MUL(ValueType)*(0.5*k*n*(n+1.0)) +
            FLOP_ADD(ValueType)*(0.5*k*n*(n+1.0)));
  }

  template<typename ValueType>
  KOKKOS_INLINE_FUNCTION
  double get_flop_trsm_lower(int mm, int nn) {
    double m = (double)mm;    double n = (double)nn;
    return (FLOP_MUL(ValueType)*(0.5*n*m*(m+1.0)) +
            FLOP_ADD(ValueType)*(0.5*n*m*(m-1.0)));
  }

  template<typename ValueType>
  KOKKOS_INLINE_FUNCTION
  double get_flop_trsm_upper(int mm, int nn) {
    double m = (double)mm;    double n = (double)nn;
    return (FLOP_MUL(ValueType)*(0.5*m*n*(n+1.0)) +
            FLOP_ADD(ValueType)*(0.5*m*n*(n-1.0)));
  }

  template<typename ValueType>
  KOKKOS_INLINE_FUNCTION
  double get_flop_trsm(int is_lower, int mm, int nn) {
    return (is_lower ? 
            get_flop_trsm_lower<ValueType>(mm, nn) : 
            get_flop_trsm_upper<ValueType>(mm, nn));
  }

  template<typename ValueType>
  KOKKOS_INLINE_FUNCTION
  double get_flop_lu(int mm, int nn) {
    double m = (double)mm;    double n = (double)nn;
    if (m > n)
      return (FLOP_MUL(ValueType)*(0.5*m*n*n-(1.0/6.0)*n*n*n+0.5*m*n-0.5*n*n+(2.0/3.0)*n) +
              FLOP_ADD(ValueType)*(0.5*m*n*n-(1.0/6.0)*n*n*n-0.5*m*n+        (1.0/6.0)*n));
    else
      return (FLOP_MUL(ValueType)*(0.5*n*m*m-(1.0/6.0)*m*m*m+0.5*n*m-0.5*m*m+(2.0/3.0)*m) +
              FLOP_ADD(ValueType)*(0.5*n*m*m-(1.0/6.0)*m*m*m-0.5*n*m+        (1.0/6.0)*m));
  }

  template<typename ValueType>
  KOKKOS_INLINE_FUNCTION
  double get_flop_chol(int nn) {
    double n = (double)nn;
    return (FLOP_MUL(ValueType)*((1.0/6.0)*n*n*n+0.5*n*n+(1.0/3.0)*n) +
            FLOP_ADD(ValueType)*((1.0/6.0)*n*n*n-        (1.0/6.0)*n));

  }

  template<typename ValueType>
  KOKKOS_INLINE_FUNCTION
  double get_flop_ldl(int nn) {
    double n = (double)nn;
    return (FLOP_MUL(ValueType)*((1.0/3.0)*n*n*n + (2.0/3.0)*n) +
            FLOP_ADD(ValueType)*((1.0/3.0)*n*n*n - (1.0/3.0)*n));

  }


}

#endif
