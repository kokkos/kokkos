#pragma once
#ifndef __GRAPH_HELPER_CAMD_HPP__
#define __GRAPH_HELPER_CAMD_HPP__

/// \file graph_helper_camd.hpp
/// \brief Interface to camd (suire sparse) ordering
/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "camd.h"
#include "util.hpp"

namespace Tacho {

  using namespace std;

  class CAMD {
  public:
    template<typename OrdinalType>
    KOKKOS_INLINE_FUNCTION
    static void run( OrdinalType n, OrdinalType Pe[], OrdinalType Iw[], 
                     OrdinalType Len[], OrdinalType iwlen, OrdinalType pfree,
                     OrdinalType Nv[], OrdinalType Next[], OrdinalType Last[],
                     OrdinalType Head[], OrdinalType Elen[], OrdinalType Degree[],
                     OrdinalType W[],
                     double Control[],
                     double Info[],
                     const OrdinalType C[],
                     OrdinalType BucketSet[] ) {
      ERROR("GraphHelper_CAMD:: CAMD does not support the ordinal type");
    }
  };
  
  template<> 
  KOKKOS_INLINE_FUNCTION
  void CAMD::run<int>( int n, int Pe[], int Iw[], 
                       int Len[], int iwlen, int pfree,
                       int Nv[], int Next[], int Last[],
                       int Head[], int Elen[], int Degree[],
                       int W[],
                       double Control[],
                       double Info[],
                       const int C[],
                       int BucketSet[] ) {
    camd_2( n, Pe, Iw, Len, iwlen, pfree, 
            Nv, Next, Last, Head, Elen, Degree, W, Control, Info, C, BucketSet );
  }
  
  template<> 
  KOKKOS_INLINE_FUNCTION
  void CAMD::run<SuiteSparse_long>( SuiteSparse_long n, SuiteSparse_long Pe[], SuiteSparse_long Iw[], 
                                    SuiteSparse_long Len[], SuiteSparse_long iwlen, SuiteSparse_long pfree,
                                    SuiteSparse_long Nv[], SuiteSparse_long Next[], SuiteSparse_long Last[],
                                    SuiteSparse_long Head[], SuiteSparse_long Elen[], SuiteSparse_long Degree[],
                                    SuiteSparse_long W[],
                                    double Control[],
                                    double Info[],
                                    const SuiteSparse_long C[],
                                    SuiteSparse_long BucketSet[] ) {
    camd_l2( n, Pe, Iw, Len, iwlen, pfree, 
             Nv, Next, Last, Head, Elen, Degree, W, Control, Info, C, BucketSet );
  }
  
  template<class CrsMatBaseType>
  class GraphHelper_CAMD : public Disp {    
  public:
    typedef typename CrsMatBaseType::ordinal_type ordinal_type;
    typedef typename CrsMatBaseType::size_type    size_type;

    typedef typename CrsMatBaseType::ordinal_type_array ordinal_type_array;
    typedef typename CrsMatBaseType::size_type_array    size_type_array;

  private:
    string _label;

    ordinal_type _m;
    size_type _nnz;
    size_type_array _rptr;
    ordinal_type_array _cidx, _cnst;

    // CAMD output
    ordinal_type_array _pe, _nv, _el, _next, _perm, _peri; // perm = last, peri = next

    double _control[CAMD_CONTROL], _info[CAMD_INFO];

    bool _is_ordered;

  public:

    void setLabel(string label) { _label = label; }
    string Label() const { return _label; }

    size_type NumNonZeros() const { return _nnz; }
    ordinal_type NumRows() const { return _m; }

    ordinal_type_array PermVector()       const { return _perm; }
    ordinal_type_array InvPermVector()    const { return _peri; }

    ordinal_type_array ConstraintVector() const { return _cnst; }

    GraphHelper_CAMD() = default;

    GraphHelper_CAMD(const string label,
                     const ordinal_type m,
                     const size_type nnz,
                     const size_type_array rptr,
                     const ordinal_type_array cidx,
                     const ordinal_type nblk,
                     const ordinal_type_array range) {
      _label = "GraphHelper_CAMD::" + label;

      // graph information
      _m     = m;
      _nnz   = nnz;

      // create a graph  structure (full without diagonals)
      _rptr  = rptr; // size_type_array(_label+"::RowPtrArray", _m+1);
      _cidx  = cidx; //ordinal_type_array(_label+"::ColIndexArray", _nnz);

      // constraints are induced from range
      _cnst  = ordinal_type_array(_label+"::ConstraintArray", _m+1);
      for (ordinal_type i=0;i<nblk;++i)
        for (ordinal_type j=range[i];j<range[i+1];++j)
          _cnst[j] = i;

      // permutation vector
      _pe    = ordinal_type_array(_label+"::EliminationArray", _m);
      _nv    = ordinal_type_array(_label+"::SupernodesArray", _m);
      _el    = ordinal_type_array(_label+"::DegreeArray", _m);
      _next  = ordinal_type_array(_label+"::InvPermSupernodesArray", _m);
      _perm  = ordinal_type_array(_label+"::PermutationArray", _m);
      _peri  = ordinal_type_array(_label+"::InvPermutationArray", _m);
    }
    GraphHelper_CAMD(const GraphHelper_CAMD &b) = default;

    int computeOrdering() {
      int r_val = 0;

      camd_defaults(_control);
      camd_control(_control);

      ordinal_type *rptr = reinterpret_cast<ordinal_type*>(_rptr.ptr_on_device());
      ordinal_type *cidx = reinterpret_cast<ordinal_type*>(_cidx.ptr_on_device());
      ordinal_type *cnst = reinterpret_cast<ordinal_type*>(_cnst.ptr_on_device());

      ordinal_type *next = reinterpret_cast<ordinal_type*>(_next.ptr_on_device());
      ordinal_type *perm = reinterpret_cast<ordinal_type*>(_perm.ptr_on_device());

      // length array
      // assume there always is diagonal and the given matrix is symmetry
      ordinal_type_array lwork(_label+"::LWorkArray", _m);
      ordinal_type *lwork_ptr = reinterpret_cast<ordinal_type*>(lwork.ptr_on_device());
      for (ordinal_type i=0;i<_m;++i)
        lwork_ptr[i] = rptr[i+1] - rptr[i];

      // workspace
      const size_type swlen = _nnz + _nnz/5 + 5*(_m+1);;
      ordinal_type_array swork(_label+"::SWorkArray", swlen);
      ordinal_type *swork_ptr = reinterpret_cast<ordinal_type*>(swork.ptr_on_device());

      ordinal_type *pe_ptr = reinterpret_cast<ordinal_type*>(_pe.ptr_on_device()); // 1) Pe
      size_type pfree = 0;
      for (ordinal_type i=0;i<_m;++i) {
        pe_ptr[i] = pfree;
        pfree += lwork_ptr[i];
      }

      if (_nnz != pfree)
        ERROR(">> nnz in the graph does not match to nnz count (pfree)");

      ordinal_type *nv_ptr = reinterpret_cast<ordinal_type*>(_nv.ptr_on_device()); // 2) Nv
      ordinal_type *hd_ptr = swork_ptr; swork_ptr += (_m+1);   // 3) Head
      ordinal_type *el_ptr = reinterpret_cast<ordinal_type*>(_el.ptr_on_device()); // 4) Elen
      ordinal_type *dg_ptr = swork_ptr; swork_ptr += _m;       // 5) Degree
      ordinal_type *wk_ptr = swork_ptr; swork_ptr += (_m+1);   // 6) W
      ordinal_type *bk_ptr = swork_ptr; swork_ptr += _m;       // 7) BucketSet

      const size_type iwlen = swlen - (4*_m+2);
      ordinal_type *iw_ptr = swork_ptr; swork_ptr += iwlen;    // Iw
      for (ordinal_type i=0;i<pfree;++i)
        iw_ptr[i] = cidx[i];

      CAMD::run(_m, pe_ptr, iw_ptr, lwork_ptr, iwlen, pfree,
                // output
                nv_ptr, next, perm, hd_ptr, el_ptr, dg_ptr, wk_ptr, 
                _control, _info, cnst, bk_ptr);
      
      r_val = (_info[CAMD_STATUS] != CAMD_OK);

      for (ordinal_type i=0;i<_m;++i)
        _peri[_perm[i]] = i;

      _is_ordered = true;

      return r_val;
    }

    ostream& showMe(ostream &os) const {
      streamsize prec = os.precision();
      os.precision(15);
      os << scientific;

      os << " -- CAMD input -- " << endl
         << "    # of Rows      = " << _m << endl
         << "    # of NonZeros  = " << _nnz << endl;

      if (_is_ordered)
        os << " -- Ordering -- " << endl
           << "  CNST     PERM     PERI       PE       NV     NEXT     ELEN" << endl;

      const int w = 6;
      for (ordinal_type i=0;i<_m;++i)
        os << setw(w) << _cnst[i] << "   "
           << setw(w) << _perm[i] << "   "
           << setw(w) << _peri[i] << "   "
           << setw(w) << _pe[i] << "   "
           << setw(w) << _nv[i] << "   "
           << setw(w) << _next[i] << "   "
           << setw(w) << _el[i] << "   "
           << endl;

      os.unsetf(ios::scientific);
      os.precision(prec);

      return os;
    }

  };

}

#endif
