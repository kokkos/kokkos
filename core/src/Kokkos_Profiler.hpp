/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact  H. Carter Edwards (hcedwar@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

/// \file Kokkos_Parallel.hpp
/// \brief Declaration of parallel operators

#ifndef KOKKOS_PROFILER_HPP
#define KOKKOS_PROFILER_HPP

#include <cstddef>
#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Macros.hpp>
#include <string>

#include <typeinfo>
#ifdef KOKKOS_COMPILER_GNU
#include <cxxabi.h>
#endif

#ifdef KOKKOS_ENABLE_PROFILING_COLLECT_KERNEL_DATA
namespace KokkosP {
namespace Experimental {
  extern void profiler_initialize();
  extern void profiler_finalize();
  extern void profiler_begin_kernel(const std::string&,const std::string&);
  extern void profiler_end_kernel(const std::string&,const std::string&);
}
}
#endif

namespace Kokkos {
namespace Experimental {

namespace Profiler {

void initialize();
void finalize();

#ifdef KOKKOS_ENABLE_PROFILING_COLLECT_KERNEL_DATA

  template<class FunctorType, class ExecPolicy>
  void begin_kernel(const int& pattern, const std::string& str_in) {
    std::string str = str_in;
    if(str.length() == 0) {
      if(pattern == 0)
        str += "parallel_for ";
      if(pattern == 1)
        str += "parallel_reduce ";
      if(pattern == 2)
        str += "parallel_scan ";

      #ifdef KOKKOS_COMPILER_GNU
      int status;
      str += abi::__cxa_demangle(typeid(FunctorType).name(),0,0,&status);
      #else
      str += typeid(FunctorType).name();
      #endif
    }
    Kokkos::fence();
    KokkosP::Experimental::profiler_begin_kernel(str,"DefaultExec");
  }

  template<class FunctorType, class ExecPolicy>
  void end_kernel(const int& pattern, const std::string& str_in) {
    std::string str = str_in;
    if(str.length() == 0) {
      if(pattern == 0)
        str += "parallel_for ";
      if(pattern == 1)
        str += "parallel_reduce ";
      if(pattern == 2)
        str += "parallel_scan ";

      #ifdef KOKKOS_COMPILER_GNU
      int status;
      str += abi::__cxa_demangle(typeid(FunctorType).name(),0,0,&status);
      #else
      str += typeid(FunctorType).name();
      #endif
    }
    Kokkos::fence();
    KokkosP::Experimental::profiler_end_kernel(str,"DefaultExec");
  }
#else
  template<class FunctorType, class ExecPolicy>
  void begin_kernel(const int& , const std::string& ) {
  }
  template<class FunctorType, class ExecPolicy>
  void end_kernel(const int& , const std::string& ) {
  }
#endif

}
}
}
#endif
