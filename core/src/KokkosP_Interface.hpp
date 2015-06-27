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

#ifndef KOKKOSP_INTERFACE_HPP
#define KOKKOSP_INTERFACE_HPP

#include <cstddef>
#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Macros.hpp>
#include <string>

#ifdef KOKKOSP_ENABLE_PROFILING
#include <dlfcn.h>
#include <iostream>
#include <stdlib.h>
#endif

#ifdef KOKKOSP_ENABLE_PROFILING
namespace KokkosP {
    typedef void (*initFunction)(const int);
    typedef void (*finalizeFunction)();
    typedef void (*beginFunction)(const char*, uint64_t*);
    typedef void (*endFunction)(uint64_t);
    
    static initFunction initProfileLibrary = NULL;
    static finalizeFunction finalizeProfileLibrary = NULL;
    static beginFunction beginForCallee = NULL;
    static beginFunction beginScanCallee = NULL;
    static beginFunction beginReduceCallee = NULL;
    static endFunction endForCallee = NULL;
    static endFunction endScanCallee = NULL;
    static endFunction endReduceCallee = NULL;
    
    void beginParallelFor(const std::string& kernelPrefix, uint64_t* kernelID);
    void endParallelFor(const uint64_t kernelID);
    void beginParallelScan(const std::string& kernelPrefix, uint64_t* kernelID);
    void endParallelScan(const uint64_t kernelID);
    void beginParallelReduce(const std::string& kernelPrefix, uint64_t* kernelID);
    void endParallelReduce(const uint64_t kernelID);
    
    void initialize();
    void finalize();
}

#endif
#endif