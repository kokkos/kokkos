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

#include <iostream>
#include <string>
#include <impl/Kokkos_Timer.hpp>
#include <KokkosP_Basic_Profiler.hpp>
#include <KokkosP_Basic_Profiler_DataBase.hpp>

namespace KokkosP {
namespace Experimental {

Kokkos::Impl::Timer* get_timer() {
  static Kokkos::Impl::Timer timer;
  return &timer;
}

void profile_begin_kernel(const std::string& kernel_name, const std::string& exec_space) {
  Kokkos::Impl::Timer* timer = get_timer();
  timer->reset();
}

void profile_end_kernel(const std::string& kernel_name, const std::string& exec_space) {
  Kokkos::Impl::Timer* timer = get_timer();
  double time = timer->seconds();

  KernelEntry* entry = get_kernel_list_head();
  if(entry == NULL) {
    KernelEntry* entry_new = new KernelEntry(entry,time,kernel_name,exec_space);
    get_kernel_list_head(entry_new);
    return;
  }

  bool found = entry->matches(kernel_name,exec_space);
  while(!found && (entry->next!=NULL) ) {
    entry = entry->next;
    found = entry->matches(kernel_name,exec_space);
  }
  if(found)
    entry->add_time(time);
  else
    entry->next = new KernelEntry(entry,time,kernel_name,exec_space);
}

void profile_initialize() {
}

void profile_finalize() {
  KernelEntry* entry = get_kernel_list_head();
  while( entry != NULL ) {
    entry->print();
    if(entry->next != NULL) {
      entry = entry->next;
      delete entry->previous;
    } else {
      delete entry;
      entry = NULL;
    }
  }
}
}
}
