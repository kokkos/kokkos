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

#include <KokkosP_Basic_Profiler_DataBase.hpp>

namespace KokkosP {
namespace Experimental {

KernelEntry::KernelEntry() {
  previous = NULL;
  next = NULL;
  time_min   = 1.0e100;
  time_max   = 0.0;
  time_total = 0.0;
  num_calls  = 0;
}

KernelEntry::KernelEntry(KernelEntry* previous_, double time, const std::string& kernel_name_, const std::string& exec_space_) {
  previous = previous_;
  next = NULL;

  if(previous!=NULL) {
    previous->next = this;
  }

  time_min   = time;
  time_max   = time;
  time_total = time;
  num_calls  = 1;

  kernel_name = kernel_name_;
  exec_space  = exec_space_;
}

bool KernelEntry::matches(const std::string& kernel_name_, const std::string& exec_space_) const {
  if( (kernel_name.compare(kernel_name_) == 0) &&
      (exec_space.compare(exec_space_) == 0) )
    return true;
  return false;
}

void KernelEntry::add_time(const double& time) {
  if (time < time_min) time_min = time;
  if (time > time_max) time_max = time;
  time_total += time;
  num_calls++;
}

void KernelEntry::print() const {
  std::cout << "KOKKOS_PROFILE: Entry: " << std::endl;
  std::cout << "KOKKOS_PROFILE:   Exec Space:  " << exec_space << std::endl;
  std::cout << "KOKKOS_PROFILE:   Kernel Name: " << kernel_name << std::endl;
  std::cout << "KOKKOS_PROFILE:   # of calls:  " << num_calls << std::endl;
  std::cout << "KOKKOS_PROFILE:   Avg Time:    " << time_total/num_calls << std::endl;
  std::cout << "KOKKOS_PROFILE:   Min Time:    " << time_min << std::endl;
  std::cout << "KOKKOS_PROFILE:   Max Time:    " << time_max << std::endl;

}

KernelEntry* get_kernel_list_head (KernelEntry* start) {
  static KernelEntry* list_head = NULL;
  if(start != NULL)
    list_head = start;
  return list_head;
}

}
}
