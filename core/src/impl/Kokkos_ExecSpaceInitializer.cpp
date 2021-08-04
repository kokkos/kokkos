/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
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
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include "Kokkos_Core.hpp"
#include "Kokkos_ExecSpaceInitializer.hpp"

#include <sstream>

namespace Kokkos {
namespace Impl {
void ExecSpaceInitializerBase::initialize(const InitArguments &args) {
  if (m_is_finalized) {
    std::stringstream abort_msg;
    abort_msg << "Calling Kokkos::initialize after ExecSpace \"";
    print_exec_space_name(abort_msg);
    abort_msg << "\" was already finalized is illegal\n";
    Kokkos::abort(abort_msg.str().c_str());
  }

  if (m_is_initialized) return;

  m_is_initialized = true;

  do_initialize(args);
}

void ExecSpaceInitializerBase::finalize(const bool all_spaces) {
  if (!m_is_initialized) {
    std::stringstream abort_msg;
    abort_msg << "Calling Kokkos::finalize without ExecSpace \"";
    print_exec_space_name(abort_msg);
    abort_msg << "\" having been initialized is illegal\n";
    Kokkos::abort(abort_msg.str().c_str());
  }

  if (m_is_finalized) return;

  m_is_finalized = true;

  do_finalize(all_spaces);
}
}  // namespace Impl
}  // namespace Kokkos
