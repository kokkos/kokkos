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

#ifndef KOKKOS_STACKTRACE_HPP
#define KOKKOS_STACKTRACE_HPP

#include <functional>
#include <ostream>
#include <string>

namespace Kokkos {
namespace Impl {

/// \brief Return the demangled version of the input symbol, or the
///   original input if demangling is not possible.
std::string demangle(const std::string& name);

/// \brief Save the current stacktrace.
///
/// You may only save one stacktrace at a time.  If you call this
/// twice, the second call will overwrite the result of the first
/// call.
void save_stacktrace();

/// \brief Print the raw form of the currently saved stacktrace, if
///   any, to the given output stream.
void print_saved_stacktrace(std::ostream& out);

/// \brief Print the currently saved, demangled stacktrace, if any, to
///   the given output stream.
///
/// Demangling is best effort only.
void print_demangled_saved_stacktrace(std::ostream& out);

/// \brief Set the std::terminate handler so that it prints the
///   currently saved stack trace, then calls user_post.
///
/// This is useful if you want to call, say, MPI_Abort instead of
/// std::abort.  The MPI Standard frowns upon calling MPI functions
/// without including their header file, and Kokkos does not depend on
/// MPI, so there's no way for Kokkos to depend on MPI_Abort in a
/// portable way.
void set_kokkos_terminate_handler(std::function<void()> user_post = nullptr);

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_STACKTRACE_HPP
