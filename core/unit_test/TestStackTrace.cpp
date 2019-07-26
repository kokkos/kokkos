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
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <iostream>
#include "Kokkos_Core.hpp"
#include "impl/Kokkos_stacktrace.hpp"

namespace { // (anonymous)

void f0 (std::ostream& out)
{
  out << "Top of f0" << std::endl;
}

int f1 (std::ostream& out) {
  out << "Top of f1" << std::endl;
  f0 (out);
  Kokkos::Impl::save_stacktrace ();
  f0 (out);
  return 42;
}

void f2 (std::ostream& out) {
  out << "Top of f2" << std::endl;
  const int result = f1 (out);
  out << "f2: f1 returned " << result << std::endl;
}

int f3 (std::ostream& out, const int level) {
  if (level <= 0) {
    return f1 (out);
  }
  else {
    return f3 (out, level-1);
  }
}

void f4 () {
  Kokkos::Impl::save_stacktrace ();
}

void
my_fancy_handler ()
{
  std::cerr << "I am the custom std::terminate handler." << std::endl;
  std::abort ();
}

} // namespace (anonymous)

int
main (int argc, char *argv[])
{
  using std::cout;
  using std::endl;

  Kokkos::ScopeGuard kokkosSession (argc, argv);
  bool success = true;

  f1 (cout);
  // TODO test by making sure that f1 and f2, but no other functions,
  // appear in the stack trace.

  cout << endl << "Mangled stacktrace:" << endl << endl;
  Kokkos::Impl::print_saved_stacktrace (cout);
  cout << endl << "Demangled stacktrace:" << endl << endl;
  Kokkos::Impl::print_demangled_saved_stacktrace (cout);

  f3 (cout, 4);
  // TODO test by making sure that f3 and f1, but no other functions,
  // appear in the stack trace, and that f3 appears 5 times.
  cout << endl << "Mangled stacktrace:" << endl << endl;
  Kokkos::Impl::print_saved_stacktrace (cout);
  cout << endl << "Demangled stacktrace:" << endl << endl;
  Kokkos::Impl::print_demangled_saved_stacktrace (cout);

  cout << endl << "Demangled version of \"main\": "
       << Kokkos::Impl::demangle ("main") << endl;

  if (argc > 1) {
    cout << "Test setting std::terminate handler that prints "
      "the last saved stack trace" << endl;

    f4 ();
    Kokkos::Impl::set_kokkos_terminate_handler (); // just test syntax
    Kokkos::Impl::set_kokkos_terminate_handler (my_fancy_handler);

    // TODO test that this prints "Oh noes!" and the correct stacktrace.
    std::terminate ();
  }
  else {
    if (success) {
      cout << "SUCCESS" << endl;
    }
    else {
      cout << "FAILED" << endl;
    }
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
  }
}
