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

#ifndef KOKKOS_CORE_PERFTEST_BENCHMARK_CONTEXT_HPP
#define KOKKOS_CORE_PERFTEST_BENCHMARK_CONTEXT_HPP

#include <string>

#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>

namespace KokkosBenchmark {

/// \brief Remove unwanted spaces and colon signs from input string. In case of
/// invalid input it will return an empty string.
std::string remove_unwanted_characters(std::string str) {
  auto from = str.find_first_not_of(" :");
  auto to   = str.find_last_not_of(" :");

  if (from == std::string::npos || to == std::string::npos) {
    return "";
  }

  // return extracted part of string without unwanted spaces and colon signs
  return str.substr(from, to + 1);
}

/// \brief Extract all key:value pairs from kokkos configuration and add it to
/// the benchmark context
void add_kokkos_configuration(bool verbose) {
  std::ostringstream msg;
  Kokkos::print_configuration(msg, verbose);

  // Iterate over lines returned from kokkos and extract key:value pairs
  std::stringstream ss{msg.str()};
  for (std::string line; std::getline(ss, line, '\n');) {
    auto found = line.find_first_of(':');
    if (found != std::string::npos) {
      auto val = remove_unwanted_characters(line.substr(found + 1));
      // Ignore line without value, for example a category name
      if (!val.empty()) {
        benchmark::AddCustomContext(
            remove_unwanted_characters(line.substr(0, found)), val);
      }
    }
  }
}

/// \brief Gather all context information and add it to benchmark context data
void add_benchmark_context(bool verbose = false) {
  // Add Kokkos configuration to benchmark context data
  add_kokkos_configuration(verbose);
}

}  // namespace KokkosBenchmark

#endif
