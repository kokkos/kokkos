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

#ifndef KOKKOS_IMPL_KOKKOS_TUNING_HPP
#define KOKKOS_IMPL_KOKKOS_TUNING_HPP
#include <Kokkos_Macros.hpp>
#include <impl/Kokkos_Tuning_Interface.hpp>
#include <string>
#include <chrono>
namespace Kokkos {
namespace Tools {
#ifdef KOKKOS_ENABLE_TUNING
// static size_t kernel_name_context_variable_id;
// static size_t kernel_type_context_variable_id;
// static size_t time_context_variable_id;

//static tuningVariableDeclarationFunction tuningVariableDeclarationCallback =
//    nullptr;
extern tuningVariableDeclarationFunction tuningVariableDeclarationCallback;
extern tuningVariableValueFunction tuningVariableValueCallback;
extern contextVariableDeclarationFunction contextVariableDeclarationCallback;
extern contextEndFunction contextEndCallback;
extern optimizationGoalDeclarationFunction optimizationGoalCallback;
using time_point = std::chrono::time_point<std::chrono::system_clock>;
// static std::stack<time_point> timer_stack;
// static int last_microseconds;
#endif

VariableValue make_variable_value(size_t id, bool val);
VariableValue make_variable_value(size_t id, int64_t val);
VariableValue make_variable_value(size_t id, double val);
VariableValue make_variable_value(size_t id, const char* val);

void declareOptimizationGoal(const OptimizationGoal& goal);

void declareTuningVariable(const std::string& variableName, size_t uniqID,
                           VariableInfo info);

void declareContextVariable(const std::string& variableName, size_t uniqID,
                            VariableInfo info,
                            Kokkos::Tools::SetOrRange candidate_values);

void declareContextVariableValues(size_t contextId, size_t count,
                                  VariableValue* values);

void endContext(size_t contextId);

void requestTuningVariableValues(size_t contextId, size_t count,
                                 VariableValue* values,
                                 Kokkos::Tools::SetOrRange* candidate_values);

bool haveTuningTool();

size_t getNewContextId();
size_t getCurrentContextId();

size_t getNewVariableId();

}  // namespace Tools
}  // namespace Kokkos
#endif  // KOKKOS_IMPL_KOKKOS_TUNING_HPP
