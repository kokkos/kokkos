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
#include <Kokkos_Macros.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <iostream>
#include <impl/Kokkos_Tuning.hpp>
namespace Kokkos {
namespace Tools {

tuningVariableDeclarationFunction tuningVariableDeclarationCallback;
tuningVariableValueFunction tuningVariableValueCallback;
contextVariableDeclarationFunction contextVariableDeclarationCallback;
contextEndFunction contextEndCallback;
optimizationGoalDeclarationFunction optimizationGoalCallback;

static size_t& getContextCounter() {
  static size_t x;
  return x;
}
static size_t& getVariableCounter() {
  static size_t x;
  return ++x;
}

size_t getNewContextId() { return ++getContextCounter(); }
size_t getCurrentContextId() { return getContextCounter(); }
void decrementCurrentContextId() { --getContextCounter(); }
size_t getNewVariableId() { return getVariableCounter(); }

static std::unordered_map<size_t, std::unordered_set<size_t>>
    features_per_context;
static std::unordered_set<size_t> active_features;
static std::unordered_map<size_t, VariableValue> feature_values;

void declareTuningVariable(const std::string& variableName, size_t uniqID,
                           VariableInfo info) {
#ifdef KOKKOS_ENABLE_TUNING
  if (tuningVariableDeclarationCallback != nullptr) {
    (*tuningVariableDeclarationCallback)(variableName.c_str(), uniqID, info);
  }
#else
  (void)variableName;
  (void)uniqID;
  (void)info;
#endif
}

void declareContextVariable(const std::string& variableName, size_t uniqID,
                            VariableInfo info,
                            Kokkos::Tools::SetOrRange candidate_values) {
#ifdef KOKKOS_ENABLE_TUNING
  if (contextVariableDeclarationCallback != nullptr) {
    (*contextVariableDeclarationCallback)(variableName.c_str(), uniqID, info,
                                          candidate_values);
  }
#else
  (void)variableName;
  (void)uniqID;
  (void)info;
  (void)candidate_values;
#endif
}

void declareContextVariableValues(size_t contextId, size_t count,
                                  VariableValue* values) {
#ifdef KOKKOS_ENABLE_TUNING
  if (features_per_context.find(contextId) == features_per_context.end()) {
    features_per_context[contextId] = std::unordered_set<size_t>();
  }
  for (size_t x = 0; x < count; ++x) {
    features_per_context[contextId].insert(values[x].id);
    active_features.insert(values[x].id);
    feature_values[values[x].id] = values[x];
  }
#else
  (void)contextId;
  (void)count;
  (void)values;
#endif
}
#include <iostream>
void requestTuningVariableValues(size_t contextId, size_t count,
                                 VariableValue* values,
                                 Kokkos::Tools::SetOrRange* candidate_values) {
#ifdef KOKKOS_ENABLE_TUNING

  std::vector<size_t> context_ids;
  std::vector<VariableValue> context_values;
  for (auto id : active_features) {
    context_values.push_back(feature_values[id]);
  }
  if (tuningVariableValueCallback != nullptr) {
    (*tuningVariableValueCallback)(contextId, context_values.size(),
                                   context_values.data(), count, values,
                                   candidate_values);
  }
#else
  (void)contextId;
  (void)count;
  (void)values;
  (void)candidate_values;
#endif
}

void endContext(size_t contextId) {
#ifdef KOKKOS_ENABLE_TUNING
  for (auto id : features_per_context[contextId]) {
    active_features.erase(id);
  }
  if (Kokkos::Tools::contextEndCallback != nullptr) {
    (*contextEndCallback)(contextId);
  }
  decrementCurrentContextId();
#else
  (void)contextId;
#endif
}

bool haveTuningTool() {
#ifdef KOKKOS_ENABLE_TUNING
  return (tuningVariableValueCallback != nullptr);
#else
  return false;
#endif
}

VariableValue make_variable_value(size_t id, bool val) {
  VariableValue variable_value;
  variable_value.id               = id;
  variable_value.value.bool_value = val;
  return variable_value;
}
VariableValue make_variable_value(size_t id, int64_t val) {
  VariableValue variable_value;
  variable_value.id              = id;
  variable_value.value.int_value = val;
  return variable_value;
}
VariableValue make_variable_value(size_t id, double val) {
  VariableValue variable_value;
  variable_value.id                 = id;
  variable_value.value.double_value = val;
  return variable_value;
}
VariableValue make_variable_value(size_t id, const char* val) {
  VariableValue variable_value;
  variable_value.id                 = id;
  variable_value.value.string_value = val;
  return variable_value;
}

size_t getNewContextId();
size_t getCurrentContextId();
void decrementCurrentContextId();
size_t getNewVariableId();

void declareOptimizationGoal(const OptimizationGoal& goal) {
#ifdef KOKKOS_ENABLE_TUNING
  if (Kokkos::Tools::optimizationGoalCallback != nullptr) {
    (*optimizationGoalCallback)(goal);
  }
#else
  (void)goal;
#endif
}

}  // end namespace Tools

}  // end namespace Kokkos
