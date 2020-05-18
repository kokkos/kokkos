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

// This file tests the primitives of the Tuning system

#include <iostream>
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>

static std::unordered_map<size_t, Kokkos::Tools::ValueType> variableTypeInfo;
static size_t expectedVariableId;
static size_t expectedNumberOfContextVariables;
static int64_t expectedContextVariableValue;

std::string variableValueString(Kokkos::Tools::VariableValue value) {
  switch (variableTypeInfo[value.id]) {
    case Kokkos::Tools::ValueType::kokkos_value_text:
      return std::string(value.value.string_value);
    case Kokkos::Tools::ValueType::kokkos_value_integer:
      return std::to_string(value.value.int_value);
    case Kokkos::Tools::ValueType::kokkos_value_floating_point:
      return std::to_string(value.value.double_value);
    case Kokkos::Tools::ValueType::kokkos_value_boolean:
      return value.value.bool_value ? "true" : "false";
    default: return "TEST ERROR";
  }
}

int main() {
  Kokkos::initialize();
  {
    auto context = Kokkos::Tools::getNewContextId();

    auto contextVariableId = Kokkos::Tools::getNewVariableId();
    Kokkos::Tools::VariableInfo contextVariableInfo;

    contextVariableInfo.category =
        Kokkos::Tools::StatisticalCategory::kokkos_value_categorical;
    contextVariableInfo.type = Kokkos::Tools::ValueType::kokkos_value_integer;
    contextVariableInfo.valueQuantity =
        Kokkos::Tools::CandidateValueType::kokkos_value_unbounded;

    Kokkos::Tools::SetOrRange empty;

    auto tuningVariableId = Kokkos::Tools::getNewVariableId();
    Kokkos::Tools::VariableInfo tuningVariableInfo;

    tuningVariableInfo.category =
        Kokkos::Tools::StatisticalCategory::kokkos_value_categorical;
    tuningVariableInfo.type = Kokkos::Tools::ValueType::kokkos_value_integer;
    tuningVariableInfo.valueQuantity =
        Kokkos::Tools::CandidateValueType::kokkos_value_unbounded;

    // test that ID's are transmitted to the tool
    Kokkos::Tools::Experimental::set_declare_tuning_variable_callback(
        [](const char* name, const size_t id,
           Kokkos::Tools::VariableInfo info) {
          if (id != expectedVariableId) {
            throw(std::runtime_error("Tuning Variable has wrong ID"));
          }
          variableTypeInfo[id] = info.type;
        });
    Kokkos::Tools::Experimental::set_declare_context_variable_callback(
        [](const char* name, const size_t id, Kokkos::Tools::VariableInfo info,
           Kokkos::Tools::SetOrRange candidates) {
          if (id != expectedVariableId) {
            throw(std::runtime_error("Context Variable has wrong ID"));
          }
          variableTypeInfo[id] = info.type;
        });
    expectedVariableId = contextVariableId;
    Kokkos::Tools::declareContextVariable("kokkos.testing.context_variable",
                                          contextVariableId,
                                          contextVariableInfo, empty);
    expectedVariableId = tuningVariableId;
    Kokkos::Tools::declareTuningVariable("kokkos.testing.tuning_variable",
                                         tuningVariableId, tuningVariableInfo);

    // test that we correctly pass context values, and receive tuning variables
    // back in return
    Kokkos::Tools::VariableValue contextValues[] = {
        Kokkos::Tools::make_variable_value(contextVariableId, int64_t(0))};
    Kokkos::Tools::declareContextVariableValues(context, 1, contextValues);

    Kokkos::Tools::Experimental::set_request_tuning_variable_values_callback(
        [](const size_t context, const size_t num_context_variables,
           const Kokkos::Tools::VariableValue* context_values,
           const size_t num_tuning_variables,
           Kokkos::Tools::VariableValue* tuning_values,
           Kokkos::Tools::SetOrRange* candidate_values) {
          if (context_values[0].value.int_value !=
              expectedContextVariableValue) {
            throw std::runtime_error(
                "Context variables not correctly passed to tuning callbacks");
          }
          int tuningVariableSetSize = candidate_values[0].set.size;
          // tuning methodology via https://xkcd.com/221/
          tuning_values[0] =
              candidate_values[0].set.values[4 % tuningVariableSetSize];
        });

    Kokkos::Tools::VariableValue tuningValues[] = {
        Kokkos::Tools::make_variable_value(tuningVariableId, int64_t(0))};

    Kokkos::Tools::VariableValue viableOptions[] = {
        Kokkos::Tools::make_variable_value(tuningVariableId, int64_t(0)),
        Kokkos::Tools::make_variable_value(tuningVariableId, int64_t(1)),
        Kokkos::Tools::make_variable_value(tuningVariableId, int64_t(2)),
        Kokkos::Tools::make_variable_value(tuningVariableId, int64_t(3)),
        Kokkos::Tools::make_variable_value(tuningVariableId, int64_t(4)),
        Kokkos::Tools::make_variable_value(tuningVariableId, int64_t(5)),
        Kokkos::Tools::make_variable_value(tuningVariableId, int64_t(6)),
        Kokkos::Tools::make_variable_value(tuningVariableId, int64_t(7)),
        Kokkos::Tools::make_variable_value(tuningVariableId, int64_t(8)),
    };

    Kokkos::Tools::SetOrRange tuningCandidates[1];

    tuningCandidates[0].set.id     = tuningVariableId;
    tuningCandidates[0].set.size   = 9;
    tuningCandidates[0].set.values = viableOptions;

    Kokkos::Tools::requestTuningVariableValues(context, 1, tuningValues,
                                               tuningCandidates);
    std::cout << tuningValues[0].value.int_value << ","
              << viableOptions[5].value.int_value << std::endl;
    if (tuningValues[0].value.int_value != viableOptions[4].value.int_value) {
      throw std::runtime_error("Tuning value return is incorrect");
    }

    Kokkos::Tools::endContext(context);

    // test nested contexts
    auto outerContext = Kokkos::Tools::getNewContextId();
    auto innerContext = Kokkos::Tools::getNewContextId();

    auto secondContextVariableId = Kokkos::Tools::getNewVariableId();

    Kokkos::Tools::VariableInfo secondContextVariableInfo;

    secondContextVariableInfo.category =
        Kokkos::Tools::StatisticalCategory::kokkos_value_categorical;
    secondContextVariableInfo.type =
        Kokkos::Tools::ValueType::kokkos_value_integer;
    secondContextVariableInfo.valueQuantity =
        Kokkos::Tools::CandidateValueType::kokkos_value_unbounded;

    expectedVariableId = secondContextVariableId;
    Kokkos::Tools::declareContextVariable(
        "kokkos.testing.second_context_variable", secondContextVariableId,
        secondContextVariableInfo, empty);

    Kokkos::Tools::VariableValue contextValueTwo[] = {
        Kokkos::Tools::make_variable_value(secondContextVariableId,
                                           int64_t(1))};

    Kokkos::Tools::Experimental::set_request_tuning_variable_values_callback(
        [](const size_t context, const size_t num_context_variables,
           const Kokkos::Tools::VariableValue* context_values,
           const size_t num_tuning_variables,
           Kokkos::Tools::VariableValue* tuning_values,
           Kokkos::Tools::SetOrRange* candidate_values) {
          std::cout << "Expect " << expectedNumberOfContextVariables
                    << ", have " << num_context_variables << std::endl;
          if (num_context_variables != expectedNumberOfContextVariables) {
            throw(
                std::runtime_error("Incorrect number of context variables in "
                                   "nested tuning contexts"));
          }
        });
    Kokkos::Tools::declareContextVariableValues(outerContext, 1, contextValues);
    expectedNumberOfContextVariables = 1;
    Kokkos::Tools::requestTuningVariableValues(outerContext, 1, tuningValues,
                                               tuningCandidates);
    Kokkos::Tools::declareContextVariableValues(innerContext, 1,
                                                contextValueTwo);
    expectedNumberOfContextVariables = 2;
    Kokkos::Tools::requestTuningVariableValues(innerContext, 1, tuningValues,
                                               tuningCandidates);
  }  // end Kokkos block

  Kokkos::finalize();
}
