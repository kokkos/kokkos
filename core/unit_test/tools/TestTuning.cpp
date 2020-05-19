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

static size_t expectedVariableId;
static size_t expectedNumberOfContextVariables;
static int64_t expectedContextVariableValue;
static std::unordered_map<size_t, Kokkos::Tools::SetOrRange>
    candidate_value_map;

int main() {
  Kokkos::initialize();
  {
    auto context = Kokkos::Tools::get_new_context_id();

    Kokkos::Tools::VariableInfo contextVariableInfo;

    contextVariableInfo.category =
        Kokkos::Tools::StatisticalCategory::kokkos_value_categorical;
    contextVariableInfo.type = Kokkos::Tools::ValueType::kokkos_value_integer;
    contextVariableInfo.valueQuantity =
        Kokkos::Tools::CandidateValueType::kokkos_value_unbounded;

    Kokkos::Tools::SetOrRange empty;

    Kokkos::Tools::VariableInfo tuningVariableInfo;

    tuningVariableInfo.category =
        Kokkos::Tools::StatisticalCategory::kokkos_value_categorical;
    tuningVariableInfo.type = Kokkos::Tools::ValueType::kokkos_value_integer;
    tuningVariableInfo.valueQuantity =
        Kokkos::Tools::CandidateValueType::kokkos_value_set;

    std::vector<int64_t> candidate_value_vector = {0, 1, 2, 3, 4,
                                                   5, 6, 7, 8, 9};

    Kokkos::Tools::SetOrRange allowed_values =
        Kokkos::Tools::make_candidate_set(candidate_value_vector.size(),
                                          candidate_value_vector.data());
    // test that ID's are transmitted to the tool
    Kokkos::Tools::Experimental::set_declare_output_type_callback(
        [](const char* name, const size_t id,
           Kokkos::Tools::VariableInfo info) {
          if (info.type != Kokkos::Tools::ValueType::kokkos_value_integer) {
            throw(std::runtime_error("Tuning Variable has wrong type"));
          }
        });
    Kokkos::Tools::Experimental::set_declare_input_type_callback(
        [](const char* name, const size_t id,
           Kokkos::Tools::VariableInfo info) {
          if (info.type != Kokkos::Tools::ValueType::kokkos_value_integer) {
            throw(std::runtime_error("Context Variable has wrong type"));
          }
        });
    contextVariableInfo.candidates = empty;
    tuningVariableInfo.candidates  = allowed_values;
    auto contextVariableId         = Kokkos::Tools::declare_input_type(
        "kokkos.testing.context_variable", contextVariableInfo);
    auto tuningVariableId = Kokkos::Tools::declare_output_type(
        "kokkos.testing.tuning_variable", tuningVariableInfo);

    // test that we correctly pass context values, and receive tuning variables
    // back in return
    Kokkos::Tools::VariableValue contextValues[] = {
        Kokkos::Tools::make_variable_value(contextVariableId, int64_t(0))};
    Kokkos::Tools::set_input_values(context, 1, contextValues);

    Kokkos::Tools::Experimental::set_request_output_values_callback(
        [](const size_t context, const size_t num_context_variables,
           const Kokkos::Tools::VariableValue* context_values,
           const size_t num_tuning_variables,
           Kokkos::Tools::VariableValue* tuning_values) {
          auto candidate_values = tuning_values[0].metadata->candidates;
          if (context_values[0].value.int_value !=
              expectedContextVariableValue) {
            throw std::runtime_error(
                "Context variables not correctly passed to tuning callbacks");
          }
          int tuningVariableSetSize = candidate_values.set.size;
          std::cout << "Set of size " << tuningVariableSetSize << std::endl;
          // tuning methodology via https://xkcd.com/221/
          tuning_values[0].value.int_value =
              candidate_values.set.values.int_value[4 % tuningVariableSetSize];
        });

    Kokkos::Tools::VariableValue tuningValues[] = {
        Kokkos::Tools::make_variable_value(tuningVariableId, int64_t(0))};

    Kokkos::Tools::request_output_values(context, 1, tuningValues);
    std::cout << tuningValues[0].value.int_value << ","
              << candidate_value_vector[4] << std::endl;
    if (tuningValues[0].value.int_value != candidate_value_vector[4]) {
      throw std::runtime_error("Tuning value return is incorrect");
    }

    Kokkos::Tools::end_context(context);

    // test nested contexts
    auto outerContext = Kokkos::Tools::get_new_context_id();
    auto innerContext = Kokkos::Tools::get_new_context_id();

    Kokkos::Tools::VariableInfo secondContextVariableInfo;

    secondContextVariableInfo.category =
        Kokkos::Tools::StatisticalCategory::kokkos_value_categorical;
    secondContextVariableInfo.type =
        Kokkos::Tools::ValueType::kokkos_value_integer;
    secondContextVariableInfo.valueQuantity =
        Kokkos::Tools::CandidateValueType::kokkos_value_unbounded;
    secondContextVariableInfo.candidates = empty;
    auto secondContextVariableId         = Kokkos::Tools::declare_output_type(
        "kokkos.testing.second_context_variable", secondContextVariableInfo);

    Kokkos::Tools::VariableValue contextValueTwo[] = {
        Kokkos::Tools::make_variable_value(secondContextVariableId,
                                           int64_t(1))};

    Kokkos::Tools::Experimental::set_request_output_values_callback(
        [](const size_t context, const size_t num_context_variables,
           const Kokkos::Tools::VariableValue* context_values,
           const size_t num_tuning_variables,
           Kokkos::Tools::VariableValue* tuning_values) {
          std::cout << "Expect " << expectedNumberOfContextVariables
                    << ", have " << num_context_variables << std::endl;
          if (num_context_variables != expectedNumberOfContextVariables) {
            throw(
                std::runtime_error("Incorrect number of context variables in "
                                   "nested tuning contexts"));
          }
        });
    Kokkos::Tools::set_input_values(outerContext, 1, contextValues);
    expectedNumberOfContextVariables = 1;
    Kokkos::Tools::request_output_values(outerContext, 1, tuningValues);
    Kokkos::Tools::set_input_values(innerContext, 1, contextValueTwo);
    expectedNumberOfContextVariables = 2;
    Kokkos::Tools::request_output_values(innerContext, 1, tuningValues);
  }  // end Kokkos block

  Kokkos::finalize();
}
