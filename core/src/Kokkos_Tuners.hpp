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

#ifndef KOKKOS_KOKKOS_TUNERS_HPP
#define KOKKOS_KOKKOS_TUNERS_HPP

#include <Kokkos_Macros.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_ExecPolicy.hpp>
#include <impl/Kokkos_Profiling_Interface.hpp>

#include <array>
#include <tuple>
#include <string>
#include <vector>

namespace Kokkos {
namespace Tools {

namespace Experimental {

// forward declarations
SetOrRange make_candidate_set(size_t size, int64_t* data);
size_t declare_output_type(const std::string&,
                           Kokkos::Tools::Experimental::VariableInfo);
void request_output_values(size_t, size_t,
                           Kokkos::Tools::Experimental::VariableValue*);
void end_context(size_t context_id);
class TeamSizeTuner {
 public:
  inline void declare_output_type(const std::string& name,
                                  std::vector<int64_t> candidates) {
    Kokkos::Tools::Experimental::VariableInfo variable_info;

    variable_info.type =
        Kokkos::Tools::Experimental::ValueType::kokkos_value_int64;
    variable_info.category =
        Kokkos::Tools::Experimental::StatisticalCategory::kokkos_value_ratio;
    variable_info.valueQuantity =
        Kokkos::Tools::Experimental::CandidateValueType::kokkos_value_set;
    variable_info.candidates = Kokkos::Tools::Experimental::make_candidate_set(
        candidates.size(), candidates.data());
    type_id = Kokkos::Tools::Experimental::declare_output_type(
        name + "_team_size", variable_info);
  }
  TeamSizeTuner& operator=(const TeamSizeTuner& other) = default;
  TeamSizeTuner()                                      = default;
  TeamSizeTuner(const TeamSizeTuner& other)            = default;
  template <typename Functor, typename TagType, typename... Properties>
  TeamSizeTuner(const std::string& name,
                Kokkos::TeamPolicy<Properties...>& policy,
                const Functor& functor, const TagType& tag) {
    using PolicyType           = Kokkos::TeamPolicy<Properties...>;
    auto initial_vector_length = policy.vector_length();
    if (initial_vector_length < 1) {
      policy.impl_set_vector_length(1);
    }
    /**
     * Here we attempt to enumerate all of the possible configurations
     * to expose to an autotuner. There are three possibilities
     *
     * 1) We're tuning both vector length and team size
     * 2) We're tuning vector length but not team size
     * 3) We're tuning team size but not vector length
     *
     * (In the fourth case where nothing is tuned
     * this function won't be called)
     *
     * The set of valid team sizes is dependent on
     * a vector length, so this leads to three
     * algorithms
     *
     * 1) Loop over vector lengths to get the set
     *    of team sizes for each vector length,
     *    add it all to the set
     * 2) Loop over vector lengths to see if the
     *    provided team size is valid for that
     *    vector length. If so, add it
     * 3) A special case of (1) in which we only
     *    have one vector length
     *
     */
    std::vector<int64_t> configuration_indices;

    if (policy.auto_vector_length()) {
      policy.impl_set_vector_length(1);  // TODO: find a heuristic
    }

    // in all cases, start by pushing the default
    configuration_indices.push_back(0);  // default option index
    if (policy.auto_team_size()) {       // case 1 or 3
      auto default_team_size = policy.team_size_recommended(functor, tag);
      configurations.push_back(
          std::make_pair(default_team_size, policy.vector_length()));
    } else {  // case 2
      configurations.push_back(
          std::make_pair(policy.team_size(), policy.vector_length()));
    }

    auto max_vector_length = PolicyType::vector_length_max();
    std::vector<int64_t> allowed_vector_lengths;

    if (policy.auto_vector_length()) {  // case 1 or 2
      for (int vector_length = max_vector_length; vector_length >= 1;
           vector_length /= 2) {
        policy.impl_set_vector_length(vector_length);
        /**
         * Figuring out whether a vector length is valid depends
         * on whether we're in case 1 (tune everything) or 2 (just tune vector
         * length)
         *
         * If we're tuning everything, all legal vector lengths are valid.
         * If we're just tuning vector length, we need to check that if we
         * set this vector length, the team size provided will be valid.
         *
         * These are the left and right hand sides of the "or" in this
         * conditional, respectively.
         */
        auto max_team_size = policy.team_size_max(functor, tag);
        if ((policy.auto_team_size()) ||
            (policy.team_size() <= max_team_size)) {
          allowed_vector_lengths.push_back(vector_length);
        }
      }
    } else {  // case 3, there's only one vector length to care about
      allowed_vector_lengths.push_back(policy.vector_length());
    }
    int index = 1;

    for (const auto vector_length : allowed_vector_lengths) {
      policy.impl_set_vector_length(vector_length);
      auto max_team_size = policy.team_size_max(functor, tag);
      if (policy.auto_team_size()) {  // case 1 or 3, try all legal team sizes
        for (int team_size = max_team_size; team_size >= 1; team_size /= 2) {
          configuration_indices.push_back(index++);
          configurations.push_back(std::make_pair(team_size, vector_length));
        }
      } else {  // case 2, just try the provided team size
        configuration_indices.push_back(index++);
        configurations.push_back(
            std::make_pair(policy.team_size(), vector_length));
      }
    }

    declare_output_type(name, configuration_indices);
    policy.impl_set_vector_length(initial_vector_length);
  }

  template <typename... Properties>
  void tune(Kokkos::TeamPolicy<Properties...>& policy,
            const size_t context_id) {
    Kokkos::Tools::Experimental::VariableValue value_index{type_id, 0};
    Kokkos::Tools::Experimental::request_output_values(context_id, 1,
                                                       &value_index);
    int index          = value_index.value.int_value;
    auto configuration = configurations[index];
    auto team_size     = configuration.first;
    auto vector_length = configuration.second;
    if (vector_length > 0) {
      policy.impl_set_team_size(team_size);
      policy.impl_set_vector_length(vector_length);
    }
  }
  void end(size_t context_id) {
    Kokkos::Tools::Experimental::end_context(context_id);
  }

 private:
  size_t type_id;
  std::vector<std::pair<int64_t, int64_t>> configurations;
};

}  // namespace Experimental
}  // namespace Tools
}  // namespace Kokkos

#endif
