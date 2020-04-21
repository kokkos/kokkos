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

#ifndef KOKKOS_TUNING_INTERFACE_HPP
#define KOKKOS_TUNING_INTERFACE_HPP

#include <impl/Kokkos_Tuning_C_Interface.h>

namespace Kokkos {
namespace Tools {

using ValueSet            = Kokkos_Tuning_ValueSet;
using ValueRange          = Kokkos_Tuning_ValueRange;
using StatisticalCategory = Kokkos_Tuning_VariableInfo_StatisticalCategory;
using ValueType           = Kokkos_Tuning_VariableInfo_ValueType;
using CandidateValueType  = Kokkos_Tuning_VariableInfo_CandidateValueType;
using SetOrRange          = Kokkos_Tuning_VariableInfo_SetOrRange;
using VariableInfo        = Kokkos_Tuning_VariableInfo;
using OptimizationGoal    = Kokkos_Tuning_OptimzationGoal;
// TODO DZP: VariableInfo subclasses to automate some of this

using VariableValue = Kokkos_Tuning_VariableValue;

VariableValue make_variable_value(size_t id, bool val);
VariableValue make_variable_value(size_t id, int64_t val);
VariableValue make_variable_value(size_t id, double val);
VariableValue make_variable_value(size_t id, const char* val);

using tuningVariableDeclarationFunction =
    Kokkos_Tuning_tuningVariableDeclarationFunction;
using contextVariableDeclarationFunction =
    Kokkos_Tuning_contextVariableDeclarationFunction;
using tuningVariableValueFunction  = Kokkos_Tuning_tuningVariableValueFunction;
using contextVariableValueFunction = Kokkos_Tuning_contextVariableValueFunction;
using contextEndFunction           = Kokkos_Tuning_contextEndFunction;
using optimizationGoalDeclarationFunction =
    Kokkos_Tuning_optimizationGoalDeclarationFunction;

}  // end namespace Tools

}  // end namespace Kokkos

#endif  // KOKKOS_TUNING_INTERFACE_HPP
