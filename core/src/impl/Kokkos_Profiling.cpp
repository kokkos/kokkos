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
#include <impl/Kokkos_Profiling.hpp>
#if defined(KOKKOS_ENABLE_PROFILING)
#include <dlfcn.h>
#endif
#include <vector>
#include <array>
#include <stack>
#include <chrono>

#if defined(KOKKOS_ENABLE_PROFILING)
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
namespace Kokkos {
namespace Tools {

#ifdef KOKKOS_ENABLE_TUNING
static size_t kernel_name_context_variable_id;
static size_t kernel_type_context_variable_id;
static size_t time_context_variable_id;
using time_point = std::chrono::time_point<std::chrono::system_clock>;
static std::stack<time_point> timer_stack;
static int last_microseconds;
#endif
}  // namespace Tools

namespace Tools {

namespace Experimental {

static EventSet current_callbacks;
static EventSet backup_callbacks;
static EventSet no_profiling;

bool eventSetsEqual(const EventSet& l, const EventSet& r) {
  return l.init == r.init && l.finalize == r.finalize &&
         l.begin_parallel_for == r.begin_parallel_for &&
         l.end_parallel_for == r.end_parallel_for &&
         l.begin_parallel_reduce == r.begin_parallel_reduce &&
         l.end_parallel_reduce == r.end_parallel_reduce &&
         l.begin_parallel_scan == r.begin_parallel_scan &&
         l.end_parallel_scan == r.end_parallel_scan &&
         l.push_region == r.push_region && l.pop_region == r.pop_region &&
         l.allocate_data == r.allocate_data &&
         l.deallocate_data == r.deallocate_data &&
         l.create_profile_section == r.create_profile_section &&
         l.start_profile_section == r.start_profile_section &&
         l.stop_profile_section == r.stop_profile_section &&
         l.destroy_profile_section == r.destroy_profile_section &&
         l.profile_event == r.profile_event &&
         l.begin_deep_copy == r.begin_deep_copy &&
         l.end_deep_copy == r.end_deep_copy &&
         l.declare_context_variable == r.declare_context_variable &&
         l.declare_tuning_variable == r.declare_tuning_variable &&
         l.end_tuning_context == r.end_tuning_context &&
         l.request_tuning_values == r.request_tuning_values &&
         l.declare_optimization_goal == r.declare_optimization_goal;
}
}  // namespace Experimental
bool profileLibraryLoaded() {
  return !Experimental::eventSetsEqual(Experimental::current_callbacks,
                                       Experimental::no_profiling);
}

void beginParallelFor(const std::string& kernelPrefix, const uint32_t devID,
                      uint64_t* kernelID) {
  if (Experimental::current_callbacks.begin_parallel_for != nullptr) {
    Kokkos::fence();
    (*Experimental::current_callbacks.begin_parallel_for)(kernelPrefix.c_str(),
                                                          devID, kernelID);
#ifdef KOKKOS_ENABLE_TUNING
    Tools::timer_stack.push(std::chrono::system_clock::now());
    Kokkos::Tools::VariableValue contextValues[] = {
        Kokkos::Tools::make_variable_value(
            Kokkos::Tools::kernel_name_context_variable_id,
            kernelPrefix.c_str()),
        Kokkos::Tools::make_variable_value(
            Kokkos::Tools::kernel_type_context_variable_id, "parallel_for")};
    Kokkos::Tools::declareContextVariableValues(Tools::getNewContextId(), 2,
                                                contextValues);
#endif
  }
}

void endParallelFor(const uint64_t kernelID) {
  if (Experimental::current_callbacks.end_parallel_for != nullptr) {
    Kokkos::fence();
    (*Experimental::current_callbacks.end_parallel_for)(kernelID);
#ifdef KOKKOS_ENABLE_TUNING
    Tools::last_microseconds =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now() - Tools::timer_stack.top())
            .count();
    Tools::timer_stack.pop();
    std::array<Kokkos::Tools::VariableValue, 1> value = {
        Kokkos::Tools::make_variable_value(
            Tools::kernel_type_context_variable_id,
            static_cast<double>(Tools::last_microseconds))};
    Tools::declareContextVariableValues(Kokkos::Tools::getCurrentContextId(), 1,
                                        value.data());
    Kokkos::Tools::endContext(Kokkos::Tools::getCurrentContextId());
#endif
  }
}

void beginParallelScan(const std::string& kernelPrefix, const uint32_t devID,
                       uint64_t* kernelID) {
  if (Experimental::current_callbacks.begin_parallel_scan != nullptr) {
    Kokkos::fence();
    (*Experimental::current_callbacks.begin_parallel_scan)(kernelPrefix.c_str(),
                                                           devID, kernelID);
#ifdef KOKKOS_ENABLE_TUNING
    Tools::timer_stack.push(std::chrono::system_clock::now());
    Kokkos::Tools::VariableValue contextValues[] = {
        Kokkos::Tools::make_variable_value(
            Kokkos::Tools::kernel_name_context_variable_id,
            kernelPrefix.c_str()),
        Kokkos::Tools::make_variable_value(
            Kokkos::Tools::kernel_type_context_variable_id, "parallel_for")};
    Kokkos::Tools::declareContextVariableValues(
        Kokkos::Tools::getNewContextId(), 2, contextValues);
#endif
  }
}

void endParallelScan(const uint64_t kernelID) {
  if (Experimental::current_callbacks.end_parallel_scan != nullptr) {
    Kokkos::fence();
    (*Experimental::current_callbacks.end_parallel_scan)(kernelID);
#ifdef KOKKOS_ENABLE_TUNING
    Tools::last_microseconds =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now() - Tools::timer_stack.top())
            .count();
    Tools::timer_stack.pop();
    std::array<Kokkos::Tools::VariableValue, 1> value = {
        Kokkos::Tools::make_variable_value(
            Tools::kernel_type_context_variable_id,
            static_cast<double>(Tools::last_microseconds))};
    Tools::declareContextVariableValues(Kokkos::Tools::getCurrentContextId(), 1,
                                        value.data());
    Kokkos::Tools::endContext(Kokkos::Tools::getCurrentContextId());
#endif
  }
}

void beginParallelReduce(const std::string& kernelPrefix, const uint32_t devID,
                         uint64_t* kernelID) {
  if (Experimental::current_callbacks.begin_parallel_reduce != nullptr) {
    Kokkos::fence();
    (*Experimental::current_callbacks.begin_parallel_reduce)(
        kernelPrefix.c_str(), devID, kernelID);
#ifdef KOKKOS_ENABLE_TUNING
    Tools::timer_stack.push(std::chrono::system_clock::now());
    Kokkos::Tools::VariableValue contextValues[] = {
        Kokkos::Tools::make_variable_value(
            Kokkos::Tools::kernel_name_context_variable_id,
            kernelPrefix.c_str()),
        Kokkos::Tools::make_variable_value(
            Kokkos::Tools::kernel_type_context_variable_id, "parallel_for")};
    Kokkos::Tools::declareContextVariableValues(
        Kokkos::Tools::getNewContextId(), 2, contextValues);
#endif
  }
}

void endParallelReduce(const uint64_t kernelID) {
  if (Experimental::current_callbacks.end_parallel_reduce != nullptr) {
    Kokkos::fence();
    (*Experimental::current_callbacks.end_parallel_reduce)(kernelID);
#ifdef KOKKOS_ENABLE_TUNING
    Tools::last_microseconds =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now() - Tools::timer_stack.top())
            .count();
    Tools::timer_stack.pop();
    std::array<Kokkos::Tools::VariableValue, 1> value = {
        Kokkos::Tools::make_variable_value(
            Tools::kernel_type_context_variable_id,
            static_cast<double>(Tools::last_microseconds))};
    Tools::declareContextVariableValues(Kokkos::Tools::getCurrentContextId(), 1,
                                        value.data());
    Kokkos::Tools::endContext(Kokkos::Tools::getCurrentContextId());
#endif
  }
}

void pushRegion(const std::string& kName) {
  if (Experimental::current_callbacks.push_region != nullptr) {
    Kokkos::fence();
    (*Experimental::current_callbacks.push_region)(kName.c_str());
  }
}

void popRegion() {
  if (Experimental::current_callbacks.pop_region != nullptr) {
    Kokkos::fence();
    (*Experimental::current_callbacks.pop_region)();
  }
}

void allocateData(const SpaceHandle space, const std::string label,
                  const void* ptr, const uint64_t size) {
  if (Experimental::current_callbacks.allocate_data != nullptr) {
    (*Experimental::current_callbacks.allocate_data)(space, label.c_str(), ptr,
                                                     size);
  }
}

void deallocateData(const SpaceHandle space, const std::string label,
                    const void* ptr, const uint64_t size) {
  if (Experimental::current_callbacks.deallocate_data != nullptr) {
    (*Experimental::current_callbacks.deallocate_data)(space, label.c_str(),
                                                       ptr, size);
  }
}

void beginDeepCopy(const SpaceHandle dst_space, const std::string dst_label,
                   const void* dst_ptr, const SpaceHandle src_space,
                   const std::string src_label, const void* src_ptr,
                   const uint64_t size) {
  if (Experimental::current_callbacks.begin_deep_copy != nullptr) {
    (*Experimental::current_callbacks.begin_deep_copy)(
        dst_space, dst_label.c_str(), dst_ptr, src_space, src_label.c_str(),
        src_ptr, size);
#ifdef KOKKOS_ENABLE_TUNING
    Kokkos::Tools::VariableValue contextValues[] = {
        Kokkos::Tools::make_variable_value(
            Kokkos::Tools::kernel_name_context_variable_id, "deep_copy_kernel"),
        Kokkos::Tools::make_variable_value(
            Kokkos::Tools::kernel_type_context_variable_id,
            "deep_copy")};  // TODO DZP: should deep copy have context
                            // variables for source and destination features?
    Kokkos::Tools::declareContextVariableValues(
        Kokkos::Tools::getNewContextId(), 2, contextValues);
#endif
  }
}

void endDeepCopy() {
  if (Experimental::current_callbacks.end_deep_copy != nullptr) {
    (*Experimental::current_callbacks.end_deep_copy)();
#ifdef KOKKOS_ENABLE_TUNING
    Kokkos::Tools::endContext(Kokkos::Tools::getCurrentContextId());
#endif
  }
}

void createProfileSection(const std::string& sectionName, uint32_t* secID) {
  if (Experimental::current_callbacks.create_profile_section != nullptr) {
    (*Experimental::current_callbacks.create_profile_section)(
        sectionName.c_str(), secID);
  }
}

void startSection(const uint32_t secID) {
  if (Experimental::current_callbacks.start_profile_section != nullptr) {
    (*Experimental::current_callbacks.start_profile_section)(secID);
  }
}

void stopSection(const uint32_t secID) {
  if (Experimental::current_callbacks.stop_profile_section != nullptr) {
    (*Experimental::current_callbacks.stop_profile_section)(secID);
  }
}

void destroyProfileSection(const uint32_t secID) {
  if (Experimental::current_callbacks.destroy_profile_section != nullptr) {
    (*Experimental::current_callbacks.destroy_profile_section)(secID);
  }
}

void markEvent(const std::string& eventName) {
  if (Experimental::current_callbacks.profile_event != nullptr) {
    (*Experimental::current_callbacks.profile_event)(eventName.c_str());
  }
}

SpaceHandle make_space_handle(const char* space_name) {
  SpaceHandle handle;
  strncpy(handle.name, space_name, 63);
  return handle;
}

}  // namespace Tools

namespace Tools {

void initialize() {
  // Make sure initialize calls happens only once
  static int is_initialized = 0;
  if (is_initialized) return;
  is_initialized = 1;

  void* firstProfileLibrary = nullptr;

  char* envProfileLibrary = getenv("KOKKOS_PROFILE_LIBRARY");

  // If we do not find a profiling library in the environment then exit
  // early.
  if (envProfileLibrary == nullptr) {
    return;
  }

  char* envProfileCopy =
      (char*)malloc(sizeof(char) * (strlen(envProfileLibrary) + 1));
  sprintf(envProfileCopy, "%s", envProfileLibrary);

  char* profileLibraryName = strtok(envProfileCopy, ";");

  if ((profileLibraryName != nullptr) &&
      (strcmp(profileLibraryName, "") != 0)) {
    firstProfileLibrary = dlopen(profileLibraryName, RTLD_NOW | RTLD_GLOBAL);

    if (firstProfileLibrary == nullptr) {
      std::cerr << "Error: Unable to load KokkosP library: "
                << profileLibraryName << std::endl;
      std::cerr << "dlopen(" << profileLibraryName
                << ", RTLD_NOW | RTLD_GLOBAL) failed with " << dlerror()
                << '\n';
    } else {
#ifdef KOKKOS_ENABLE_PROFILING_LOAD_PRINT
      std::cout << "KokkosP: Library Loaded: " << profileLibraryName
                << std::endl;
#endif

      // dlsym returns a pointer to an object, while we want to assign to
      // pointer to function A direct cast will give warnings hence, we have to
      // workaround the issue by casting pointer to pointers.
      auto p1 = dlsym(firstProfileLibrary, "kokkosp_begin_parallel_for");
      Experimental::set_begin_parallel_for_callback(
          *reinterpret_cast<beginFunction*>(&p1));
      auto p2 = dlsym(firstProfileLibrary, "kokkosp_begin_parallel_scan");
      Experimental::set_begin_parallel_scan_callback(
          *reinterpret_cast<beginFunction*>(&p2));
      auto p3 = dlsym(firstProfileLibrary, "kokkosp_begin_parallel_reduce");
      Experimental::set_begin_parallel_reduce_callback(
          *reinterpret_cast<beginFunction*>(&p3));

      auto p4 = dlsym(firstProfileLibrary, "kokkosp_end_parallel_scan");
      Experimental::set_end_parallel_scan_callback(
          *reinterpret_cast<endFunction*>(&p4));
      auto p5 = dlsym(firstProfileLibrary, "kokkosp_end_parallel_for");
      Experimental::set_end_parallel_for_callback(
          *reinterpret_cast<endFunction*>(&p5));
      auto p6 = dlsym(firstProfileLibrary, "kokkosp_end_parallel_reduce");
      Experimental::set_end_parallel_reduce_callback(
          *reinterpret_cast<endFunction*>(&p6));

      auto p7 = dlsym(firstProfileLibrary, "kokkosp_init_library");
      Experimental::set_init_callback(*reinterpret_cast<initFunction*>(&p7));
      auto p8 = dlsym(firstProfileLibrary, "kokkosp_finalize_library");
      Experimental::set_finalize_callback(
          *reinterpret_cast<finalizeFunction*>(&p8));

      auto p9 = dlsym(firstProfileLibrary, "kokkosp_push_profile_region");
      Experimental::set_push_region_callback(
          *reinterpret_cast<pushFunction*>(&p9));
      auto p10 = dlsym(firstProfileLibrary, "kokkosp_pop_profile_region");
      Experimental::set_pop_region_callback(
          *reinterpret_cast<popFunction*>(&p10));

      auto p11 = dlsym(firstProfileLibrary, "kokkosp_allocate_data");
      Experimental::set_allocate_data_callback(
          *reinterpret_cast<allocateDataFunction*>(&p11));
      auto p12 = dlsym(firstProfileLibrary, "kokkosp_deallocate_data");
      Experimental::set_deallocate_data_callback(
          *reinterpret_cast<deallocateDataFunction*>(&p12));

      auto p13 = dlsym(firstProfileLibrary, "kokkosp_begin_deep_copy");
      Experimental::set_begin_deep_copy_callback(
          *reinterpret_cast<beginDeepCopyFunction*>(&p13));
      auto p14 = dlsym(firstProfileLibrary, "kokkosp_end_deep_copy");
      Experimental::set_end_deep_copy_callback(
          *reinterpret_cast<endDeepCopyFunction*>(&p14));

      auto p15 = dlsym(firstProfileLibrary, "kokkosp_create_profile_section");
      Experimental::set_create_profile_section_callback(
          *(reinterpret_cast<createProfileSectionFunction*>(&p15)));
      auto p16 = dlsym(firstProfileLibrary, "kokkosp_start_profile_section");
      Experimental::set_start_profile_section_callback(
          *reinterpret_cast<startProfileSectionFunction*>(&p16));
      auto p17 = dlsym(firstProfileLibrary, "kokkosp_stop_profile_section");
      Experimental::set_stop_profile_section_callback(
          *reinterpret_cast<stopProfileSectionFunction*>(&p17));
      auto p18 = dlsym(firstProfileLibrary, "kokkosp_destroy_profile_section");
      Experimental::set_destroy_profile_section_callback(
          *(reinterpret_cast<destroyProfileSectionFunction*>(&p18)));

      auto p19 = dlsym(firstProfileLibrary, "kokkosp_profile_event");
      Experimental::set_profile_event_callback(
          *reinterpret_cast<profileEventFunction*>(&p19));

#ifdef KOKKOS_ENABLE_TUNING
      auto p20 = dlsym(firstProfileLibrary, "kokkosp_declare_tuning_variable");
      Experimental::set_declare_tuning_variable_callback(
          *reinterpret_cast<Kokkos::Tools::tuningVariableDeclarationFunction*>(
              &p20));

      auto p21 = dlsym(firstProfileLibrary, "kokkosp_declare_context_variable");
      Experimental::set_declare_context_variable_callback(
          *reinterpret_cast<Kokkos::Tools::contextVariableDeclarationFunction*>(
              &p21));
      auto p22 =
          dlsym(firstProfileLibrary, "kokkosp_request_tuning_variable_values");
      Experimental::set_request_tuning_variable_values_callback(
          *reinterpret_cast<Kokkos::Tools::tuningVariableValueFunction*>(&p22));
      auto p23 = dlsym(firstProfileLibrary, "kokkosp_end_context");
      Experimental::set_end_context_callback(
          *reinterpret_cast<Kokkos::Tools::contextEndFunction*>(&p23));
      auto p24 =
          dlsym(firstProfileLibrary, "kokkosp_declare_optimization_goal");
      Experimental::set_declare_optimization_goal_callback(
          *reinterpret_cast<
              Kokkos::Tools::optimizationGoalDeclarationFunction*>(&p24));
#endif
    }
  }

  if (Experimental::current_callbacks.init != nullptr) {
    (*Experimental::current_callbacks.init)(
        0, (uint64_t)KOKKOSP_INTERFACE_VERSION, (uint32_t)0, nullptr);
  }

#ifdef KOKKOS_ENABLE_TUNING
  Kokkos::Tools::VariableInfo kernel_name;
  kernel_name.type = Kokkos::Tools::ValueType::kokkos_value_text;
  kernel_name.category =
      Kokkos::Tools::StatisticalCategory::kokkos_value_categorical;
  kernel_name.valueQuantity =
      Kokkos::Tools::CandidateValueType::kokkos_value_unbounded;
  Kokkos::Tools::kernel_name_context_variable_id =
      Kokkos::Tools::getNewVariableId();
  Kokkos::Tools::kernel_type_context_variable_id =
      Kokkos::Tools::getNewVariableId();
  Kokkos::Tools::time_context_variable_id = Kokkos::Tools::getNewVariableId();

  Kokkos::Tools::SetOrRange kernel_type_variable_candidates;
  kernel_type_variable_candidates.set.size = 4;
  kernel_type_variable_candidates.set.id =
      Kokkos::Tools::kernel_type_context_variable_id;

  std::array<Kokkos::Tools::VariableValue, 4> candidate_values = {
      Kokkos::Tools::make_variable_value(
          Kokkos::Tools::kernel_type_context_variable_id, "parallel_for"),
      Kokkos::Tools::make_variable_value(
          Kokkos::Tools::kernel_type_context_variable_id, "parallel_reduce"),
      Kokkos::Tools::make_variable_value(
          Kokkos::Tools::kernel_type_context_variable_id, "parallel_scan"),
      Kokkos::Tools::make_variable_value(
          Kokkos::Tools::kernel_type_context_variable_id, "parallel_copy"),
  };

  kernel_type_variable_candidates.set.values = candidate_values.data();

  Kokkos::Tools::SetOrRange
      kernel_name_candidates;  // TODO DZP: an empty set in SetOrRange if
                               // things are unbounded? Maybe an empty
                               // struct in the union just for
                               // clarification? Or unify the tag and the
                               // data
  kernel_name_candidates.set.size = 0;
  kernel_name_candidates.set.id =
      Kokkos::Tools::kernel_name_context_variable_id;

  Kokkos::Tools::declareContextVariable(
      "kokkos.kernel_name", Kokkos::Tools::kernel_name_context_variable_id,
      kernel_name, kernel_name_candidates);

  Kokkos::Tools::VariableInfo kernel_type;
  kernel_type.type = Kokkos::Tools::ValueType::kokkos_value_text;
  kernel_type.category =
      Kokkos::Tools::StatisticalCategory::kokkos_value_categorical;
  kernel_type.valueQuantity =
      Kokkos::Tools::CandidateValueType::kokkos_value_set;

  Kokkos::Tools::declareContextVariable(
      "kokkos.kernel_type", Kokkos::Tools::kernel_type_context_variable_id,
      kernel_type, kernel_type_variable_candidates);

  Kokkos::Tools::SetOrRange
      time_candidates;  // TODO DZP: an empty set in SetOrRange if
                        // things are unbounded? Maybe an empty
                        // struct in the union just for
                        // clarification? Or unify the tag and the
                        // data
  time_candidates.set.size = 0;
  time_candidates.set.id   = Kokkos::Tools::time_context_variable_id;

  Kokkos::Tools::VariableInfo wall_clock_time;
  wall_clock_time.type = Kokkos::Tools::ValueType::kokkos_value_floating_point;
  wall_clock_time.category =
      Kokkos::Tools::StatisticalCategory::kokkos_value_ratio;
  wall_clock_time.valueQuantity =
      Kokkos::Tools::CandidateValueType::kokkos_value_unbounded;

  Kokkos::Tools::declareContextVariable("kokkos.wall_time",
                                        Kokkos::Tools::time_context_variable_id,
                                        wall_clock_time, time_candidates);

  Kokkos::Tools::OptimizationGoal initial_goal{
      Kokkos::Tools::kernel_type_context_variable_id, Kokkos_Tuning_Minimize};

  Kokkos::Tools::declareOptimizationGoal(initial_goal);
#endif

  Experimental::no_profiling.init     = nullptr;
  Experimental::no_profiling.finalize = nullptr;

  Experimental::no_profiling.begin_parallel_for    = nullptr;
  Experimental::no_profiling.begin_parallel_scan   = nullptr;
  Experimental::no_profiling.begin_parallel_reduce = nullptr;
  Experimental::no_profiling.end_parallel_scan     = nullptr;
  Experimental::no_profiling.end_parallel_for      = nullptr;
  Experimental::no_profiling.end_parallel_reduce   = nullptr;

  Experimental::no_profiling.push_region     = nullptr;
  Experimental::no_profiling.pop_region      = nullptr;
  Experimental::no_profiling.allocate_data   = nullptr;
  Experimental::no_profiling.deallocate_data = nullptr;

  Experimental::no_profiling.begin_deep_copy = nullptr;
  Experimental::no_profiling.end_deep_copy   = nullptr;

  Experimental::no_profiling.create_profile_section  = nullptr;
  Experimental::no_profiling.start_profile_section   = nullptr;
  Experimental::no_profiling.stop_profile_section    = nullptr;
  Experimental::no_profiling.destroy_profile_section = nullptr;

  Experimental::no_profiling.profile_event = nullptr;

  Experimental::no_profiling.declare_context_variable = nullptr;
  Experimental::no_profiling.declare_tuning_variable  = nullptr;
  Experimental::no_profiling.request_tuning_values    = nullptr;
  Experimental::no_profiling.end_tuning_context       = nullptr;

  free(envProfileCopy);
}

void finalize() {
  // Make sure finalize calls happens only once
  static int is_finalized = 0;
  if (is_finalized) return;
  is_finalized = 1;

  if (Experimental::current_callbacks.finalize != nullptr) {
    (*Experimental::current_callbacks.finalize)();

    Experimental::pause_tools();
  }
}

}  // namespace Tools

namespace Tools {
namespace Experimental {
void set_init_callback(initFunction callback) {
  current_callbacks.init = callback;
}
void set_finalize_callback(finalizeFunction callback) {
  current_callbacks.finalize = callback;
}
void set_begin_parallel_for_callback(beginFunction callback) {
  current_callbacks.begin_parallel_for = callback;
}
void set_end_parallel_for_callback(endFunction callback) {
  current_callbacks.end_parallel_for = callback;
}
void set_begin_parallel_reduce_callback(beginFunction callback) {
  current_callbacks.begin_parallel_reduce = callback;
}
void set_end_parallel_reduce_callback(endFunction callback) {
  current_callbacks.end_parallel_reduce = callback;
}
void set_begin_parallel_scan_callback(beginFunction callback) {
  current_callbacks.begin_parallel_scan = callback;
}
void set_end_parallel_scan_callback(endFunction callback) {
  current_callbacks.end_parallel_scan = callback;
}
void set_push_region_callback(pushFunction callback) {
  current_callbacks.push_region = callback;
}
void set_pop_region_callback(popFunction callback) {
  current_callbacks.pop_region = callback;
}
void set_allocate_data_callback(allocateDataFunction callback) {
  current_callbacks.allocate_data = callback;
}
void set_deallocate_data_callback(deallocateDataFunction callback) {
  current_callbacks.deallocate_data = callback;
}
void set_create_profile_section_callback(
    createProfileSectionFunction callback) {
  current_callbacks.create_profile_section = callback;
}
void set_start_profile_section_callback(startProfileSectionFunction callback) {
  current_callbacks.start_profile_section = callback;
}
void set_stop_profile_section_callback(stopProfileSectionFunction callback) {
  current_callbacks.stop_profile_section = callback;
}
void set_destroy_profile_section_callback(
    destroyProfileSectionFunction callback) {
  current_callbacks.destroy_profile_section = callback;
}
void set_profile_event_callback(profileEventFunction callback) {
  current_callbacks.profile_event = callback;
}
void set_begin_deep_copy_callback(beginDeepCopyFunction callback) {
  current_callbacks.begin_deep_copy = callback;
}
void set_end_deep_copy_callback(endDeepCopyFunction callback) {
  current_callbacks.end_deep_copy = callback;
}

void set_declare_tuning_variable_callback(
    tuningVariableDeclarationFunction callback) {
  current_callbacks.declare_tuning_variable = callback;
}
void set_declare_context_variable_callback(
    contextVariableDeclarationFunction callback) {
  current_callbacks.declare_context_variable = callback;
}
void set_declare_tuning_variable_values_callback(
    tuningVariableValueFunction callback) {
  current_callbacks.request_tuning_values = callback;
}
void set_end_context_callback(contextEndFunction callback) {
  current_callbacks.end_tuning_context = callback;
}
void set_declare_optimization_goal_callback(
    optimizationGoalDeclarationFunction callback) {
  current_callbacks.declare_optimization_goal = callback;
}

void pause_tools() {
  backup_callbacks  = current_callbacks;
  current_callbacks = no_profiling;
}

void resume_tools() { current_callbacks = backup_callbacks; }

EventSet get_callbacks() { return current_callbacks; }
void set_callbacks(EventSet new_events) { current_callbacks = new_events; }
}  // namespace Experimental
}  // namespace Tools

}  // namespace Kokkos

// TODO DZP, handle the off case
// Tuning

namespace Kokkos {
namespace Tools {

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
  if (Experimental::current_callbacks.declare_tuning_variable != nullptr) {
    (*Experimental::current_callbacks.declare_tuning_variable)(
        variableName.c_str(), uniqID, info);
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
  if (Experimental::current_callbacks.declare_context_variable != nullptr) {
    (*Experimental::current_callbacks.declare_context_variable)(
        variableName.c_str(), uniqID, info, candidate_values);
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
  if (Experimental::current_callbacks.request_tuning_values != nullptr) {
    (*Experimental::current_callbacks.request_tuning_values)(
        contextId, context_values.size(), context_values.data(), count, values,
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
  if (Experimental::current_callbacks.end_tuning_context != nullptr) {
    (*Experimental::current_callbacks.end_tuning_context)(contextId);
  }
  decrementCurrentContextId();
#else
  (void)contextId;
#endif
}

bool haveTuningTool() {
#ifdef KOKKOS_ENABLE_TUNING
  return (Experimental::current_callbacks.request_tuning_values != nullptr);
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
  if (Experimental::current_callbacks.declare_optimization_goal != nullptr) {
    (*Experimental::current_callbacks.declare_optimization_goal)(goal);
  }
#else
  (void)goal;
#endif
}

}  // end namespace Tools

}  // end namespace Kokkos

#else
#include <impl/Kokkos_Profiling_Interface.hpp>
#include <cstring>

namespace Kokkos {
namespace Tools {

bool profileLibraryLoaded() { return false; }

void beginParallelFor(const std::string&, const uint32_t, uint64_t*) {}
void endParallelFor(const uint64_t) {}
void beginParallelScan(const std::string&, const uint32_t, uint64_t*) {}
void endParallelScan(const uint64_t) {}
void beginParallelReduce(const std::string&, const uint32_t, uint64_t*) {}
void endParallelReduce(const uint64_t) {}

void pushRegion(const std::string&) {}
void popRegion() {}
void createProfileSection(const std::string&, uint32_t*) {}
void startSection(const uint32_t) {}
void stopSection(const uint32_t) {}
void destroyProfileSection(const uint32_t) {}

void markEvent(const std::string&) {}

void allocateData(const SpaceHandle, const std::string, const void*,
                  const uint64_t) {}
void deallocateData(const SpaceHandle, const std::string, const void*,
                    const uint64_t) {}

void beginDeepCopy(const SpaceHandle, const std::string, const void*,
                   const SpaceHandle, const std::string, const void*,
                   const uint64_t) {}
void endDeepCopy() {}

void initialize() {}
void finalize() {}

namespace Experimental {
static EventSet current_callbacks;

void set_init_callback(initFunction) {}
void set_finalize_callback(finalizeFunction) {}
void set_begin_parallel_for_callback(beginFunction) {}
void set_end_parallel_for_callback(endFunction) {}
void set_begin_parallel_reduce_callback(beginFunction) {}
void set_end_parallel_reduce_callback(endFunction) {}
void set_begin_parallel_scan_callback(beginFunction) {}
void set_end_parallel_scan_callback(endFunction) {}
void set_push_region_callback(pushFunction) {}
void set_pop_region_callback(popFunction) {}
void set_allocate_data_callback(allocateDataFunction) {}
void set_deallocate_data_callback(deallocateDataFunction) {}
void set_create_profile_section_callback(createProfileSectionFunction) {}
void set_start_profile_section_callback(startProfileSectionFunction) {}
void set_stop_profile_section_callback(stopProfileSectionFunction) {}
void set_destroy_profile_section_callback(destroyProfileSectionFunction) {}
void set_profile_event_callback(profileEventFunction) {}
void set_begin_deep_copy_callback(beginDeepCopyFunction) {}
void set_end_deep_copy_callback(endDeepCopyFunction) {}
void set_declare_tuning_variable_callback(tuningVariableDeclarationFunction) {}
void set_declare_context_variable_callback(contextVariableDeclarationFunction) {
}
void set_declare_tuning_variable_values_callback(tuningVariableValueFunction) {}
void set_declare_optimization_goal_callback(
    optimizationGoalDeclarationFunction) {}
void set_end_context_callback(contextEndFunction) {}

void pause_tools() {}

void resume_tools() {}

EventSet get_callbacks() { return current_callbacks; }
void set_callbacks(EventSet) {}
}  // namespace Experimental
}  // namespace Tools
}  // namespace Kokkos

namespace Kokkos {
namespace Tools {

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

void declareTuningVariable(const std::string&, size_t, VariableInfo) {}

void declareContextVariable(const std::string&, size_t, VariableInfo,
                            Kokkos::Tools::SetOrRange) {}

void declareContextVariableValues(size_t, size_t, VariableValue*) {}
void requestTuningVariableValues(size_t, size_t, VariableValue*,
                                 Kokkos::Tools::SetOrRange*) {}

void endContext(size_t) {}

bool haveTuningTool() { return false; }

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

void declareOptimizationGoal(const OptimizationGoal&) {}

}  // end namespace Tools

}  // end namespace Kokkos

#endif
