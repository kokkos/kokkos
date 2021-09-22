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

#define KOKKOS_TOOLS_INDEPENDENT_BUILD

#ifndef KOKKOS_TOOLS_INDEPENDENT_BUILD
#include <Kokkos_Macros.hpp>
#include <Kokkos_Tuners.hpp>
#endif

#include <impl/Kokkos_Profiling.hpp>
#include <impl/Kokkos_Profiling_Interface.hpp>

#if defined(KOKKOS_ENABLE_LIBDL) || defined(KOKKOS_TOOLS_INDEPENDENT_BUILD)
#include <dlfcn.h>
#define KOKKOS_TOOLS_ENABLE_LIBDL
#endif

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <memory>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <sstream>
#include <iostream>
namespace Kokkos {

namespace Tools {

InitArguments tool_arguments;

namespace Impl {
void parse_command_line_arguments(int& narg, char* arg[],
                                  InitArguments& arguments) {
  int iarg = 0;
  using Kokkos::Impl::check_arg;
  using Kokkos::Impl::check_int_arg;
  using Kokkos::Impl::check_str_arg;

  auto& lib            = arguments.lib;
  auto& args           = arguments.args;
  auto& help           = arguments.help;
  auto& tune_internals = arguments.tune_internals;
  while (iarg < narg) {
    if (check_arg(arg[iarg], "--kokkos-tune-internals")) {
      tune_internals = true;
      for (int k = iarg; k < narg - 1; k++) {
        arg[k] = arg[k + 1];
      }
      narg--;
    } else if (check_str_arg(arg[iarg], "--kokkos-tools-library", lib)) {
      for (int k = iarg; k < narg - 1; k++) {
        arg[k] = arg[k + 1];
      }
      narg--;
    } else if (check_str_arg(arg[iarg], "--kokkos-tools-args", args)) {
      for (int k = iarg; k < narg - 1; k++) {
        arg[k] = arg[k + 1];
      }
      narg--;
      // strip any leading and/or trailing quotes if they were retained in the
      // string because this will very likely cause parsing issues for tools.
      // If the quotes are retained (via bypassing the shell):
      //    <EXE> --kokkos-tools-args="-c my example"
      // would be tokenized as:
      //    "<EXE>" "\"-c" "my" "example\""
      // instead of:
      //    "<EXE>" "-c" "my" "example"
      if (!args.empty()) {
        if (args.front() == '"') args = args.substr(1);
        if (args.back() == '"') args = args.substr(0, args.length() - 1);
      }
      // add the name of the executable to the beginning
      if (narg > 0) args = std::string(arg[0]) + " " + args;
    } else if (check_arg(arg[iarg], "--kokkos-tools-help")) {
      help = true;
      for (int k = iarg; k < narg - 1; k++) {
        arg[k] = arg[k + 1];
      }
      narg--;
    }
  }
}



} // namespace Impl
void initialize(int argc, char* argv[]){
  InitArguments arguments;
  Impl::parse_command_line_arguments(argc, argv, arguments);
  initialize_profiling(arguments);
}

namespace Experimental {

namespace Impl {
void tool_invoked_fence(const uint32_t /* devID */) {
  /**
   * Currently the function ignores the device ID,
   * Eventually we want to support fencing only
   * a given stream/resource
   */
#ifndef KOKKOS_TOOLS_INDEPENDENT_BUILD
  Kokkos::fence(
      "Kokkos::Tools::Experimental::Impl::tool_invoked_fence: Tool Requested "
      "Fence");
#endif
}
}  // namespace Impl

#ifdef KOKKOS_ENABLE_TUNING
static size_t kernel_name_context_variable_id;
static size_t kernel_type_context_variable_id;
static std::unordered_map<size_t, std::unordered_set<size_t>>
    features_per_context;
static std::unordered_set<size_t> active_features;
static std::unordered_map<size_t, VariableValue> feature_values;
static std::unordered_map<size_t, VariableInfo> variable_metadata;
#endif
static Profiling::Experimental::EventSet current_callbacks;
static Profiling::Experimental::EventSet backup_callbacks;
static Profiling::Experimental::EventSet no_profiling;
static Kokkos::Tools::Experimental::ToolSettings tool_requirements;
bool eventSetsEqual(const Profiling::Experimental::EventSet& l, const Profiling::Experimental::EventSet& r) {
  return l.init == r.init && l.finalize == r.finalize &&
         l.parse_args == r.parse_args && l.print_help == r.print_help &&
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
         l.end_deep_copy == r.end_deep_copy && l.begin_fence == r.begin_fence &&
         l.end_fence == r.end_fence && l.sync_dual_view == r.sync_dual_view &&
         l.modify_dual_view == r.modify_dual_view &&
         l.declare_metadata == r.declare_metadata &&
         l.request_tool_settings == r.request_tool_settings &&
         l.provide_tool_programming_interface ==
             r.provide_tool_programming_interface &&
         l.declare_input_type == r.declare_input_type &&
         l.declare_output_type == r.declare_output_type &&
         l.end_tuning_context == r.end_tuning_context &&
         l.begin_tuning_context == r.begin_tuning_context &&
         l.request_output_values == r.request_output_values &&
         l.declare_optimization_goal == r.declare_optimization_goal;
}
enum class MayRequireGlobalFencing : bool { No, Yes };
template <typename Callback, typename... Args>
inline void invoke_kokkosp_callback(
    MayRequireGlobalFencing may_require_global_fencing,
    const Callback& callback, Args&&... args) {
  if (callback != nullptr) {
    // two clause if statement
    // may_require_global_fencing: "if this callback ever needs a fence", AND
    // if the tool requires global fencing (default true, but tools can
    // overwrite)
    if (may_require_global_fencing == MayRequireGlobalFencing::Yes &&
        (tool_requirements
             .requires_global_fencing)) {
#ifndef KOKKOS_TOOLS_INDEPENDENT_BUILD
      Kokkos::fence(
          "Kokkos::Tools::invoke_kokkosp_callback: Kokkos Profile Tool Fence");
#endif
    }
    (*callback)(std::forward<Args>(args)...);
  }
}
}  // namespace Experimental
bool profileLibraryLoaded() {
  return !Experimental::eventSetsEqual(Experimental::current_callbacks,
                                       Experimental::no_profiling);
}

void beginParallelFor(const std::string& kernelPrefix, const uint32_t devID,
                      uint64_t* kernelID) {
  Experimental::invoke_kokkosp_callback(
      Experimental::MayRequireGlobalFencing::Yes,
      Experimental::current_callbacks.begin_parallel_for, kernelPrefix.c_str(),
      devID, kernelID);
#ifdef KOKKOS_ENABLE_TUNING
  if (Kokkos::tune_internals()) {
    auto context_id = Experimental::get_new_context_id();
    Experimental::begin_context(context_id);
    Experimental::VariableValue contextValues[] = {
        Experimental::make_variable_value(
            Experimental::kernel_name_context_variable_id, kernelPrefix),
        Experimental::make_variable_value(
            Experimental::kernel_type_context_variable_id, "parallel_for")};
    Experimental::set_input_values(context_id, 2, contextValues);
  }
#endif
}

void endParallelFor(const uint64_t kernelID) {
  Experimental::invoke_kokkosp_callback(
      Experimental::MayRequireGlobalFencing::Yes,
      Experimental::current_callbacks.end_parallel_for, kernelID);
#ifdef KOKKOS_ENABLE_TUNING
  if (Kokkos::tune_internals()) {
    Experimental::end_context(Experimental::get_current_context_id());
  }
#endif
}

void beginParallelScan(const std::string& kernelPrefix, const uint32_t devID,
                       uint64_t* kernelID) {
  Experimental::invoke_kokkosp_callback(
      Experimental::MayRequireGlobalFencing::Yes,
      Experimental::current_callbacks.begin_parallel_scan, kernelPrefix.c_str(),
      devID, kernelID);
#ifdef KOKKOS_ENABLE_TUNING
  if (Kokkos::tune_internals()) {
    auto context_id = Experimental::get_new_context_id();
    Experimental::begin_context(context_id);
    Experimental::VariableValue contextValues[] = {
        Experimental::make_variable_value(
            Experimental::kernel_name_context_variable_id, kernelPrefix),
        Experimental::make_variable_value(
            Experimental::kernel_type_context_variable_id, "parallel_for")};
    Experimental::set_input_values(context_id, 2, contextValues);
  }
#endif
}

void endParallelScan(const uint64_t kernelID) {
  Experimental::invoke_kokkosp_callback(
      Experimental::MayRequireGlobalFencing::Yes,
      Experimental::current_callbacks.end_parallel_scan, kernelID);
#ifdef KOKKOS_ENABLE_TUNING
  if (Kokkos::tune_internals()) {
    Experimental::end_context(Experimental::get_current_context_id());
  }
#endif
}

void beginParallelReduce(const std::string& kernelPrefix, const uint32_t devID,
                         uint64_t* kernelID) {
  Experimental::invoke_kokkosp_callback(
      Experimental::MayRequireGlobalFencing::Yes,
      Experimental::current_callbacks.begin_parallel_reduce,
      kernelPrefix.c_str(), devID, kernelID);
#ifdef KOKKOS_ENABLE_TUNING
  if (Kokkos::tune_internals()) {
    auto context_id = Experimental::get_new_context_id();
    Experimental::begin_context(context_id);
    Experimental::VariableValue contextValues[] = {
        Experimental::make_variable_value(
            Experimental::kernel_name_context_variable_id, kernelPrefix),
        Experimental::make_variable_value(
            Experimental::kernel_type_context_variable_id, "parallel_for")};
    Experimental::set_input_values(context_id, 2, contextValues);
  }
#endif
}

void endParallelReduce(const uint64_t kernelID) {
  Experimental::invoke_kokkosp_callback(
      Experimental::MayRequireGlobalFencing::Yes,
      Experimental::current_callbacks.end_parallel_reduce, kernelID);
#ifdef KOKKOS_ENABLE_TUNING
  if (Kokkos::tune_internals()) {
    Experimental::end_context(Experimental::get_current_context_id());
  }
#endif
}

void pushRegion(const std::string& kName) {
  Experimental::invoke_kokkosp_callback(
      Experimental::MayRequireGlobalFencing::Yes,
      Experimental::current_callbacks.push_region, kName.c_str());
}

void popRegion() {
  Experimental::invoke_kokkosp_callback(
      Experimental::MayRequireGlobalFencing::Yes,
      Experimental::current_callbacks.pop_region);
}

void allocateData(const SpaceHandle space, const std::string label,
                  const void* ptr, const uint64_t size) {
  Experimental::invoke_kokkosp_callback(
      Experimental::MayRequireGlobalFencing::No,
      Experimental::current_callbacks.allocate_data, space, label.c_str(), ptr,
      size);
}

void deallocateData(const SpaceHandle space, const std::string label,
                    const void* ptr, const uint64_t size) {
  Experimental::invoke_kokkosp_callback(
      Experimental::MayRequireGlobalFencing::No,
      Experimental::current_callbacks.deallocate_data, space, label.c_str(),
      ptr, size);
}

void beginDeepCopy(const SpaceHandle dst_space, const std::string dst_label,
                   const void* dst_ptr, const SpaceHandle src_space,
                   const std::string src_label, const void* src_ptr,
                   const uint64_t size) {
  Experimental::invoke_kokkosp_callback(
      Experimental::MayRequireGlobalFencing::No,
      Experimental::current_callbacks.begin_deep_copy, dst_space,
      dst_label.c_str(), dst_ptr, src_space, src_label.c_str(), src_ptr, size);
#ifdef KOKKOS_ENABLE_TUNING
  if (Experimental::current_callbacks.begin_deep_copy != nullptr) {
    if (Kokkos::tune_internals()) {
      auto context_id = Experimental::get_new_context_id();
      Experimental::begin_context(context_id);
      Experimental::VariableValue contextValues[] = {
          Experimental::make_variable_value(
              Experimental::kernel_name_context_variable_id,
              "deep_copy_kernel"),
          Experimental::make_variable_value(
              Experimental::kernel_type_context_variable_id, "deep_copy")};
      Experimental::set_input_values(context_id, 2, contextValues);
    }
  }
#endif
}

void endDeepCopy() {
  Experimental::invoke_kokkosp_callback(
      Experimental::MayRequireGlobalFencing::No,
      Experimental::current_callbacks.end_deep_copy);
#ifdef KOKKOS_ENABLE_TUNING
  if (Experimental::current_callbacks.end_deep_copy != nullptr) {
    if (Kokkos::tune_internals()) {
      Experimental::end_context(Experimental::get_current_context_id());
    }
  }
#endif
}

void beginFence(const std::string name, const uint32_t deviceId,
                uint64_t* handle) {
  Experimental::invoke_kokkosp_callback(
      Experimental::MayRequireGlobalFencing::No,
      Experimental::current_callbacks.begin_fence, name.c_str(), deviceId,
      handle);
}

void endFence(const uint64_t handle) {
  Experimental::invoke_kokkosp_callback(
      Experimental::MayRequireGlobalFencing::No,
      Experimental::current_callbacks.end_fence, handle);
}

void createProfileSection(const std::string& sectionName, uint32_t* secID) {
  Experimental::invoke_kokkosp_callback(
      Experimental::MayRequireGlobalFencing::No,
      Experimental::current_callbacks.create_profile_section,
      sectionName.c_str(), secID);
}

void startSection(const uint32_t secID) {
  Experimental::invoke_kokkosp_callback(
      Experimental::MayRequireGlobalFencing::No,
      Experimental::current_callbacks.start_profile_section, secID);
}

void stopSection(const uint32_t secID) {
  Experimental::invoke_kokkosp_callback(
      Experimental::MayRequireGlobalFencing::No,
      Experimental::current_callbacks.stop_profile_section, secID);
}

void destroyProfileSection(const uint32_t secID) {
  Experimental::invoke_kokkosp_callback(
      Experimental::MayRequireGlobalFencing::No,
      Experimental::current_callbacks.destroy_profile_section, secID);
}

void markEvent(const std::string& eventName) {
  Experimental::invoke_kokkosp_callback(
      Experimental::MayRequireGlobalFencing::No,
      Experimental::current_callbacks.profile_event, eventName.c_str());
}

bool printHelp(const std::string& args) {
  if (Experimental::current_callbacks.print_help == nullptr) {
    return false;
  }
  std::string arg0  = args.substr(0, args.find_first_of(' '));
  const char* carg0 = arg0.c_str();
  Experimental::invoke_kokkosp_callback(
      Experimental::MayRequireGlobalFencing::No,
      Experimental::current_callbacks.print_help, const_cast<char*>(carg0));
  return true;
}

void parseArgs(int _argc, char** _argv) {
  if (Experimental::current_callbacks.parse_args != nullptr && _argc > 0) {
    Experimental::invoke_kokkosp_callback(
        Experimental::MayRequireGlobalFencing::No,
        Experimental::current_callbacks.parse_args, _argc, _argv);
  }
}

void parseArgs(const std::string& args) {
  if (Experimental::current_callbacks.parse_args == nullptr) {
    return;
  }
  using strvec_t = std::vector<std::string>;
  auto tokenize  = [](const std::string& line, const std::string& delimiters) {
    strvec_t _result{};
    std::size_t _bidx = 0;  // position that is the beginning of the new string
    std::size_t _didx = 0;  // position of the delimiter in the string
    while (_bidx < line.length() && _didx < line.length()) {
      // find the first character (starting at _didx) that is not a delimiter
      _bidx = line.find_first_not_of(delimiters, _didx);
      // if no more non-delimiter chars, done
      if (_bidx == std::string::npos) break;
      // starting at the position of the new string, find the next delimiter
      _didx = line.find_first_of(delimiters, _bidx);
      // starting at the position of the new string, get the characters
      // between this position and the next delimiter
      std::string _tmp = line.substr(_bidx, _didx - _bidx);
      // don't add empty strings
      if (!_tmp.empty()) _result.emplace_back(_tmp);
    }
    return _result;
  };
  auto vargs = tokenize(args, " \t");
  if (vargs.size() == 0) return;
  auto _argc          = static_cast<int>(vargs.size());
  char** _argv        = new char*[_argc + 1];
  _argv[vargs.size()] = nullptr;
  for (int i = 0; i < _argc; ++i) {
    auto& _str = vargs.at(i);
    _argv[i]   = new char[_str.length() + 1];
    std::memcpy(_argv[i], _str.c_str(), _str.length() * sizeof(char));
    _argv[i][_str.length()] = '\0';
  }
  parseArgs(_argc, _argv);
  for (int i = 0; i < _argc; ++i) {
    delete[] _argv[i];
  }
  delete[] _argv;
}

SpaceHandle make_space_handle(const char* space_name) {
  SpaceHandle handle;
  strncpy(handle.name, space_name, 63);
  return handle;
}

template <typename Callback>
void lookup_function(void* dlopen_handle, const std::string& basename,
                     Callback& callback) {
#ifdef KOKKOS_TOOLS_ENABLE_LIBDL
  // dlsym returns a pointer to an object, while we want to assign to
  // pointer to function A direct cast will give warnings hence, we have to
  // workaround the issue by casting pointer to pointers.
  void* p  = dlsym(dlopen_handle, basename.c_str());
  callback = *reinterpret_cast<Callback*>(&p);
#endif
}

void initialize(const std::string& profileLibrary) {
  // Make sure initialize calls happens only once
  static int is_initialized = 0;
  if (is_initialized) return;
  is_initialized = 1;

  auto invoke_init_callbacks = []() {
    Experimental::invoke_kokkosp_callback(
        Experimental::MayRequireGlobalFencing::No,
        Experimental::current_callbacks.init, 0,
        (uint64_t)KOKKOSP_INTERFACE_VERSION, (uint32_t)0, nullptr);

    Experimental::tool_requirements.requires_global_fencing = true;

    Experimental::invoke_kokkosp_callback(
        Experimental::MayRequireGlobalFencing::No,
        Experimental::current_callbacks.request_tool_settings, 1,
        &Experimental::tool_requirements);

    Kokkos::Tools::Experimental::ToolProgrammingInterface actions;
    actions.fence = &Experimental::Impl::tool_invoked_fence;

    Experimental::invoke_kokkosp_callback(
        Experimental::MayRequireGlobalFencing::No,
        Experimental::current_callbacks.provide_tool_programming_interface, 1,
        actions);
  };

#ifdef KOKKOS_TOOLS_ENABLE_LIBDL
  void* firstProfileLibrary = nullptr;

  if (profileLibrary.empty()) {
    invoke_init_callbacks();
    return;
  }

  char* envProfileLibrary = const_cast<char*>(profileLibrary.c_str());

  const auto envProfileCopy =
      std::make_unique<char[]>(strlen(envProfileLibrary) + 1);
  sprintf(envProfileCopy.get(), "%s", envProfileLibrary);

  char* profileLibraryName = strtok(envProfileCopy.get(), ";");

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
      lookup_function(
          firstProfileLibrary, "kokkosp_begin_parallel_scan",
          Experimental::current_callbacks.begin_parallel_scan);
      lookup_function(
          firstProfileLibrary, "kokkosp_begin_parallel_for",
          Experimental::current_callbacks.begin_parallel_for);
      lookup_function(
          firstProfileLibrary, "kokkosp_begin_parallel_reduce",
          Experimental::current_callbacks.begin_parallel_reduce);
      lookup_function(
          firstProfileLibrary, "kokkosp_end_parallel_scan",
          Experimental::current_callbacks.end_parallel_scan);
      lookup_function(
          firstProfileLibrary, "kokkosp_end_parallel_for",
          Experimental::current_callbacks.end_parallel_for);
      lookup_function(
          firstProfileLibrary, "kokkosp_end_parallel_reduce",
          Experimental::current_callbacks.end_parallel_reduce);

      lookup_function(firstProfileLibrary, "kokkosp_init_library",
                      Experimental::current_callbacks.init);
      lookup_function(firstProfileLibrary, "kokkosp_finalize_library",
                      Experimental::current_callbacks.finalize);

      lookup_function(
          firstProfileLibrary, "kokkosp_push_profile_region",
          Experimental::current_callbacks.push_region);
      lookup_function(
          firstProfileLibrary, "kokkosp_pop_profile_region",
          Experimental::current_callbacks.pop_region);
      lookup_function(
          firstProfileLibrary, "kokkosp_allocate_data",
          Experimental::current_callbacks.allocate_data);
      lookup_function(
          firstProfileLibrary, "kokkosp_deallocate_data",
          Experimental::current_callbacks.deallocate_data);

      lookup_function(
          firstProfileLibrary, "kokkosp_begin_deep_copy",
          Experimental::current_callbacks.begin_deep_copy);
      lookup_function(
          firstProfileLibrary, "kokkosp_end_deep_copy",
          Experimental::current_callbacks.end_deep_copy);
      lookup_function(
          firstProfileLibrary, "kokkosp_begin_fence",
          Experimental::current_callbacks.begin_fence);
      lookup_function(firstProfileLibrary, "kokkosp_end_fence",
                      Experimental::current_callbacks.end_fence);
      lookup_function(
          firstProfileLibrary, "kokkosp_dual_view_sync",
          Experimental::current_callbacks.sync_dual_view);
      lookup_function(
          firstProfileLibrary, "kokkosp_dual_view_modify",
          Experimental::current_callbacks.modify_dual_view);

      lookup_function(
          firstProfileLibrary, "kokkosp_declare_metadata",
          Experimental::current_callbacks.declare_metadata);
      lookup_function(firstProfileLibrary, "kokkosp_create_profile_section",
                      Experimental::current_callbacks
                          .create_profile_section);
      lookup_function(
          firstProfileLibrary, "kokkosp_start_profile_section",
          Experimental::current_callbacks.start_profile_section);
      lookup_function(
          firstProfileLibrary, "kokkosp_stop_profile_section",
          Experimental::current_callbacks.stop_profile_section);
      lookup_function(firstProfileLibrary, "kokkosp_destroy_profile_section",
                      Experimental::current_callbacks
                          .destroy_profile_section);

      lookup_function(
          firstProfileLibrary, "kokkosp_profile_event",
          Experimental::current_callbacks.profile_event);
#ifdef KOKKOS_ENABLE_TUNING
      lookup_function(
          firstProfileLibrary, "kokkosp_declare_output_type",
          Experimental::current_callbacks.declare_output_type);

      lookup_function(
          firstProfileLibrary, "kokkosp_declare_input_type",
          Experimental::current_callbacks.declare_input_type);
      lookup_function(
          firstProfileLibrary, "kokkosp_request_values",
          Experimental::current_callbacks.request_output_values);
      lookup_function(
          firstProfileLibrary, "kokkosp_end_context",
          Experimental::current_callbacks.end_tuning_context);
      lookup_function(
          firstProfileLibrary, "kokkosp_begin_context",
          Experimental::current_callbacks.begin_tuning_context);
      lookup_function(firstProfileLibrary, "kokkosp_declare_optimization_goal",
                      Experimental::current_callbacks
                          .declare_optimization_goal);
#endif  // KOKKOS_ENABLE_TUNING

      lookup_function(
          firstProfileLibrary, "kokkosp_print_help",
          Experimental::current_callbacks.print_help);
      lookup_function(
          firstProfileLibrary, "kokkosp_parse_args",
          Experimental::current_callbacks.parse_args);
      lookup_function(firstProfileLibrary,
                      "kokkosp_provide_tool_programming_interface",
                      Experimental::current_callbacks
                          .provide_tool_programming_interface);
      lookup_function(
          firstProfileLibrary, "kokkosp_request_tool_settings",
          Experimental::current_callbacks.request_tool_settings);
    }
  }
#else
  (void)profileLibrary;
#endif  // KOKKOS_ENABLE_LIBDL

  invoke_init_callbacks();

#ifdef KOKKOS_ENABLE_TUNING
  Experimental::VariableInfo kernel_name;
  kernel_name.type = Experimental::ValueType::kokkos_value_string;
  kernel_name.category =
      Experimental::StatisticalCategory::kokkos_value_categorical;
  kernel_name.valueQuantity =
      Experimental::CandidateValueType::kokkos_value_unbounded;

  std::array<std::string, 4> candidate_values = {
      "parallel_for",
      "parallel_reduce",
      "parallel_scan",
      "parallel_copy",
  };

  Experimental::SetOrRange kernel_type_variable_candidates =
      Experimental::make_candidate_set(4, candidate_values.data());

  Experimental::kernel_name_context_variable_id =
      Experimental::declare_input_type("kokkos.kernel_name", kernel_name);

  Experimental::VariableInfo kernel_type;
  kernel_type.type = Experimental::ValueType::kokkos_value_string;
  kernel_type.category =
      Experimental::StatisticalCategory::kokkos_value_categorical;
  kernel_type.valueQuantity =
      Experimental::CandidateValueType::kokkos_value_set;
  kernel_type.candidates = kernel_type_variable_candidates;
  Experimental::kernel_type_context_variable_id =
      Experimental::declare_input_type("kokkos.kernel_type", kernel_type);

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

  Experimental::no_profiling.declare_input_type    = nullptr;
  Experimental::no_profiling.declare_output_type   = nullptr;
  Experimental::no_profiling.request_output_values = nullptr;
  Experimental::no_profiling.end_tuning_context    = nullptr;
}

void finalize() {
  // Make sure finalize calls happens only once
  static int is_finalized = 0;
  if (is_finalized) return;
  is_finalized = 1;

  if (Experimental::current_callbacks.finalize != nullptr) {
    Experimental::invoke_kokkosp_callback(
        Experimental::MayRequireGlobalFencing::No,
        Experimental::current_callbacks.finalize);

    Profiling::Experimental::pause_tools();
  }
#ifdef KOKKOS_ENABLE_TUNING
  // clean up string candidate set
  for (auto& metadata_pair : Experimental::variable_metadata) {
    auto metadata = metadata_pair.second;
    if ((metadata.type == Experimental::ValueType::kokkos_value_string) &&
        (metadata.valueQuantity ==
         Experimental::CandidateValueType::kokkos_value_set)) {
      auto candidate_set = metadata.candidates.set;
      delete[] candidate_set.values.string_value;
    }
  }
#endif
}

void syncDualView(const std::string& label, const void* const ptr,
                  bool to_device) {
  Experimental::invoke_kokkosp_callback(
      Experimental::MayRequireGlobalFencing::No,
      Experimental::current_callbacks.sync_dual_view, label.c_str(), ptr,
      to_device);
}
void modifyDualView(const std::string& label, const void* const ptr,
                    bool on_device) {
  Experimental::invoke_kokkosp_callback(
      Experimental::MayRequireGlobalFencing::No,
      Experimental::current_callbacks.modify_dual_view, label.c_str(), ptr,
      on_device);
}

void declareMetadata(const std::string& key, const std::string& value) {
  Experimental::invoke_kokkosp_callback(
      Experimental::MayRequireGlobalFencing::No,
      Experimental::current_callbacks.declare_metadata, key.c_str(),
      value.c_str());
}

}  // namespace Impl
} // namespace Tools
namespace Tools {
namespace Experimental {
void set_init_callback(Kokkos::Tools::initFunction callback) {
  Kokkos::Tools::Experimental::current_callbacks.init = callback;
}
void set_finalize_callback(Kokkos::Tools::finalizeFunction callback) {
  Kokkos::Tools::Experimental::current_callbacks.finalize = callback;
}
void set_parse_args_callback(Kokkos::Tools::parseArgsFunction callback) {
  Kokkos::Tools::Experimental::current_callbacks.parse_args = callback;
}
void set_print_help_callback(Kokkos::Tools::printHelpFunction callback) {
  Kokkos::Tools::Experimental::current_callbacks.print_help = callback;
}
void set_begin_parallel_for_callback(Kokkos::Tools::beginFunction callback) {
  Kokkos::Tools::Experimental::current_callbacks.begin_parallel_for = callback;
}
void set_end_parallel_for_callback(Kokkos::Tools::endFunction callback) {
  Kokkos::Tools::Experimental::current_callbacks.end_parallel_for = callback;
}
void set_begin_parallel_reduce_callback(Kokkos::Tools::beginFunction callback) {
  Kokkos::Tools::Experimental::current_callbacks.begin_parallel_reduce = callback;
}
void set_end_parallel_reduce_callback(Kokkos::Tools::endFunction callback) {
  Kokkos::Tools::Experimental::current_callbacks.end_parallel_reduce = callback;
}
void set_begin_parallel_scan_callback(Kokkos::Tools::beginFunction callback) {
  Kokkos::Tools::Experimental::current_callbacks.begin_parallel_scan = callback;
}
void set_end_parallel_scan_callback(Kokkos::Tools::endFunction callback) {
  Kokkos::Tools::Experimental::current_callbacks.end_parallel_scan = callback;
}
void set_push_region_callback(Kokkos::Tools::pushFunction callback) {
  Kokkos::Tools::Experimental::current_callbacks.push_region = callback;
}
void set_pop_region_callback(Kokkos::Tools::popFunction callback) {
  Kokkos::Tools::Experimental::current_callbacks.pop_region = callback;
}
void set_allocate_data_callback(Kokkos::Tools::allocateDataFunction callback) {
  Kokkos::Tools::Experimental::current_callbacks.allocate_data = callback;
}
void set_deallocate_data_callback(Kokkos::Tools::deallocateDataFunction callback) {
  Kokkos::Tools::Experimental::current_callbacks.deallocate_data = callback;
}
void set_create_profile_section_callback(
    Kokkos::Tools::createProfileSectionFunction callback) {
  Kokkos::Tools::Experimental::current_callbacks.create_profile_section = callback;
}
void set_start_profile_section_callback(Kokkos::Tools::startProfileSectionFunction callback) {
  Kokkos::Tools::Experimental::current_callbacks.start_profile_section = callback;
}
void set_stop_profile_section_callback(Kokkos::Tools::stopProfileSectionFunction callback) {
  Kokkos::Tools::Experimental::current_callbacks.stop_profile_section = callback;
}
void set_destroy_profile_section_callback(
    Kokkos::Tools::destroyProfileSectionFunction callback) {
  Kokkos::Tools::Experimental::current_callbacks.destroy_profile_section = callback;
}
void set_profile_event_callback(Kokkos::Tools::profileEventFunction callback) {
 Kokkos::Tools::Experimental::current_callbacks.profile_event = callback;
}
void set_begin_deep_copy_callback(Kokkos::Tools::beginDeepCopyFunction callback) {
  Kokkos::Tools::Experimental::current_callbacks.begin_deep_copy = callback;
}
void set_end_deep_copy_callback(Kokkos::Tools::endDeepCopyFunction callback) {
  Kokkos::Tools::Experimental::current_callbacks.end_deep_copy = callback;
}
void set_begin_fence_callback(Kokkos::Tools::beginFenceFunction callback) {
  Kokkos::Tools::Experimental::current_callbacks.begin_fence = callback;
}
void set_end_fence_callback(Kokkos::Tools::endFenceFunction callback) {
  Kokkos::Tools::Experimental::current_callbacks.end_fence = callback;
}

void set_dual_view_sync_callback(Kokkos::Tools::dualViewSyncFunction callback) {
  Kokkos::Tools::Experimental::current_callbacks.sync_dual_view = callback;
}
void set_dual_view_modify_callback(Kokkos::Tools::dualViewModifyFunction callback) {
  Kokkos::Tools::Experimental::current_callbacks.modify_dual_view = callback;
}
void set_declare_metadata_callback(Kokkos::Tools::declareMetadataFunction callback) {
  Kokkos::Tools::Experimental::current_callbacks.declare_metadata = callback;
}

void set_declare_output_type_callback(Kokkos::Tools::Experimental::outputTypeDeclarationFunction callback) {
  Kokkos::Tools::Experimental::current_callbacks.declare_output_type = callback;
}
void set_declare_input_type_callback(Kokkos::Tools::Experimental::inputTypeDeclarationFunction callback) {
  Kokkos::Tools::Experimental::current_callbacks.declare_input_type = callback;
}
void set_request_output_values_callback(Kokkos::Tools::Experimental::requestValueFunction callback) {
  Kokkos::Tools::Experimental::current_callbacks.request_output_values = callback;
}
void set_end_context_callback(Kokkos::Tools::Experimental::contextEndFunction callback) {
  Kokkos::Tools::Experimental::current_callbacks.end_tuning_context = callback;
}
void set_begin_context_callback(Kokkos::Tools::Experimental::contextBeginFunction callback) {
  Kokkos::Tools::Experimental::current_callbacks.begin_tuning_context = callback;
}
void set_declare_optimization_goal_callback(
    Kokkos::Tools::Experimental::optimizationGoalDeclarationFunction callback) {
  Kokkos::Tools::Experimental::current_callbacks.declare_optimization_goal = callback;
}

void pause_tools() {
  Kokkos::Tools::Experimental::backup_callbacks  = Kokkos::Tools::Experimental::current_callbacks;
  Kokkos::Tools::Experimental::current_callbacks = Kokkos::Tools::Experimental::no_profiling;
}

void resume_tools() { Kokkos::Tools::Experimental::current_callbacks = Kokkos::Tools::Experimental::backup_callbacks; }

Kokkos::Tools::Experimental::EventSet get_callbacks() { return Kokkos::Tools::Experimental::current_callbacks; }
void set_callbacks(Kokkos::Tools::Experimental::EventSet new_events) { Kokkos::Tools::Experimental::current_callbacks = new_events; }
}  // namespace Experimental
}  // namespace Tools

namespace Profiling {
bool profileLibraryLoaded() { return Kokkos::Tools::profileLibraryLoaded(); }

void beginParallelFor(const std::string& kernelPrefix, const uint32_t devID,
                      uint64_t* kernelID) {
  Kokkos::Tools::beginParallelFor(kernelPrefix, devID, kernelID);
}
void beginParallelReduce(const std::string& kernelPrefix, const uint32_t devID,
                         uint64_t* kernelID) {
  Kokkos::Tools::beginParallelReduce(kernelPrefix, devID, kernelID);
}
void beginParallelScan(const std::string& kernelPrefix, const uint32_t devID,
                       uint64_t* kernelID) {
  Kokkos::Tools::beginParallelScan(kernelPrefix, devID, kernelID);
}
void endParallelFor(const uint64_t kernelID) {
  Kokkos::Tools::endParallelFor(kernelID);
}
void endParallelReduce(const uint64_t kernelID) {
  Kokkos::Tools::endParallelReduce(kernelID);
}
void endParallelScan(const uint64_t kernelID) {
  Kokkos::Tools::endParallelScan(kernelID);
}

void pushRegion(const std::string& kName) { Kokkos::Tools::pushRegion(kName); }
void popRegion() { Kokkos::Tools::popRegion(); }

void createProfileSection(const std::string& sectionName, uint32_t* secID) {
  Kokkos::Tools::createProfileSection(sectionName, secID);
}
void destroyProfileSection(const uint32_t secID) {
  Kokkos::Tools::destroyProfileSection(secID);
}

void startSection(const uint32_t secID) { Kokkos::Tools::startSection(secID); }

void stopSection(const uint32_t secID) { Kokkos::Tools::stopSection(secID); }

void markEvent(const std::string& eventName) {
  Kokkos::Tools::markEvent(eventName);
}
void allocateData(const Kokkos::Tools::SpaceHandle handle, const std::string name,
                  const void* data, const uint64_t size) {
  Kokkos::Tools::allocateData(handle, name, data, size);
}
void deallocateData(const Kokkos::Tools::SpaceHandle space, const std::string label,
                    const void* ptr, const uint64_t size) {
  Kokkos::Tools::deallocateData(space, label, ptr, size);
}

void beginDeepCopy(const Kokkos::Tools::SpaceHandle dst_space, const std::string dst_label,
                   const void* dst_ptr, const Kokkos::Tools::SpaceHandle src_space,
                   const std::string src_label, const void* src_ptr,
                   const uint64_t size) {
  Kokkos::Tools::beginDeepCopy(dst_space, dst_label, dst_ptr, src_space,
                               src_label, src_ptr, size);
}
void endDeepCopy() { Kokkos::Tools::endDeepCopy(); }

void finalize() { Kokkos::Tools::finalize(); }
void initialize(const std::string& profileLibrary) {
  Kokkos::Tools::initialize(profileLibrary);
}

bool printHelp(const std::string& args) {
  return Kokkos::Tools::printHelp(args);
}
void parseArgs(const std::string& args) { Kokkos::Tools::parseArgs(args); }
void parseArgs(int _argc, char** _argv) {
  Kokkos::Tools::parseArgs(_argc, _argv);
}

Kokkos::Tools::SpaceHandle make_space_handle(const char* space_name) {
  return Kokkos::Tools::make_space_handle(space_name);
}
}  // namespace Profiling

// Tuning

namespace Kokkos {
namespace Tools {
namespace Experimental {
static size_t& get_context_counter() {
  static size_t x;
  return x;
}
static size_t& get_variable_counter() {
  static size_t x;
  return ++x;
}

size_t get_new_context_id() { return ++get_context_counter(); }
size_t get_current_context_id() { return get_context_counter(); }
void decrement_current_context_id() { --get_context_counter(); }
size_t get_new_variable_id() { return get_variable_counter(); }

size_t declare_output_type(const std::string& variableName, Kokkos::Tools::Experimental::VariableInfo info) {
  size_t variableId = get_new_variable_id();
#ifdef KOKKOS_ENABLE_TUNING
  Experimental::invoke_kokkosp_callback(
      Experimental::MayRequireGlobalFencing::No,
      Experimental::current_callbacks.declare_output_type, variableName.c_str(),
      variableId, &info);
  variable_metadata[variableId] = info;
#else
  (void)variableName;
  (void)info;
#endif
  return variableId;
}

size_t declare_input_type(const std::string& variableName, VariableInfo info) {
  size_t variableId = get_new_variable_id();
#ifdef KOKKOS_ENABLE_TUNING
  Experimental::invoke_kokkosp_callback(
      Experimental::MayRequireGlobalFencing::No,
      Experimental::current_callbacks.declare_input_type, variableName.c_str(),
      variableId, &info);
  variable_metadata[variableId] = info;
#else
  (void)variableName;
  (void)info;
#endif
  return variableId;
}

void set_input_values(size_t contextId, size_t count, VariableValue* values) {
#ifdef KOKKOS_ENABLE_TUNING
  if (features_per_context.find(contextId) == features_per_context.end()) {
    features_per_context[contextId] = std::unordered_set<size_t>();
  }
  for (size_t x = 0; x < count; ++x) {
    values[x].metadata = &variable_metadata[values[x].type_id];
    features_per_context[contextId].insert(values[x].type_id);
    active_features.insert(values[x].type_id);
    feature_values[values[x].type_id] = values[x];
  }
#else
  (void)contextId;
  (void)count;
  (void)values;
#endif
}
#include <iostream>
void request_output_values(size_t contextId, size_t count,
                           VariableValue* values) {
#ifdef KOKKOS_ENABLE_TUNING
  std::vector<size_t> context_ids;
  std::vector<VariableValue> context_values;
  for (auto id : active_features) {
    context_values.push_back(feature_values[id]);
  }
  if (Experimental::current_callbacks.request_output_values != nullptr) {
    for (size_t x = 0; x < count; ++x) {
      values[x].metadata = &variable_metadata[values[x].type_id];
    }
    Experimental::invoke_kokkosp_callback(
        Experimental::MayRequireGlobalFencing::No,
        Experimental::current_callbacks.request_output_values, contextId,
        context_values.size(), context_values.data(), count, values);
  }
#else
  (void)contextId;
  (void)count;
  (void)values;
#endif
}

#ifdef KOKKOS_ENABLE_TUNING
static std::unordered_map<size_t, size_t> optimization_goals;
#endif

void begin_context(size_t contextId) {
  Experimental::invoke_kokkosp_callback(
      Experimental::MayRequireGlobalFencing::No,
      Experimental::current_callbacks.begin_tuning_context, contextId);
}
void end_context(size_t contextId) {
#ifdef KOKKOS_ENABLE_TUNING
  for (auto id : features_per_context[contextId]) {
    active_features.erase(id);
  }
  Experimental::invoke_kokkosp_callback(
      Experimental::MayRequireGlobalFencing::No,
      Experimental::current_callbacks.end_tuning_context, contextId,
      feature_values[optimization_goals[contextId]]);
  optimization_goals.erase(contextId);
  decrement_current_context_id();
#else
  (void)contextId;
#endif
}

bool have_tuning_tool() {
#ifdef KOKKOS_ENABLE_TUNING
  return (Experimental::current_callbacks.request_output_values != nullptr);
#else
  return false;
#endif
}

VariableValue make_variable_value(size_t id, int64_t val) {
  VariableValue variable_value;
  variable_value.type_id         = id;
  variable_value.value.int_value = val;
  return variable_value;
}
VariableValue make_variable_value(size_t id, double val) {
  VariableValue variable_value;
  variable_value.type_id            = id;
  variable_value.value.double_value = val;
  return variable_value;
}
VariableValue make_variable_value(size_t id, const std::string& val) {
  VariableValue variable_value;
  variable_value.type_id = id;
  strncpy(variable_value.value.string_value, val.c_str(),
          KOKKOS_TOOLS_TUNING_STRING_LENGTH - 1);
  return variable_value;
}
SetOrRange make_candidate_set(size_t size, std::string* data) {
  SetOrRange value_set;
  value_set.set.values.string_value = new TuningString[size];
  for (size_t x = 0; x < size; ++x) {
    strncpy(value_set.set.values.string_value[x], data[x].c_str(),
            KOKKOS_TOOLS_TUNING_STRING_LENGTH - 1);
  }
  value_set.set.size = size;
  return value_set;
}
SetOrRange make_candidate_set(size_t size, int64_t* data) {
  SetOrRange value_set;
  value_set.set.size             = size;
  value_set.set.values.int_value = data;
  return value_set;
}
SetOrRange make_candidate_set(size_t size, double* data) {
  SetOrRange value_set;
  value_set.set.size                = size;
  value_set.set.values.double_value = data;
  return value_set;
}
SetOrRange make_candidate_range(double lower, double upper, double step,
                                bool openLower = false,
                                bool openUpper = false) {
  SetOrRange value_range;
  value_range.range.lower.double_value = lower;
  value_range.range.upper.double_value = upper;
  value_range.range.step.double_value  = step;
  value_range.range.openLower          = openLower;
  value_range.range.openUpper          = openUpper;
  return value_range;
}

SetOrRange make_candidate_range(int64_t lower, int64_t upper, int64_t step,
                                bool openLower = false,
                                bool openUpper = false) {
  SetOrRange value_range;
  value_range.range.lower.int_value = lower;
  value_range.range.upper.int_value = upper;
  value_range.range.step.int_value  = step;
  value_range.range.openLower       = openLower;
  value_range.range.openUpper       = openUpper;
  return value_range;
}

size_t get_new_context_id();
size_t get_current_context_id();
void decrement_current_context_id();
size_t get_new_variable_id();
void declare_optimization_goal(const size_t context,
                               const OptimizationGoal& goal) {
#ifdef KOKKOS_ENABLE_TUNING
  Experimental::invoke_kokkosp_callback(
      Experimental::MayRequireGlobalFencing::No,
      Experimental::current_callbacks.declare_optimization_goal, context, goal);
  optimization_goals[context] = goal.type_id;
#else
  (void)context;
  (void)goal;
#endif
}
}  // end namespace Experimental
}  // end namespace Tools

namespace Impl {

void traceback_callstack(std::ostream& msg) {
  msg << std::endl << "Traceback functionality not available" << std::endl;
}

bool is_unsigned_int(const char* str) {
  const size_t len = strlen(str);
  for (size_t i = 0; i < len; ++i) {
    if (!isdigit(str[i])) {
      return false;
    }
  }
  return true;
}

bool check_arg(char const* arg, char const* expected) {
  std::size_t arg_len = std::strlen(arg);
  std::size_t exp_len = std::strlen(expected);
  if (arg_len < exp_len) return false;
  if (std::strncmp(arg, expected, exp_len) != 0) return false;
  if (arg_len == exp_len) return true;
  
  if (std::isalnum(arg[exp_len]) || arg[exp_len] == '-' ||
      arg[exp_len] == '_') {
    return false;
  }
  return true;
}
void throw_runtime_exception(const std::string& msg) {
  std::ostringstream o;
  o << msg;
  traceback_callstack(o);
  throw std::runtime_error(o.str());
}

bool check_int_arg(char const* arg, char const* expected, int* value) {
  if (!check_arg(arg, expected)) return false;
  std::size_t arg_len = std::strlen(arg);
  std::size_t exp_len = std::strlen(expected);
  bool okay           = true;
  if (arg_len == exp_len || arg[exp_len] != '=') okay = false;
  char const* number = arg + exp_len + 1;
  if (!Kokkos::Impl::is_unsigned_int(number) || strlen(number) == 0)
    okay = false;
  *value = std::stoi(number);
  if (!okay) {
    std::ostringstream ss;
    ss << "Error: expecting an '=INT' after command line argument '" << expected
       << "'";
    ss << ". Raised by Kokkos::initialize(int narg, char* argc[]).";
    Impl::throw_runtime_exception(ss.str());
  }
  return true;
}
bool check_str_arg(char const* arg, char const* expected, std::string& value) {
  if (!check_arg(arg, expected)) return false;
  std::size_t arg_len = std::strlen(arg);
  std::size_t exp_len = std::strlen(expected);
  bool okay           = true;
  if (arg_len == exp_len || arg[exp_len] != '=') okay = false;
  char const* remain = arg + exp_len + 1;
  value              = remain;
  if (!okay) {
    std::ostringstream ss;
    ss << "Error: expecting an '=STRING' after command line argument '"
       << expected << "'";
    ss << ". Raised by Kokkos::initialize(int narg, char* argc[]).";
    Impl::throw_runtime_exception(ss.str());
  }
  return true;
}
void warn_deprecated_command_line_argument(std::string deprecated,
                                           std::string valid) {
  std::cerr
      << "Warning: command line argument '" << deprecated
      << "' is deprecated. Use '" << valid
      << "' instead. Raised by Kokkos::initialize(int narg, char* argc[])."
      << std::endl;
}
}  // namespace Impl


}  // end namespace Kokkos
