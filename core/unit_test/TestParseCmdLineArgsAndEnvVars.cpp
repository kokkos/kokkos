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

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <TestDefaultDeviceType_Category.hpp>

#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace Kokkos {
namespace Impl {
void parse_command_line_arguments(int& narg, char* arg[],
                                  InitArguments& arguments);
void parse_environment_variables(InitArguments& arguments);
}  // namespace Impl
}  // namespace Kokkos

namespace {

class EnvVarsHelper {
  // do not let GTest run unit tests that set the environment concurently
  static std::mutex mutex_;
  std::vector<std::string> vars_;
  // FIXME_CXX17 prefer optional
  // store name of env var that was already set (if any)
  // in which case unit test is skipped
  std::unique_ptr<std::string> skip_;

 public:
  auto& skip() { return skip_; }
  EnvVarsHelper(std::unordered_map<std::string, std::string> const& vars) {
    mutex_.lock();
    for (auto const& x : vars) {
      auto const& name  = x.first;
      auto const& value = x.second;
      // skip unit test if env var is already set
      if (getenv(name.c_str())) {
        skip_ = std::make_unique<std::string>(name);
        break;
      }
#ifdef _WIN32
      int const error_code = _putenv((name + "=" + value).c_str());
#else
      int const error_code =
          setenv(name.c_str(), value.c_str(), /*overwrite=*/0);
#endif
      if (error_code != 0) {
        std::cerr << "failed to set environment variable '" << name << "="
                  << value << "'\n";
        std::abort();
      }
      vars_.push_back(name);
    }
  }
  ~EnvVarsHelper() {
    for (auto const& name : vars_) {
#ifdef _WIN32
      int const error_code = _putenv((name + "=").c_str());
#else
      int const error_code = unsetenv(name.c_str());
#endif
      if (error_code != 0) {
        std::cerr << "failed to unset environment variable '" << name << "'\n";
        std::abort();
      }
    }
    mutex_.unlock();
  }
};
std::mutex EnvVarsHelper::mutex_;

class CmdLineArgsHelper {
  int argc_;
  std::vector<char*> argv_;
  std::vector<std::unique_ptr<char[]>> args_;

 public:
  CmdLineArgsHelper(std::vector<std::string> const& args) : argc_(args.size()) {
    for (auto const& x : args) {
      args_.emplace_back(new char[x.size() + 1]);
      char* ptr = args_.back().get();
      strcpy(ptr, x.c_str());
      argv_.push_back(ptr);
    }
  }
  int& argc() { return argc_; }
  char** argv() { return argv_.data(); }
};

TEST(defaultdevicetype, cmd_line_args_num_threads) {
  CmdLineArgsHelper cla = {{
      "--foo=bar",
      "--kokkos-num-threads=1",
      "--kokkos-num-threads=2",
  }};
  Kokkos::InitArguments ia;
  Kokkos::Impl::parse_command_line_arguments(cla.argc(), cla.argv(), ia);
  EXPECT_EQ(ia.num_threads, 2);
  EXPECT_EQ(cla.argc(), 1);
  EXPECT_STREQ(*cla.argv(), "--foo=bar");

  ia  = {};
  cla = {{
      {"--kokkos-num-threads=-1"},
  }};
  EXPECT_THROW(  // consider calling abort instead
      Kokkos::Impl::parse_command_line_arguments(cla.argc(), cla.argv(), ia),
      std::runtime_error
      // expecting an '=INT' after command line argument '--kokkos-num-threads'
  );
}

TEST(defaultdevicetype, cmd_line_args_device_id) {
  CmdLineArgsHelper cla = {{
      "--kokkos-device-id=3",
      "--dummy",
      "--kokkos-device-id=4",
  }};
  Kokkos::InitArguments ia;
  Kokkos::Impl::parse_command_line_arguments(cla.argc(), cla.argv(), ia);
  EXPECT_EQ(ia.device_id, 4);
  EXPECT_EQ(cla.argc(), 1);
  EXPECT_STREQ(*cla.argv(), "--dummy");
}

TEST(defaultdevicetype, cmd_line_args_num_devices) {
  CmdLineArgsHelper cla = {{
      "--kokkos-num-devices=5,6",
      "--kokkos-num-devices=7",
      "-v",
  }};
  Kokkos::InitArguments ia;
  Kokkos::Impl::parse_command_line_arguments(cla.argc(), cla.argv(), ia);
  EXPECT_EQ(ia.ndevices, 7);
  // this is the current behavior, not suggesting this cannot be revisited
  EXPECT_EQ(ia.skip_device, 6) << "behavior changed see comment";
  EXPECT_EQ(cla.argc(), 1);
  EXPECT_STREQ(*cla.argv(), "-v");
}

TEST(defaultdevicetype, cmd_line_args_disable_warning) {
  CmdLineArgsHelper cla = {{
      "--kokkos-disable-warnings=0",
  }};
  Kokkos::InitArguments ia;
  Kokkos::Impl::parse_command_line_arguments(cla.argc(), cla.argv(), ia);
  // this is the current behavior, not suggesting this cannot be revisited
  // essentially here the =BOOL is ignored
  EXPECT_TRUE(ia.disable_warnings) << "behavior changed see comment";
}

TEST(defaultdevicetype, cmd_line_args_tune_internals) {
  CmdLineArgsHelper cla = {{
      "--kokkos-tune-internals",
      "--kokkos-num-threads=3",
  }};
  Kokkos::InitArguments ia;
  Kokkos::Impl::parse_command_line_arguments(cla.argc(), cla.argv(), ia);
  EXPECT_TRUE(ia.tune_internals);
  EXPECT_EQ(ia.num_threads, 3);
}

TEST(defaultdevicetype, env_vars_num_threads) {
  EnvVarsHelper ev = {{
      {"KOKKOS_NUM_THREADS", "24"},
      {"KOKKOS_DISABLE_WARNINGS", "1"},
  }};
  if (ev.skip()) {
    GTEST_SKIP() << "environment variable '" << *ev.skip()
                 << "' is already set";
  }
  Kokkos::InitArguments ia;
  Kokkos::Impl::parse_environment_variables(ia);
  EXPECT_EQ(ia.num_threads, 24);
  EXPECT_TRUE(ia.disable_warnings);
}

TEST(defaultdevicetype, env_vars_device_id) {
  EnvVarsHelper ev = {{
      {"KOKKOS_DEVICE_ID", "33"},
  }};
  if (ev.skip()) {
    GTEST_SKIP() << "environment variable '" << *ev.skip()
                 << "' is already set";
  }
  Kokkos::InitArguments ia;
  Kokkos::Impl::parse_environment_variables(ia);
  EXPECT_EQ(ia.device_id, 33);
}

TEST(defaultdevicetype, env_vars_num_devices) {
  EnvVarsHelper ev = {{
      {"KOKKOS_NUM_DEVICES", "4"},
      {"KOKKOS_SKIP_DEVICE", "1"},
  }};
  if (ev.skip()) {
    GTEST_SKIP() << "environment variable '" << *ev.skip()
                 << "' is already set";
  }
  Kokkos::InitArguments ia;
  Kokkos::Impl::parse_environment_variables(ia);
  EXPECT_EQ(ia.ndevices, 4);
  EXPECT_EQ(ia.skip_device, 1);
}

TEST(defaultdevicetype, env_vars_disable_warnings) {
  {
    EnvVarsHelper ev = {{
        {"KOKKOS_DISABLE_WARNINGS", "1"},
    }};
    if (ev.skip()) {
      GTEST_SKIP() << "environment variable '" << *ev.skip()
                   << "' is already set";
    }
    Kokkos::InitArguments ia;
    Kokkos::Impl::parse_environment_variables(ia);
    EXPECT_TRUE(ia.disable_warnings);
  }

  {
    EnvVarsHelper ev = {{
        {"KOKKOS_DISABLE_WARNINGS", "0"},
    }};
    if (ev.skip()) {
      GTEST_SKIP() << "environment variable '" << *ev.skip()
                   << "' is already set";
    }
    Kokkos::InitArguments ia;
    Kokkos::Impl::parse_environment_variables(ia);
    EXPECT_FALSE(ia.disable_warnings);
  }
}

TEST(defaultdevicetype, env_vars_tune_internals) {
  EnvVarsHelper ev = {{
      {"KOKKOS_TUNE_INTERNALS", "1"},
  }};
  if (ev.skip()) {
    GTEST_SKIP() << "environment variable '" << *ev.skip()
                 << "' is already set";
  }
  Kokkos::InitArguments ia;
  Kokkos::Impl::parse_environment_variables(ia);
  EXPECT_TRUE(ia.tune_internals);
}

}  // namespace
