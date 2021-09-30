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
/**
 * Before digging in to the code, it's worth taking a moment to review this
 * design. Fundamentally, what we're looking to do is allow people to test that
 * a piece of code Produces some expected series of tool events. Maybe we want
 * to check that deep_copy on an execution space instance only causes the
 * expected types of fences, or that calls to resize(WithoutInitializing,...)
 * don't call an initialization kernel.
 *
 * This design is realized with an interface in which you provide a code region,
 * and a set of matchers that consume the events. These matchers are lambdas
 * that accept some set of tool events, and analyze their content, and return
 * success or failure.
 *
 * Digging into implementation, this works by having a class hierarchy of Tool
 * Events, rooted at EventBase. Every Tool event inherits from this
 * (BeginParallelForEvent, PushRegionEvent, etc). We subscribe a Kokkos Tool
 * that pushes instances of these events into a vector as a code region runs. We
 * then iterate over the list of events and the matchers, first making sure that
 * every event is of the right type to be used in the matcher, and then passing
 * it to the matcher.
 *
 * Current examples are in TestEventCorrectness.hpp
 */

#include <Kokkos_Core.hpp>
#include <sstream>
#include <iostream>
#include <utility>
#include <type_traits>
namespace Kokkos {

namespace Test {

namespace Tools {

/**
 * @brief This is what a matcher should return
 * It is a two-part struct, with a bool representing
 * success (true if the match holds), and a vector of
 * strings representing the diagnostics that should be
 * printed in case of a failure
 */
struct MatchDiagnostic {
  bool success                      = true;
  std::vector<std::string> messages = {};
};

struct EventBase;  // forward declaration
using EventBasePtr = std::shared_ptr<EventBase>;
using EventSet     = std::vector<EventBasePtr>;
using event_vector = EventSet;

/**
 * @brief Base case of a recursive reduction using templates
 * Should be replaced with a fold in C++17
 */

bool is_nonnull() { return true; }

/**
 * @brief Recursive reduction to check whether any pointer in a set is null
 *
 * @tparam Head Type of the pointer to examine
 * @tparam Tail Types of the rest of the pointers
 * @param head The pointer to examine
 * @param tail The rest of the pointers
 * @return true if no pointer is null, false otherwise
 *
 */
template <class Head, class... Tail>
bool is_nonnull(const Head& head, const Tail... tail) {
  return (head != nullptr) && (is_nonnull(tail...));
}

/**
 * @brief In order to call some arbitrary set of lambdas representing matchers,
 * we need the ability to look at a lambda, and deduce its arguments.
 *
 * This is the base template, and will be specialized. All specializations
 * should define A return type R, an args pack A, a num_args, and a function
 * "invoke_as" that takes a functor and an arg-pack, and tries to call the
 * functor with that arg-pack
 *
 * The main original intent here is two-fold, one to allow us to look at how
 * many args a functor takes, and two to look at the types of its args. The
 * second of these is used to do a series of dynamic_casts, making sure that the
 * EventBase's captured in our event vectors are of the types being looked for
 * by our matchers
 *
 * @tparam T a functor-like object
 * @tparam typename used for specialization shenanigans
 */
template <typename T, typename = void>
struct function_traits;

/**
 * @brief Specialization of function traits, representing a free function.
 * See the base template for info on what this struct is doing
 *
 * @tparam R return type of the function
 * @tparam A arg pack
 */
template <typename R, typename... A>
struct function_traits<R (*)(A...)> {
  using return_type                  = R;
  using class_type                   = void;
  using args_type                    = std::tuple<A...>;
  constexpr static int num_arguments = std::tuple_size<args_type>::value;
  template <class Call, class... Args>
  static auto invoke_as(Call call, Args... args) {
    if (!is_nonnull(std::dynamic_pointer_cast<A>(args)...)) {
      return false;
    }
    return call(*std::dynamic_pointer_cast<A>(args)...);
  }
};

/**
 * @brief Specialization of function traits, representing a class member
 * function. See the base template for info on what this struct is doing
 *
 * @tparam R return type of the function
 * @tparam C the class function being represented
 * @tparam A arg pack
 */

template <typename R, typename C, typename... A>
struct function_traits<R (C::*)(A...)> {
  using return_type                  = R;
  using class_type                   = void;
  using args_type                    = std::tuple<A...>;
  constexpr static int num_arguments = std::tuple_size<args_type>::value;
  template <class Call, class... Args>
  static auto invoke_as(Call call, Args... args) {
    if (!is_nonnull(std::dynamic_pointer_cast<A>(args)...)) {
      return false;
    }
    return call(*std::dynamic_pointer_cast<A>(args)...);
  }
};

/**
 * @brief Specialization of function traits, representing a *const* class member
 * function. See the base template for info on what this struct is doing
 *
 * @tparam R return type of the function
 * @tparam C the class function being represented
 * @tparam A arg pack
 */

template <typename R, typename C, typename... A>
struct function_traits<R (C::*)(A...) const>  // const
{
  using return_type                  = R;
  using class_type                   = C;
  using args_type                    = std::tuple<A...>;
  constexpr static int num_arguments = std::tuple_size<args_type>::value;
  template <class Call, class... Args>
  static auto invoke_as(Call call, const Args&... args) {
    if (!is_nonnull(std::dynamic_pointer_cast<A>(args)...)) {
      return MatchDiagnostic{false, {"Types didn't match on arguments"}};
    }
    return call(*std::dynamic_pointer_cast<A>(args)...);
  }
};

/**
 * @brief Specialization of function traits, representing a T that has a
 * non-generic call operator, i.e. a functor/lambda whose operator() has no auto
 * or template on it See the base template for info on what this struct is doing
 *
 * @tparam T The functor type
 */
template <typename T>
struct function_traits<T, Kokkos::Impl::void_t<decltype(&T::operator())> >
    : public function_traits<decltype(&T::operator())> {};

/**
 * @brief A struct to extract events from an event vector, and invoke a matcher
 * with them
 *
 * This one is a bit funky, you can't do std::get's or the like with a vector.
 * So this takes in a number of arguments to pull from the vector, and a start
 * index at which to begin taking from. It then makes an index sequence of tht
 * number of elements {0, 1, 2, ..., num}, and then uses the function_traits
 * trick above to invoke the matcher with
 * {events[index+0],events[index+1],...,events[num-1]}
 *
 * @tparam num number of arguments to the functor
 * @tparam Matcher the lambda we want to call with events from our event vector
 */
template <int num, class Matcher>
struct invoke_helper {
 private:
  // private helper with an index_sequence, invokes the matcher
  template <class Traits, size_t... Indices>
  static auto call(int index, event_vector events,
                   std::index_sequence<Indices...>, Matcher matcher) {
    return Traits::invoke_as(matcher, events[index + Indices]...);
  }

 public:
  // the entry point to the class, takes in a Traits class that knows how to
  // invoke the matcher,
  template <class Traits>
  static auto call(int index, event_vector events, Matcher matcher) {
    return call<Traits>(index, events, std::make_index_sequence<num>{},
                        matcher);
  }
};

/**
 * @brief This is the base case of a recursive check of matchers, meaning no
 * more matchers exist. The only check now should be that we made it all the way
 * through the list of events captured by our lambda
 *
 * @param index how many events we scanned
 * @param events the vector containing our events
 * @return MatchDiagnostic success if we scanned all events, failure otherwise
 */
MatchDiagnostic check_match(event_vector::size_type index,
                            event_vector events) {
  return (index == events.size())
             ? MatchDiagnostic{true}
             : MatchDiagnostic{false, {"Wrong number of events encountered"}};
}

/**
 * @brief
 *
 * @tparam Matcher a functor that accepts a set of events, and returns whether
 * they meet an expected structure
 * @tparam Matchers additional matchers to invoke, supposing the current one is
 * fine
 * @param index What position in our vector of events to begin pulling events
 * from
 * @param events A vector of events we want to match against our matchers
 * @param matcher the instance of Matcher (see above)
 * @param matchers the instances of Matchers (see above)
 * @return MatchDiagnostic success if the matcher matches, failure otherwise
 */
template <class Matcher, class... Matchers>
MatchDiagnostic check_match(event_vector::size_type index, event_vector events,
                            Matcher matcher, Matchers... matchers) {
  // struct that tells us what we want to know about our matcher, and helps us
  // invoke it
  using Traits = function_traits<Matcher>;
  // how many args does that lambda have?
  constexpr static event_vector::size_type num_args = Traits::num_arguments;
  // make sure that we don't have insufficient events in our event vector
  if (index + num_args > events.size()) {
    return {false, {"Too many events encounted"}};
  }
  // Call the lambda, if it's callable with our args. Store the resulting
  // MatchDiagnostic
  auto result = invoke_helper<num_args, Matcher>::template call<Traits>(
      index, events, matcher);
  // If we fail, don't continue looking for more matches, just return
  if (!result.success) {
    return result;
  }
  // Otherwise, call with the next matcher
  return check_match(index + num_args, events, matchers...);
}

/**
 * @brief Small utility helper, an entry point into "check_match."
 * The real "check_match" needs an index at which to start checking,
 * this just tells it "hey, start at 0"
 *
 */
template <class... Matchers>
auto check_match(event_vector events, Matchers... matchers) {
  return check_match(0, events, matchers...);
}

/**
 * @brief Base class of representing everything you can do with an Event
 * checked by this system. Not much is required, just
 *
 * 1) You can compare to some other EventBase
 * 2) You can represent yourself as a string
 */
struct EventBase {
  template <typename T>
  constexpr static uint64_t unspecified_sentinel =
      std::numeric_limits<T>::max();
  virtual ~EventBase()             = default;
  virtual std::string repr() const = 0;
};

/**
 * @brief There are an unholy number of begin events in Kokkos, this is a base
 * class for them (BeginParallel[For/Reduce/Scan], BeginFence)
 *
 * @tparam Derived CRTP, intended for use with dynamic_casts
 */
template <class Derived>
struct BeginOperation : public EventBase {
  using ThisType = BeginOperation;
  const std::string name;
  const uint32_t deviceID;
  uint64_t kID;
  BeginOperation(const std::string& n,
                 const uint32_t devID = unspecified_sentinel<uint32_t>,
                 uint64_t k           = unspecified_sentinel<uint64_t>)
      : name(n), deviceID(devID), kID(k) {}
  virtual ~BeginOperation() = default;
  virtual std::string repr() const {
    std::stringstream s;
    s << Derived::begin_op_name() << " { " << name << ", ";
    if (deviceID == unspecified_sentinel<uint32_t>) {
      s << "(any deviceID) ";
    } else {
      s << deviceID;
    }
    s << ",";
    if (kID == unspecified_sentinel<uint64_t>) {
      s << "(any kernelID) ";
    } else {
      s << kID;
    }
    s << "}";
    return s.str();
  }
};
/**
 * @brief Analogous to BeginOperation, there are a lot of things in Kokkos
 * of roughly this structure
 *
 * @tparam Derived CRTP, used for comparing that EventBase's are of the same
 * type
 */
template <class Derived>
struct EndOperation : public EventBase {
  using ThisType = EndOperation;
  uint64_t kID;
  EndOperation(uint64_t k = unspecified_sentinel<uint64_t>) : kID(k) {}
  virtual ~EndOperation() = default;

  virtual std::string repr() const {
    std::stringstream s;
    s << Derived::end_op_name() << " { ";
    if (kID == unspecified_sentinel<uint64_t>) {
      s << "(any kernelID) ";
    } else {
      s << kID;
    }
    s << "}";
    return s.str();
  }
};

/**
 * Note, the following classes look identical, and they are. They exist because
 * we're using dynamic_casts up above to check whether events are of the same
 * type. So the different type names here are meaningful, even though the
 * classes are empty
 */
struct BeginParallelForEvent : public BeginOperation<BeginParallelForEvent> {
  static const std::string& begin_op_name() {
    static std::string value = "BeginParallelFor";
    return value;
  }
  BeginParallelForEvent(
      std::string n,
      const uint32_t devID = EventBase::unspecified_sentinel<uint32_t>,
      uint64_t k           = EventBase::unspecified_sentinel<uint64_t>)
      : BeginOperation<BeginParallelForEvent>(n, devID, k) {}
  virtual ~BeginParallelForEvent() = default;
};
struct BeginParallelReduceEvent
    : public BeginOperation<BeginParallelReduceEvent> {
  static const std::string& begin_op_name() {
    static std::string value = "BeginParallelReduce";
    return value;
  }

  BeginParallelReduceEvent(
      std::string n,
      const uint32_t devID = EventBase::unspecified_sentinel<uint32_t>,
      uint64_t k           = EventBase::unspecified_sentinel<uint64_t>)
      : BeginOperation<BeginParallelReduceEvent>(n, devID, k) {}
  virtual ~BeginParallelReduceEvent() = default;
};
struct BeginParallelScanEvent : public BeginOperation<BeginParallelScanEvent> {
  static const std::string& begin_op_name() {
    static std::string value = "BeginParallelScan";
    return value;
  }

  BeginParallelScanEvent(
      std::string n,
      const uint32_t devID = EventBase::unspecified_sentinel<uint32_t>,
      uint64_t k           = EventBase::unspecified_sentinel<uint64_t>)
      : BeginOperation<BeginParallelScanEvent>(n, devID, k) {}
  virtual ~BeginParallelScanEvent() = default;
};
struct BeginFenceEvent : public BeginOperation<BeginFenceEvent> {
  static const std::string& begin_op_name() {
    static std::string value = "BeginFence";
    return value;
  }

  BeginFenceEvent(
      std::string n,
      const uint32_t devID = EventBase::unspecified_sentinel<uint32_t>,
      uint64_t k           = EventBase::unspecified_sentinel<uint64_t>)
      : BeginOperation<BeginFenceEvent>(n, devID, k) {}
  virtual ~BeginFenceEvent() = default;
};

struct EndParallelForEvent : public EndOperation<EndParallelForEvent> {
  static const std::string& end_op_name() {
    static std::string value = "EndParallelFor";
    return value;
  }

  EndParallelForEvent(uint64_t k = EventBase::unspecified_sentinel<uint64_t>)
      : EndOperation<EndParallelForEvent>(k) {}
  virtual ~EndParallelForEvent() = default;
};
struct EndParallelReduceEvent : public EndOperation<EndParallelReduceEvent> {
  static const std::string& end_op_name() {
    static std::string value = "EndParallelReduce";
    return value;
  }

  EndParallelReduceEvent(uint64_t k = EventBase::unspecified_sentinel<uint64_t>)
      : EndOperation<EndParallelReduceEvent>(k) {}
  virtual ~EndParallelReduceEvent() = default;
};
struct EndParallelScanEvent : public EndOperation<EndParallelScanEvent> {
  static const std::string& end_op_name() {
    static std::string value = "EndParallelScan";
    return value;
  }

  EndParallelScanEvent(uint64_t k = EventBase::unspecified_sentinel<uint64_t>)
      : EndOperation<EndParallelScanEvent>(k) {}
  virtual ~EndParallelScanEvent() = default;
};
struct EndFenceEvent : public EndOperation<EndFenceEvent> {
  static const std::string& end_op_name() {
    static std::string value = "EndFence";
    return value;
  }

  EndFenceEvent(uint64_t k = EventBase::unspecified_sentinel<uint64_t>)
      : EndOperation<EndFenceEvent>(k) {}
  virtual ~EndFenceEvent() = default;
};

struct InitEvent : public EventBase {
  int load_sequence;
  uint64_t version_number;
  uint32_t num_device_infos;
  Kokkos::Profiling::KokkosPDeviceInfo* device_infos;
  virtual std::string repr() const override {
    std::stringstream s;
    s << "InitEvent { load_sequence: " << load_sequence << ", version_number "
      << version_number << ", num_device_infos " << num_device_infos << "}";
    return s.str();
  }
  InitEvent(int l, uint64_t v_n, uint32_t n_d_i,
            Kokkos::Profiling::KokkosPDeviceInfo* d_i)
      : load_sequence(l),
        version_number(v_n),
        num_device_infos(n_d_i),
        device_infos(d_i) {}
};
struct FinalizeEvent : public EventBase {
  virtual std::string repr() const override { return "FinalizeEvent{}"; }
};

struct ParseArgsEvent : public EventBase {
  int num_args;
  char** args;

  std::string repr() const override {
    std::stringstream s;
    s << "ParseArgsEvent { num_args : " << num_args << std::endl;
    for (int x = 0; x < num_args; ++x) {
      s << "  " << args[x] << std::endl;
    }
    s << "}";
    return s.str();
  }
  ParseArgsEvent(int n_a, char** a) : num_args(n_a), args(a) {}
};
struct PrintHelpEvent : public EventBase {
  char* prog_name;
  std::string repr() const override {
    return "PrintHelpEvent { Program Name: " + std::string(prog_name) + "}";
  }
  PrintHelpEvent(char* p_n) : prog_name(p_n) {}
};
struct PushRegionEvent : public EventBase {
  std::string name;
  std::string repr() const override {
    return "PushRegionEvent { Region Name: " + name + " }";
  }
  PushRegionEvent(std::string n) : name(n) {}
};
struct PopRegionEvent : public EventBase {
  std::string repr() const override { return "PopRegionEvent{}"; }
};

template <class Derived>
struct DataEvent : public EventBase {
  using SpaceHandleType = Kokkos::Profiling::SpaceHandle;
  SpaceHandleType handle;
  std::string name;
  void* const ptr;
  uint64_t size;

  std::string repr() const override {
    std::stringstream s;
    s << Derived::event_name() << "{ In space " << handle.name
      << ", name: " << name << ", ptr: " << ptr << ", size: " << size;
    return s.str();
  }
  DataEvent(SpaceHandleType h, std::string n, void* const p, uint64_t s)
      : handle(h), name(n), ptr(p), size(s) {}
};

struct AllocateDataEvent : public DataEvent<AllocateDataEvent> {
  static std::string event_name() { return "AllocateDataEvent"; }
  AllocateDataEvent(SpaceHandleType h, std::string n, void* const p, uint64_t s)
      : DataEvent<AllocateDataEvent>(h, n, p, s) {}
};
struct DeallocateDataEvent : public DataEvent<DeallocateDataEvent> {
  static std::string event_name() { return "DeallocateDataEvent"; }
  DeallocateDataEvent(SpaceHandleType h, std::string n, void* const p,
                      uint64_t s)
      : DataEvent<DeallocateDataEvent>(h, n, p, s) {}
};

struct CreateProfileSectionEvent : public EventBase {
  std::string name;
  uint32_t section_id;
  std::string repr() const override {
    return "CreateProfileSectionEvent {" + name + ", " +
           std::to_string(section_id) + "}";
  }
  CreateProfileSectionEvent(std::string n, uint32_t s_i)
      : name(n), section_id(s_i) {}
};

template <class Derived>
struct ProfileSectionManipulationEvent : public EventBase {
  uint32_t device_id;
  std::string repr() const override {
    std::stringstream s;
    s << Derived::event_name() << "{ " << device_id << "}";
    return s.str();
  }
  ProfileSectionManipulationEvent(uint32_t d_i) : device_id(d_i){};
};

struct BeginProfileSectionEvent
    : public ProfileSectionManipulationEvent<BeginProfileSectionEvent> {
  static std::string event_name() { return "BeginProfileSectionEvent"; }
  BeginProfileSectionEvent(uint32_t d_i)
      : ProfileSectionManipulationEvent<BeginProfileSectionEvent>(d_i){};
};
struct EndProfileSectionEvent
    : public ProfileSectionManipulationEvent<EndProfileSectionEvent> {
  static std::string event_name() { return "EndProfileSectionEvent"; }
  EndProfileSectionEvent(uint32_t d_i)
      : ProfileSectionManipulationEvent<EndProfileSectionEvent>(d_i){};
};
struct DestroyProfileSectionEvent
    : public ProfileSectionManipulationEvent<DestroyProfileSectionEvent> {
  static std::string event_name() { return "DestroyProfileSectionEvent"; }
  DestroyProfileSectionEvent(uint32_t d_i)
      : ProfileSectionManipulationEvent<DestroyProfileSectionEvent>(d_i){};
};

struct ProfileEvent : public EventBase {
  std::string name;
  std::string repr() const override { return "ProfileEvent {" + name + "}"; }
  ProfileEvent(std::string n) : name(n) {}
};

struct BeginDeepCopyEvent : public EventBase {
  using SpaceHandleType = Kokkos::Profiling::SpaceHandle;
  SpaceHandleType src_handle;
  std::string src_name;
  void* const src_ptr;
  SpaceHandleType dst_handle;
  std::string dst_name;
  void* const dst_ptr;
  uint64_t size;
  std::string repr() const override {
    std::stringstream s;
    s << "BeginDeepCopyEvent { size: " << size << std::endl;
    s << "  dst: { " << dst_handle.name << ", " << dst_name << ", " << dst_ptr
      << "}\n";
    s << "  src: { " << src_handle.name << ", " << src_name << ", " << src_ptr
      << "}\n";
    s << "}";
    return s.str();
  }
  BeginDeepCopyEvent(SpaceHandleType s_h, std::string s_n, void* const s_p,
                     SpaceHandleType d_h, std::string d_n, void* const d_p,
                     uint64_t s)
      : src_handle(s_h),
        src_name(s_n),
        src_ptr(s_p),
        dst_handle(d_h),
        dst_name(d_n),
        dst_ptr(d_p),
        size(s) {}
};
struct EndDeepCopyEvent : public EventBase {
  std::string repr() const override { return "EndDeepCopyEvent{}"; }
};

template <class Derived>
struct DualViewEvent : public EventBase {
  std::string name;
  void* const ptr;
  bool is_device;
  DualViewEvent(std::string n, void* const p, bool i_d)
      : name(n), ptr(p), is_device(i_d) {}
  std::string repr() const override {
    std::stringstream s;
    s << Derived::event_name() << " { " << name << ", " << std::hex << ptr
      << ", " << std::boolalpha << is_device << "}";
    return s.str();
  }
};
struct DualViewModifyEvent : public DualViewEvent<DualViewModifyEvent> {
  static std::string event_name() { return "DualViewModifyEvent"; }
  DualViewModifyEvent(std::string n, void* const p, bool i_d)
      : DualViewEvent(n, p, i_d) {}
};
struct DualViewSyncEvent : public DualViewEvent<DualViewSyncEvent> {
  static std::string event_name() { return "DualViewSyncEvent"; }
  DualViewSyncEvent(std::string n, void* const p, bool i_d)
      : DualViewEvent(n, p, i_d) {}
};

struct DeclareMetadataEvent : public EventBase {
  std::string key;
  std::string value;
  std::string repr() const override {
    return "DeclareMetadataEvent {" + key + ", " + value + "}";
  }
  DeclareMetadataEvent(std::string k, std::string v) : key(k), value(v) {}
};

struct ProvideToolProgrammingInterfaceEvent : public EventBase {
  using Interface = Kokkos::Tools::Experimental::ToolProgrammingInterface;

  uint32_t num_functions;
  Interface interface;
  ProvideToolProgrammingInterfaceEvent(uint32_t n_f, Interface i)
      : num_functions(n_f), interface(i) {}
  std::string repr() const override {
    return "ProvideToolProgrammingInterfaceEvent {" +
           std::to_string(num_functions) + "}";
  }
};
struct RequestToolSettingsEvent : public EventBase {
  using Settings = Kokkos::Tools::Experimental::ToolSettings;

  uint32_t num_settings;
  Settings settings;
  RequestToolSettingsEvent(uint32_t n_s, Settings s)
      : num_settings(n_s), settings(s) {}
  std::string repr() const override {
    return "RequestToolSettingsEvent {" + std::to_string(num_settings) + "}";
  }
};

template <class Derived>
struct TypeDeclarationEvent : public EventBase {
  std::string name;
  uint32_t device_id;
  Kokkos::Tools::Experimental::VariableInfo info;
  std::string repr() const override {
    return Derived::event_name() + "{ " + name + "," +
           std::to_string(device_id) + "}";
  }
  TypeDeclarationEvent(std::string n, uint32_t d_i,
                       Kokkos::Tools::Experimental::VariableInfo i)
      : name(n), device_id(d_i), info(i) {}
};
struct OutputTypeDeclarationEvent
    : public TypeDeclarationEvent<OutputTypeDeclarationEvent> {
  static std::string event_name() { return "OutputTypeDeclarationEvent"; }
  OutputTypeDeclarationEvent(std::string n, uint32_t d_i,
                             Kokkos::Tools::Experimental::VariableInfo i)
      : TypeDeclarationEvent(n, d_i, i) {}
};
struct InputTypeDeclarationEvent
    : public TypeDeclarationEvent<InputTypeDeclarationEvent> {
  static std::string event_name() { return "InputTypeDeclarationEvent"; }
  InputTypeDeclarationEvent(std::string n, uint32_t d_i,
                            Kokkos::Tools::Experimental::VariableInfo i)
      : TypeDeclarationEvent(n, d_i, i) {}
};

struct RequestOutputValuesEvent : public EventBase {
  size_t context;
  size_t num_inputs;
  std::vector<Kokkos::Tools::Experimental::VariableValue> inputs;
  size_t num_outputs;
  std::vector<Kokkos::Tools::Experimental::VariableValue> outputs;
  std::string repr() const override {
    std::stringstream s;
    s << "RequestOutputValuesEvent { ";
    s << num_inputs << " inputs,";
    s << num_outputs << " outputs}";
    return s.str();
  }
  RequestOutputValuesEvent(
      size_t c, size_t n_i,
      std::vector<Kokkos::Tools::Experimental::VariableValue> i, size_t n_o,
      std::vector<Kokkos::Tools::Experimental::VariableValue> o)
      : context(c), num_inputs(n_i), inputs(i), num_outputs(n_o), outputs(o) {}
};

struct ContextBeginEvent : public EventBase {
  size_t context;
  std::string repr() const override {
    return "ContextBeginEvent{ " + std::to_string(context) + "}";
  }
  ContextBeginEvent(size_t c) : context(c) {}
};
struct ContextEndEvent : public EventBase {
  size_t context;
  Kokkos::Tools::Experimental::VariableValue value;
  std::string repr() const override {
    return "ContextEndEvent {" + std::to_string(context) + "}";
  }
  ContextEndEvent(size_t c, Kokkos::Tools::Experimental::VariableValue v)
      : context(c), value(v) {}
};

struct OptimizationGoalDeclarationEvent : public EventBase {
  size_t context;
  Kokkos::Tools::Experimental::OptimizationGoal goal;
  std::string repr() const override {
    return "OptimizationGoalDeclarationEvent{" + std::to_string(context) + "}";
  }
  OptimizationGoalDeclarationEvent(
      size_t c, Kokkos::Tools::Experimental::OptimizationGoal g)
      : context(c), goal(g) {}
};

/**
 * @brief Takes a vector of events, a set of matchers, and checks whether
 *        that event vector matches what those matchers expect
 *
 * @tparam Matchers types of our matchers
 * @param events A vector containing events
 * @param matchers A set of functors that match those Events
 * @return true on successful match, false otherwise
 */
template <class... Matchers>
bool compare_event_vectors(event_vector events, Matchers... matchers) {
  // leans on check_match to do the bulk of the work
  auto diagnostic = check_match(events, matchers...);
  // On failure, print out the error messages
  if (!diagnostic.success) {
    for (const auto& message : diagnostic.messages) {
      std::cerr << "Error matching event vectors: " << message << std::endl;
    }
  }
  return diagnostic.success;
}

/**
 * @brief This tells the testing tool which events to listen to.
 * My strong recommendation is to make this "all events" in most cases,
 * but if there is an event that is hard to match in some cases, a stray
 * deep_copy or the like, this will let you ignore that event
 */

struct ToolValidatorConfiguration {
  struct Profiling {
    bool kernels       = true;
    bool fences        = true;
    bool allocs        = true;
    bool copies        = true;
    bool dual_view_ops = true;
  };
  struct Tuning {
    bool contexts          = true;
    bool type_declarations = true;
    bool request_values    = true;
  };
  struct Infrastructure {
    bool init                  = true;
    bool finalize              = true;
    bool programming_interface = true;
    bool request_settings      = true;
  };
  Profiling profiling           = {false, false, false, false, false};
  Tuning tuning                 = Tuning();
  Infrastructure infrastructure = Infrastructure();
};

namespace Config {
#define KOKKOS_IMPL_TOOLS_TEST_CONFIG_OPTION(name, value, depth)    \
  template <bool target_value>                                      \
  struct Toggle##name : public std::integral_constant<int, depth> { \
    void operator()(ToolValidatorConfiguration& config) {           \
      config.value = target_value;                                  \
    }                                                               \
  };                                                                \
  using Enable##name  = Toggle##name<true>;                         \
  using Disable##name = Toggle##name<false>

KOKKOS_IMPL_TOOLS_TEST_CONFIG_OPTION(Kernels, profiling.kernels, 2);
KOKKOS_IMPL_TOOLS_TEST_CONFIG_OPTION(Fences, profiling.fences, 2);
KOKKOS_IMPL_TOOLS_TEST_CONFIG_OPTION(Allocs, profiling.allocs, 2);
KOKKOS_IMPL_TOOLS_TEST_CONFIG_OPTION(Copies, profiling.copies, 2);
KOKKOS_IMPL_TOOLS_TEST_CONFIG_OPTION(DualViewOps, profiling.dual_view_ops, 2);
KOKKOS_IMPL_TOOLS_TEST_CONFIG_OPTION(Contexts, tuning.contexts, 2);
KOKKOS_IMPL_TOOLS_TEST_CONFIG_OPTION(TypeDeclarations, tuning.type_declarations,
                                     2);
KOKKOS_IMPL_TOOLS_TEST_CONFIG_OPTION(RequestValues, tuning.request_values, 2);
KOKKOS_IMPL_TOOLS_TEST_CONFIG_OPTION(Init, infrastructure.init, 2);
KOKKOS_IMPL_TOOLS_TEST_CONFIG_OPTION(Finalize, infrastructure.finalize, 2);
KOKKOS_IMPL_TOOLS_TEST_CONFIG_OPTION(ProgrammingInterface,
                                     infrastructure.programming_interface, 2);
KOKKOS_IMPL_TOOLS_TEST_CONFIG_OPTION(RequestSettings,
                                     infrastructure.request_settings, 2);

template <bool target_value>
struct ToggleInfrastructure : public std::integral_constant<int, 1> {
  void operator()(ToolValidatorConfiguration& config) {
    config.infrastructure = {target_value, target_value, target_value,
                             target_value};
  }
};

using EnableInfrastructure  = ToggleInfrastructure<true>;
using DisableInfrastructure = ToggleInfrastructure<false>;

template <bool target_value>
struct ToggleProfiling : public std::integral_constant<int, 1> {
  void operator()(ToolValidatorConfiguration& config) {
    config.profiling = {target_value, target_value, target_value, target_value,
                        target_value};
  }
};

using EnableProfiling  = ToggleProfiling<true>;
using DisableProfiling = ToggleProfiling<false>;

template <bool target_value>
struct ToggleTuning : public std::integral_constant<int, 1> {
  void operator()(ToolValidatorConfiguration& config) {
    config.tuning = {target_value, target_value, target_value};
  }
};

using EnableTuning  = ToggleTuning<true>;
using DisableTuning = ToggleTuning<false>;

template <bool target_value>
struct ToggleAll : public std::integral_constant<int, 0> {
  void operator()(ToolValidatorConfiguration& config) {
    ToggleProfiling<target_value>{}(config);
    ToggleTuning<target_value>{}(config);
    ToggleInfrastructure<target_value>{}(config);
  }
};

using EnableAll  = ToggleAll<true>;
using DisableAll = ToggleAll<false>;
}  // namespace Config

/**
 * Needs to stand outside of functions, this is the vector tool callbacks will
 * push events into
 */
std::vector<EventBasePtr> found_events;
/**
 * Needs to stand outside of functions, this is the kID of the last encountered
 * begin event
 */
static uint64_t last_kid;
/** Subscribes to all of the requested callbacks */
void set_tool_events_impl(ToolValidatorConfiguration& config) {
  Kokkos::Tools::Experimental::pause_tools();  // remove all events
  if (config.profiling.kernels) {
    Kokkos::Tools::Experimental::set_begin_parallel_for_callback(
        [](const char* n, const uint32_t d, uint64_t* k) {
          *k = ++last_kid;
          found_events.push_back(
              std::make_shared<BeginParallelForEvent>(std::string(n), d, *k));
        });
    Kokkos::Tools::Experimental::set_begin_parallel_reduce_callback(
        [](const char* n, const uint32_t d, uint64_t* k) {
          *k = ++last_kid;
          found_events.push_back(std::make_shared<BeginParallelReduceEvent>(
              std::string(n), d, *k));
        });
    Kokkos::Tools::Experimental::set_begin_parallel_scan_callback(
        [](const char* n, const uint32_t d, uint64_t* k) {
          *k = ++last_kid;

          found_events.push_back(
              std::make_shared<BeginParallelScanEvent>(std::string(n), d, *k));
        });
    Kokkos::Tools::Experimental::set_end_parallel_for_callback(
        [](const uint64_t k) {
          found_events.push_back(std::make_shared<EndParallelForEvent>(k));
        });
    Kokkos::Tools::Experimental::set_end_parallel_reduce_callback(
        [](const uint64_t k) {
          found_events.push_back(std::make_shared<EndParallelReduceEvent>(k));
        });
    Kokkos::Tools::Experimental::set_end_parallel_scan_callback(
        [](const uint64_t k) {
          found_events.push_back(std::make_shared<EndParallelScanEvent>(k));
        });
  }  // if profiling.kernels
  if (config.profiling.fences) {
    Kokkos::Tools::Experimental::set_begin_fence_callback(
        [](const char* n, const uint32_t d, uint64_t* k) {
          found_events.push_back(
              std::make_shared<BeginFenceEvent>(std::string(n), d, *k));
        });

    Kokkos::Tools::Experimental::set_end_fence_callback([](const uint64_t k) {
      found_events.push_back(std::make_shared<EndFenceEvent>(k));
    });
  }  // profiling.fences
}
template <int priority>
void listen_tool_events_impl(std::integral_constant<int, priority>,
                             ToolValidatorConfiguration&) {}

template <class Config>
void invoke_config(ToolValidatorConfiguration& in, Config conf,
                   std::true_type) {
  conf(in);
}
template <class Config>
void invoke_config(ToolValidatorConfiguration&, Config, std::false_type) {}

template <int priority, class Config, class... Configs>
void listen_tool_events_impl(std::integral_constant<int, priority> prio,
                             ToolValidatorConfiguration& in, Config conf,
                             Configs... configs) {
  invoke_config(in, conf,
                std::integral_constant<bool, priority == conf.value>{});
  listen_tool_events_impl(prio, in, configs...);
}
template <class... Configs>
void listen_tool_events(Configs... confs) {
  ToolValidatorConfiguration conf;
  listen_tool_events_impl(std::integral_constant<int, 0>{}, conf, confs...);
  listen_tool_events_impl(std::integral_constant<int, 1>{}, conf, confs...);
  listen_tool_events_impl(std::integral_constant<int, 2>{}, conf, confs...);
  set_tool_events_impl(conf);
}

/**
 * @brief This is the main entry point people will use to test their programs
 *        Given a lambda representing a code region, and a set of matchers on
 * tools events, verify that the given lambda produces events that match those
 * expected by the matchers
 *
 * @tparam Lambda Type of lam
 * @tparam Matchers Type of matchers
 * @param lam The code region that will produce events
 * @param matchers Matchers for those events, lambdas that expect events and
 * compare them
 * @return true if all events are consumed, all matchers are invoked, and all
 * matchers success, false otherwise
 */
template <class Lambda, class... Matchers>
bool validate_event_set(const Lambda& lam, const Matchers... matchers) {
  // First, erase events from previous invocations
  found_events.clear();
  // Invoke the lambda (this will populate found_events, via tooling)
  lam();
  // compare the found events against the expected ones
  auto success = compare_event_vectors(found_events, matchers...);
  if (!success) {
    // on failure, print out the events we found
    for (const auto& event : found_events) {
      std::cout << event->repr() << std::endl;
    }
  }
  return success;
}
/**
 * @brief Analogous to validate_event_set up above, except rather than
 *        comparing to matchers, this just returns the found event vector
 *
 * @tparam Lambda as in validate_event_set
 * @param lam as in validate_event_set
 * @return auto
 */
template <class Lambda>
auto get_event_set(const Lambda& lam) {
  found_events.clear();
  lam();
  // return compare_event_vectors(expected, found_events);
  std::vector<EventBasePtr> events;
  std::copy(found_events.begin(), found_events.end(),
            std::back_inserter(events));
  return events;
}

}  // namespace Tools
}  // namespace Test
}  // namespace Kokkos
