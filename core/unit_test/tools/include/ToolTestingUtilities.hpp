#include <Kokkos_Core.hpp>
#include <sstream>
#include <iostream>
#include <utility>
#include <type_traits>
namespace Kokkos {

namespace Test {

namespace Tools {

struct MatchDiagnostic {
  bool success                      = true;
  std::vector<std::string> messages = {};
};

// Originally found at https://stackoverflow.com/a/39717241
template <typename... Ts>
struct make_void {
  using type = void;
};
template <typename... Ts>
using void_t = typename make_void<Ts...>::type;

template <typename T, typename = void>
struct function_traits;

struct EventBase;
using EventBasePtr = std::shared_ptr<EventBase>;
using EventSet     = std::vector<EventBasePtr>;
using event_vector = EventSet;

bool is_nonnull() { return true; }

template <class Head, class... Tail>
bool is_nonnull(const Head& head, const Tail... tail) {
  return (head != nullptr) && (is_nonnull(tail...));
}

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

template <typename T>
struct function_traits<T, void_t<decltype(&T::operator())> >
    : public function_traits<decltype(&T::operator())> {};

MatchDiagnostic check_match(event_vector::size_type index,
                            event_vector events) {
  return (index == events.size())
             ? MatchDiagnostic{true}
             : MatchDiagnostic{false, {"Wrong number of events encountered"}};
}

template <int num, class Matcher>
struct invoke_helper {
  template <class Traits, class... Args>
  static auto call(int index, event_vector events, Matcher matcher,
                   Args... args) {
    return invoke_helper<num - 1, Matcher>::template call<Traits>(
        index + 1, events, matcher, args..., events[index]);
  }
};

template <class Matcher>
struct invoke_helper<0, Matcher> {
  template <class Traits, class... Args>
  static auto call(int, event_vector, Matcher matcher, Args... args) {
    return Traits::invoke_as(matcher, args...);
  }
};

template <class Matcher, class... Matchers>
MatchDiagnostic check_match(event_vector::size_type index, event_vector events,
                            Matcher matcher, Matchers... matchers) {
  using Traits                                      = function_traits<Matcher>;
  constexpr static event_vector::size_type num_args = Traits::num_arguments;
  if (index + num_args > events.size()) {
    return {false, {"Too many events encounted"}};
  }
  auto result = invoke_helper<num_args, Matcher>::template call<Traits>(
      index, events, matcher);
  if (!result.success) {
    return result;
  }
  return check_match(index + num_args, events, matchers...);
}

template <class... Matchers>
auto check_match(event_vector events, Matchers... matchers) {
  return check_match(0, events, matchers...);
}

struct EventBase {
  virtual bool isEqual(const EventBase&) const = 0;
  template <typename T>
  constexpr static uint64_t unspecified_sentinel =
      std::numeric_limits<T>::max();
  virtual ~EventBase()             = default;
  virtual std::string repr() const = 0;
};

struct InitEvent : public EventBase {
  virtual bool isEqual(const EventBase&) const = 0;
};

template <class Derived>
struct BeginOperation : public EventBase {
  using ThisType = BeginOperation;
  const std::string name;
  const uint32_t deviceID;
  uint64_t kID;
  virtual bool isEqual(const EventBase& other_base) const {
    try {
      const auto& other = dynamic_cast<const ThisType&>(other_base);
      bool matches      = true;
      matches &= (kID == unspecified_sentinel<uint64_t>) || (kID == other.kID);
      matches &= (deviceID == unspecified_sentinel<uint32_t>) ||
                 (deviceID == other.deviceID);
      matches &= name == other.name;
      return matches;
    } catch (std::bad_cast) {
      return false;
    }
    return false;
  }
  BeginOperation(const std::string& n,
                 const uint32_t devID = unspecified_sentinel<uint32_t>,
                 uint64_t k           = unspecified_sentinel<uint64_t>)
      : name(n), deviceID(devID), kID(k) {}
  virtual ~BeginOperation() = default;
  virtual std::string repr() const {
    std::stringstream s;
    s << "BeginOp { " << name << ", ";
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

template <class Derived>
struct EndOperation : public EventBase {
  using ThisType = EndOperation;
  uint64_t kID;
  virtual bool isEqual(const EventBase& other_base) const {
    try {
      const auto& other = dynamic_cast<const ThisType&>(other_base);
      bool matches =
          (kID == unspecified_sentinel<uint64_t>) || (kID == other.kID);
      return matches;
    } catch (std::bad_cast) {
      return false;
    }
    return false;
  }
  EndOperation(uint64_t k = unspecified_sentinel<uint64_t>) : kID(k) {}
  virtual ~EndOperation() = default;

  virtual std::string repr() const {
    std::stringstream s;
    s << "EndOp { ";
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
  BeginParallelForEvent(
      std::string n,
      const uint32_t devID = EventBase::unspecified_sentinel<uint32_t>,
      uint64_t k           = EventBase::unspecified_sentinel<uint64_t>)
      : BeginOperation<BeginParallelForEvent>(n, devID, k) {}
  virtual ~BeginParallelForEvent() = default;
};
struct BeginParallelReduceEvent
    : public BeginOperation<BeginParallelReduceEvent> {
  BeginParallelReduceEvent(
      std::string n,
      const uint32_t devID = EventBase::unspecified_sentinel<uint32_t>,
      uint64_t k           = EventBase::unspecified_sentinel<uint64_t>)
      : BeginOperation<BeginParallelReduceEvent>(n, devID, k) {}
  virtual ~BeginParallelReduceEvent() = default;
};
struct BeginParallelScanEvent : public BeginOperation<BeginParallelScanEvent> {
  BeginParallelScanEvent(
      std::string n,
      const uint32_t devID = EventBase::unspecified_sentinel<uint32_t>,
      uint64_t k           = EventBase::unspecified_sentinel<uint64_t>)
      : BeginOperation<BeginParallelScanEvent>(n, devID, k) {}
  virtual ~BeginParallelScanEvent() = default;
};
struct BeginFenceEvent : public BeginOperation<BeginFenceEvent> {
  BeginFenceEvent(
      std::string n,
      const uint32_t devID = EventBase::unspecified_sentinel<uint32_t>,
      uint64_t k           = EventBase::unspecified_sentinel<uint64_t>)
      : BeginOperation<BeginFenceEvent>(n, devID, k) {}
  virtual ~BeginFenceEvent() = default;
};

struct EndParallelForEvent : public EndOperation<EndParallelForEvent> {
  EndParallelForEvent(uint64_t k = EventBase::unspecified_sentinel<uint64_t>)
      : EndOperation<EndParallelForEvent>(k) {}
  virtual ~EndParallelForEvent() = default;
};
struct EndParallelReduceEvent : public EndOperation<EndParallelReduceEvent> {
  EndParallelReduceEvent(uint64_t k = EventBase::unspecified_sentinel<uint64_t>)
      : EndOperation<EndParallelReduceEvent>(k) {}
  virtual ~EndParallelReduceEvent() = default;
};
struct EndParallelScanEvent : public EndOperation<EndParallelScanEvent> {
  EndParallelScanEvent(uint64_t k = EventBase::unspecified_sentinel<uint64_t>)
      : EndOperation<EndParallelScanEvent>(k) {}
  virtual ~EndParallelScanEvent() = default;
};
struct EndFenceEvent : public EndOperation<EndFenceEvent> {
  EndFenceEvent(uint64_t k = EventBase::unspecified_sentinel<uint64_t>)
      : EndOperation<EndFenceEvent>(k) {}
  virtual ~EndFenceEvent() = default;
};

template <class... Matchers>
bool compare_event_vectors(event_vector events, Matchers... matchers) {
  auto diagnostic = check_match(0, events, matchers...);
  if (!diagnostic.success) {
    for (const auto& message : diagnostic.messages) {
      std::cerr << message;
    }
  }
  return diagnostic.success;
}

std::vector<EventBasePtr> found_events;

struct ToolValidatorConfiguration {
  struct Profiling {
    bool kernels = true;
    bool fences  = true;
    Profiling(bool k = true, bool f = true) : kernels(k), fences(f) {}
  };
  struct Tuning {};
  struct Infrastructure {};
  Profiling profiling;
  Tuning tuning;
  Infrastructure infrastructure;
  ToolValidatorConfiguration(Profiling p = Profiling(), Tuning t = Tuning(),
                             Infrastructure i = Infrastructure())
      : profiling(p), tuning(t), infrastructure(i) {}
};
static uint64_t last_kid;
void listen_tool_events(ToolValidatorConfiguration config) {
  Kokkos::Tools::Experimental::pause_tools();
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
     /**
   if(config.tuning) {
     // TODO
   } // if tuning
   if(config.infrastructure){
     // TODO
   } // if infrastructure
   */
}

template <class Lambda, class... Matchers>
bool validate_event_set(const Lambda& lam, const Matchers... matchers) {
  found_events.clear();
  lam();
  auto success = compare_event_vectors(found_events, matchers...);
  if (!success) {
    for (const auto& event : found_events) {
      std::cout << event->repr() << std::endl;
    }
  }
  return success;
}

template <class Lambda, class... Args>
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
