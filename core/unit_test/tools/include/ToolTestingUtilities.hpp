#include <Kokkos_Core.hpp>
#include <sstream>
#include <iostream>
namespace Kokkos {

namespace Test {

namespace Tools {

struct EventBase;
using EventBasePtr = std::shared_ptr<EventBase>;
using EventSet     = std::vector<EventBasePtr>;

struct MatchDiagnostic {
  bool success                      = true;
  std::vector<std::string> messages = {};
};

struct EventMatcher {
  using Pattern = std::vector<std::shared_ptr<EventMatcher>>;
  struct MatchResult {
    bool match_success;
    int increment_pattern;
    int increment_event;
    std::string diagnostic;
  };
  static MatchResult SimpleMatchResult(bool success,
                                       const std::string& diagnostic = "") {
    return MatchResult{success, 0, 1, diagnostic};
  }
  virtual std::string repr() const                                     = 0;
  virtual MatchResult matches(const EventSet& events, EventSet::size_type event_index) = 0;
  virtual ~EventMatcher() {}
  void checkMatchAt(MatchDiagnostic& builder, const Pattern& pattern,
                    const EventSet& events, EventSet::size_type pattern_index,
                    EventSet::size_type event_index) {
    if (pattern_index >= pattern.size()) {
      builder.messages.push_back("Error: stepped off end of pattern\n");
      return;
    }
    if (event_index >= events.size()) {
      builder.messages.push_back(
          "Error: event set completed, additional events found in the "
          "pattern\n");
      for (EventSet::size_type x = pattern_index; x < pattern.size(); ++x) {
        std::stringstream s;
        s << "Additional pattern entry " << x << ": " << pattern[x]->repr()
          << std::endl;
        builder.messages.push_back(s.str());
      }
      return;
    }
    auto match = pattern[pattern_index]->matches(events, event_index);
    if (!match.match_success) {
      builder.messages.push_back(match.diagnostic);
      return;
    } else {
      if (((pattern_index + match.increment_pattern) == pattern.size()) &&
          ((event_index + match.increment_event) == events.size())) {
        builder.success = true;
        return;
      }
      checkMatchAt(builder, pattern, events,
                   pattern_index + match.increment_pattern,
                   event_index + match.increment_event);
    }
  }
  MatchDiagnostic checkMatch(const Pattern& pattern, const EventSet& events) {
    MatchDiagnostic diagnostic{false};
    checkMatchAt(diagnostic, pattern, events, 0, 0);
    return diagnostic;
  }
};

struct EventRoot : public EventMatcher {
  virtual std::string repr() const { return ""; };
  virtual MatchResult matches(const EventSet&, EventSet::size_type) {
    return {false, 1, 1, ""};
  };
  virtual ~EventRoot() {}
};

struct EventBase : public EventMatcher {
  virtual MatchResult matches(const EventSet& events,
                              EventSet::size_type event_index) override {
    bool is_match = isEqual(*events[event_index]);
    if (is_match) {
      return {true, 1, 1, ""};
    } else {
      std::stringstream s;
      s << "Failed match on event " << event_index << ", expected " << repr()
        << ", got " << events[event_index]->repr() << std::endl;
      return {false, 1, 1, s.str()};
    }
  }
  virtual bool isEqual(const EventBase&) const = 0;
  template <typename T>
  constexpr static uint64_t unspecified_sentinel =
      std::numeric_limits<T>::max();
  virtual ~EventBase()                      = default;
  virtual std::string repr() const override = 0;
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

template <typename MetaMatcher>
struct EventPair : public EventMatcher {
  using Ev = EventBasePtr;
  Ev match;
  using MatchType = std::shared_ptr<EventMatcher>;
  MetaMatcher comparator;
  std::string descriptor;
  EventPair(const Ev& f, const MetaMatcher& n, const std::string& d)
      : match(f), comparator(n), descriptor(d) {}
  virtual std::string repr() const override {
    std::stringstream s;
    s << "A pair of (" << match->repr() << "), then (" << descriptor << ")"
      << std::endl;
    return s.str();
  }
  virtual MatchResult matches(const EventSet& events,
                              EventSet::size_type event_index) override {
    MatchResult result{false, 1, 1};
    if (event_index >= (events.size() - 1)) {
      result.diagnostic = std::string(
          "EventPair searching for a pair of events, but not enough events "
          "left in pattern\n");
      return result;
    }
    if (!match->isEqual(*events[event_index])) {
      std::stringstream s;
      s << "EventPair failed to match first item, expected " << match->repr()
        << ", got " << events[event_index]->repr() << "\n";
      result.diagnostic = s.str();
      return result;
    }
    auto submatch = comparator(events[event_index], events[event_index + 1]);
    if (!submatch.match_success) {
      result.diagnostic = submatch.diagnostic;
      return result;
    }
    result.increment_pattern += submatch.increment_pattern;
    result.increment_event += submatch.increment_event;
    result.match_success = true;
    return result;
  }
};

template <typename MetaMatcher>
auto make_event_pair(const EventBasePtr& first, const MetaMatcher& second,
                     const std::string& descriptor) {
  return std::make_shared<EventPair<MetaMatcher>>(first, second, descriptor);
}

template <typename IntendedEventType1, typename IntendedEventType2,
          typename MetaMatcher>
auto make_event_pair(const EventBasePtr& first, const MetaMatcher& second,
                     const std::string& descriptor) {
  auto comparator = [&](const auto e1, const auto e2) {
    auto first_event  = std::dynamic_pointer_cast<IntendedEventType1>(e1);
    auto second_event = std::dynamic_pointer_cast<IntendedEventType2>(e2);
    if ((first_event == nullptr) || (second_event == nullptr)) {
      return EventMatcher::MatchResult{false, 0, 1,
                                       "Types of events don't match"};
    }
    return second(first_event, second_event);
  };
  return std::make_shared<EventPair<decltype(comparator)>>(first, comparator,
                                                           descriptor);
}

bool compare_event_vectors(const EventMatcher::Pattern& expected,
                           const std::vector<EventBasePtr>& found) {
  EventMatcher* matcher = new EventRoot();
  auto diagnostic       = matcher->checkMatch(expected, found);
  if (!diagnostic.success) {
    for (const auto& message : diagnostic.messages) {
      std::cerr << message;
    }
  }
  delete matcher;
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

template <class Lambda, class... Args>
bool validate_event_set(const EventMatcher::Pattern& expected,
                        const Lambda& lam, Args... args) {
  found_events.clear();
  lam(args...);
  return compare_event_vectors(expected, found_events);
}

template <class Lambda, class... Args>
auto get_event_set(const Lambda& lam, Args... args) {
  found_events.clear();
  lam(args...);
  // return compare_event_vectors(expected, found_events);
  std::vector<EventBasePtr> events;
  std::copy(found_events.begin(), found_events.end(),
            std::back_inserter(events));
  return events;
}

}  // namespace Tools
}  // namespace Test
}  // namespace Kokkos
