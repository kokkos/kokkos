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

// Originally found at https://stackoverflow.com/a/39717241
// make_void is in C++17
template <typename... Ts>
struct make_void {
  using type = void;
};
template <typename... Ts>
using void_t = typename make_void<Ts...>::type;

struct EventBase; // forward declaration
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
 * This is the base template, and will be specialized. All specializations should define
 * A return type R, an args pack A, a num_args, and a function "invoke_as" that takes a
 * functor and an arg-pack, and tries to call the functor with that arg-pack
 *
 * The main original intent here is two-fold, one to allow us to look at how many args a
 * functor takes, and two to look at the types of its args. The second of these is used
 * to do a series of dynamic_casts, making sure that the EventBase's captured in our event
 * vectors are of the types being looked for by our matchers
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
 * @brief Specialization of function traits, representing a class member function.
 * See the base template for info on what this struct is doing
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
 * @brief Specialization of function traits, representing a *const* class member function.
 * See the base template for info on what this struct is doing
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
 * @brief Specialization of function traits, representing a T that has a non-generic call operator,
 * i.e. a functor/lambda whose operator() has no auto or template on it
 * See the base template for info on what this struct is doing
 * 
 * @tparam T The functor type
 */
template <typename T>
struct function_traits<T, void_t<decltype(&T::operator())> >
    : public function_traits<decltype(&T::operator())> {};


/**
 * @brief A struct to extract events from an event vector, and invoke a matcher with them
 *
 * This one is a bit funky, you can't do std::get's or the like with a vector. So this takes
 * in a number of arguments to pull from the vector, and a start index at which to begin taking from.
 * It then makes an index sequence of tht number of elements {0, 1, 2, ..., num}, and then uses
 * the function_traits trick above to invoke the matcher with 
 * {events[index+0],events[index+1],...,events[num-1]}
 * 
 * @tparam num number of arguments to the functor
 * @tparam Matcher the lambda we want to call with events from our event vector
 */
template <int num, class Matcher>
struct invoke_helper {
  private:
  // private helper with an index_sequence, invokes the matcher
  template<class Traits, size_t... Indices>
  static auto call(int index, event_vector events, std::index_sequence<Indices...>, Matcher matcher){
    return Traits::invoke_as(matcher, events[index+Indices]...);
  }
  public:
  // the entry point to the class, takes in a Traits class that knows how to invoke
  // the matcher,
  template <class Traits>
  static auto call(int index, event_vector events, Matcher matcher) {
    return call<Traits>(index, events, std::make_index_sequence<num>{}, matcher);
  }
};

/**
 * @brief This is the base case of a recursive check of matchers, meaning no more matchers
 * exist. The only check now should be that we made it all the way through the list of events
 * captured by our lambda
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
 * @tparam Matcher a functor that accepts a set of events, and returns whether they meet an expected structure
 * @tparam Matchers additional matchers to invoke, supposing the current one is fine 
 * @param index What position in our vector of events to begin pulling events from
 * @param events A vector of events we want to match against our matchers
 * @param matcher the instance of Matcher (see above)
 * @param matchers the instances of Matchers (see above)
 * @return MatchDiagnostic success if the matcher matches, failure otherwise
 */
template <class Matcher, class... Matchers>
MatchDiagnostic check_match(event_vector::size_type index, event_vector events,
                            Matcher matcher, Matchers... matchers) {
  // struct that tells us what we want to know about our matcher, and helps us invoke it
  using Traits                                      = function_traits<Matcher>;
  // how many args does that lambda have?
  constexpr static event_vector::size_type num_args = Traits::num_arguments;
  // make sure that we don't have insufficient events in our event vector
  if (index + num_args > events.size()) {
    return {false, {"Too many events encounted"}};
  }
  // Call the lambda, if it's callable with our args. Store the resulting MatchDiagnostic
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
  virtual bool isEqual(const EventBase&) const = 0;
  template <typename T>
  constexpr static uint64_t unspecified_sentinel =
      std::numeric_limits<T>::max();
  virtual ~EventBase()             = default;
  virtual std::string repr() const = 0;
};

// TODO
struct InitEvent : public EventBase {
  virtual bool isEqual(const EventBase&) const = 0;
};

/**
 * @brief There are an unholy number of begin events in Kokkos, this is a base class for them
 * (BeginParallel[For/Reduce/Scan], BeginFence)
 * 
 * @tparam Derived CRTP, intended for use with dynamic_casts
 */
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
/**
 * @brief Analogous to BeginOperation, there are a lot of things in Kokkos
 * of roughly this structure
 * 
 * @tparam Derived CRTP, used for comparing that EventBase's are of the same type
 */
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
      std::cerr << message;
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
/**
 * Needs to stand outside of functions, this is the vector tool callbacks will push events into
 */
std::vector<EventBasePtr> found_events;
/**
 * Needs to stand outside of functions, this is the kID of the last encountered begin event
 */
static uint64_t last_kid;
/** Subscribes to all of the requested callbacks */
void listen_tool_events(ToolValidatorConfiguration config) {
  Kokkos::Tools::Experimental::pause_tools(); // remove all events
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

/**
 * @brief This is the main entry point people will use to test their programs
 *        Given a lambda representing a code region, and a set of matchers on tools events,
 *        verify that the given lambda produces events that match those expected by the matchers
 * 
 * @tparam Lambda Type of lam
 * @tparam Matchers Type of matchers
 * @param lam The code region that will produce events
 * @param matchers Matchers for those events, lambdas that expect events and compare them
 * @return true if all events are consumed, all matchers are invoked, and all matchers success, false otherwise
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
