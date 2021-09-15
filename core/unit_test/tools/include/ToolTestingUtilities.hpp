#include <Kokkos_Core.hpp>
#include <sstream>
#include <iostream>
namespace Kokkos {

namespace Test {

namespace Tools {


struct EventBase {
  virtual bool isEqual(const EventBase&) const = 0;
  template<typename T>
  constexpr static const uint64_t unspecified_sentinel = std::numeric_limits<T>::max();
  virtual ~EventBase() = default;
  virtual std::string repr() const = 0;

};

using EventBasePtr = std::shared_ptr<EventBase>;


template<class Derived>
struct BeginOperation : public EventBase { 
   using ThisType = BeginOperation;
   const std::string name;
   const uint32_t deviceID;
   uint64_t kID;
   virtual bool isEqual(const EventBase& other_base) const {
     try {
       const auto& other = dynamic_cast<const ThisType&>(other_base);
       bool matches = true;
       matches &= (kID == unspecified_sentinel<uint64_t>) || (kID == other.kID);
       matches &= (deviceID == unspecified_sentinel<uint32_t>) || (deviceID == other.deviceID);
       matches &= name == other.name;
       return matches;
     } catch (std::bad_cast) {
       return false;
     }
     return false;
   }
   BeginOperation(const std::string& n, const uint32_t devID = unspecified_sentinel<uint32_t>, uint64_t k = unspecified_sentinel<uint64_t>) :
   name(n), deviceID(devID), kID(k) {} 
   virtual ~BeginOperation() = default;
   virtual std::string repr () const {
     std::stringstream s;
     
     s << "BeginOp { "  << name <<  ", " <<deviceID <<"," <<kID <<"}";
     return s.str();
   }
};

template<class Derived>
struct EndOperation : public EventBase { 
   using ThisType = EndOperation;
   uint64_t kID;
   virtual bool isEqual(const EventBase& other_base) const {
     try {
       const auto& other = dynamic_cast<const ThisType&>(other_base);
       bool matches = (kID == unspecified_sentinel<uint64_t>) || (kID == other.kID);
       return matches;
     } catch (std::bad_cast) {
       return false;
     }
     return false;
   }
   EndOperation(uint64_t k = unspecified_sentinel<uint64_t>) :
   kID(k) {} 
   virtual ~EndOperation() = default;

      virtual std::string repr () const {
        std::stringstream s;
     s<< "EndOp { "<<kID<<"}";
     return s.str();
   }


};

/**
 * Note, the following classes look identical, and they are. They exist because we're using
 * dynamic_casts up above to check whether events are of the same type. So the different
 * type names here are meaningful, even though the classes are empty
 */
struct BeginParallelForEvent : public BeginOperation<BeginParallelForEvent> {
  BeginParallelForEvent(std::string n, const uint32_t devID = EventBase::unspecified_sentinel<uint32_t>, uint64_t k = EventBase::unspecified_sentinel<uint64_t>) :
  BeginOperation<BeginParallelForEvent>(n, devID, k) {}
  virtual ~BeginParallelForEvent() = default;
};
struct BeginParallelReduceEvent : public BeginOperation<BeginParallelReduceEvent> {
  BeginParallelReduceEvent(std::string n, const uint32_t devID = EventBase::unspecified_sentinel<uint32_t>, uint64_t k = EventBase::unspecified_sentinel<uint64_t>) :
  BeginOperation<BeginParallelReduceEvent>(n, devID, k) {}
  virtual ~BeginParallelReduceEvent() = default;
};
struct BeginParallelScanEvent : public BeginOperation<BeginParallelScanEvent> {
  BeginParallelScanEvent(std::string n, const uint32_t devID = EventBase::unspecified_sentinel<uint32_t>, uint64_t k = EventBase::unspecified_sentinel<uint64_t>) :
  BeginOperation<BeginParallelScanEvent>(n, devID, k) {}
  virtual ~BeginParallelScanEvent() = default;
};
struct BeginFenceEvent : public BeginOperation<BeginFenceEvent> {
  BeginFenceEvent(std::string n, const uint32_t devID = EventBase::unspecified_sentinel<uint32_t>, uint64_t k = EventBase::unspecified_sentinel<uint64_t>) :
  BeginOperation<BeginFenceEvent>(n, devID, k) {}
  virtual ~BeginFenceEvent() = default;
};

struct EndParallelForEvent : public EndOperation<EndParallelForEvent> {
  EndParallelForEvent(uint64_t k = EventBase::unspecified_sentinel<uint64_t>) :
  EndOperation<EndParallelForEvent>(k) {}
  virtual ~EndParallelForEvent() = default;
};
struct EndParallelReduceEvent : public EndOperation<EndParallelReduceEvent> {
  EndParallelReduceEvent(uint64_t k = EventBase::unspecified_sentinel<uint64_t>) :
  EndOperation<EndParallelReduceEvent>(k) {}
  virtual ~EndParallelReduceEvent() = default;
};
struct EndParallelScanEvent : public EndOperation<EndParallelScanEvent> {
  EndParallelScanEvent(uint64_t k = EventBase::unspecified_sentinel<uint64_t>) :
  EndOperation<EndParallelScanEvent>(k) {}
  virtual ~EndParallelScanEvent() = default;
};
struct EndFenceEvent : public EndOperation<EndFenceEvent> {
  EndFenceEvent(uint64_t k = EventBase::unspecified_sentinel<uint64_t>) :
  EndOperation<EndFenceEvent>(k) {}
  virtual ~EndFenceEvent() = default;
};

bool compare_event_vectors(
  const std::vector<EventBasePtr>& expected,
  const std::vector<EventBasePtr>& found 
) {
  
  auto expected_size = expected.size();
  if(found.size() != expected_size) {
    std::cout << "Expected "<<expected_size<<" events, got "<<found.size()<<std::endl;
    for(auto entry : found ){
      std::cout << "Entry: "<<entry->repr()<<std::endl;
    }
    return false; 
    }
  for(int x =0 ; x<expected_size;++x){

    if(!expected[x]->isEqual(*found[x])) {
        for(int y =0;y<expected_size;++y){
          std::cout <<"Expected ["<<y<<"] : "<<expected[y]->repr()<<std::endl;
          std::cout <<"Actual   ["<<y<<"] : "<<found[y]->repr()<<std::endl;

        } 
      return false; 
      }
  }
  return true;
}

std::vector<EventBasePtr> found_events;

struct ToolValidatorConfiguration {
  struct Profiling {
    bool kernels = true;
    bool fences = true;
    Profiling(bool k = true, bool f = true) : kernels(k), fences(f) {}
  };
  struct Tuning {
  };
  struct Infrastructure {

  };
  Profiling profiling;
  Tuning tuning;
  Infrastructure infrastructure;
  ToolValidatorConfiguration(Profiling p = Profiling(), Tuning t = Tuning(), Infrastructure i = Infrastructure()) :
  profiling(p), tuning(t), infrastructure(i) {}
};

void listen_tool_events(ToolValidatorConfiguration config){
  Kokkos::Tools::Experimental::pause_tools();
  if(config.profiling.kernels){
    Kokkos::Tools::Experimental::set_begin_parallel_for_callback(
      [](const char* n, const uint32_t d, uint64_t* k){
        found_events.push_back(std::make_shared<BeginParallelForEvent>(std::string(n), d, *k));
      }
    );
    Kokkos::Tools::Experimental::set_begin_parallel_reduce_callback(
      [](const char* n, const uint32_t d, uint64_t* k){
        found_events.push_back(std::make_shared<BeginParallelReduceEvent>(std::string(n), d, *k));
      }
    );
    Kokkos::Tools::Experimental::set_begin_parallel_scan_callback(
      [](const char* n, const uint32_t d, uint64_t* k){
        found_events.push_back(std::make_shared<BeginParallelScanEvent>(std::string(n), d, *k));
      }
    );
    Kokkos::Tools::Experimental::set_end_parallel_for_callback([](const uint64_t k){
      found_events.push_back(std::make_shared<EndParallelForEvent>(k));
    });
    Kokkos::Tools::Experimental::set_end_parallel_reduce_callback([](const uint64_t k){
      found_events.push_back(std::make_shared< EndParallelReduceEvent>(k));
    });
    Kokkos::Tools::Experimental::set_end_parallel_scan_callback([](const uint64_t k){
      found_events.push_back(std::make_shared< EndParallelScanEvent>(k));
    });
    } // if profiling.kernels
    if(config.profiling.fences) {
        Kokkos::Tools::Experimental::set_begin_fence_callback(
      [](const char* n, const uint32_t d, uint64_t* k){
        found_events.push_back(std::make_shared< BeginFenceEvent>(std::string(n), d, *k));
      }
    );

    Kokkos::Tools::Experimental::set_end_fence_callback([](const uint64_t k){
      found_events.push_back(std::make_shared< EndFenceEvent>(k));
    });
    } // profiling.fences
    /**
  if(config.tuning) {
    // TODO
  } // if tuning
  if(config.infrastructure){
    // TODO
  } // if infrastructure
  */
}

template<class Lambda, class... Args>
bool validate_event_set(
  const std::vector<EventBasePtr>& expected,
  const Lambda& lam, Args... args
) {
  found_events.clear();
  lam(args...);
  return compare_event_vectors(expected, found_events);
}

template<class Lambda, class... Args>
auto get_event_set(
  const Lambda& lam, Args... args
) {

  found_events.clear();
  lam(args...);
  //return compare_event_vectors(expected, found_events);
  std::vector<EventBasePtr> events;
  std::copy(found_events.begin(), found_events.end(), std::back_inserter(events));
  return events;
}

} // namespace Tools
} // namespace Test
} // namespace Kokkos