# Design Notes for Execution and Memory Space Instances

## Resources

A collection of low-level APIs to manage the resources for various *control threads*.

Each device type has a seperate resources manager (currently only HWLOC_PU_Manger and GPU_Device_Manager)

### HWLOC_PU_Manager

All calls are thread-safe (though they may lock).  Initalize must be called after any call which could change the *process cpuset* (e.g. MPI_Init) but before any other function in this namespace.

The thread requesting resources a *control thread*.  A sucessful request will bind the control thread to the requested resources.  Additional requests from the control thread will release any existing resourses before trying to aquire the new resources.


```
namespace Experiment::Resources::hwloc {

void initialize(); 

void finalize();

hwloc_topology_t topology();

hwloc_const_cpuset_t process_cpuset();

hwloc_const_cpuset_t available_cpuset();

// returns true if the requested cpuset was available
// if true the available_cpuset is modified to exclude 
// the requested cpuset
bool request_cpuset( hwloc_const_cpuset_t cpuset );

// release a previously requested cpuset
// returns true if the cpuset is a subset of the process_cpuset
// and is disjoint from the available_cpuset
bool release_cpuset( hwloc_const_cpuset_t cpuset );

// binds this thread to the requested cpuset
// if promote to cores is true the cpuset will be promoted to the union of all the cores 
// which have at least one active pu in the given cpuset
// returns true if bind was successfull
bool bind_this_thread( hwloc_const_cpuset_t cpuset, bool promote_to_cores = true );

}
```

### GPU_Device_Manager

Manages which device and numbers of streams a control thread is using

```
namespace Experiment::Resources::GPU {

struct  Device {
  int gpu = -1;
  int num_stream = -1;
};

void initialize(); 

void finalize();

// the visible devices are set through an environment variable
std::vector<Device> process_gpus();

// the devices which are still available
std::vector<Device available_devices();

// returns true if the requested device was available
// if true the available_devices is modified to exclude 
// the requested device
bool request_device( Device device );

// release a previously requested device
bool release_device( Device device );


bool bind_to_device( Device );

}
```

### Thread

A very shallow wrapper on top of std::threads for spawning detached threads which are bound to a particular cpuset.
Can work independently from the HWLOC_PU_Manager, but most effective when hwloc is present.  Avoids thread-creation by reusing existing blocked threads if possible.

```
namespace Experiment::Resources::Threads {

// must be called before spawned_detached
void initialize(); 

// destroy any blocked thread
void finalize();

// if location is nullptr or hwloc is not available then the thread will be unbound
// when the function completes the thread will be hard blocked until it is spawned again
template <typename Function, typename... Args>
void spawn_detached( hwloc_const_cpuset_t loc, Function && f, Args &&... args)

// aquire an already existing thread, the thread will be hard blocked until 
// it is lauched as a spawned_thread or the Threads::finalize() is called 
// this_thread::id() must be unique
void aquire_thread( hwloc_const_cpuset_t loc = nullptr)

}
```

## Kokkos

* A *control thread* is a user thread which has requested resources from a resource manager or is the main thread
 
* Each control thread **must** call Kokkos::initialize(...) and Kokkos::finalize()
  If a control thread has not yet requested resources it will examine the arguments and make a suitable request 
  Otherwise it will validate that its current resources does not conflict with the requested resources
  
* Kokkos::initialize(...) and Kokkos::finalize() are thread-safe, but may lock

* **Execution Space** instances are unique to a control thread

* Default instances for the requested execution spaces are created for each control thread and only use resources requested by the control thread

* Parallel alorgithms use the control thread's default instances by default
  e.g.
  ```
  parallel_for( n, functor );
  ```
  
### Execution Spaces

  *  Work is *dispatched* to an execution space instance
  
  * Execution space instances can be subseted to create new instances
  
  *  Exactly one control thread is associated with an instance and only that control thread may dispatch work to to that instance
  
  * A control thread may be a member of an instance,if so then it is also the control thread associated with that instance
  
  * parallel algorithms will block at launching until the resources associated with the instance are available
  
  *  An instance may be masked

    -  Allows work to be dispatched to a subset of the pool

    -  Example: only one hyperthread per core of the instance

    -  When a mask is applied to an instance that mask
       remains until cleared or another mask is applied

    -  Masking is portable by defining it as using a fraction
       of the available resources (threads)
  
  e.g.
  
  ```
  auto A = Kokkos::default_instance<Threads>();
  auto B = A.split( ... );
  auto C = A.split( ... );
  
  // assuming B and C are disjoint subsets and the control thread is a member of B
  
  // asynchronous launch
  parallel_for( Policy<...>(C,...), functor_c ); 
  
  // blocks until the control thread is done -- other threads my still be running
  parallel_for( Policy<...>(B,...), functor_b);
  
  // blocks until both B and C are done since it requires overlaping resources
  parallel_for( Policy<...>(A,...), functor_a);
 ```
 

## Host Associated Execution Space Instances


### Requesting an Execution Space Instance

  *  `Space::request(` *who* `,` *what* `,` *control-opt* `)`

  *  *who* is an identifier for subsquent queries regarding
    who requested each instance

  *  *what* is the number of threads and how they should be placed

    -  Placement within locality-topology hierarchy; e.g., HWLOC

    -  Compact within a level of hierarchy, or striped across that level;
       e.g., socket or NUMA region

    -  Granularity of request is core

  *  *control-opt*  optionally specifies whether the instance
     has a new control thread

    -  *control-opt* includes a control function / closure

    -  The new control thread is a member of the instance

    -  The control function is called by the new control thread
       and is passed a `const` instance

    -  The instance is **not** returned to the creating control thread

  *  `std::thread` that is not a member of an instance is
     *hard blocked* on a `std::mutex`

    -  One global mutex or one mutex per thread?

  *  `std::thread` that is a member of an instance is
     *spinning* waiting for work, or are working

```
struct StdThread {

  struct Resource ;

  static StdThread request(); // default

  static StdThread request( const std::string & , const Resource & );

  // If the instance can be reserved then
  // allocate a copy of ControlClosure and invoke
  //   ControlClosure::operator()( const StdThread intance ) const
  template< class ControlClosure >
  static bool request( const std::string & , const Resource &
                     , const ControlClosure & );
};
```

### Relinquishing an Execution Space Instance

  *  De-referencing the last reference-counted instance
     relinquishes the pool of threads

  *  If a control thread was created for the instance then
     it is relinquished when that control thread returns
     from the control function

    -  Requires the reference count to be zero, an error if not

  *  No *forced* relinquish



## CUDA Associated Execution Space Instances

  *  Only a signle CUDA architecture

  *  An instance is a device + stream

  *  A stream is exclusive to an instance

  *  Only a host-side control thread can dispatch work to an instance

  *  Finite number of streams per device

  *  ISSUE:  How to use CUDA `const` memory with multiple streams?

  *  Masking can be mapped to restricting the number of CUDA blocks
     to the fraction of available resources; e.g., maximum resident blocks


### Requesting an Execution Space Instance

  *  `Space::request(` *who* `,` *what* `)`

  *  *who* is an identifier for subsquent queries regarding
    who requested each instance

  *  *what* is which device, the stream is a requested/relinquished resource


```
struct Cuda {

  struct Resource ;

  static Cuda request();

  static Cuda request( const std::string & , const Resource & );
};
```


