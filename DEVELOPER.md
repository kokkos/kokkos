![Kokkos](https://avatars2.githubusercontent.com/u/10199860?s=200&v=4)

# Developing Kokkos

This document contains build instructions for developers.
For build system details for users, refer to the [build instructions](BUILD.md).

## Build System

Kokkos uses CMake to configure, build, and install.
Rather than being a completely straightforward use of modern CMake,
Kokkos has several extra complications, primarily due to:
* Kokkos must support linking to an installed version or in-tree builds as a subdirectory of a larger project.
* Kokkos must configure a special compiler `nvcc_wrapper` that allows `nvcc` to accept all C++ flags (which it currently does not).
* Kokkos must work as a part of TriBITS, a CMake library providing a particular build idiom for Trilinos.
* Kokkos has many pre-existing users. We need to be careful about breaking previous versions or generating meaningful error messags if we do break backwards compatibility.

If you are looking at the build system code... pulling your hair out... wondering why on Earth we did something that looks so silly and obtuse... the answer is technical debt. Everything in the build system was done for a reason, trying to adhere as closely as possible to modern CMake best practices while meeting all customer requirements. 

### Modern CMake Philosophy

Modern CMake relies on understanding the principle of *building* and *using* a code project.
What preprocessor, compiler, and linker flags do I need to *build* my project?
What flags does a downstream project that links to me need to *use* my project?
In CMake terms, flags that are only needed for building are `PRIVATE`.
Only Kokkos needs these flags, not a package that depends on Kokkos.
Flags that must be used in a downstream project are `PUBLIC`.
Kokkos must tell other projects to use them.

In Kokkos, almost everything is a public flag since Kokkos is driven by headers and Kokkos is in charge of optimizing your code - performance portability!
Include paths, C++ standard flags, architecture-specific optimizations, or OpenMP and CUDA flags are all example of flags that Kokkos configures and adds to your project.

Modern CMake now automatically propagates flags through the `target_link_libraries` command.
Suppose you have a library `stencil` that needs to build with Kokkos.
Consider the following CMake:

````
find_package(Kokkos)
add_library(stencil stencil.cpp)
target_link_libraries(stencil Kokkos::kokkos)
````

This locates the Kokkos package, adds your library, and tells CMake to link Kokkos to your library.
All public build flags get added automatically through the `target_link_libraries` command.
There is nothing to do. You can be happily oblivious to how Kokkos was configured. 
Everything should just work.

As a Kokkos developer who wants to add new public compiler flags, how do you ensure that CMake does this properly? Modern CMake works through targets and properties. 
Each target has a set of standard properties:
* `INTERFACE_COMPILE_OPTIONS` contains all the compiler options that Kokkos should add to downstream projects
* `INTERFACE_INCLUDE_DIRECTORIES` contains all the directories downstream projects must include from Kokkos
* `INTERFACE_COMPILE_DEFINITIONS` contains the list of preprocessor `-D` flags
* `INTERFACE_LINK_LIBRARIES` contains all the libraries downstream projects need to link
* `INTERFACE_COMPILE_FEATURES` essentially adds compiler flags, but with extra complications. Features names are specific to CMake. More later.

CMake makes it easy to append to these properties using:
* `target_compile_options(kokkos PUBLIC -fmyflag)`
* `target_include_directories(kokkos PUBLIC mySpecialFolder)`
* `target_compile_definitions(kokkos PUBLIC -DmySpecialFlag=0)`
* `target_link_libraries(kokkos PUBLIC mySpecialLibrary)`
* `target_compile_features(kokkos PUBLIC mySpecialFeature)`
Note that all of these use `PUBLIC`! Almost every Kokkos flag is not private to Kokkos, but must also be used by downstream projects.


### Compiler Features and Compiler Options
Compiler options are flags like `-fopenmp` that do not need to be "resolved." 
The flag is either on or off.
Compiler features are more fine-grained and require conflicting requests to be resolved.
Suppose I have
````
add_library(A a.cpp)
target_compile_features(A PUBLIC cxx_std_11)
````
then another target
````
add_library(B b.cpp)
target_compile_features(B PUBLIC cxx_std_14)
target_link_libraries(A B)
````
I have requested two diferent features.
CMake understands the requests and knows that `cxx_std_11` is a subset of `cxx_std_14`.
CMake then picks C++14 for library `B`. 
CMake would not have been able to do feature resolution if we had directly done:
````
target_compile_options(A PUBLIC -std=c++11)
````

### Adding Kokkos Options
After configuring for the first time, 
CMake creates a cache of configure variables in `CMakeCache.txt`.
Reconfiguring in the folder "restarts" from those variables.
All flags passed as `-DKokkos_SOME_OPTION=X` to `cmake` become variables in the cache.
All Kokkos options begin with camel case `Kokkos` followed by an upper case option name.

CMake best practice is to avoid cache variables, if possible.
In essence, you want the minimal amount of state cached between configurations.
And never, ever have behavior influenced by multiple cache variables.
If you want to change the Kokkos configuration, have a single unique variable that needs to be changed.
Never require two cache variables to be changed.

Kokkos provides a function `KOKKOS_OPTION` for defining valid cache-level variables,
proofreading them, and defining local project variables.
The most common variables are called `Kokkos_ENABLE_X`, 
for which a helper function `KOKKOS_ENABLE_OPTION` is provided, e.g.
````
KOKKOS_ENABLE_OPTION(TESTS OFF  "Whether to build tests")
````
The function checks if `-DKokkos_ENABLE_TESTS` was given,
whether it was given with the wrong case, e.g. `-DKokkos_Enable_Tests`,
and then defines a regular (non-cache) variable `KOKKOS_ENABLE_TESTS` to `ON` or `OFF`
depending on the given default and whether the option was specified.

### Defining Kokkos Config Macros

Sometimes you may want to add `#define Kokkos_X` macros to the config header.
This is straightforward with CMake.
Suppose you want to define an optional macro `KOKKOS_SUPER_SCIENCE`.
Simply go into `KokkosCore_config.h.in` and add
````
#cmakedefine KOKKOS_SUPER_SCIENCE
````
I can either add
````
KOKKOS_OPTION(SUPER_SCIENCE ON "Whether to do some super science")
````
to directly set the variable as a command-line `-D` option.
Alternatively, based on other logic, I could add to a `CMakeLists.txt`
````
SET(KOKKOS_SUPER_SCIENCE ON)
````
If not set as a command-line option (cache variable), you must make sure the variable is visible in the top-level scope.
If set in a function, you would need:
````
SET(KOKKOS_SUPER_SCIENCE ON PARENT_SCOPE)
````

### Third-Party Libraries
In much the same way that compiler flags transitively propagate to dependent projects,
modern CMake allows us to propagate dependent libraries.
If Kokkos depends on, e.g. `hwloc` the downstream project will also need to link `hwloc`.
There are three stages in adding a new third-party library (TPL):
* Finding: find the desired library on the system and verify the installation is correct
* Importing: create a CMake target (if not already CMake) that is compatible with `target_link_libraries`
* Exporting: make the desired library visible to downstream projects 

#### Finding TPLs

If the TPL is a modern CMake project, you can just use `find_package` and `target_link_libraries` and you are done.
If finding a TPL that is not a modern CMake project, refer to the `FindHWLOC.cmake` file in `cmake/Modules` for reference. 
You will ususally need to verify the expected headers with `find_path`
````
find_path(TPL_INCLUDE_DIR mytpl.h PATHS "${KOKKOS_MYTPL_DIR}/include")
````
This insures that the library header is in the expected include directory and defines the variable `TPL_INCLUDE_DIR` with a valid path if successful.
Similarly, you can verify a library
````
find_library(TPL_LIBRARIES mytpl PATHS "${KOKKOS_MYTPL_DIR/lib")
````
that then defines the variable `TPL_LIBRARIES` with a valid path if successful.
CMake provides a utility for easily checking if the `find_path` and `find_library` calls were successful.
````
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MYTPL DEFAULT_MSG
                                  MYTPL_INCLUDE_DIR MYTPL_LIBRARIES)
````

#### Importing TPLs

The library must now be made into a CMake target. CMake allows libraries to be added that are built externally as follows:
````
add_library(Kokkos::mytpl UNKNOWN IMPORTED)
````
Importantly, we use a `Kokkos::` namespace to avoid name conflicts and identify this specifically as the version imported by Kokkos.
Because we are importing a non-CMake target, we must populate all the target properties that would have been automatically populated for a CMake target.
````
set_target_properties(Kokkos::mytpl PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${MYTPL_INCLUDE_DIR}"
  IMPORTED_LOCATION "${MYTPL_LIBRARIES}"
)
````

#### Exporting TPLs

Kokkos may now depend on the target `Kokkos::mytpl` as a `PUBLIC` library (remember building and using).
This means that downstream projects must also know about `Kokkos::myptl` - so Kokkos must export them.
In the `KokkosConfig.cmake.in` file, we need to add code like the following:
````
set(MYTPL_LIBRARIES @MYTPL_LIBRARIES@)
set(MYTPL_INCLUDE_DIR @MYTPL_INCLUDE_DIR@)
add_library(Kokkos::mytpl UNKNOWN IMPORTED)
set_target_properties(Kokkos::mytpl PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${MYTPL_INCLUDE_DIR}"
  IMPORTED_LOCATION "${MYTPL_LIBRARIES}"
)
````
If this looks familiar, that's because it is exactly the same code as above for importing the TPL.
Exporting a TPL really just means importing the TPL when Kokkos is loaded by an external project.
We are working on ways to avoid this redundancy.

### The Great TriBITS Compromise

TriBITS was a masterpiece of CMake version 2 before the modern CMake idioms of building and using.
TriBITS greatly limited verbosity of CMake files, handled complicated dependency trees between packages, and handled automatically setting up include and linker paths for dependent libraries.

Kokkos is now used by numerous projects that don't (and won't) depend on TriBITS for their build systems.
Kokkos has to work outside of TriBITS and provide a standard CMake 3+ build system.
At the same time, Kokkos is used by numerous projects that depend on TriBITS and don't (and won't) switch to a standard CMake 3+ build system.

Nevertheless, Kokkos must satisfy all customers for now.
Kokkos is implemented as something TriBITS-like.
Instead of calling functions `TRIBITS_X(...)` it calls a function `KOKKOS_X(...)`.
If TriBITS is available (as in Trilinos), `KOKKOS_X` will just be a thin wrapper around `TRIBITS_X`.
If TriBITS is not available, Kokkos maps `KOKKOS_X` calls to native CMake that complies with CMake 3 idioms.
For the time being, this seems the most sensible way to handle the competing requirements of a standalone modern CMake and TriBITS build system.

##### [LICENSE](https://github.com/kokkos/kokkos/blob/devel/LICENSE)

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Under the terms of Contract DE-NA0003525 with NTESS,
the U.S. Government retains certain rights in this software.
