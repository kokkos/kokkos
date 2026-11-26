# Backwards & Future Compatibility

For a sufficiently clever user, effectively any change we make to Kokkos will be a breaking change. The intent of this document is to make it clear about what does and does not constitute supported use of Kokkos, as well as how Kokkos moves forward.

There is a tension between the freedom to make improvements and backward compatibility.  We are presenting a set of rules that allows the Kokkos Team to make improvements going forward while maintaining a high level of backward compatibility (which avoids frustration and pain on the part of users).  While we do not deliberately set out to break users, we'd like to minimize accidental breakage while still allowing the Kokkos Team a good path forward.

Unless we document otherwise, please:

* Avoid adding into `namespace Kokkos`
* Avoid adding/removing/modifying macros starting with `KOKKOS_`
* Avoid creating/removing/modifying files whose names start with `Kokkos_`

This minimizes the chances that either Kokkos or user code is inadvertently broken by future changes.  

We reserve for the private use of the Kokkos Team:

* Any nested `namespace Impl` inside `namespace Kokkos` (`Kokkos::Impl`, `Kokkos::Experimental::Impl`)
* Any macro starting with `KOKKOS_IMPL_`

These things contain the implementation details of Kokkos.  They are subject to change without notice, either in name or in behavior, even in minor point releases.  They should never be referred to directly in user code.

## API Compatibility

The public supported interface for Kokkos is:

* The top-level `namespace Kokkos`
* Macros starting with `KOKKOS_` (excluding those starting with `KOKKOS_IMPL_`)

While the implementation details may change, the Kokkos Team puts its best effort into limiting changes to either having no functional behavioral differences (apart from bug fixes) or if they must, are changed with compile-time (preferred) or run-time warning and a suitable deprecation period (if possible).

The experimentally support interface for Kokkos is in

* `namespace Kokkos::Experimental` 

This namespace houses experimental features that are not yet ready for prime time.  The feature may be incomplete, and the interface may change between releases.  The intent is to eventually move them into the top level namespace Kokkos (with a suitable deprecation period where they are both in `namespace Kokkos` and `namespace Kokkos::Experimental`).  If you need the functionality (e.g., a new backend), you may use it knowing that you may have to change your code for newer minor releases of Kokkos (and eventually will have to change your code when it moves to the top-level Kokkos namespace). 

## User Defined Macros & Compatibility

User defined macros can be particularly problematic, as they change what is lexically seen by the compiler and do not obey the language scoping rules.  They could interfere with variable names, functions, etc., including private ones used in Kokkos and other libraries.

In order to minimize the risk of collisions, user defined macros should be prefaced with `MYPROJECT_` (or a similar way to disambiguate them) and be in all caps (this informs code readers that macros don't obey the usual syntactic and semantic rules of C++).

## C++ Compatibility

It is the intent of the Kokkos team for minimal C++ support to be one revision behind the latest published C++ standard (they are published every three years starting with C++11).  These releases are generally considered major.  This drives increasing the minimal supported compiler versions, as well as allowing the Kokkos Team to take advantage of new library and language features, as well removing workarounds for older compiler bugs and limitations.  Kokkos may also optionally support later versions of the C++ standard, giving users features should they be compiling in those modes.

## ABI Compatibility

It is expected that Kokkos users recompile their code against new releases or builds of Kokkos.  There are no ABI (Application Binary Interface) guarantees at this level.

An exception to this are Kokkos Tools, where much care is taken to ensure that already compiled older versions of tools work with newer versions of Kokkos.

## Deprecation
Occasionally the Kokkos Team needs to remove things for overall improvements to the Kokkos code base.  When doing so, the Kokkos Team puts in a best effort with deprecation warnings as well as a migratory, evolutionary path (ideally both the deprecated version and the new version co-exist for a suitable period of time) for moving to the improved interface and functionality.

## Headers

Avoid creating/removing/modifying header files whose names start with `Kokkos_`, even under other parts of your project.  Depending on how your build system is set up, this can, for instance, cause the wrong file to be included and lead to many hours of wasted debugging time.

The following are public headers:

    // Core API
    Kokkos_Core.hpp

    Kokkos_Abort.hpp                     // since Kokkos 4.2
    Kokkos_Array.hpp
    Kokkos_Assert.hpp                    // since Kokkos 4.2
    Kokkos_Atomic.hpp
    Kokkos_BitManipulation.hpp           // since Kokkos 4.1
    Kokkos_Clamp.hpp                     // since Kokkos 4.3
    Kokkos_Complex.hpp
    Kokkos_DetectionIdiom.hpp
    Kokkos_Macros.hpp
    Kokkos_MathematicalConstants.hpp
    Kokkos_MathematicalFunctions.hpp
    Kokkos_MinMax.hpp                    // since Kokkos 4.3
    Kokkos_Pair.hpp
    Kokkos_Printf.hpp                    // since Kokkos 4.2
    Kokkos_Profiling_ProfileSection.hpp
    Kokkos_Profiling_ScopedRegion.hpp
    Kokkos_Swap.hpp                      // since Kokkos 4.3
    Kokkos_Timer.hpp

    // Containers API
    Kokkos_Bitset.hpp
    Kokkos_DualView.hpp
    Kokkos_DynamicView.hpp
    Kokkos_DynRankView.hpp
    Kokkos_OffsetView.hpp
    Kokkos_ScatterView.hpp
    Kokkos_UnorderedMap.hpp

    // Algorithms API
    Kokkos_NestedSort.hpp
    Kokkos_Random.hpp
    Kokkos_Sort.cppm
    Kokkos_Sort.hpp
    Kokkos_StdAlgorithms.hpp


If a header is not public, please do not directly `#include` it.  It is not guaranteed to work now or continue to work in the future.  This includes any headers found in subdirectories.

### Other rights the Kokkos Team reserves

* Add new names and entities to `namespace Kokkos`, including but not limited to:
  * Functions (this includes new member functions and overloads to existing functions)
  * Enumerations
  * Namespaces
  * Aliases (`using`, `typedef`, etc.)
  * Classes (`struct`/`class`/`union`)
  * Concepts
  * Variables
* Add new default arguments to functions and templates
* Change return-types of functions in compatible ways (void to anything, etc).
* Make changes to existing interfaces in a fashion that will be backward compatible, if those interfaces are solely used to instantiate types and invoke functions. Implementation details (the primary name of a type, the implementation details for a function callable) may not be depended upon.

### Miscellaneous future proofing

* Avoid taking the address of a function or variable in `namespace Kokkos`
* Avoid `using namespace` declarations (`using namespace Kokkos;`, `using namespace Kokkos::Experimental;`)

