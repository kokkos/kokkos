# Code Organization

```
“To err is human, to forgive, divine.”
```

All that to say, I apologize for the state of code organization in Kokkos Tools. The state of code organization of Kokkos as a whole is iffy, so don't be terribly surprised if something works in a weird way.

That said, our attempt at organization is the following tree, with parent nodes including the listed child nodes

- Kokkos_Parallel.hpp
  - Kokkos_Tools_Generic.hpp
    - Kokkos_Tuners.hpp
    - Backends (CUDA and the like)
      - **Kokkos_Profiling.hpp/.cpp**
        - **Kokkos_Command_Line_Parsing.hpp/.cpp**
        - *Kokkos_Profiling_Interface.hpp*
          - *Kokkos_Profiling_C_Interface.h*

Note that `Kokkos_Tools_Generic` also directly includes `Kokkos_Profiling.hpp.` I'll be talking through these files individually. Note that all descriptions are aspirational, if something is wrong by all means fix it.

## Kokkos_Profiling_C_Interface.h

In [Design Overview](DesignOverview) I talk about the need for C in the callback signatures this interface supports. To guarantee that, Kokkos has exactly one `.h` file rather than `.hpp`, and we test compiling it with `gcc` (not `g++`). In this file are all the base-level definitions of Kokkos structures. Everything will look weird, because C is weird. I imagine you'll be able to eventually grok the syntax, if only through repetition (or knowledge of C), but remember that things like "oh, I'll add a helpful class method to this struct" is strictly verboten. Also forbidden: namespaces, so the structs being named `Kokkos_Profiling_TheRealName` is a convention we should stick to.

## Kokkos_Profiling_Interface.hpp

So, the above comment about namespaces makes the interface really uncomfortable in C++, everything else in Kokkos is `Kokkos::`, this `Kokkos_` is pretty gross. `Kokkos_Profiling_Interface.hpp` primarily contains `using` statements that re-export these names in namespaces.

## Kokkos_Command_Line_Parsing.[ch]pp

These files contain the command line parsing primitives used by both Core and Tools.

## Kokkos_Profiling.[ch]pp

This files contains the implementations of _calling_ callbacks, setting them, basically anything that doesn't depend on Kokkos Core. This file is intended to be included by backends, for cases when they need to directly call beginParallelFor or the like for some unfortunate reason.

## Kokkos_Tools_Generic.hpp

This is the "outermost" Kokkos Tools file, intended to be included only by Kokkos_Parallel.hpp, not by backends. This is where we do all the shiny tuning and high-level tricks in Kokkos. This file _requires_ visibility into backends, so backends including it leads to a circular dependency.