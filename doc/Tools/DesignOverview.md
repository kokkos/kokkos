# Design Overview

This document covers the high-level design goals and implementation of the Kokkos Tools subsystem.

## History

Much of the design of Kokkos Tools has influences from its history. As Kokkos was being developed, the developers wanted ways to check the performance of Kokkos code in a platform and backend-agnostic way. Suppose Christian had some clever new optimization in Kokkos itself, and wanted to check whether this was (a) brilliant or (b) deeply stupid. As this capability was developed, it became more and more clear that these facilities were useful to developers of Kokkos applications. 

## Goals

One of the early goals of the Kokkos Tools subsystem was to make it really easy to develop simple tools for simple tasks. Specifically, it was seen as deeply obnoxious to require

- A tool to need some version-dependent Kokkos_Make_The_Tool_Work.hpp file to achieve simple goals
- An app developer to recompile their code to change or add a tool
- Any noticeable overhead when a tool was not in use

The first is for the convenience of tool developers, and it is critically important. As of today, we _do_ have header files for some advanced Kokkos Tools functionality in the Kokkos_Profiling_C_Interface.h and Kokkos_Profiling.hpp files, but these are backwards compatible for almost five years. If suddenly TAU needs to know whether it's being used by Kokkos 3.6 or Kokkos 3.6.1, we will have failed in a promise to tool developers. This matters because right now, Tools have to guard profiling layers like OMPT behind an "ENABLE_OMPT" build config option, which defaults to off. Meanwhile, an increasing set of tools either have no build option to disable Kokkos Tools (see: TAU), or at least to default to building Kokkos Tools (see: Caliper). We are also discussing with vendors (see Christian Trott for details) the option of them supporting Kokkos Tools on a first-party basis, something that _won't_ happen if we aren't backwards-compatible.

The second comes from convenience to Kokkos app developers, specifically those who write million line codes that take days to compile. It will be extremely tempting to you to say "ahhhh, if I have such-and-such advanced C++ template metaprogramming in the tool and require a recompile, I can change the world," but if it takes developers six hours to get feedback from a tool, it won't matter. This isn't a caution against doing _anything_ that requires a recompile, the resilience work is an effective example of tooling that requires recompilation, but the tools layer should not require recompilation to change or add tools, if such a design is feasible.

The final goal is simplest, but extremely important. If an app developer sees a slowdown from Kokkos Tools without actually having a tool loaded, they're going to ask us to remove the Kokkos Tools layer, which would be very sad. Making sure that we keep overhead near-zero when not in use is critical. As an example of how this can be complex, consider the name passed to begin\_parallel\_x events. If a user doesn't provide a name, Kokkos generates one. Critically, Kokkos only generates that name _if a tool is loaded_, even the overhead of generating the name would be too much. Further, our current implementation is inefficient, it should only generate the name if the loaded tool implements begin\_parallel\_x. Keeping an eye on overheads is truly critical, our users are awesome and will be friendly about it if they see a slowdown so long as we fix it, but there is likely a breaking point where they ask to compile out the tools layer, which is an absolute nightmare.

## Implementation

The fundamental design in Kokkos Tools is a function-pointer based callback interface. This arises largely out of the low-overhead goal. Consider the following implementation sketch:

```c++
begin_function_ptr begin;
end_function_ptr end;
void parallel_for(const std::string& label, boring_parallelism_stuff stuff) {
  if(begin != nullptr){
      fence();
      (*begin)(label);
  }
  // do boring parallelism things
  if(end != nullptr){
      fence();
      (*end)();
  }
}
```

The 

## A Tool's Perspective: What do Tools Get From Kokkos, What Comes from Other Sources?

Request from Bill Williams, currently a stub section